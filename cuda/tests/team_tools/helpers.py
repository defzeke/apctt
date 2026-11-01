"""Shared helpers for the team tool tests: load a tool by path, build tiny wheels and locks, serve a
fake of the Hugging Face API and write a small model catalogue.

Every wheel is synthetic (a METADATA, WHEEL and RECORD file plus the given modules) and every
environment is built offline from a local wheelhouse; models come from a local HTTP fake on
127.0.0.1. Nothing here needs network access.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import threading
import zipfile
from collections.abc import Callable, Iterable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
STAMP = (2026, 10, 10, 0, 0, 0)


def load_tool(name: str) -> ModuleType:
    """tools/<name>.py as a module (registered in sys.modules so dataclasses resolve)."""
    if name in sys.modules:
        return sys.modules[name]
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def uv_or_skip() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH (the kit's step 1 installs it)")
    return uv


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def build_wheel(
    directory: Path,
    name: str,
    version: str,
    files: dict[str, str],
    requires: Iterable[str] = (),
    licence: str = "MIT",
) -> Path:
    """A minimal, valid py3-none-any wheel with the given files."""
    directory.mkdir(parents=True, exist_ok=True)
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
    metadata += f"License-Expression: {licence}\n"
    metadata += "".join(f"Requires-Dist: {r}\n" for r in requires)
    wheel = "Wheel-Version: 1.0\nGenerator: apctt-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    contents = {**files, f"{info}/METADATA": metadata, f"{info}/WHEEL": wheel}
    record = "".join(
        f"{path},{_record_hash(text.encode())},{len(text.encode())}\n"
        for path, text in contents.items()
    )
    record += f"{info}/RECORD,,\n"
    path = directory / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, text in [*contents.items(), (f"{info}/RECORD", record)]:
            archive.writestr(zipfile.ZipInfo(member, date_time=STAMP), text)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_line(name: str, version: str, wheel: Path) -> str:
    return f"{name}=={version} --hash=sha256:{sha256_file(wheel)}"


class FakeBase:
    """A wheelhouse that imitates the common base: torch in its cu128 and CPU builds, CUDA-only
    packages, numpy and a single-file module. Every team uses this one base; there are no per-team
    packages (D64)."""

    def __init__(self, root: Path) -> None:
        self.wheelhouse = root / "wheelhouse"
        house = self.wheelhouse

        def module(version: str) -> str:
            return f'__version__ = "{version}"\n'

        self.wheels = {
            ("torch", "2.11.0+cu128"): build_wheel(
                house,
                "torch",
                "2.11.0+cu128",
                {"torch/__init__.py": module("2.11.0+cu128")},
                licence="BSD-3-Clause",
            ),
            ("torch", "2.11.0"): build_wheel(
                house,
                "torch",
                "2.11.0",
                {"torch/__init__.py": module("2.11.0")},
                licence="BSD-3-Clause",
            ),
            ("torch", "2.11.0+cpu"): build_wheel(
                house,
                "torch",
                "2.11.0+cpu",
                {"torch/__init__.py": module("2.11.0+cpu")},
                licence="BSD-3-Clause",
            ),
            ("triton", "3.6.0"): build_wheel(
                house, "triton", "3.6.0", {"triton/__init__.py": module("3.6.0")}
            ),
            ("nvidia-fake-cu12", "1.0"): build_wheel(
                house,
                "nvidia-fake-cu12",
                "1.0",
                {"nvidia/fake/__init__.py": ""},
                licence="LicenseRef-NVIDIA-Proprietary",
            ),
            ("numpy", "2.5.3"): build_wheel(
                house,
                "numpy",
                "2.5.3",
                {"numpy/__init__.py": module("2.5.3")},
                licence="BSD-3-Clause",
            ),
            ("six", "1.17.0"): build_wheel(house, "six", "1.17.0", {"six.py": module("1.17.0")}),
        }
        base = ["numpy==2.5.3", "nvidia-fake-cu12==1.0", "six==1.17.0", "torch==2.11.0+cu128"]
        base.append("triton==3.6.0")
        self.lock = root / "requirements-base.lock"
        lines = [
            "# fake base lock for the tests",
            "--find-links https://download.pytorch.org/whl/cu128/torch/",
            "",
        ]
        for pin in base:
            name, version = pin.split("==")
            digest = sha256_file(self.wheels[(name, version)])
            lines += [f"{name}=={version} \\", f"    --hash=sha256:{digest}", "    # via test"]
        self.lock.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def line(self, name: str, version: str) -> str:
        return lock_line(name, version, self.wheels[(name, version)])


# ------------------------------------------------------------------ fake Hugging Face API

Route = Callable[[str, str, bytes], tuple[int, dict[str, str], bytes]]


@contextlib.contextmanager
def serve(route: Route) -> Iterator[str]:
    """A local HTTP server on 127.0.0.1 that answers every request with route(method, path,
    body) -> (status, headers, body). Yields the base URL; stopped on exit."""

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, headers, payload = route(method, self.path, body)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self._answer("GET")

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            self._answer("POST")

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def as_json(status: int, data: Any, **headers: str) -> tuple[int, dict[str, str], bytes]:
    return status, {"Content-Type": "application/json", **headers}, json.dumps(data).encode()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


TINY_REPO = "test-org/tiny-encoder"
TINY_REVISION = "a" * 40
TINY_FILES = {
    "config.json": b'{"hidden_size": 8}\n',
    "model.safetensors": b"\x00weights\x01" * 100,
    "tokenizer.json": b'{"version": "1.0"}\n',
    "1_Pooling/config.json": b'{"pooling_mode_mean_tokens": true}\n',
    "LICENSE": b"Apache License\nVersion 2.0, January 2004\n",
    "README.md": b"---\nlicense: apache-2.0\n---\n# tiny\n",
    "pytorch_model.bin": b"not listed in the catalogue",
}
OTHER_REPO = "test-org/tiny-mit"
OTHER_REVISION = "b" * 40
OTHER_FILES = {
    "config.json": b'{"hidden_size": 4}\n',
    "model.safetensors": b"\x02mit-weights\x03" * 50,
}
LFS = frozenset({"model.safetensors", "pytorch_model.bin"})


class FakeHub:
    """A fake of the two Hugging Face endpoints the model tools use: the model info API at a
    revision (?blobs=true: sizes, LFS sha256, git blob ids) and /<repo>/resolve/<rev>/<path>.
    corrupt: names served with flipped bytes (same size). calls: every path asked for."""

    def __init__(self, repos: dict[tuple[str, str], dict[str, bytes]] | None = None) -> None:
        self.repos = repos or {
            (TINY_REPO, TINY_REVISION): dict(TINY_FILES),
            (OTHER_REPO, OTHER_REVISION): dict(OTHER_FILES),
        }
        self.corrupt: set[str] = set()
        self.answer_sha: dict[str, str] = {}
        self.extra_siblings: list[dict[str, Any]] = []
        self.calls: list[str] = []

    def route(self, method: str, path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        self.calls.append(path)
        for (repo, revision), files in self.repos.items():
            if path.split("?")[0] == f"/api/models/{repo}/revision/{revision}":
                siblings: list[dict[str, Any]] = []
                for name, data in files.items():
                    entry: dict[str, Any] = {"rfilename": name, "size": len(data)}
                    entry["blobId"] = git_blob(data)
                    if name in LFS:
                        digest = hashlib.sha256(data).hexdigest()
                        entry["lfs"] = {"sha256": digest, "size": len(data), "pointerSize": 133}
                    siblings.append(entry)
                sha = self.answer_sha.get(repo, revision)
                info = {"id": repo, "sha": sha, "siblings": siblings + self.extra_siblings}
                return as_json(200, info)
            prefix = f"/{repo}/resolve/{revision}/"
            if path.startswith(prefix):
                name = unquote(path[len(prefix) :])
                if name not in files:
                    return 404, {}, b"not found"
                data = files[name]
                if name in self.corrupt:  # same size, other bytes: only a hash can tell
                    data = bytes(b ^ 0xFF for b in data)
                return 200, {"Content-Type": "application/octet-stream"}, data
        return 404, {}, b"not found"


# ------------------------------------------------------------------- a small catalogue

BUDGET = {
    "reference_gpu": "RTX 3070 Laptop GPU 8 GiB (D40)",
    "vram_gib": 8,
    "target_peak_gib": 7.0,
    "llm_context_tokens": 8192,
    "encoder_batch": "32 x 512 tokens",
    "overhead_gib": 1.0,
    "measured": False,
}


def weights_of(files: dict[str, bytes], names: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The catalogue's "weights" entry: size and sha256 of every weight file among names."""
    return {
        name: {"size": len(files[name]), "sha256": hashlib.sha256(files[name]).hexdigest()}
        for name in names
        if name.endswith(".safetensors")
    }


def tiny_model(**changes: Any) -> dict[str, Any]:
    """A valid catalogue entry for TINY_REPO (an embedding model with a LICENSE file)."""
    entry: dict[str, Any] = {
        "id": "tiny-encoder",
        "repo": TINY_REPO,
        "revision": TINY_REVISION,
        "license": "Apache-2.0",
        "license_evidence": "card metadata license: apache-2.0; LICENSE file in the repository",
        "license_file_in_repo": True,
        "gated": False,
        "category": "embedding",
        "task": "sentence embeddings",
        "params_b": 0.001,
        "files": ["1_Pooling/config.json", "config.json", "model.safetensors", "tokenizer.json"],
        "download_bytes": sum(
            len(TINY_FILES[n])
            for n in ("1_Pooling/config.json", "config.json", "model.safetensors", "tokenizer.json")
        ),
        "weights": weights_of(TINY_FILES, ["model.safetensors"]),
        "load": {
            "library": "sentence-transformers",
            "class": "SentenceTransformer",
            "dtype": "float32",
            "quantization": None,
            "trust_remote_code": False,
        },
        "vram_estimate_gib": {
            "weights": 0.01,
            "kv_cache_or_activations": 0.01,
            "overhead": 1.0,
            "peak": 1.02,
        },
        "fits_8gb": True,
        "cpu_feasible": "yes: tiny",
        "verified_load_with_base": "not run",
        "use": "tests",
        "notes": "",
    }
    entry.update(changes)
    return entry


def mit_model(**changes: Any) -> dict[str, Any]:
    """A valid catalogue entry for OTHER_REPO (MIT, no licence file in the repository)."""
    entry = tiny_model(
        id="tiny-mit",
        repo=OTHER_REPO,
        revision=OTHER_REVISION,
        license="MIT",
        license_evidence="card metadata license: mit; no LICENSE file in the repository",
        license_file_in_repo=False,
        category="llm",
        files=["config.json", "model.safetensors"],
        download_bytes=sum(len(v) for v in OTHER_FILES.values()),
        weights=weights_of(OTHER_FILES, ["model.safetensors"]),
        load={
            "library": "transformers",
            "class": "AutoModelForCausalLM",
            "dtype": "bfloat16",
            "quantization": None,
            "trust_remote_code": False,
        },
    )
    entry.update(changes)
    return entry


def catalog_data(*models: dict[str, Any], **changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "catalog_version": "apctt-model-catalog-v1",
        "status": "seed",
        "gpu_budget": copy.deepcopy(BUDGET),
        "store_layout": "<APCTT_MODEL_STORE>/<org>/<name>/<revision>/",
        "models": list(models) if models else [tiny_model(), mit_model()],
    }
    data.update(changes)
    return data


def write_catalog(path: Path, *models: dict[str, Any], **changes: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog_data(*models, **changes), indent=2), encoding="utf-8")
    return path
