"""tools/get_models.py: catalogue models into a local store with the organizer's layout (D64).

Downloads come from a local fake of the Hugging Face API (tests/team_tools/helpers.FakeHub) and a
small catalogue written per test; nothing here needs network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.team_tools.helpers import (
    OTHER_FILES,
    OTHER_REPO,
    OTHER_REVISION,
    ROOT,
    TINY_FILES,
    TINY_REPO,
    TINY_REVISION,
    FakeHub,
    load_tool,
    mit_model,
    serve,
    tiny_model,
    write_catalog,
)

get_models = load_tool("get_models")
model_catalog = load_tool("model_catalog")
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


@pytest.fixture
def hub() -> Iterator[tuple[FakeHub, str]]:
    fake = FakeHub()
    with serve(fake.route) as url:
        yield fake, url


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    return write_catalog(tmp_path / "catalog.json")


@pytest.fixture(autouse=True)
def writable(tmp_path: Path) -> Iterator[None]:
    yield
    for directory, _, _ in os.walk(tmp_path):  # let pytest remove read-only stores
        os.chmod(directory, 0o755)


def run(
    catalog: Path, *argv: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any]]:
    code = get_models.main([*argv, "--catalog", str(catalog), "--json"])
    return code, json.loads(capsys.readouterr().out)


def fetch(
    catalog: Path, store: Path, url: str, capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, Any]]:
    return run(catalog, *argv, "--store", str(store), "--hf-endpoint", url, capsys=capsys)


def test_the_kit_ships_the_standard_licence_texts() -> None:
    apache = (ROOT / "licenses/Apache-2.0.txt").read_bytes()
    assert hashlib.sha256(apache).hexdigest() == APACHE_SHA256  # apache.org LICENSE-2.0.txt
    mit = (ROOT / "licenses/MIT.txt").read_text(encoding="utf-8")
    assert mit.startswith("MIT License\n") and "Permission is hereby granted" in mit
    assert "<copyright holders>" in mit


def test_list_shows_sizes_vram_and_cpu(catalog: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(catalog, "--list", capsys=capsys)
    assert code == 0
    rows = {m["id"]: m for m in out["models"]}
    assert set(rows) == {"tiny-encoder", "tiny-mit"}
    assert rows["tiny-encoder"]["download_bytes"] > 0
    assert rows["tiny-encoder"]["vram_peak_gib"] == 1.02
    assert rows["tiny-mit"]["cpu_feasible"].startswith("yes")
    assert out["status"] == "seed"
    assert get_models.main(["--list", "--catalog", str(catalog)]) == 0
    text = capsys.readouterr().out
    assert "tiny-encoder" in text and "1.02 GiB" in text and "973 B" in text
    assert "reference GPU RTX 3070 Laptop GPU 8 GiB (D40)" in text and "estimates" in text
    assert rows["tiny-mit"]["quantization"] is None and rows["tiny-mit"]["measured"] is None
    assert get_models.main(["--list"]) == 0  # the published catalogue
    published = capsys.readouterr().out
    assert "qwen2.5-1.5b-instruct" in published and "3.1 GB" in published
    four_bit = next(line for line in published.splitlines() if "qwen3-4b-instruct-2507-nf4" in line)
    assert "nf4" in four_bit and " no " in four_bit  # a GPU-only 4-bit option (D67)


def test_download_check_reuse_and_read_only(
    tmp_path: Path,
    catalog: Path,
    hub: tuple[FakeHub, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, url = hub
    store = tmp_path / "store"
    code, out = fetch(catalog, store, url, capsys, "tiny-encoder", "tiny-mit", "--read-only")
    assert code == 0, out
    assert out["ok"] is True
    assert out["env"] == {"APCTT_MODEL_STORE": str(store.absolute())}
    tiny = store / TINY_REPO / TINY_REVISION
    other = store / OTHER_REPO / OTHER_REVISION
    results = {m["id"]: m for m in out["models"]}
    assert results["tiny-encoder"]["path"] == str(tiny.absolute())
    assert results["tiny-encoder"]["status"] == "downloaded"
    # Only the listed files, plus the repository's licence file; nothing else.
    listed = {p.relative_to(tiny).as_posix() for p in tiny.rglob("*") if p.is_file()}
    assert listed == {
        ".apctt-model.json",
        "1_Pooling/config.json",
        "LICENSE",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    assert (tiny / "model.safetensors").read_bytes() == TINY_FILES["model.safetensors"]
    assert not any("pytorch_model.bin" in c or "README.md" in c for c in fake.calls)
    record = json.loads((tiny / model_catalog.MARKER).read_text())
    assert record["record_version"] == "apctt-model-store-v1"
    assert (record["id"], record["repo"], record["revision"]) == (
        "tiny-encoder",
        TINY_REPO,
        TINY_REVISION,
    )
    assert record["license"] == "Apache-2.0" and record["licence_added"] is None
    assert record["licence_files"] == ["LICENSE"]
    files = {f["path"]: f for f in record["files"]}
    assert (
        files["model.safetensors"]["sha256"]
        == hashlib.sha256(TINY_FILES["model.safetensors"]).hexdigest()
    )
    assert files["LICENSE"]["origin"] == "repository"
    # The MIT model has no licence file in its repository: the kit's MIT text is added.
    added = (other / "LICENSE").read_text(encoding="utf-8")
    assert OTHER_REPO in added and "Permission is hereby granted" in added
    other_record = json.loads((other / model_catalog.MARKER).read_text())
    assert other_record["licence_added"]["source"] == "licenses/MIT.txt"
    assert {f["path"] for f in other_record["files"]} == {*OTHER_FILES, "LICENSE"}
    # Read-only: files and folders lose their write bits.
    assert not os.access(tiny / "config.json", os.W_OK)
    assert not os.access(tiny, os.W_OK)
    assert record["read_only"] is True

    # A second run reuses both copies: no download.
    fake.calls.clear()
    code, again = fetch(catalog, store, url, capsys, "--all")
    assert code == 0, again
    assert {m["status"] for m in again["models"]} == {"reused"}
    assert not any("/resolve/" in c for c in fake.calls)

    # --check reads the store without the network; --deep re-hashes every file.
    code, checked = run(catalog, "--check", "--store", str(store), "--deep", capsys=capsys)
    assert code == 0, checked
    assert {m["status"] for m in checked["models"]} == {"ok"}
    os.chmod(tiny, 0o755)
    os.chmod(tiny / "model.safetensors", 0o644)
    data = (tiny / "model.safetensors").read_bytes()
    (tiny / "model.safetensors").write_bytes(bytes(b ^ 0xFF for b in data))  # same size
    code, shallow = run(catalog, "--check", "--store", str(store), capsys=capsys)
    assert code == 0, shallow  # sizes only
    code, deep = run(catalog, "--check", "--store", str(store), "--deep", capsys=capsys)
    assert code == 1
    failed = next(m for m in deep["models"] if m["id"] == "tiny-encoder")
    assert failed["status"] == "failed" and "sha256" in " ".join(failed["problems"])
    (tiny / "tokenizer.json").unlink()
    code, missing = run(catalog, "--check", "--store", str(store), capsys=capsys)
    assert code == 1 and "tokenizer.json" in json.dumps(missing)
    # A broken copy is fetched again on the next download.
    code, repaired = fetch(catalog, store, url, capsys, "tiny-encoder")
    assert code == 0 and repaired["models"][0]["status"] == "downloaded", repaired
    # A copy whose recorded weights are not the catalogue's is not a catalogue copy.
    weights = {"model.safetensors": {"size": len(data), "sha256": "0" * 64}}
    changed = write_catalog(tmp_path / "changed.json", tiny_model(weights=weights), mit_model())
    code, stale = run(changed, "--check", "--store", str(store), capsys=capsys)
    assert code == 1 and "differs from the catalogue" in json.dumps(stale), stale


def test_text_output_names_the_store_and_a_loading_snippet(
    tmp_path: Path, catalog: Path, hub: tuple[FakeHub, str]
) -> None:
    _, url = hub
    store = tmp_path / "store"
    argv = [sys.executable, str(ROOT / "tools/get_models.py"), "tiny-mit", "--store", str(store)]
    argv += ["--hf-endpoint", url, "--catalog", str(catalog)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"APCTT_MODEL_STORE={store.absolute()}" in proc.stdout
    assert '"extra_env": {"APCTT_MODEL_STORE"' in proc.stdout
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in proc.stdout
    assert "local_files_only=True" in proc.stdout
    snippet = subprocess.run(
        [sys.executable, str(ROOT / "tools/get_models.py"), "--snippet", "tiny-mit",
         "--catalog", str(catalog)],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert snippet.returncode == 0 and "AutoModelForCausalLM" in snippet.stdout


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda fake: fake.corrupt.add("model.safetensors"), "sha256"),
        (lambda fake: fake.corrupt.add("config.json"), "git blob"),
        (lambda fake: fake.answer_sha.update({TINY_REPO: "f" * 40}), "revision"),
        (
            lambda fake: fake.extra_siblings.append(
                {"rfilename": "../escape.txt", "size": 1, "blobId": "0" * 40}
            ),
            "unsafe",
        ),
        (lambda fake: fake.repos[(TINY_REPO, TINY_REVISION)].pop("tokenizer.json"), "tokenizer"),
        (  # the repository's own record changed: the catalogue's weights hash decides
            lambda fake: fake.repos[(TINY_REPO, TINY_REVISION)].update(
                {"model.safetensors": b"other weights"}
            ),
            "catalogue's weights record",
        ),
    ],
)
def test_a_bad_download_leaves_nothing_in_the_store(
    tmp_path: Path,
    catalog: Path,
    hub: tuple[FakeHub, str],
    capsys: pytest.CaptureFixture[str],
    setup: Any,
    message: str,
) -> None:
    fake, url = hub
    setup(fake)
    store = tmp_path / "store"
    code, out = fetch(catalog, store, url, capsys, "tiny-encoder")
    assert code == 1, out
    model = out["models"][0]
    assert model["status"] == "failed" and message in " ".join(model["problems"]), model
    target = store / TINY_REPO / TINY_REVISION
    assert not target.exists()
    assert not target.with_name(TINY_REVISION + ".partial").exists()


def test_unknown_ids_invalid_catalogues_and_usage(
    tmp_path: Path, catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = str(tmp_path / "store")
    code, out = run(catalog, "no-such-model", "--store", store, capsys=capsys)
    assert code == 2 and "no-such-model" in out["error"]
    bad = write_catalog(tmp_path / "bad.json", tiny_model(gated=True))
    code, out = run(bad, "--list", capsys=capsys)
    assert code == 2 and "gated" in out["error"]
    tool = [sys.executable, str(ROOT / "tools/get_models.py")]

    def exit_code(*argv: str) -> int:
        proc = subprocess.run([*tool, *argv], capture_output=True, timeout=60, check=False)
        return proc.returncode

    assert exit_code() == 3  # nothing to do
    assert exit_code("tiny-encoder", "--catalog", str(catalog)) == 3  # no --store
    assert exit_code("--all", "tiny-encoder", "--store", store) == 3
    assert exit_code("--no-such-option") == 3


def test_a_notice_file_is_not_a_licence_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = {**OTHER_FILES, "NOTICE": b"Copyright the authors.\n"}
    fake = FakeHub({(OTHER_REPO, OTHER_REVISION): files})
    catalog = write_catalog(tmp_path / "catalog.json")
    store = tmp_path / "store"
    with serve(fake.route) as url:
        code, out = fetch(catalog, store, url, capsys, "tiny-mit")
    assert code == 0, out
    folder = store / OTHER_REPO / OTHER_REVISION
    assert (folder / "NOTICE").is_file()  # notices travel with the weights
    assert "Permission is hereby granted" in (folder / "LICENSE").read_text(encoding="utf-8")
    record = json.loads((folder / model_catalog.MARKER).read_text())
    assert record["licence_added"]["source"] == "licenses/MIT.txt"
    (folder / "LICENSE").chmod(0o644)
    folder.chmod(0o755)
    (folder / "LICENSE").unlink()
    code, checked = run(catalog, "--check", "--store", str(store), capsys=capsys)
    assert code == 1 and "LICENSE" in json.dumps(checked)
