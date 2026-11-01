"""tools/make_env.py: the common base environment built the organizer's way (D60, D64).

Every team runs in the same base: there are no per-team overlays or extra packages (D64). Unit tests
cover lock parsing, the install plan and the schema validator. The build tests create real virtual
environments with uv from a local wheelhouse of synthetic wheels (tests/team_tools/helpers.py),
offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.team_tools.helpers import ROOT, FakeBase, load_tool, uv_or_skip

make_env = load_tool("make_env")


# ----------------------------------------------------------------------------- lock files


def test_parse_lock_reads_pins_hashes_and_options() -> None:
    text = (
        "# header\n"
        "--find-links https://download.pytorch.org/whl/cu128/torch/\n"
        "\n"
        "Numpy==2.5.3 \\\n"
        f"    --hash=sha256:{'a' * 64} \\\n"
        f"    --hash=sha256:{'b' * 64}\n"
        "    # via\n"
        "    #   -r requirements-base.in\n"
        f"six==1.17.0 --hash=sha256:{'c' * 64}  # trailing comment\n"
        "torch==2.11.0+cu128 \\\n"
        f"    --hash=sha256:{'d' * 64}\n"
    )
    lock = make_env.parse_lock(text, "test.lock")
    assert lock.options == ("--find-links https://download.pytorch.org/whl/cu128/torch/",)
    assert [(e.name, e.version) for e in lock.entries] == [
        ("numpy", "2.5.3"),
        ("six", "1.17.0"),
        ("torch", "2.11.0+cu128"),
    ]
    assert lock.entries[0].hashes == ("a" * 64, "b" * 64)
    assert lock.entries[0].line == f"numpy==2.5.3 --hash=sha256:{'a' * 64} --hash=sha256:{'b' * 64}"
    assert lock.names() == {"numpy", "six", "torch"}


@pytest.mark.parametrize(
    "line",
    ["numpy>=2", "numpy", "numpy @ https://example.org/numpy.whl", "-e .", "./local.whl"],
)
def test_parse_lock_refuses_anything_but_exact_pins(line: str) -> None:
    with pytest.raises(make_env.InputError, match="exact"):
        make_env.parse_lock(line + "\n", "bad.lock")


def test_base_id_and_flavour_helpers(tmp_path: Path) -> None:
    lock = tmp_path / "a.lock"
    lock.write_text("six==1.17.0\n")
    first = make_env.base_id(lock)
    assert first.startswith("apctt-base-") and len(first) == 23
    lock.write_text("six==1.16.0\n")
    assert make_env.base_id(lock) != first
    for name in (
        "triton",
        "nvidia-cublas-cu12",
        "nvidia-nccl-cu12",
        "cuda-bindings",
        "cuda-toolkit",
    ):
        assert make_env.is_cuda_only(name), name
    for name in ("torch", "numpy", "nvidiax"):
        assert not make_env.is_cuda_only(name), name
    assert make_env.cpu_torch("2.11.0+cu128", "darwin") == ("torch===2.11.0", None)
    assert make_env.cpu_torch("2.11.0+cu128", "linux") == (
        "torch===2.11.0+cpu",
        "https://download.pytorch.org/whl/cpu/torch/",
    )
    assert make_env.choose_flavor("auto", "darwin", "arm64", gpu=False) == "cpu"
    assert make_env.choose_flavor("auto", "linux", "x86_64", gpu=False) == "cpu"
    assert make_env.choose_flavor("auto", "linux", "x86_64", gpu=True) == "cuda"
    assert make_env.choose_flavor("cuda", "darwin", "arm64", gpu=False) == "cuda"


def test_dist_name_reads_installed_metadata_folders() -> None:
    assert make_env.dist_name("accelerate-1.15.0.dist-info") == "accelerate"
    assert make_env.dist_name("torch-2.11.0+cu128.dist-info") == "torch"
    assert make_env.dist_name("nvidia_cublas_cu12-12.8.4.1.dist-info") == "nvidia-cublas-cu12"
    assert make_env.dist_name("PyYAML-6.0.3.dist-info") == "pyyaml"


def test_install_plan_cuda_uses_the_whole_lock_with_hashes(tmp_path: Path) -> None:
    fake = FakeBase(tmp_path)
    lock = make_env.read_lock(fake.lock)
    online = make_env.install_steps(lock, "cuda", "linux", None)
    assert len(online) == 1 and online[0].hashed and not online[0].no_index
    assert online[0].text.startswith("--find-links https://download.pytorch.org/whl/cu128/torch/\n")
    assert "torch==2.11.0+cu128 --hash=sha256:" in online[0].text
    offline = make_env.install_steps(lock, "cuda", "linux", fake.wheelhouse)
    assert offline[0].no_index and offline[0].find_links == (str(fake.wheelhouse),)
    assert "--find-links" not in offline[0].text  # only the local wheelhouse is used


@pytest.mark.parametrize("system", ["darwin", "linux"])
def test_install_plan_cpu_drops_cuda_and_swaps_torch(tmp_path: Path, system: str) -> None:
    lock = make_env.read_lock(FakeBase(tmp_path).lock)
    hashed, torch = make_env.install_steps(lock, "cpu", system, None)
    assert hashed.hashed and "numpy==2.5.3" in hashed.text and "six==1.17.0" in hashed.text
    for name in ("torch", "triton", "nvidia-fake-cu12"):
        assert name + "==" not in hashed.text
    assert "--find-links" not in hashed.text
    assert not torch.hashed
    requirement, links = make_env.cpu_torch("2.11.0+cu128", system)
    assert torch.text == requirement + "\n"
    assert torch.find_links == ((links,) if links else ())


# ------------------------------------------------------------------------ schema subset


def test_the_log_names_an_inline_script_without_its_body(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """README step 5 tells teams to read the last lines of the check: the 25-line check script
    is not echoed on every build and check."""
    script = "import sys\nprint('first')\nprint('second')\n"
    names = json.dumps([f"package-{n}" for n in range(30)])
    out = make_env.run([sys.executable, "-s", "-c", script, names, '["torch"]'], capture=True)
    assert out.splitlines() == ["first", "second"]
    logged = capfd.readouterr().err
    assert f'$ {sys.executable} -s -c <inline script, 3 lines> <list of 30> ["torch"]' in logged
    assert "print(" not in logged and "package-29" not in logged


def test_validator_supports_every_keyword_the_schemas_use() -> None:
    for name in ("submission.schema.json", "model-catalog.schema.json"):
        schema = json.loads((ROOT / "contracts" / name).read_text())
        used: set[str] = set()

        def walk(node: Any, used: set[str] = used) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    used.add(key)
                    if key in ("properties", "$defs"):
                        for child in value.values():
                            walk(child)
                    elif key not in ("enum", "const", "required", "examples", "default"):
                        walk(value)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(schema)
        assert used <= make_env.SUPPORTED_KEYWORDS, (name, used - make_env.SUPPORTED_KEYWORDS)


def test_validator_reports_paths() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": {
            "a": {"type": "array", "items": {"type": "string", "pattern": "^x"}, "maxItems": 2},
            "b": {"type": "integer", "minimum": 1, "maximum": 3},
            "c": {"$ref": "#/$defs/flag"},
            "d": {
                "type": "object",
                "propertyNames": {"pattern": "^[A-Z]+$", "not": {"enum": ["PATH"]}},
                "additionalProperties": {"type": "string"},
            },
            "e": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}, "why": {"type": ["string", "null"]}},
                "if": {"properties": {"on": {"const": True}}},
                "then": {"required": ["why"], "properties": {"why": {"type": "string"}}},
            },
        },
        "$defs": {"flag": {"enum": [True]}},
    }
    assert make_env.validate({"a": ["x1"], "b": 2}, schema) == []
    errors = make_env.validate(
        {
            "a": ["y", "x", "x"],
            "b": True,
            "c": False,
            "z": 1,
            "d": {"PATH": "1", "ok": "2"},
            "e": {"on": True},
        },
        schema,
    )
    text = "\n".join(errors)
    for expected in (
        "$.a: more than 2 items",
        "$.a[0]: does not match",
        "$.b: expected integer",
        "$.c: must be one of",
        "$: unknown property 'z'",
        "$.d: property name 'PATH'",
        "$.d: property name 'ok'",
        "$.e: missing required property 'why'",
    ):
        assert expected in text, (expected, errors)


# ------------------------------------------------------------------------- real builds


def build(fake: FakeBase, tmp_path: Path, *extra: str) -> tuple[int, dict[str, Any]]:
    uv_or_skip()
    argv = [
        sys.executable,
        str(ROOT / "tools/make_env.py"),
        "--base-dir",
        str(tmp_path / "env/base"),
        "--base-lock",
        str(fake.lock),
        "--wheelhouse",
        str(fake.wheelhouse),
        "--python",
        sys.executable,
        "--json",
        *extra,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    data = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
    data["_stderr"] = proc.stderr
    return proc.returncode, data


def check_argv(fake: FakeBase, tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/make_env.py"),
        "--check",
        "--base-dir",
        str(tmp_path / "env/base"),
        "--base-lock",
        str(fake.lock),
        *extra,
    ]


def test_build_cpu_base_and_check(tmp_path: Path) -> None:
    fake = FakeBase(tmp_path)
    code, out = build(fake, tmp_path, "--flavor", "cpu")
    assert code == 0, out
    system = "darwin" if sys.platform == "darwin" else "linux"
    torch_version = make_env.cpu_torch("2.11.0+cu128", system)[0].split("=")[-1]
    worker = Path(out["worker_python"])
    assert worker == tmp_path / "env/base/bin/python" and worker.exists()  # the base itself
    assert set(out) - {"_stderr"} == {"base", "check", "worker_python"}
    assert out["base"]["flavor"] == "cpu" and out["base"]["built"] is True
    assert out["base"]["base_id"] == make_env.base_id(fake.lock)
    check = out["check"]
    assert check["ok"] is True, check
    assert check["worker_python"] == str(worker)
    assert check["imports"]["torch"]["version"] == torch_version
    assert check["imports"]["torch"]["from"] == "base"
    assert check["imports"]["numpy"]["from"] == "base"
    marker = json.loads((tmp_path / "env/base" / make_env.BASE_MARKER).read_text())
    assert marker["flavor"] == "cpu" and marker["lock_sha256"]
    # The worker runs the way the kit server starts it; CUDA-only packages are not in a CPU base.
    probe = subprocess.run(
        [str(worker), "-s", "-c", "import numpy, six, torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        env={"PATH": f"{worker.parent}:/usr/bin:/bin"},
        timeout=60,
        check=True,
    )
    assert probe.stdout.strip() == torch_version
    missing = subprocess.run(
        [str(worker), "-s", "-c", "import triton"], capture_output=True, timeout=60, check=False
    )
    assert missing.returncode != 0

    # A second run reuses the base.
    code, again = build(fake, tmp_path, "--flavor", "cpu")
    assert code == 0, again
    assert again["base"]["built"] is False

    # --check on its own, then with a module the base lacks, then with a base package removed.
    ok = subprocess.run(check_argv(fake, tmp_path), capture_output=True, text=True, timeout=120,
                        check=False)  # fmt: skip
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "check   OK" in ok.stdout
    assert 'worker python for config/my.local.json ("worker": {"python": ...})' in ok.stdout
    absent = subprocess.run(check_argv(fake, tmp_path, "--import", "rank_fake"),
                            capture_output=True, text=True, timeout=120, check=False)  # fmt: skip
    assert absent.returncode == 1 and "import rank_fake failed" in absent.stdout
    site = Path(out["base"]["site_packages"])
    for dist in site.glob("six-*.dist-info"):
        for item in dist.iterdir():
            item.unlink()
        dist.rmdir()
    gone = subprocess.run(check_argv(fake, tmp_path), capture_output=True, text=True,
                          timeout=120, check=False)  # fmt: skip
    assert gone.returncode == 1 and "base package six is not installed" in gone.stdout


def test_build_cuda_base_installs_the_exact_lock_read_only(tmp_path: Path) -> None:
    fake = FakeBase(tmp_path)
    code, out = build(fake, tmp_path, "--flavor", "cuda", "--read-only")
    assert code == 0, out
    assert out["check"]["imports"]["torch"]["version"] == "2.11.0+cu128"
    assert out["check"]["base_writable"] is False
    worker = out["worker_python"]
    probe = subprocess.run(
        [worker, "-s", "-c", "import triton, nvidia.fake, six; print(six.__version__)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert probe.stdout.strip() == "1.17.0"
    base = tmp_path / "env/base"
    assert not os.access(base / make_env.BASE_MARKER, os.W_OK)
    # Another lock for the same base directory is refused, not mixed in.
    fake.lock.write_text(fake.lock.read_text().replace("six==1.17.0", "six==1.17.0 ") + "\n")
    code, refused = build(fake, tmp_path, "--flavor", "cuda")
    assert code == 2 and "another lock" in refused["_stderr"]
    for directory, _, _ in os.walk(base):  # let pytest remove the read-only tree
        os.chmod(directory, 0o755)


def test_the_overlay_and_proposal_options_are_gone(tmp_path: Path) -> None:
    tool = [sys.executable, str(ROOT / "tools/make_env.py")]
    for option in (
        ["--overlay-dir", str(tmp_path / "o")],
        ["--proposal", "p.json"],
        ["--extra-lock", "x.lock"],
        ["--lock-extras"],
        ["--validate"],
        ["--stage2"],
    ):
        proc = subprocess.run(
            [*tool, "--base-dir", str(tmp_path / "b"), *option],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 3 and "unrecognized arguments" in proc.stderr, option
    help_text = subprocess.run([*tool, "--help"], capture_output=True, text=True, timeout=60,
                               check=True).stdout  # fmt: skip
    for word in ("overlay", "proposal", "approv"):
        assert word not in help_text.lower(), word


def test_usage_errors(tmp_path: Path) -> None:
    tool = [sys.executable, str(ROOT / "tools/make_env.py")]
    none = subprocess.run(tool, capture_output=True, text=True, timeout=60, check=False)
    assert none.returncode == 3
    check = subprocess.run([*tool, "--check"], capture_output=True, text=True, timeout=60,
                           check=False)  # fmt: skip
    assert check.returncode == 3
    missing = subprocess.run(
        [*tool, "--check", "--base-dir", str(tmp_path / "nothing")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert missing.returncode == 2 and "apctt-base.json" in missing.stderr
