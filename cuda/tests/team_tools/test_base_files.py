"""The common base environment files (decision D60) agree with each other and with the policy.

requirements-base.in -> requirements-base.lock (hashed, Linux x86_64, CPython 3.12) ->
constraints-base.txt (plain pins) and contracts/base-environment.json (identity, import names,
measured sizes and licence summary). No network: the files are read as committed.
"""

from __future__ import annotations

import hashlib
import json
import re

from tests.team_tools.helpers import ROOT, load_tool

make_env = load_tool("make_env")
check_licenses = load_tool("check_licenses")

LOCK = ROOT / "requirements-base.lock"
CONSTRAINTS = ROOT / "constraints-base.txt"
RECORD = json.loads((ROOT / "contracts/base-environment.json").read_text(encoding="utf-8"))
CURATED = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "sentence-transformers",
    "accelerate",
    "peft",
    "pypdf",
    "pdfplumber",
    "pypdfium2",
    "onnxruntime",
    "rapidocr-onnxruntime",
    "opencv-python-headless",
    "fastapi",
    "uvicorn",
    "jinja2",
    "python-multipart",
    "httpx",
    "bitsandbytes",  # D67: the catalogue's 4-bit (nf4) LLMs load with it on the GPU
)


def pins(text: str) -> dict[str, str]:
    return {
        m.group(1): m.group(2) for m in re.finditer(r"^([a-z0-9][a-z0-9._-]*)==(\S+)", text, re.M)
    }


def test_lock_is_complete_hashed_and_uses_only_the_torch_page() -> None:
    lock = make_env.read_lock(LOCK)
    assert lock.options == ("--find-links https://download.pytorch.org/whl/cu128/torch/",)
    assert "--index-url" not in LOCK.read_text() and "--extra-index-url" not in LOCK.read_text()
    assert all(entry.hashes for entry in lock.entries)
    versions = {e.name: e.version for e in lock.entries}
    assert versions["torch"] == "2.11.0+cu128"
    for name in CURATED:
        assert name in versions, name
    assert "opencv-python" not in versions  # overrides-base.txt: the headless build provides cv2
    assert len(versions) == len(lock.entries)
    head = LOCK.read_text().splitlines()[:12]
    assert head[0].startswith("# APCTT common base environment lock")
    assert any("--no-deps --require-hashes" in line for line in head)


def test_lock_keeps_the_d52_closure_and_the_kit_server_versions() -> None:
    versions = {e.name: e.version for e in make_env.read_lock(LOCK).entries}
    # Every exact pin of the input is kept; the D52 closure is pinned there (the repository's
    # release tests compare those pins with requirements-cuda-example.lock).
    pinned = pins((ROOT / "requirements-base.in").read_text())
    for name, version in pinned.items():
        assert versions.get(name) == version, name
    policy = json.loads((ROOT / "contracts/license-policy.json").read_text())
    d52 = policy["organizer_base"]["packages"]
    assert len(d52) == 29 and set(d52) <= set(pinned)
    server = pins((ROOT / "requirements-dev.lock").read_text())
    for name in ("fastapi", "uvicorn", "pydantic", "pypdf", "httpx", "starlette"):
        assert versions[name] == server[name], name


def test_constraints_are_plain_pins_of_the_lock() -> None:
    text = CONSTRAINTS.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# APCTT common base environment constraints")
    body = [line for line in lines if line and not line.startswith("#")]
    for line in body:
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*==[A-Za-z0-9.+!_-]+", line), line
    lock = {e.name: e.version for e in make_env.read_lock(LOCK).entries}
    assert dict(line.split("==") for line in body) == lock


def test_input_lists_the_curated_set_and_overrides() -> None:
    names = set(pins((ROOT / "requirements-base.in").read_text()))
    names |= {
        line.strip()
        for line in (ROOT / "requirements-base.in").read_text().splitlines()
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", line.strip())
    }
    for name in CURATED:
        assert name in names, name
    overrides = [
        line
        for line in (ROOT / "overrides-base.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert overrides == ["opencv-python ; sys_platform == 'never'"]


def test_record_matches_the_files() -> None:
    assert RECORD["record_version"] == "apctt-base-environment-v1"
    assert RECORD["base_id"] == make_env.base_id(LOCK)
    for key, entry in RECORD["files"].items():
        data = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], key
    lock = {e.name: e.version for e in make_env.read_lock(LOCK).entries}
    assert {n: p["version"] for n, p in RECORD["packages"].items()} == lock
    # Import names, for the bundle validator; cuda-toolkit is a metapackage without modules.
    empty = [name for name, package in RECORD["packages"].items() if not package["modules"]]
    assert empty == ["cuda-toolkit"], empty
    assert "torch" in RECORD["packages"]["torch"]["modules"]
    assert "cv2" in RECORD["packages"]["opencv-python-headless"]["modules"]
    assert "sklearn" in RECORD["packages"]["scikit-learn"]["modules"]
    assert RECORD["cuda_only_patterns"] == list(make_env.CUDA_ONLY)
    assert RECORD["torch"]["cuda"]["find_links"] == make_env.TORCH_CUDA_FIND_LINKS
    for system in ("darwin", "linux"):
        requirement, links = make_env.cpu_torch(lock["torch"], system)
        assert RECORD["torch"]["cpu_dev"][system] == {
            "requirement": requirement,
            "find_links": links,
        }


def test_measured_sizes_and_licences() -> None:
    measured = RECORD["measured"]
    assert measured["lock_sha256"] == RECORD["files"]["lock"]["sha256"]
    assert measured["wheel_count"] == len(RECORD["packages"])
    assert 0 < measured["wheel_bytes_other"] < measured["wheel_bytes"]
    assert measured["installed_bytes"] < measured["installed_bytes_with_bytecode"]
    licences = measured["licences"]
    assert licences["denied_other"] == []  # D53: nothing outside the D52 closure is denied
    assert licences["wheels"] == measured["wheel_count"]
    policy = check_licenses.load_policy(check_licenses.DEFAULT_POLICY)
    for name in licences["denied_d52_installed_by_organizer"]:
        assert check_licenses.is_base(name, "", policy, "bundle"), name


def test_no_policy_name_rule_denies_a_base_package() -> None:
    policy = check_licenses.load_policy(check_licenses.DEFAULT_POLICY)
    for entry in make_env.read_lock(LOCK).entries:
        item = check_licenses.Item("web", "lock", entry.name, entry.version)
        check_licenses.apply_rules(item, "pypi", policy)
        assert item.status != check_licenses.DENIED, (entry.name, item.reasons)
