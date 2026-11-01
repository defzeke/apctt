"""tools/make_submission.py: the minimal Stage 1 submission zip (D59, D63, D64) and its validation.

Every source folder is synthetic. The package hash is compared with the kit server's own
verify_package(), the licence rule with tools/check_licenses.py. Models are catalogue ids
(contracts/model-catalog.json) or team-trained weights; the rules are mechanical and the same for
every team (no per-team approvals, exceptions or environments).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apctt_contracts.config import load_config
from apctt_contracts.hashing import package_digest, package_manifest
from apctt_server.state import verify_package
from tests.team_tools.helpers import OTHER_FILES, ROOT, TINY_FILES, load_tool, write_catalog

make_submission = load_tool("make_submission")
make_env = load_tool("make_env")
BASE_ID = make_env.base_id(make_env.BASE_LOCK)
CATALOG_ID = "qwen2.5-1.5b-instruct"
FORBIDDEN_WORDS = ("approv", "exception", "proposal", "env_id")
SUBMISSION_SCHEMA = json.loads((ROOT / "contracts/submission.schema.json").read_text())
# The full MIT text with the team's copyright line (D72: the LICENSE must hold the MIT text).
MIT_TEXT = (
    (ROOT / "licenses/MIT.txt").read_text().replace("<year> <copyright holders>", "2026 Team 01")
)
ADAPTER = "class AssessmentModule:\n    pass\n"
TEAM = "TEAM-01"
ZIP_NAME = f"{TEAM}_submission.zip"
EXPECTED_ENTRIES = [
    "README.md",
    "assets/",
    "assets/rules.json",
    "declaration.json",
    "participant_module/",
    "participant_module/LICENSE",
    "participant_module/THIRD_PARTY_NOTICES.md",
    "participant_module/__init__.py",
    "participant_module/adapter.py",
    "participant_module/rules/",
    "participant_module/rules/criteria.yaml",
    "submission.json",
]


def submission_json(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "submission_version": "apctt-submission-v3",
        "team": TEAM,
        "package_id": "team-01-module",
        "package_version": "1.0.0",
        "base_id": BASE_ID,
        "declared_package_sha256": None,
        "license": "MIT",
        "compute": {"device": "cpu", "precision": "float32"},
        "settings": {"threshold": 0.5},
        "models": [{"catalog_id": CATALOG_ID}],
    }
    data.update(changes)
    return {k: v for k, v in data.items() if v is not ...}


def make_source(tmp_path: Path, **changes: Any) -> Path:
    source = tmp_path / "my_submission"
    module = source / "participant_module"
    (module / "rules").mkdir(parents=True)
    (module / "__init__.py").write_text("")
    (module / "adapter.py").write_text(ADAPTER)
    (module / "rules/criteria.yaml").write_text("criteria: []\n")
    (module / "LICENSE").write_text(MIT_TEXT)
    (module / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices\n\nNo third-party code.\n"
    )
    (source / "assets").mkdir()
    (source / "assets/rules.json").write_text('{"threshold": 0.5}\n')
    (source / "submission.json").write_text(json.dumps(submission_json(**changes), indent=2))
    declaration = {
        "declaration_version": "apctt-declaration-example-v1",
        "team": TEAM,
        "package_id": "team-01-module",
        "package_version": "1.0.0",
        "declared_package_sha256": None,
        "team_training": "no_team_training",
        "training": None,
        "data_sources": [{"source": "Example open data", "license": "CC0-1.0"}],
    }
    (source / "declaration.json").write_text(json.dumps(declaration, indent=2))
    (source / "README.md").write_text("# Team 01 module\n\nRule-based TRL assessment.\n")
    # Things a team folder often holds that are never packed.
    (source / ".venv-worker").mkdir()
    (source / ".venv-worker/pyvenv.cfg").write_text("home = /usr/bin\n")
    (source / "wheels").mkdir()
    (source / "wheels/numpy-2.5.3-py3-none-any.whl").write_bytes(b"PK")
    (source / "requirements.lock").write_text("numpy==2.5.3\n")
    (module / "__pycache__").mkdir()
    (module / "__pycache__/adapter.cpython-312.pyc").write_bytes(b"\0")
    (module / ".DS_Store").write_bytes(b"\0")
    return source


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = make_submission.main([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


def pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, dict[str, Any]]:
    return run([str(tmp_path / "my_submission"), "--output", str(tmp_path / "out"), *extra], capsys)


def text_of(result: dict[str, Any]) -> str:
    return "\n".join(result["errors"] + result["reviews"] + result["notes"] + result["passed"])


def assert_no_approval_wording(result: dict[str, Any]) -> None:
    text = text_of(result).lower()
    for word in FORBIDDEN_WORDS:
        assert word not in text, (word, text)


def test_pack_writes_a_minimal_deterministic_zip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_source(tmp_path)
    code, result = pack(tmp_path, capsys)
    assert code == 0, result
    archive = tmp_path / "out" / ZIP_NAME
    assert result["zip"] == str(archive) and result["result"] == "VALID"
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        assert [i.filename for i in infos] == EXPECTED_ENTRIES
        assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)}
        modes = {i.filename: i.external_attr >> 16 for i in infos}
        assert modes["participant_module/adapter.py"] == 0o100644
        assert modes["assets/"] == 0o040755
        packed = json.loads(bundle.read("submission.json"))
        declaration = json.loads(bundle.read("declaration.json"))
    for left_out in (".venv-worker/", "wheels/", "requirements.lock"):
        assert left_out in text_of(result)
    # The hash in submission.json is the kit's package hash of exactly the packed files.
    digest = package_digest(
        package_manifest(
            [("package", source / "participant_module"), ("assets", source / "assets")]
        )
    )
    assert packed["declared_package_sha256"] == digest == result["declared_package_sha256"]
    assert declaration["declared_package_sha256"] == digest
    assert json.loads((source / "submission.json").read_text())["declared_package_sha256"] == digest
    assert make_env.validate(packed, SUBMISSION_SCHEMA) == []
    # Packing again gives the same bytes.
    first = archive.read_bytes()
    code, again = pack(tmp_path, capsys)
    assert code == 0 and archive.read_bytes() == first
    assert again["zip_sha256"] == result["zip_sha256"]
    assert f"models: catalogue {CATALOG_ID}" in text_of(result)
    assert f"base_id {BASE_ID}" in text_of(result)
    assert_no_approval_wording(result)
    # The organizer's check of the uploaded zip passes.
    code, checked = run(["--check", str(archive)], capsys)
    assert code == 0, checked
    assert checked["declared_package_sha256"] == digest
    # D66: packing and checking remind the team who answers for included third-party code.
    for answer in (result, again, checked):
        assert answer["reminder"] == make_submission.REMINDER
    reminder = " ".join(make_submission.REMINDER.split())
    for phrase in ("pure-Python", "licence or security problem", "legal responsibility",
                   "award is revoked", "D66"):  # fmt: skip
        assert phrase in reminder, phrase


def test_packed_hash_is_what_the_kit_server_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_source(tmp_path)
    code, result = pack(tmp_path, capsys)
    assert code == 0, result
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(result["zip"]) as bundle:
        bundle.extractall(unpacked)
    config = json.loads((ROOT / "config/dev.json").read_text())
    config.update(
        data_dir=str(tmp_path / "data"),
        worker_tmp_dir=str(tmp_path / "tmp"),
        checklist_path=str(ROOT / "contracts/checklist.official.json"),
    )
    config["module"].update(
        package_id="team-01-module",
        entrypoint="participant_module.adapter:AssessmentModule",
        module_root=str(unpacked),
        package_root=str(unpacked / "participant_module"),
        asset_root=str(unpacked / "assets"),
        declared_package_sha256=result["declared_package_sha256"],
        example_only=False,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    state = verify_package(load_config(path, profile="dev"))
    assert state.status == "verified" and state.verified == result["declared_package_sha256"]


def symlink(source: Path) -> None:
    os.symlink("adapter.py", source / "participant_module/link.py")


def write(relative: str, data: str = "") -> Callable[[Path], None]:
    def change(source: Path) -> None:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    return change


def remove(relative: str) -> Callable[[Path], None]:
    return lambda source: (source / relative).unlink()


def edit_json(name: str, **changes: Any) -> Callable[[Path], None]:
    def change(source: Path) -> None:
        data = json.loads((source / name).read_text())
        for key, value in changes.items():
            if value is ...:
                data.pop(key, None)
            else:
                data[key] = value
        (source / name).write_text(json.dumps(data))

    return change


def shipped_public_model(source: Path) -> None:
    write("assets/qwen/model.safetensors", "w")(source)
    edit_json("submission.json", models=[{"catalog_id": CATALOG_ID, "path": "assets/qwen"}])(source)


REFUSALS: dict[str, tuple[Callable[[Path], None], str]] = {
    "symlink": (symlink, "symbolic link"),
    "wheel": (write("assets/vendor/numpy-2.5.3-py3-none-any.whl"), "wheel"),
    "venv": (write("assets/env/pyvenv.cfg", "home = /usr/bin\n"), "virtual environment"),
    "site-packages": (write("assets/lib/site-packages/x.py"), "site-packages"),
    "dist-info": (write("participant_module/vendor/foo-1.0.dist-info/METADATA"), "installed"),
    "numpy copy": (write("participant_module/vendor/numpy/__init__.py"), "numpy"),
    "torch copy": (write("assets/lib/torch/__init__.py"), "torch"),
    "sklearn copy": (write("assets/sklearn/__init__.py"), "scikit-learn"),
    "transformers copy": (write("participant_module/transformers/__init__.py"), "transformers"),
    "so": (write("participant_module/native/_fast.cpython-312-x86_64-linux-gnu.so"), "compiled"),
    "pyd": (write("participant_module/native/_fast.pyd"), "compiled"),
    "dylib": (write("assets/lib/libfast.dylib"), "compiled"),
    "dll": (write("assets/lib/fast.dll"), "compiled"),
    "git": (write("participant_module/.git/HEAD", "ref: x\n"), ".git"),
    "no licence file": (remove("participant_module/LICENSE"), "LICENSE"),
    "gpl": (edit_json("submission.json", license="GPL-3.0-only"), "GPL-3.0-only"),
    # D72 (2026-10-13): the team's own work is MIT only; the message says what changed.
    "apache": (
        edit_json("submission.json", license="Apache-2.0"),
        "Apache-2.0 is no longer accepted for the team's own work",
    ),
    "apache or mit": (edit_json("submission.json", license="MIT OR Apache-2.0"), "must be MIT"),
    # D72: "license": "MIT" with another text in participant_module/LICENSE.
    "apache text under mit": (
        write("participant_module/LICENSE", (ROOT / "licenses/Apache-2.0.txt").read_text()),
        "participant_module/LICENSE holds the Apache License 2.0 text",
    ),
    "short mit text": (
        write("participant_module/LICENSE", "MIT License\n\nPermission is hereby granted...\n"),
        "does not contain the MIT licence text",
    ),
    "no notices": (remove("participant_module/THIRD_PARTY_NOTICES.md"), "THIRD_PARTY_NOTICES.md"),
    "no adapter": (remove("participant_module/adapter.py"), "adapter.py"),
    "no readme": (remove("README.md"), "README.md"),
    "no declaration": (remove("declaration.json"), "declaration.json"),
    "no base_id": (edit_json("submission.json", base_id=...), "base_id"),
    "another base": (edit_json("submission.json", base_id="apctt-base-000000000000"), "base_id"),
    "env_id key": (edit_json("submission.json", env_id="env-" + "0" * 64), "unknown property"),
    "old version": (edit_json("submission.json", submission_version="apctt-submission-v2"), "v3"),
    "entrypoint": (edit_json("submission.json", entrypoint="my.module:Main"), "entrypoint"),
    "wheels key": (edit_json("submission.json", wheels=["x.whl"]), "unknown property"),
    "team mismatch": (edit_json("declaration.json", team="OTHER"), "declaration.json"),
    "unknown catalogue id": (
        edit_json("submission.json", models=[{"catalog_id": "llama-3.2-1b"}]),
        "llama-3.2-1b",
    ),
    "public repo": (
        edit_json("submission.json", models=[{"repo": "Qwen/Qwen2.5-3B", "revision": "a" * 40}]),
        "unknown property",
    ),
    "shipped public model": (shipped_public_model, "not shipped"),
    "base_model outside the catalogue": (
        edit_json(
            "submission.json",
            models=[
                {
                    "name": "ours",
                    "team_trained": True,
                    "path": "assets/rules.json",
                    "base_model": "meta-llama-3",
                }
            ],
        ),
        "base_model",
    ),
    "team-trained without path": (
        edit_json("submission.json", models=[{"name": "ours", "team_trained": True}]),
        "path",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_pack_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str) -> None:
    change, expected = REFUSALS[case]
    source = make_source(tmp_path)
    change(source)
    code, result = pack(tmp_path, capsys)
    assert code == 1, result
    assert result["result"] == "REFUSED"
    assert expected in "\n".join(result["errors"]), result["errors"]
    assert not (tmp_path / "out" / ZIP_NAME).exists() and result["zip"] is None
    if case != "env_id key":  # that message repeats the key the submission itself holds
        assert_no_approval_wording(result)


def test_large_weights_must_be_declared_team_trained(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_submission, "WEIGHT_LIMIT", 1000)
    source = make_source(tmp_path)
    (source / "assets/model.safetensors").write_bytes(b"w" * 2000)
    (source / "assets/extra.bin").write_bytes(b"b" * 2000)
    (source / "assets/small.safetensors").write_bytes(b"s" * 10)
    code, result = pack(tmp_path, capsys)
    assert code == 1
    errors = "\n".join(result["errors"])
    assert "assets/model.safetensors" in errors and "assets/extra.bin" in errors
    assert "team-trained" in errors and "small.safetensors" not in errors
    models = [
        {"name": "my-classifier", "path": "assets/model.safetensors", "team_trained": True},
        {
            "name": "my-adapter",
            "path": "assets/extra.bin",
            "team_trained": True,
            "base_model": CATALOG_ID,  # an adapter of a catalogue model is the team's own work
        },
        {"name": "my-small", "path": "assets/small.safetensors", "team_trained": True},
        {"catalog_id": "bge-base-en-v1.5"},
    ]
    edit_json("submission.json", models=models)(source)
    code, result = pack(tmp_path, capsys)
    assert code == 0, result


def test_catalogue_weights_are_refused_however_they_are_named_or_declared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bundle file with the bytes of a catalogue weight file (its size and sha256 in the
    catalogue's "weights") is refused, even when models[] declares it team-trained or it is
    renamed: public weights are not shipped (D64). Other bytes of the same size are the team's."""
    catalog = write_catalog(tmp_path / "catalog.json")  # tiny-encoder and tiny-mit
    trained = {"name": "my-encoder", "path": "assets/minilm", "team_trained": True,
               "license": "MIT", "base_model": "tiny-encoder"}  # fmt: skip
    source = make_source(tmp_path, models=[trained, {"catalog_id": "tiny-mit"}])
    (source / "assets/minilm").mkdir()
    (source / "assets/minilm/model.safetensors").write_bytes(TINY_FILES["model.safetensors"])
    (source / "participant_module/blob.dat").write_bytes(OTHER_FILES["model.safetensors"])
    code, result = pack(tmp_path, capsys, "--catalog", str(catalog))
    assert code == 1, result
    errors = "\n".join(result["errors"])
    assert (
        "assets/minilm/model.safetensors is the weights of catalogue model tiny-encoder "
        "(model.safetensors); name it by catalog_id, public weights are not shipped (D64)"
    ) in errors
    assert "participant_module/blob.dat is the weights of catalogue model tiny-mit" in errors
    assert_no_approval_wording(result)
    # The same sizes with other bytes: the team's own files.
    own = bytes(b ^ 0xFF for b in TINY_FILES["model.safetensors"])
    (source / "assets/minilm/model.safetensors").write_bytes(own)
    (source / "participant_module/blob.dat").unlink()
    code, result = pack(tmp_path, capsys, "--catalog", str(catalog))
    assert code in (0, 2), result
    # The organizer's check of a zip (intake) applies the same rule.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = rewrite(
            Path(result["zip"]),
            tmp_path / "bad.zip",
            add("assets/copy.bin", TINY_FILES["model.safetensors"]),
        )
    code, checked = run(["--check", str(bad), "--catalog", str(catalog)], capsys)
    assert code == 1
    assert "assets/copy.bin is the weights of catalogue model tiny-encoder" in "\n".join(
        checked["errors"]
    )


def test_an_example_declaration_is_noted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """README step 10: copy the example declaration and delete example_only. A copy that still
    says example_only is packed (the data declaration is reviewed by people) with a NOTE."""
    source = make_source(tmp_path)
    edit_json("declaration.json", example_only=True)(source)
    code, result = pack(tmp_path, capsys)
    assert code == 0, result
    assert len([n for n in result["notes"] if '"example_only": true' in n]) == 1, result["notes"]


def test_review_items_keep_the_zip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = make_source(tmp_path)
    (source / "assets/index.pkl").write_bytes(b"\x80\x04")  # small, undeclared: licence review
    code, result = pack(tmp_path, capsys)
    assert code == 2, result
    assert result["result"] == "VALID_WITH_REVIEW" and result["reviews"]
    assert (tmp_path / "out" / ZIP_NAME).exists()


def test_size_cap_is_fixed_for_every_team(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert make_submission.SIZE_CAP == 2_000_000_000
    assert not hasattr(make_submission, "SIZE_CAP_EXCEPTION")
    monkeypatch.setattr(make_submission, "SIZE_CAP", 20_000)
    source = make_source(tmp_path)
    (source / "assets/blob.dat").write_bytes(os.urandom(40_000))
    code, result = pack(tmp_path, capsys)
    assert code == 1 and "size cap" in "\n".join(result["errors"])
    assert result["cap_bytes"] == 20_000
    assert_no_approval_wording(result)
    tool = [sys.executable, str(ROOT / "tools/make_submission.py")]
    old = subprocess.run(
        [*tool, str(source), "--approval", str(tmp_path / "approval.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert old.returncode == 3 and "unrecognized arguments" in old.stderr


def test_the_tool_has_no_approval_paths() -> None:
    source = (ROOT / "tools/make_submission.py").read_text(encoding="utf-8").lower()
    for word in ("approv", "proposal", "env_id", "exception approved", "size_exception"):
        assert word not in source, word
    schema = json.dumps(SUBMISSION_SCHEMA).lower()
    for word in ("approv", "proposal", "env_id"):
        assert word not in schema, word


def rewrite(source: Path, target: Path, change: Callable[[zipfile.ZipFile], None]) -> Path:
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "w") as new:
        for info in old.infolist():
            if info.filename == "participant_module/adapter.py" and change is tamper:
                new.writestr(info, ADAPTER + "# changed\n")
            else:
                new.writestr(info, old.read(info))
        change(new)
    return target


def tamper(bundle: zipfile.ZipFile) -> None:
    pass


def add(name: str, data: bytes = b"", mode: int = 0o100644) -> Callable[[zipfile.ZipFile], None]:
    def change(bundle: zipfile.ZipFile) -> None:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = mode << 16
        bundle.writestr(info, data)

    return change


ZIP_REFUSALS: dict[str, tuple[Callable[[zipfile.ZipFile], None], str]] = {
    "extra top level": (add("requirements.lock", b"numpy==2.5.3\n"), "top level"),
    "traversal": (add("../evil.py"), "unsafe"),
    "absolute": (add("/etc/evil"), "unsafe"),
    "symlink": (add("participant_module/link.py", b"adapter.py", 0o120777), "symbolic link"),
    "cache": (add("participant_module/__pycache__/a.pyc", b"\0"), "__pycache__"),
    "duplicate": (add("README.md", b"again"), "duplicate"),
    "tampered": (tamper, "hash"),
}


@pytest.mark.parametrize("case", sorted(ZIP_REFUSALS))
def test_check_refuses_bad_zips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    make_source(tmp_path)
    code, result = pack(tmp_path, capsys)
    assert code == 0, result
    change, expected = ZIP_REFUSALS[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate name it writes
        bad = rewrite(Path(result["zip"]), tmp_path / "bad.zip", change)
    code, checked = run(["--check", str(bad)], capsys)
    assert code == 1, checked
    assert expected in "\n".join(checked["errors"]), checked["errors"]


def test_cli_usage_errors(tmp_path: Path) -> None:
    tool = [sys.executable, str(ROOT / "tools/make_submission.py")]

    def code(*argv: str) -> int:
        return subprocess.run(
            [*tool, *argv], capture_output=True, timeout=120, check=False
        ).returncode

    assert code() == 3
    assert code(str(tmp_path / "missing")) == 3
    not_zip = tmp_path / "x.zip"
    not_zip.write_bytes(b"not a zip")
    assert code("--check", str(not_zip)) == 3


def test_cli_pack_and_check(tmp_path: Path) -> None:
    make_source(tmp_path)
    tool = [sys.executable, str(ROOT / "tools/make_submission.py")]
    packed = subprocess.run(
        [*tool, str(tmp_path / "my_submission"), "--output", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert packed.returncode == 0, packed.stdout + packed.stderr
    assert "declared_package_sha256" in packed.stdout and "VALID" in packed.stdout
    checked = subprocess.run(
        [*tool, "--check", str(tmp_path / "out" / ZIP_NAME)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert packed.stdout.count("PASS") >= 5
    for output in (packed.stdout, checked.stdout):  # one line, every time (D66)
        lines = [line for line in output.splitlines() if line.startswith("REMINDER")]
        assert len(lines) == 1 and "award is revoked" in lines[0], output
