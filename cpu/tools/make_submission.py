"""Pack and validate the minimal Stage 1 submission zip (decisions D59, D63, D64, D72, D73).

    python tools/make_submission.py SOURCE_DIR [--output DIR] [--json]
    python tools/make_submission.py --check TEAM_submission.zip [--json]

The submission contains no environment: the organizer copies it into its reference kit and runs it
in the one common base environment that every team uses (D64), with the catalogue models in the
read-only model store (APCTT_MODEL_STORE). Each team may submit up to two times, and the latest
valid submission received by the deadline is its final model (D73); valid means accepted by these
rules, checked in full for every submission and the same for every team: this tool is the check
the organizer runs. Packing takes only these entries from SOURCE_DIR and leaves everything else
out:

    participant_module/   code and rules: __init__.py, adapter.py (entry point
                          participant_module.adapter:AssessmentModule), LICENSE (the MIT
                          text, D72), THIRD_PARTY_NOTICES.md (also the licence of any
                          third-party pure-Python source you copied in)
    assets/               only weights and rules the team trained or wrote (may be empty)
    submission.json       team, package_id, package_version, base_id, license, compute, settings,
                          models (contracts/submission.schema.json)
    declaration.json      the data declaration
    README.md             what the module does and how it was built

Caches (__pycache__, *.pyc, .DS_Store, tool caches) are skipped, as the package hash skips them.
The package hash of participant_module/ and assets/ is computed with the kit's own hashing
(apctt_contracts.hashing, as the server does) and written into submission.json (and into
declaration.json when it has declared_package_sha256), in SOURCE_DIR and in the zip. The zip
<team>_submission.zip (in --output, default the parent of SOURCE_DIR) is deterministic: sorted
entries, fixed timestamps and permissions. After packing, the zip is checked as --check does.

Refused: other top-level entries in a zip; symbolic links and special files; compiled extensions
and binaries (.so, .pyd, .dylib, .dll, .exe), wheels (*.whl), virtual environments (pyvenv.cfg),
site-packages, installed package metadata (*.dist-info, *.egg-info), .git and node_modules folders,
and copies of base packages (a folder named like a base package's import name, with __init__.py)
anywhere inside; any file with the bytes of a catalogue weight file (the size and sha256 that
contracts/model-catalog.json records in "weights"), however it is named and even when models[]
declares it team-trained; weight files (.safetensors, .bin, .gguf, .onnx, .pt, .pth, .ckpt, ...)
over 50 MiB that no submission.json models[] entry declares as team-trained (an undeclared weight
file under 50 MiB that looks like a public model is a licence review item, decided by the organizer
after the submission); a models[] entry that names a
model outside the catalogue (contracts/model-catalog.json), ships a catalogue model (public model
weights are never shipped) or names a base_model outside the catalogue; a base_id other than this
kit's common base; a zip over 2,000,000,000 bytes; a package hash that differs from the declared
one; and anything tools/check_licenses.py --bundle denies (the team licence rule of decision D72:
submission.json "license" MIT and a non-empty participant_module/LICENSE; Apache-2.0, accepted
for the team's own work until D72, is refused with a message saying so).

Third-party pure-Python modules may be copied into participant_module/ with their licences in
THIRD_PARTY_NOTICES.md; if a licence or security problem arises later, the team bears the legal
responsibility and the award is revoked (decision D66). Every pack and check prints this as one
REMINDER line (JSON: "reminder").

Exit codes: 0 valid, 1 refused, 2 valid but the licence checker lists review items (decided by the
organizer after the submission), 3 usage or input error. Needs the kit's apctt_contracts (run it
with the kit's .venv python); no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))  # the kit's sources, when the kit is not installed

import check_licenses  # noqa: E402
import make_env  # noqa: E402
import model_catalog  # noqa: E402

from apctt_contracts.hashing import (  # noqa: E402
    EXCLUDED_DIRS,
    EXCLUDED_FILES,
    EXCLUDED_SUFFIXES,
    PackageDigestError,
    package_digest,
    package_manifest,
)

SUBMISSION_SCHEMA = ROOT / "contracts" / "submission.schema.json"
BASE_RECORD = ROOT / "contracts" / "base-environment.json"
ENTRYPOINT = "participant_module.adapter:AssessmentModule"
FOLDERS = ("participant_module", "assets")
FILES = ("submission.json", "declaration.json", "README.md")
MODULE_FILES = ("__init__.py", "adapter.py", "LICENSE", "THIRD_PARTY_NOTICES.md")
WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin", ".gguf", ".onnx", ".pt", ".pth")
WEIGHT_SUFFIXES += (".ckpt", ".h5", ".keras", ".tflite", ".pb", ".joblib", ".pkl", ".npz", ".npy")
WEIGHT_LIMIT = 50 * 1024 * 1024
SIZE_CAP = 2_000_000_000
MAX_UNPACKED = 8_000_000_000
STAMP = (1980, 1, 1, 0, 0, 0)
STORED_SUFFIXES = (*WEIGHT_SUFFIXES, ".zip", ".gz", ".png", ".jpg", ".jpeg")
FORBIDDEN_DIRS = {
    ".git": "a .git folder",
    "node_modules": "a node_modules folder",
    "site-packages": "a site-packages folder (an installed environment)",
    "wheels": "a wheels/ folder (every team runs in the common base; ship no packages)",
}
COMPILED = re.compile(r"\.(so(\.\d+)*|pyd|dylib|dll|exe)$", re.I)
GENERIC_NAMES = frozenset({"tests", "test", "docs", "examples", "benchmarks", "scripts", "bin"})
EXIT_VALID, EXIT_REFUSED, EXIT_REVIEW, EXIT_USAGE = 0, 1, 2, 3
REMINDER = (
    "third-party pure-Python code in your package is your responsibility: if a licence or "
    "security problem arises later, your team bears the legal responsibility and the award is "
    "revoked (D66)"
)


class UsageError(Exception):
    """A bad command line, a missing input or a file that is not a zip (exit code 3)."""


@dataclass
class Outcome:
    errors: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    def ok(self, label: str) -> None:
        self.passed.append(label)


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000
    return f"{size:.1f} GB"


def mib(size: float) -> str:
    """Binary units for the weight-file limit, which the documents state as 50 MiB."""
    text = f"{size / 1024**2:.1f}"
    return f"{text.removesuffix('.0')} MiB"


# ---------------------------------------------------------------------------- base names


def base_modules() -> dict[str, str]:
    """Import name -> base package, from contracts/base-environment.json (else the lock)."""
    names: dict[str, str] = {}
    try:
        record = json.loads(BASE_RECORD.read_text(encoding="utf-8"))
        packages = record.get("packages", {}) if isinstance(record, dict) else {}
    except (OSError, ValueError):
        packages = {}
    if not packages:
        try:
            lock = make_env.read_lock(make_env.BASE_LOCK)
            packages = {e.name: {"modules": [e.name.replace("-", "_")]} for e in lock.entries}
        except make_env.InputError:
            packages = {}
    for package, info in packages.items():
        for module in info.get("modules") or [package.replace("-", "_")]:
            if module not in GENERIC_NAMES:
                names.setdefault(module, package)
    return names


# ------------------------------------------------------------------------- tree checks


def walk_files(root: Path) -> Iterable[tuple[Path, list[str], list[str]]]:
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        yield Path(directory), subdirs, sorted(files)


def is_cache(name: str, directory: bool) -> bool:
    if directory:
        return name in EXCLUDED_DIRS
    return name in EXCLUDED_FILES or name.endswith(EXCLUDED_SUFFIXES)


def covered(path: str, models: list[dict[str, Any]]) -> bool:
    for entry in models:
        target = str(entry.get("path", "")).rstrip("/")
        if (
            entry.get("team_trained") is True
            and target
            and (path == target or path.startswith(target + "/"))
        ):
            return True
    return False


def catalogue_copy(
    path: Path, size: int, weights: dict[tuple[int, str], tuple[str, str]]
) -> tuple[str, str] | None:
    """(model id, file) when path holds the bytes of a catalogue weight file; only files of a
    catalogue weight's size are hashed."""
    if not any(size == known for known, _ in weights):
        return None
    return weights.get((size, file_sha256(path)))


def check_tree(
    root: Path,
    models: list[dict[str, Any]],
    outcome: Outcome,
    packing: bool,
    catalog: model_catalog.Catalog | None = None,
) -> None:
    """participant_module/ and assets/ under root: no environment, no base copies, no catalogue
    weights, no undeclared large weights, no links. Caches are refused in a zip (they are not
    packed)."""
    modules = base_modules()
    weights = catalog.weight_files() if catalog is not None else {}
    errors = len(outcome.errors)
    for folder in FOLDERS:
        top = root / folder
        if top.is_symlink():
            outcome.errors.append(f"{folder}/ is a symbolic link")
            continue
        if not top.is_dir():
            outcome.errors.append(f"{folder}/ is missing")
            continue
        for directory, subdirs, files in walk_files(top):
            relative = directory.relative_to(root).as_posix()
            for name in list(subdirs):
                path = directory / name
                where = f"{relative}/{name}"
                if path.is_symlink():
                    outcome.errors.append(f"{where} is a symbolic link")
                    subdirs.remove(name)
                elif is_cache(name, directory=True):
                    if not packing:
                        outcome.errors.append(f"{where}/ is a cache folder ({name}); remove it")
                    subdirs.remove(name)
                elif name in FORBIDDEN_DIRS:
                    outcome.errors.append(f"{where}/ is {FORBIDDEN_DIRS[name]}")
                    subdirs.remove(name)
                elif name.endswith((".dist-info", ".egg-info")):
                    outcome.errors.append(
                        f"{where}/ is installed package metadata; ship no installed packages"
                    )
                    subdirs.remove(name)
                elif name in modules and (path / "__init__.py").exists():
                    package = modules[name]
                    outcome.errors.append(
                        f"{where}/ looks like a copy of the base package {package} (import name "
                        f"{name}); the base provides it, so remove the copy (rename the folder if "
                        "it is your own code)"
                    )
                    subdirs.remove(name)
            if "pyvenv.cfg" in files:
                outcome.errors.append(f"{relative}/ is a virtual environment (pyvenv.cfg)")
                subdirs.clear()
                continue
            for name in files:
                path = directory / name
                where = f"{relative}/{name}"
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    outcome.errors.append(f"{where} is a symbolic link")
                elif not stat.S_ISREG(info.st_mode):
                    outcome.errors.append(f"{where} is not a regular file")
                elif is_cache(name, directory=False):
                    if not packing:
                        outcome.errors.append(f"{where} is a cache file; remove it")
                elif name.endswith((".whl", ".egg")):
                    outcome.errors.append(f"{where} is a Python wheel; ship no packages")
                elif COMPILED.search(name):
                    outcome.errors.append(
                        f"{where} is a compiled extension or binary; ship only pure-Python "
                        "source (the common base provides the compiled packages)"
                    )
                elif copy := catalogue_copy(path, info.st_size, weights):
                    model_id, original = copy
                    outcome.errors.append(
                        f"{where} is the weights of catalogue model {model_id} ({original}); "
                        "name it by catalog_id, public weights are not shipped (D64): the "
                        "organizer provides catalogue models in APCTT_MODEL_STORE"
                    )
                elif name.lower().endswith(WEIGHT_SUFFIXES) and info.st_size > WEIGHT_LIMIT:
                    if not covered(where, models):
                        outcome.errors.append(
                            f"{where} ({mib(info.st_size)}) is a weight file over "
                            f"{mib(WEIGHT_LIMIT)} that no submission.json models[] entry "
                            'declares as team-trained ("team_trained": true, "path"); public '
                            "model weights are not shipped: name catalogue models by catalog_id "
                            "(contracts/model-catalog.json)"
                        )
    module = root / "participant_module"
    for name in MODULE_FILES:
        target = module / name
        if module.is_dir() and not target.is_file():
            outcome.errors.append(f"participant_module/{name} is missing")
        elif name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "adapter.py") and target.is_file():
            if not target.read_bytes().strip():
                outcome.errors.append(f"participant_module/{name} is empty")
    if len(outcome.errors) == errors:
        outcome.ok(
            "contents: no environment, packages, caches, links, catalogue weights or undeclared "
            "weights"
        )


# ---------------------------------------------------------------------------- JSON files


def read_object(path: Path, outcome: Outcome) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        outcome.errors.append(f"{path.name} is missing")
        return None
    except (OSError, ValueError) as exc:
        outcome.errors.append(f"{path.name} is not valid JSON ({exc})")
        return None
    if not isinstance(data, dict):
        outcome.errors.append(f"{path.name} is not a JSON object")
        return None
    return data


def check_models(
    submission: dict[str, Any], catalog: model_catalog.Catalog, outcome: Outcome
) -> None:
    """models[]: catalogue ids that exist and ship nothing; team-trained weights in the bundle
    whose base_model, when given, is a catalogue model (decision D64)."""
    listed: list[str] = []
    trained: list[str] = []
    errors = len(outcome.errors)
    for index, entry in enumerate(submission.get("models", [])):
        where = f"submission.json models[{index}]"
        if entry.get("team_trained") is True:
            trained.append(str(entry.get("name")))
            base = entry.get("base_model")
            if base is not None and catalog.get(base) is None:
                outcome.errors.append(
                    f"{where}: base_model {base} is not in the model catalogue; fine-tunes and "
                    "adapters may be made only from catalogue models (D64)"
                )
            continue
        model_id = entry.get("catalog_id")
        if catalog.get(model_id) is None:
            outcome.errors.append(
                f"{where}: catalog_id {model_id} is not in the model catalogue "
                "(python tools/get_models.py --list); other public models are not available "
                "offline and are refused (D64)"
            )
        elif entry.get("path"):
            outcome.errors.append(
                f"{where}: {model_id} has a path, but public model weights are not shipped: the "
                "organizer provides catalogue models in APCTT_MODEL_STORE"
            )
        listed.append(str(model_id))
    if len(outcome.errors) == errors:
        parts = [f"catalogue {', '.join(listed)}" if listed else "no catalogue model"]
        parts += [f"team-trained {', '.join(trained)}"] if trained else []
        outcome.ok("models: " + "; ".join(parts))


def check_documents(
    root: Path, outcome: Outcome, base_id: str, catalog: model_catalog.Catalog
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    submission = read_object(root / "submission.json", outcome)
    declaration = read_object(root / "declaration.json", outcome)
    readme = root / "README.md"
    if not readme.is_file() or not readme.read_bytes().strip():
        outcome.errors.append("README.md is missing or empty")
    if submission is None:
        return None, declaration
    schema = json.loads(SUBMISSION_SCHEMA.read_text(encoding="utf-8"))
    problems = make_env.validate(submission, schema)
    outcome.errors += [f"submission.json {p}" for p in problems]
    if problems:
        return submission, declaration
    if submission["base_id"] != base_id:
        outcome.errors.append(
            f"submission.json base_id {submission['base_id']} is not this kit's common base "
            f"{base_id}: build and test against the kit's base (tools/make_env.py)"
        )
    else:
        outcome.ok(f"submission.json: team {submission['team']}, base_id {base_id}")
    check_models(submission, catalog, outcome)
    if declaration is not None:
        if declaration.get("example_only") is True:
            outcome.notes.append(
                'declaration.json still says "example_only": true, so it is the kit\'s example: '
                "delete example_only and declare your own team, package and data sources "
                "(README.participant.md step 10)"
            )
        for key in ("team", "package_id", "package_version"):
            if key in declaration and declaration[key] != submission[key]:
                outcome.errors.append(
                    f'declaration.json "{key}" is {declaration[key]!r}, submission.json says '
                    f"{submission[key]!r}"
                )
    return submission, declaration


def kit_base_id() -> str:
    """base_id of this kit's common base (requirements-base.lock)."""
    return str(make_env.base_id(make_env.BASE_LOCK))


def check_licence(root: Path, outcome: Outcome, catalog: Path) -> None:
    policy = check_licenses.load_policy(check_licenses.DEFAULT_POLICY, catalog)
    report = check_licenses.Report(policy, {"bundle": str(root)})
    check_licenses.check_bundle(root, policy, report)
    for item in report.sorted_items():
        reasons = "; ".join(item.reasons)
        label = f"licence {item.kind} {item.name}{' ' + item.version if item.version else ''}"
        if item.status == check_licenses.DENIED:
            outcome.errors.append(f"{label}: denied: {reasons}")
        elif item.status == check_licenses.REVIEW:
            outcome.reviews.append(f"{label}: review: {reasons}")
    team = next(i for i in report.items if i.kind == "team")
    if team.status == check_licenses.ALLOWED:
        outcome.ok(f"licence: {team.license}, participant_module/LICENSE present")


def digest_of(root: Path) -> str:
    roots = [("package", root / "participant_module"), ("assets", root / "assets")]
    return package_digest(package_manifest(roots))


# --------------------------------------------------------------------------------- packing


def collect(source: Path) -> list[tuple[str, Path]]:
    """(zip name, file) of everything packed, sorted; folders get their own entries."""
    entries: list[tuple[str, Path]] = []
    for folder in FOLDERS:
        top = source / folder
        entries.append((folder + "/", top))
        for directory, subdirs, files in walk_files(top):
            subdirs[:] = [d for d in subdirs if not is_cache(d, directory=True)]
            for name in subdirs:
                path = directory / name
                entries.append((path.relative_to(source).as_posix() + "/", path))
            for name in files:
                if not is_cache(name, directory=False):
                    path = directory / name
                    entries.append((path.relative_to(source).as_posix(), path))
    entries += [(name, source / name) for name in FILES]
    return sorted(entries)


def zip_info(name: str, directory: bool, size: int = 0) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.create_system = 3
    if directory:
        info.external_attr = (0o040755 << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.external_attr = 0o100644 << 16
        stored = name.lower().endswith(STORED_SUFFIXES)
        info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        info.file_size = size
    return info


def write_zip(target: Path, entries: list[tuple[str, Path]], replaced: dict[str, bytes]) -> None:
    partial = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(partial, "w", allowZip64=True) as bundle:
        for name, path in entries:
            if name.endswith("/"):
                bundle.writestr(zip_info(name, True), b"")
            elif name in replaced:
                data = replaced[name]
                bundle.writestr(zip_info(name, False, len(data)), data)
            else:
                size = path.stat().st_size
                with path.open("rb") as src, bundle.open(zip_info(name, False, size), "w") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    partial.replace(target)


def filled(path: Path, digest: str) -> bytes:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["declared_package_sha256"] = digest
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def pack(
    source: Path, output: Path, *, base_id: str | None = None, catalog_path: Path | None = None
) -> dict[str, Any]:
    """Pack SOURCE_DIR into <output>/<team>_submission.zip and check the zip."""
    outcome = Outcome()
    base_id = base_id or kit_base_id()
    catalog_path = catalog_path or model_catalog.CATALOG
    catalog = load_catalog(catalog_path)
    left_out = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in source.iterdir()
        if p.name not in (*FOLDERS, *FILES) and p.name != ".DS_Store"
    )
    if left_out:
        outcome.notes.append(
            "not packed (the submission holds no environment; decision D59): " + ", ".join(left_out)
        )
    submission, declaration = check_documents(source, outcome, base_id, catalog)
    models = list(submission.get("models", [])) if submission else []
    check_tree(source, models, outcome, packing=True, catalog=catalog)
    result: dict[str, Any] = {"source": str(source), "zip": None}
    if outcome.errors or submission is None:
        return finish(result, outcome, None)
    try:
        digest = digest_of(source)
    except (PackageDigestError, OSError) as exc:
        outcome.errors.append(f"the package hash cannot be computed: {exc}")
        return finish(result, outcome, None)
    replaced = {"submission.json": filled(source / "submission.json", digest)}
    if declaration is not None and "declared_package_sha256" in declaration:
        replaced["declaration.json"] = filled(source / "declaration.json", digest)
    for name, data in replaced.items():
        (source / name).write_bytes(data)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{submission['team']}_submission.zip"
    write_zip(target, collect(source), replaced)
    checked = check_zip(target, base_id=base_id, catalog_path=catalog_path)
    checked["notes"] = outcome.notes + [n for n in checked["notes"] if n not in outcome.notes]
    checked["source"] = str(source)
    if checked["exit_code"] == EXIT_REFUSED:
        target.unlink()
        checked["zip"] = None
        checked["notes"].append(f"{target.name} was refused and removed")
    return checked


# --------------------------------------------------------------------------------- checking


def unsafe(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        not name
        or name.startswith(("/", "\\"))
        or "\\" in name
        or "\0" in name
        or re.match(r"^[A-Za-z]:", name) is not None
        or any(part in ("..", ".") for part in pure.parts)
    )


def check_entries(bundle: zipfile.ZipFile, outcome: Outcome) -> None:
    seen: set[str] = set()
    lowered: set[str] = set()
    total = 0
    tops: set[str] = set()
    for info in bundle.infolist():
        name = info.filename
        if unsafe(name):
            outcome.errors.append(f"{name!r}: unsafe path (absolute, .., or backslash)")
            continue
        if name in seen or name.lower() in lowered:
            outcome.errors.append(f"{name}: duplicate entry")
        seen.add(name)
        lowered.add(name.lower())
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            outcome.errors.append(f"{name}: symbolic link")
        top = name.split("/", 1)[0]
        tops.add(top + ("/" if "/" in name else ""))
        if top not in (*FOLDERS, *FILES) or (top in FOLDERS and "/" not in name):
            outcome.errors.append(f"{name}: not allowed at the top level of the zip")
        parts = name.rstrip("/").split("/")
        caches = [p for p in parts[:-1] if is_cache(p, directory=True)]
        if caches or (not name.endswith("/") and is_cache(parts[-1], directory=False)):
            what = caches[0] if caches else parts[-1]
            outcome.errors.append(f"{name}: cache entry ({what}); caches are not submitted")
        total += info.file_size
    if total > MAX_UNPACKED:
        outcome.errors.append(f"the zip unpacks to {human(total)}, more than {human(MAX_UNPACKED)}")
    missing = [f + "/" for f in FOLDERS if f + "/" not in tops] + [
        f for f in FILES if f not in tops
    ]
    if missing:
        outcome.errors.append("missing at the top level: " + ", ".join(missing))
    if not outcome.errors:
        outcome.ok(
            "layout: participant_module/, assets/, submission.json, declaration.json, README.md"
        )


def extract(bundle: zipfile.ZipFile, target: Path) -> None:
    for info in bundle.infolist():
        path = target / info.filename
        if info.filename.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(info) as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> model_catalog.Catalog:
    try:
        return model_catalog.load(path)
    except model_catalog.CatalogError as exc:
        raise UsageError(f"the model catalogue is not valid: {exc}") from None


def check_zip(
    archive: Path, *, base_id: str | None = None, catalog_path: Path | None = None
) -> dict[str, Any]:
    """Every rule of a submission zip; the organizer runs this same check on each submission."""
    outcome = Outcome()
    base_id = base_id or kit_base_id()
    catalog_path = catalog_path or model_catalog.CATALOG
    catalog = load_catalog(catalog_path)
    size = archive.stat().st_size
    limit = SIZE_CAP
    result: dict[str, Any] = {
        "zip": str(archive),
        "zip_sha256": file_sha256(archive),
        "size_bytes": size,
        "cap_bytes": limit,
    }
    if size > limit:
        outcome.errors.append(
            f"the zip is {human(size)} ({size} bytes), over the size cap of {human(limit)} "
            f"({limit} bytes), which is the same for every team"
        )
    else:
        outcome.ok(f"size: {human(size)} <= {human(limit)}")
    try:
        bundle = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise UsageError(f"{archive}: not a readable zip file ({exc})") from None
    with bundle, tempfile.TemporaryDirectory(prefix="apctt-submission-") as scratch:
        check_entries(bundle, outcome)
        if outcome.errors:
            return finish(result, outcome, None)
        root = Path(scratch)
        try:
            extract(bundle, root)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            outcome.errors.append(f"the zip cannot be unpacked: {exc}")
            return finish(result, outcome, None)
        submission, declaration = check_documents(root, outcome, base_id, catalog)
        models = list(submission.get("models", [])) if submission else []
        check_tree(root, models, outcome, packing=False, catalog=catalog)
        digest = None
        try:
            digest = digest_of(root)
        except (PackageDigestError, OSError) as exc:
            outcome.errors.append(f"the package hash cannot be computed: {exc}")
        if digest and submission is not None:
            declared = submission.get("declared_package_sha256")
            if declared != digest:
                outcome.errors.append(
                    f"package hash {digest} differs from submission.json "
                    f"declared_package_sha256 {declared}"
                )
            else:
                outcome.ok(f"package hash matches submission.json: {digest}")
            if declaration is not None and declaration.get("declared_package_sha256") not in (
                None,
                digest,
            ):
                outcome.errors.append("declaration.json declared_package_sha256 differs")
        check_licence(root, outcome, catalog_path)
        return finish(result, outcome, digest)


def finish(result: dict[str, Any], outcome: Outcome, digest: str | None) -> dict[str, Any]:
    code = EXIT_REFUSED if outcome.errors else EXIT_REVIEW if outcome.reviews else EXIT_VALID
    status = {EXIT_VALID: "VALID", EXIT_REVIEW: "VALID_WITH_REVIEW", EXIT_REFUSED: "REFUSED"}
    result.update(
        {
            "declared_package_sha256": digest,
            "passed": outcome.passed,
            "errors": outcome.errors,
            "reviews": outcome.reviews,
            "notes": outcome.notes,
            "result": status[code],
            "exit_code": code,
        }
    )
    return result


# ------------------------------------------------------------------------------------ CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def show(result: dict[str, Any]) -> None:
    if result.get("zip"):
        print(f"zip                      {result['zip']}")
        print(f"zip_sha256               {result.get('zip_sha256')}")
        print(f"size                     {human(result.get('size_bytes', 0))}")
    print(f"declared_package_sha256  {result['declared_package_sha256']}")
    for line in result["passed"]:
        print(f"PASS    {line}")
    for line in result["errors"]:
        print(f"ERROR   {line}")
    for line in result["reviews"]:
        print(f"REVIEW  {line}")
    for line in result["notes"]:
        print(f"NOTE    {line}")
    print(f"REMINDER {result['reminder']}")
    print(f"result: {result['result']} (exit {result['exit_code']})")


def main(argv: list[str] | None = None) -> int:
    parser = Parser(
        prog="make_submission.py",
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Exit codes: 0 valid, 1 refused, 2 valid with licence review items, 3 usage error.",
    )
    parser.add_argument("source", nargs="?", type=Path, help="the folder to pack")
    parser.add_argument("--check", type=Path, metavar="ZIP", help="validate an existing zip")
    parser.add_argument("--output", type=Path, help="where to write <team>_submission.zip")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--base-lock", type=Path, default=make_env.BASE_LOCK,
                        help=argparse.SUPPRESS)  # fmt: skip
    parser.add_argument("--catalog", type=Path, default=model_catalog.CATALOG,
                        help=argparse.SUPPRESS)  # fmt: skip
    args = parser.parse_args(argv)
    try:
        if (args.source is None) == (args.check is None):
            raise UsageError("give either SOURCE_DIR or --check ZIP")
        base_id = str(make_env.base_id(args.base_lock))
        if args.check is not None:
            if not args.check.is_file():
                raise UsageError(f"{args.check}: not a file")
            result = check_zip(args.check.absolute(), base_id=base_id, catalog_path=args.catalog)
        else:
            source = args.source.absolute()
            if not source.is_dir():
                raise UsageError(f"{args.source}: not a directory")
            output = (args.output or source.parent).absolute()
            result = pack(source, output, base_id=base_id, catalog_path=args.catalog)
    except UsageError as exc:
        print(f"make_submission.py: {exc}", file=sys.stderr)
        return EXIT_USAGE
    result["reminder"] = REMINDER  # D66, on every pack and check
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        show(result)
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
