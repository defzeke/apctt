"""Build the common base environment the way the organizer builds it (decisions D60 and D64).

    python tools/make_env.py --base-dir DIR [--flavor auto|cuda|cpu] [--python PYTHON]
                             [--wheelhouse DIR] [--read-only] [--import MODULE ...] [--uv UV]
                             [--json]
    python tools/make_env.py --check --base-dir DIR [--import MODULE ...] [--json]

Every team runs in the same fixed environment: the common base of requirements-base.lock. There are
no per-team packages (D64); public models come from the model catalogue (tools/get_models.py).
The base is a virtual environment with every package of the lock, installed --no-deps
--require-hashes --compile-bytecode. Flavour cuda installs the lock exactly (Linux x86_64, torch
2.11.0+cu128 and the NVIDIA CUDA packages, about 4 GB). Flavour cpu is the development base for
macOS, or Linux and WSL2 without an NVIDIA GPU: the same versions without the CUDA-only packages,
and the CPU build of torch 2.11.0 (PyPI on macOS, the PyTorch CPU page on Linux). auto picks cuda on
Linux x86_64 when nvidia-smi is found, else cpu. A base directory is built once and reused while
its lock and flavour match (marker file apctt-base.json). --read-only makes it read-only, as on the
organizer's machines.

The worker python is the base's own interpreter, <base-dir>/bin/python: put it into
config/my.local.json "worker": {"python": ...}. --check runs it the way the kit server does
(python -s, PATH limited to the interpreter's folder, /usr/bin and /bin) and verifies that torch
and numpy (and every --import) import from the base and that every base package is installed at
the lock's version.

Network is needed only while building (none with --wheelhouse, a folder of wheels); evaluation runs
offline. Exit codes: 0 done, 1 check failed, 2 invalid lock or base directory, 3 usage or tool
error. Standard library and uv only; Linux, WSL2 or macOS (not native Windows).
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
BASE_LOCK = ROOT / "requirements-base.lock"

PYTHON_VERSION = "3.12"
EVENT_PLATFORM = "x86_64-manylinux_2_28"
EVENT_PYTHON = "/usr/bin/python3.12"
TORCH_CUDA_FIND_LINKS = "https://download.pytorch.org/whl/cu128/torch/"
TORCH_CPU_FIND_LINKS = "https://download.pytorch.org/whl/cpu/torch/"
CUDA_ONLY = ("triton", "nvidia-*", "cuda-*")
BASE_MARKER = "apctt-base.json"
BUILDING = ".apctt-building"
DEFAULT_IMPORTS = ("torch", "numpy")

EXIT_OK, EXIT_CHECK, EXIT_INVALID, EXIT_TOOL = 0, 1, 2, 3
LOCK_OPTIONS = ("--find-links", "-f", "--index-url", "-i", "--extra-index-url", "--no-index")

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema", "$id", "$defs", "$ref", "$comment", "title", "description", "examples",
        "default", "type", "const", "enum", "pattern", "minLength", "maxLength", "minimum",
        "maximum", "exclusiveMinimum", "exclusiveMaximum", "items", "minItems", "maxItems",
        "uniqueItems", "properties", "required", "additionalProperties", "propertyNames", "not",
        "anyOf", "if", "then", "else",
    }
)  # fmt: skip


class InputError(Exception):
    """An invalid lock file or base directory (exit code 2)."""


class ToolError(Exception):
    """A missing tool, a failed uv command or an unusable interpreter (exit code 3)."""


# ------------------------------------------------------------------------------ lock files


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9.+!_-]+)"
    r"\s*(?P<marker>;[^-]*?)?(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})*)\s*$"
)


@dataclass(frozen=True)
class LockEntry:
    name: str
    version: str
    hashes: tuple[str, ...]
    marker: str = ""

    @property
    def line(self) -> str:
        marker = f" {self.marker}" if self.marker else ""
        hashes = "".join(f" --hash=sha256:{h}" for h in self.hashes)
        return f"{self.name}=={self.version}{marker}{hashes}"


@dataclass(frozen=True)
class Lock:
    options: tuple[str, ...]
    entries: tuple[LockEntry, ...]

    def names(self) -> set[str]:
        return {e.name for e in self.entries}

    def get(self, name: str) -> LockEntry | None:
        return next((e for e in self.entries if e.name == canonical(name)), None)


def logical_lines(text: str) -> Iterable[tuple[int, str]]:
    """Requirement-file lines with continuations joined and comments removed."""
    number, pending = 0, ""
    for index, raw in enumerate(text.splitlines(), 1):
        number = number or index
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        joined = " ".join((pending + line).split())
        pending = ""
        if joined:
            yield number, joined
        number = 0
    if pending.strip():
        yield number, " ".join(pending.split())


def parse_lock(text: str, source: str) -> Lock:
    """A pip-style lock: exact name==version pins with optional sha256 hashes, plus
    --find-links/--index-url style options. Anything else is an input error."""
    options: list[str] = []
    entries: list[LockEntry] = []
    for number, line in logical_lines(text):
        if line.split(" ", 1)[0].split("=", 1)[0] in LOCK_OPTIONS:
            options.append(line)
            continue
        found = PIN.match(line)
        if found is None or "://" in line or "@" in line.split("--hash", 1)[0]:
            raise InputError(f"{source} line {number}: not an exact name==version pin: {line[:80]}")
        hashes = tuple(re.findall(r"--hash=sha256:([0-9a-f]{64})", found.group("hashes")))
        marker = (found.group("marker") or "").strip()
        entries.append(
            LockEntry(canonical(found.group("name")), found.group("version"), hashes, marker)
        )
    names = [e.name for e in entries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise InputError(f"{source}: packages listed twice: {', '.join(duplicates)}")
    return Lock(tuple(options), tuple(entries))


def read_lock(path: Path) -> Lock:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path}: {exc.strerror or exc}") from None
    return parse_lock(text, path.name)


def dist_name(folder: str) -> str:
    """The package name of an installed *.dist-info folder (name-version.dist-info)."""
    stem = folder[: -len(".dist-info")] if folder.endswith(".dist-info") else folder
    return canonical(stem.rsplit("-", 1)[0])


def base_id(lock_path: Path) -> str:
    return "apctt-base-" + hashlib.sha256(lock_path.read_bytes()).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_cuda_only(name: str) -> bool:
    return any(fnmatch.fnmatchcase(canonical(name), pattern) for pattern in CUDA_ONLY)


def cpu_torch(version: str, system: str) -> tuple[str, str | None]:
    """The CPU build of the base's torch version: PyPI on macOS, the CPU page on Linux. Arbitrary
    equality (===), because ==2.11.0 would also accept 2.11.0+cu128."""
    public = version.split("+", 1)[0]
    if system == "darwin":
        return f"torch==={public}", None
    return f"torch==={public}+cpu", TORCH_CPU_FIND_LINKS


def host_system() -> str:
    return "darwin" if sys.platform == "darwin" else "linux"


def choose_flavor(requested: str, system: str, machine: str, gpu: bool) -> str:
    if requested != "auto":
        return requested
    linux_x86 = system == "linux" and machine.lower() in ("x86_64", "amd64")
    return "cuda" if linux_x86 and gpu else "cpu"


@dataclass(frozen=True)
class InstallStep:
    """One `uv pip install --no-deps` run: the requirement file text and where wheels come from."""

    text: str
    hashed: bool
    find_links: tuple[str, ...] = ()
    no_index: bool = False


def install_steps(
    lock: Lock, flavor: str, system: str, wheelhouse: Path | None
) -> list[InstallStep]:
    local = (str(wheelhouse),) if wheelhouse else ()
    if flavor == "cuda":
        options = "" if wheelhouse else "".join(o + "\n" for o in lock.options)
        body = "".join(e.line + "\n" for e in lock.entries)
        return [InstallStep(options + body, True, local, bool(wheelhouse))]
    kept = [e for e in lock.entries if e.name != "torch" and not is_cuda_only(e.name)]
    steps = [InstallStep("".join(e.line + "\n" for e in kept), True, local, bool(wheelhouse))]
    torch = lock.get("torch")
    if torch is not None:
        requirement, links = cpu_torch(torch.version, system)
        sources = local or ((links,) if links else ())
        steps.append(InstallStep(requirement + "\n", False, sources, bool(wheelhouse)))
    return steps


# ---------------------------------------------------------------------- schema validation


def _is_type(value: object, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    types: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    return isinstance(value, types[name])


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        reference = str(schema["$ref"])
        if not reference.startswith("#/"):
            raise InputError(f"schema: unsupported $ref {reference}")
        target: Any = root
        for part in reference[2:].split("/"):
            target = target[part]
        schema = target
    return schema


def validate(
    instance: object, schema: dict[str, Any], root: dict[str, Any] | None = None, path: str = "$"
) -> list[str]:
    """Errors of instance against the subset of JSON Schema 2020-12 in SUPPORTED_KEYWORDS."""
    root = schema if root is None else root
    schema = _resolve(schema, root)
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_is_type(instance, n) for n in names):
            return [f"{path}: expected {' or '.join(names)}"]
    errors: list[str] = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: must be {json.dumps(schema['const'])}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: must be one of {', '.join(json.dumps(v) for v in schema['enum'])}")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if _is_type(instance, "number"):
        number = float(instance)  # type: ignore[arg-type]
        if "minimum" in schema and number < schema["minimum"]:
            errors.append(f"{path}: less than {schema['minimum']}")
        if "maximum" in schema and number > schema["maximum"]:
            errors.append(f"{path}: more than {schema['maximum']}")
        if "exclusiveMinimum" in schema and number <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: must be more than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and number >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: must be less than {schema['exclusiveMaximum']}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in instance]
            if len(set(seen)) != len(seen):
                errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors += validate(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required property '{name}'")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if "propertyNames" in schema:
                problems = validate(key, schema["propertyNames"], root, "")
                if problems:
                    detail = _resolve(schema["propertyNames"], root)
                    reason = detail.get("not", {}).get("description", "not allowed")
                    errors.append(f"{path}: property name '{key}' is {reason} or malformed")
            if key in properties:
                errors += validate(value, properties[key], root, f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}: unknown property '{key}'")
            elif isinstance(extra, dict):
                errors += validate(value, extra, root, f"{path}.{key}")
    if "not" in schema and not validate(instance, schema["not"], root, path):
        errors.append(f"{path}: {schema['not'].get('description', 'is not allowed')}")
    if "anyOf" in schema and all(validate(instance, s, root, path) for s in schema["anyOf"]):
        errors.append(f"{path}: matches none of the allowed forms")
    if "if" in schema:
        branch = "then" if not validate(instance, schema["if"], root, path) else "else"
        if branch in schema:
            errors += validate(instance, schema[branch], root, path)
    return errors


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ToolError(f"{path}: not found") from None
    except (OSError, ValueError) as exc:
        raise InputError(f"{path}: not readable JSON ({exc})") from None


# ------------------------------------------------------------------------------ processes


def log(message: str) -> None:
    print(f"make_env: {message}", file=sys.stderr, flush=True)


def shown(argv: list[str]) -> str:
    """The command line for the log, with an inline -c script replaced by its line count and a
    long JSON list argument by its length (the check script and the package list would bury the
    result lines that teams are told to read)."""
    parts = []
    for index, arg in enumerate(argv):
        if index and argv[index - 1] == "-c":
            parts.append(f"<inline script, {len(arg.strip().splitlines())} lines>")
        elif arg.startswith("[") and len(arg) > 80:
            try:
                parts.append(f"<list of {len(json.loads(arg))}>")
            except ValueError:
                parts.append(arg)
        else:
            parts.append(arg)
    return " ".join(parts)


def run(argv: list[str], *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    log("$ " + shown(argv))
    try:
        proc = subprocess.run(  # noqa: S603 - argv lists built here, shell=False
            argv,
            stdout=subprocess.PIPE if capture else sys.stderr,
            stderr=sys.stderr,
            text=True,
            env=env,
            check=False,
        )
    except OSError as exc:
        raise ToolError(f"cannot run {argv[0]}: {exc}") from None
    if proc.returncode != 0:
        raise ToolError(f"command failed (exit {proc.returncode}): {' '.join(argv[:4])} ...")
    return proc.stdout if capture else ""


def find_uv(option: str | None) -> str:
    uv = option or shutil.which("uv")
    if not uv:
        raise ToolError("uv is not on PATH: install it (https://docs.astral.sh/uv/) or pass --uv")
    return uv


def find_python(option: str | None, uv: str) -> str:
    if option:
        return option
    if sys.platform.startswith("linux") and Path(EVENT_PYTHON).exists():
        return EVENT_PYTHON
    # --system: never a virtual environment (such as the kit's own .venv) as the base interpreter
    found = run([uv, "python", "find", "--system", PYTHON_VERSION], capture=True).strip()
    if not found:
        raise ToolError(f"no Python {PYTHON_VERSION}: run uv python install {PYTHON_VERSION}")
    return found


def python_info(python: str) -> dict[str, str]:
    script = (
        "import json, sys, sysconfig; print(json.dumps({'version': '%d.%d' % sys.version_info[:2],"
        " 'purelib': sysconfig.get_paths()['purelib']}))"
    )
    data = json.loads(run([python, "-s", "-c", script], capture=True))
    return {str(k): str(v) for k, v in data.items()}


def gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def write_requirements(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def uv_install(uv: str, python: str, requirements: Path, step: InstallStep) -> None:
    argv = [uv, "pip", "install", "--python", python, "--no-deps", "--compile-bytecode"]
    argv += ["--link-mode", "copy"]
    if step.hashed:
        argv.append("--require-hashes")
    if step.no_index:
        argv.append("--no-index")
    for link in step.find_links:
        argv += ["--find-links", link]
    run([*argv, "-r", str(requirements)])


def make_writable(path: Path) -> None:
    for directory, _, files in os.walk(path):
        os.chmod(directory, 0o755)  # noqa: S103 - the owner removes the tree
        for name in files:
            target = Path(directory) / name
            if not target.is_symlink():
                os.chmod(target, target.stat().st_mode | stat.S_IWUSR)


def make_read_only(path: Path) -> None:
    for directory, _, files in os.walk(path, topdown=False):
        for name in files:
            target = Path(directory) / name
            if not target.is_symlink():
                mode = target.stat().st_mode
                os.chmod(target, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        os.chmod(directory, 0o555)  # noqa: S103 - read-only base, like the organizer


def remove_tree(path: Path) -> None:
    make_writable(path)
    shutil.rmtree(path)


# ----------------------------------------------------------------------------- the build


@dataclass
class Settings:
    uv: str
    python: str | None
    lock_path: Path
    lock: Lock
    flavor: str
    wheelhouse: Path | None
    read_only: bool


def build_base(directory: Path, settings: Settings) -> dict[str, Any]:
    lock_sha = sha256_file(settings.lock_path)
    marker = directory / BASE_MARKER
    if marker.is_file():
        data = load_json(marker)
        if data.get("lock_sha256") != lock_sha or data.get("flavor") != settings.flavor:
            raise InputError(
                f"{directory} was built from another lock or flavour ({data.get('flavor')}, "
                f"{data.get('base_id')}); remove it or choose another --base-dir"
            )
        if settings.python and settings.python != data.get("python"):
            raise InputError(
                f"{directory} was built with {data.get('python')}, not {settings.python}"
            )
        log(f"base {directory} is up to date ({data.get('base_id')}, {settings.flavor})")
        return {**data, "built": False, "path": str(directory)}
    if directory.exists():
        if (directory / BUILDING).exists():
            log(f"removing the unfinished base {directory}")
            remove_tree(directory)
        elif any(directory.iterdir()):
            raise InputError(f"{directory} exists and is not a base made by make_env.py")
        else:
            directory.rmdir()
    python = find_python(settings.python, settings.uv)
    run([settings.uv, "venv", "--python", python, "--no-config", str(directory)])
    (directory / BUILDING).write_text("")
    target = str(directory / "bin" / "python")
    info = python_info(target)
    if info["version"] != PYTHON_VERSION:
        raise ToolError(f"{python} is Python {info['version']}, not {PYTHON_VERSION}")
    steps = install_steps(settings.lock, settings.flavor, host_system(), settings.wheelhouse)
    with tempfile.TemporaryDirectory(prefix="apctt-make-env-") as scratch:
        for index, step in enumerate(steps, 1):
            requirements = write_requirements(Path(scratch), f"base-{index}.txt", step.text)
            uv_install(settings.uv, target, requirements, step)
    torch = settings.lock.get("torch")
    installed_torch = torch.version if torch else None
    if torch is not None and settings.flavor == "cpu":
        installed_torch = cpu_torch(torch.version, host_system())[0].split("=")[-1]
    record = {
        "record_version": "apctt-base-install-v1",
        "base_id": base_id(settings.lock_path),
        "flavor": settings.flavor,
        "lock": str(settings.lock_path),
        "lock_sha256": lock_sha,
        "python": python,
        "python_version": info["version"],
        "site_packages": info["purelib"],
        "torch": installed_torch,
        "packages": len(settings.lock.entries)
        - (
            sum(1 for e in settings.lock.entries if is_cuda_only(e.name))
            if settings.flavor == "cpu"
            else 0
        ),
        "created": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "read_only": settings.read_only,
    }
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (directory / BUILDING).unlink()
    if settings.read_only:
        make_read_only(directory)
    return {**record, "built": True, "path": str(directory)}


# ------------------------------------------------------------------------------ the check

CHECK_SCRIPT = r"""
import importlib, importlib.metadata as metadata, json, sys, sysconfig
modules, dists = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = {"version": "%d.%d" % sys.version_info[:2], "path": sys.path,
       "purelib": sysconfig.get_paths()["purelib"], "imports": {}, "dists": {}}
for name in modules:
    try:
        module = importlib.import_module(name)
        out["imports"][name] = {"file": getattr(module, "__file__", None) or "",
                                "version": str(getattr(module, "__version__", ""))}
    except BaseException as exc:
        out["imports"][name] = {"error": (type(exc).__name__ + ": " + str(exc))[:300]}
if "torch" in sys.modules:
    try:
        torch = sys.modules["torch"]
        out["torch_cuda"] = {"build": torch.version.cuda, "available": torch.cuda.is_available()}
    except BaseException:
        pass
for name in dists:
    try:
        found = metadata.distribution(name)
        out["dists"][name] = {"version": found.version, "location": str(found.locate_file(""))}
    except metadata.PackageNotFoundError:
        out["dists"][name] = None
print(json.dumps(out))
"""


def within(path: str, folder: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(folder).resolve())
    except (OSError, ValueError):
        return False


def check(base_dir: Path, lock: Lock, lock_path: Path, imports: Sequence[str]) -> dict[str, Any]:
    """Run the base's python as the kit server runs the worker and check the base."""
    problems: list[str] = []
    marker = base_dir / BASE_MARKER
    if not marker.is_file():
        raise InputError(f"{base_dir} has no {BASE_MARKER}: build it with make_env.py first")
    base = load_json(marker)
    if base.get("lock_sha256") != sha256_file(lock_path):
        problems.append(f"the base was built from another lock than {lock_path.name}")
    base_site = str(base["site_packages"])
    worker = base_dir / "bin" / "python"
    modules = [m for m in imports if m not in DEFAULT_IMPORTS or lock.get(m) is not None]
    expected = [
        e.name for e in lock.entries if not (base.get("flavor") == "cpu" and is_cuda_only(e.name))
    ]
    env = {"PATH": os.pathsep.join([str(worker.parent), "/usr/bin", "/bin"]), "LANG": "C.UTF-8"}
    argv = [str(worker), "-s", "-c", CHECK_SCRIPT, json.dumps(modules), json.dumps(expected)]
    data = json.loads(run(argv, capture=True, env=env))
    if data["version"] != PYTHON_VERSION:
        problems.append(f"the worker runs Python {data['version']}, not {PYTHON_VERSION}")
    paths = [str(Path(p).resolve()) for p in data["path"] if p]
    if str(Path(base_site).resolve()) not in paths:
        problems.append(f"the base site-packages {base_site} is not on the worker's sys.path")
    imported: dict[str, dict[str, str]] = {}
    for name, result in data["imports"].items():
        if "error" in result:
            problems.append(f"import {name} failed: {result['error']}")
            continue
        origin = "base" if within(result["file"], base_site) else "elsewhere"
        imported[name] = {"version": result["version"], "file": result["file"], "from": origin}
        if lock.get(name) is not None and origin != "base":
            problems.append(f"{name} imports from {result['file']}, not from the base")
    for name, result in data["dists"].items():
        entry = lock.get(name)
        if result is None:
            problems.append(f"base package {name} is not installed")
        elif not within(result["location"], base_site):
            problems.append(f"base package {name} is found outside the base")
        elif entry is not None and name != "torch" and result["version"] != entry.version:
            problems.append(
                f"base package {name} is {result['version']}, the lock says {entry.version}"
            )
    return {
        "ok": not problems,
        "problems": problems,
        "worker_python": str(worker),
        "imports": imported,
        "torch_cuda": data.get("torch_cuda"),
        "base_writable": os.access(marker, os.W_OK),
    }


def print_check(result: dict[str, Any]) -> None:
    for name, info in result["imports"].items():
        print(f"import  {name} {info['version']} from the {info['from']}: {info['file']}")
    if result.get("torch_cuda"):
        cuda = result["torch_cuda"]
        print(f"torch   CUDA build {cuda['build']}, CUDA available: {cuda['available']}")
    for problem in result["problems"]:
        print(f"PROBLEM {problem}")
    if result["ok"]:
        print("check   OK: imports come from the base and every base package is installed")
    else:
        print(f"check   FAILED: {len(result['problems'])} problem(s)")


# ------------------------------------------------------------------------------------ CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_TOOL, f"{self.prog}: error: {message}\n")


def parser() -> Parser:
    p = Parser(
        prog="make_env.py",
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Exit codes: 0 done, 1 check failed, 2 invalid input, 3 usage or tool error.",
    )
    p.add_argument("--check", action="store_true", help="only check an existing base")
    p.add_argument("--base-dir", type=Path, help="base environment folder (built once, reused)")
    p.add_argument("--flavor", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--python", help=f"interpreter of the base (default {EVENT_PYTHON} if "
                   "present, else uv python find --system 3.12)")  # fmt: skip
    p.add_argument("--wheelhouse", type=Path, help="install offline from this folder of wheels")
    p.add_argument("--read-only", action="store_true", help="make a new base read-only")
    p.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE",
                   help="also check that MODULE imports (repeatable)")  # fmt: skip
    p.add_argument("--base-lock", type=Path, default=BASE_LOCK, help=argparse.SUPPRESS)
    p.add_argument("--uv", help="uv executable (default: uv on PATH)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    return p


def print_worker(worker: str) -> None:
    print('worker python for config/my.local.json ("worker": {"python": ...}):')
    print(worker)


def build(args: argparse.Namespace) -> int:
    if os.name == "nt":
        raise ToolError("native Windows is not supported: use WSL2")
    uv = find_uv(args.uv)
    lock = read_lock(args.base_lock)
    flavor = choose_flavor(args.flavor, host_system(), platform.machine(), gpu_present())
    settings = Settings(uv, args.python, args.base_lock, lock, flavor, args.wheelhouse,
                        args.read_only)  # fmt: skip
    base_dir = args.base_dir.absolute()
    base = build_base(base_dir, settings)
    result = check(base_dir, lock, args.base_lock, [*DEFAULT_IMPORTS, *args.imports])
    worker = str(base_dir / "bin" / "python")
    if args.json:
        print(json.dumps({"base": base, "check": result, "worker_python": worker}, indent=2))
    else:
        state = "built" if base["built"] else "reused"
        print(f"base    {base['path']} ({state}; {base['flavor']}, {base['base_id']}, "
              f"torch {base['torch']})")  # fmt: skip
        print_check(result)
        print_worker(worker)
    return EXIT_OK if result["ok"] else EXIT_CHECK


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.base_dir:
            raise ToolError("give --base-dir (and --check to check an existing base)")
        if args.check:
            lock = read_lock(args.base_lock)
            imports = [*DEFAULT_IMPORTS, *args.imports]
            result = check(args.base_dir.absolute(), lock, args.base_lock, imports)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_check(result)
                print_worker(result["worker_python"])
            return EXIT_OK if result["ok"] else EXIT_CHECK
        return build(args)
    except InputError as exc:
        log(f"error: {exc}")
        return EXIT_INVALID
    except ToolError as exc:
        log(f"error: {exc}")
        return EXIT_TOOL


if __name__ == "__main__":
    sys.exit(main())
