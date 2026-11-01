"""Supervisor-side worker protocol helpers (SPEC-09, SPEC-17): argv, env, envelope, collection.

Files written by the worker are read relative to a directory descriptor opened when the scratch
directory was created, with O_NOFOLLOW and O_NONBLOCK. Symlinks, FIFOs, directories and oversized
files are rejected. Raw result bytes are preserved (bounded) before any parsing.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from apctt_contracts import WORKER_PROTOCOL_VERSION
from apctt_contracts.config import Checklist
from apctt_contracts.hashing import sha256_hex
from apctt_contracts.ipc import Observation, RunOutcome, RunSpec, WorkerEnvelope, WorkerFailure
from apctt_contracts.pdf import document_filename, worker_case
from apctt_contracts.results import build_supervisor_failure, validate_result
from apctt_contracts.strict_json import StrictJSONError, loads_strict

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"
EXIT_MODULE_FAILURE = 3
RAW_ARTIFACT = "raw_output.bin"
BASE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "LANG",
        "LC_ALL",
        "PYTHONUTF8",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
    }
)


# argv / env / envelope -----------------------------------------------------------------------


def worker_argv(spec: RunSpec) -> list[str]:
    """Fixed argv; nothing here comes from the request. Used with shell=False only."""
    return [
        *spec.launcher_argv,
        str(spec.worker_python),
        "-s",
        str(WORKER_SCRIPT),
        "--run-id",
        spec.run_id,
    ]


def worker_env(spec: RunSpec, scratch: Path) -> dict[str, str]:
    """A new allowlisted environment; the API process environment is never inherited."""
    tmp = str(scratch / "tmp")
    env = {
        "PATH": os.pathsep.join([str(spec.worker_python.parent), "/usr/bin", "/bin"]),
        "HOME": str(scratch / "home"),
        "TMPDIR": tmp,
        "TMP": tmp,
        "TEMP": tmp,
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONHASHSEED": str(spec.compute.seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if spec.compute.visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = spec.compute.visible_devices
    elif spec.compute.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(spec.extra_env)
    return env


def time_budget_ms(spec: RunSpec, now: float | None = None) -> int:
    """Remaining per-case budget at hand-off. The worker computes its own local deadline."""
    remaining = spec.deadline_monotonic - (time.monotonic() if now is None else now)
    return max(1, int(remaining * 1000))


def build_envelope(
    spec: RunSpec, scratch: Path, budget_ms: int, document_paths: Sequence[Path] = ()
) -> bytes:
    case = worker_case(spec.case, document_paths) if document_paths else dict(spec.case)
    value = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "run_id": spec.run_id,
        "case": case,
        "config": {
            "checklist": spec.checklist.model_dump(),
            "module": dict(spec.module),
            "compute": spec.compute.model_dump(),
        },
        "context": {
            "run_id": spec.run_id,
            "seed": spec.compute.seed,
            "time_budget_ms": budget_ms,
            "scratch_dir": str(scratch),
        },
    }
    return WorkerEnvelope.model_validate(value).model_dump_json().encode("utf-8")


# scratch -------------------------------------------------------------------------------------


class Scratch:
    """Private per-run directory with a descriptor that pins it against later path swaps."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self.fd = fd

    def read_regular(self, name: str, max_bytes: int) -> tuple[bytes | None, str | None]:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            fd = os.open(name, flags, dir_fd=self.fd)
        except FileNotFoundError:
            return None, "missing"
        except OSError as exc:
            return None, "symlink" if exc.errno == errno.ELOOP else "unreadable"
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None, "not_regular"
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(remaining, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
        if len(data) > max_bytes:
            return data[:max_bytes], "oversize"
        return data, None

    def close(self, *, remove: bool) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass
        if remove:
            shutil.rmtree(self.path, ignore_errors=True)


def stage_documents(scratch: Path, files: Sequence[Path]) -> list[Path]:
    """Read-only copies of the case PDFs in `<scratch>/documents/`, named by position only."""
    if not files:
        return []
    directory = scratch / "documents"
    directory.mkdir(mode=0o700)
    staged: list[Path] = []
    for index, source in enumerate(files):
        target = directory / document_filename(index)
        shutil.copyfile(source, target, follow_symlinks=False)
        os.chmod(target, 0o400)
        staged.append(target)
    return staged


def prepare_scratch(root: Path, run_id: str) -> Scratch:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / run_id
    path.mkdir(mode=0o700)
    for name in ("home", "tmp", "cache"):
        (path / name).mkdir(mode=0o700)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    return Scratch(path, fd)


# stdout / stderr -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSummary:
    total_bytes: int
    stored_bytes: int
    truncated: bool


class StreamSink:
    """Keep the first `limit` bytes of a stream in a local artifact; count and drop the rest."""

    def __init__(self, path: Path, limit: int) -> None:
        self.limit = limit
        self.total = 0
        self.stored = 0
        self._handle = open(path, "wb")  # noqa: SIM115 - closed explicitly by the owner
        os.chmod(path, 0o600)

    def write(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.limit - self.stored
        if room > 0:
            part = chunk[:room]
            self._handle.write(part)
            self.stored += len(part)

    def close(self) -> None:
        self._handle.close()

    def summary(self) -> StreamSummary:
        return StreamSummary(self.total, self.stored, self.total > self.stored)


# collection ----------------------------------------------------------------------------------


def read_observation(
    scratch: Scratch, run_id: str, max_bytes: int
) -> tuple[Observation | None, list[str]]:
    data, problem = scratch.read_regular("observation.json", max_bytes)
    if problem == "missing":
        return None, ["SIDECAR_MISSING"]
    if problem is not None or data is None:
        return None, ["SIDECAR_INVALID"]
    try:
        observation = Observation.model_validate(loads_strict(data))
    except (StrictJSONError, ValidationError):
        return None, ["SIDECAR_INVALID"]
    if observation.run_id != run_id:
        return None, ["SIDECAR_INVALID"]
    return observation, []


def read_failure(scratch: Scratch, run_id: str) -> tuple[WorkerFailure | None, list[str]]:
    data, problem = scratch.read_regular("failure.json", 4096)
    if problem == "missing":
        return None, []
    if problem is not None or data is None:
        return None, ["FAILURE_RECORD_INVALID"]
    try:
        failure = WorkerFailure.model_validate(loads_strict(data))
    except (StrictJSONError, ValidationError):
        return None, ["FAILURE_RECORD_INVALID"]
    if failure.run_id != run_id:
        return None, ["FAILURE_RECORD_INVALID"]
    return failure, []


def preserve_raw(artifact_dir: Path, data: bytes) -> str:
    """Write raw worker bytes once (O_EXCL) and return the artifact name."""
    fd = os.open(artifact_dir / RAW_ARTIFACT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return RAW_ARTIFACT


def collect_outcome(
    spec: RunSpec,
    scratch: Scratch,
    *,
    exit_code: int | None,
    exit_signal: int | None,
    timed_out: bool,
    cleanup_confirmed: bool,
    total_ms: float,
    stdout: StreamSummary,
    stderr: StreamSummary,
) -> RunOutcome:
    warnings: list[str] = []
    observation, sidecar_warnings = read_observation(
        scratch, spec.run_id, spec.limits.max_observation_bytes
    )
    warnings.extend(sidecar_warnings)
    failure, failure_warnings = read_failure(scratch, spec.run_id)
    warnings.extend(failure_warnings)
    raw, raw_problem = scratch.read_regular("result.json", spec.limits.max_result_bytes)

    raw_name: str | None = None
    raw_sha: str | None = None
    if raw is not None:
        raw_name = preserve_raw(spec.artifact_dir, raw)
        raw_sha = sha256_hex(raw)
        if raw_problem == "oversize":
            warnings.extend(["RESULT_OVERSIZE", "RAW_TRUNCATED"])
    if raw_problem in ("symlink", "not_regular", "unreadable"):
        warnings.append("RESULT_FILE_REJECTED")

    parsed: Any = None
    issues: tuple[Any, ...] = ()
    if timed_out:
        code: str | None = "TIMEOUT"
    elif exit_code == EXIT_MODULE_FAILURE and failure is not None:
        code = failure.error_code
    elif exit_code != 0 or exit_signal is not None:
        code = "WORKER_CRASH"
    elif failure is not None:
        code = "WORKER_PROTOCOL_ERROR"
    elif raw_problem is not None or raw is None:
        code = "WORKER_PROTOCOL_ERROR"
    else:
        try:
            parsed = loads_strict(raw)
        except StrictJSONError:
            code = "INVALID_OUTPUT"
        else:
            checklist = Checklist.model_validate(spec.checklist.model_dump())
            found = validate_result(parsed, case=spec.case, checklist=checklist)
            issues = tuple(found)
            code = "INVALID_OUTPUT" if found else None
    if stdout.truncated or stderr.truncated:
        warnings.append("LOG_TRUNCATED")

    if code is None:
        origin = "module"
        result: Mapping[str, Any] = parsed
    else:
        origin = "supervisor"
        result = build_supervisor_failure(spec.case_meta, code)
    return RunOutcome(
        run_id=spec.run_id,
        result_origin=origin,  # type: ignore[arg-type]
        result=result,
        error_code=code,
        raw_output_sha256=raw_sha,
        raw_output_bytes=len(raw) if raw is not None else None,
        raw_artifact=raw_name,
        output_issues=issues,
        exit_code=exit_code,
        exit_signal=exit_signal,
        timed_out=timed_out,
        cleanup_confirmed=cleanup_confirmed,
        total_ms=round(total_ms, 3),
        observation=observation,
        warning_codes=tuple(dict.fromkeys(warnings)),
        stdout_bytes=stdout.total_bytes,
        stderr_bytes=stderr.total_bytes,
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
    )
