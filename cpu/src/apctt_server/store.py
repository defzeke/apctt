"""SPEC-08 SQLite job store: durable admission, duplicates, single slot, restart state.

Admission runs in one `BEGIN IMMEDIATE` transaction that checks the request ID, readiness,
capacity and the singleton active reservation before inserting. The caller answers 202 only after
`admit` returns, i.e. after COMMIT. A lifetime file lock prevents a second API instance from
opening the same data directory. Finished rows are immutable at the database level.
"""

from __future__ import annotations

import enum
import fcntl
import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apctt_contracts.hashing import canonical_bytes, canonical_sha256
from apctt_contracts.results import CaseMeta, build_supervisor_failure
from apctt_diagnostics.responses import job_failure

SCHEMA_VERSION = "apctt-store-v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    assessment_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    effective_sha256 TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    verified_package_sha256 TEXT,
    package_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    checklist_version TEXT NOT NULL,
    case_id TEXT NOT NULL,
    target_json TEXT NOT NULL,
    case_json TEXT NOT NULL,
    job_status TEXT NOT NULL CHECK (job_status IN ('queued', 'running', 'finished')),
    result_origin TEXT CHECK (result_origin IN ('module', 'supervisor')),
    result_json TEXT,
    result_sha256 TEXT,
    error_code TEXT,
    raw_output_sha256 TEXT,
    raw_output_bytes INTEGER,
    raw_artifact TEXT,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    accepted_trace_id TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    run_id TEXT,
    worker_pid INTEGER,
    worker_pgid INTEGER,
    worker_start_token TEXT,
    CHECK ((job_status = 'finished') = (result_json IS NOT NULL AND result_origin IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS active_slot (
    slot INTEGER PRIMARY KEY CHECK (slot = 1),
    assessment_id TEXT NOT NULL UNIQUE REFERENCES jobs (assessment_id)
);
CREATE TRIGGER IF NOT EXISTS jobs_terminal_immutable
BEFORE UPDATE ON jobs WHEN OLD.job_status = 'finished'
BEGIN
    SELECT RAISE(ABORT, 'terminal job state is immutable');
END;
CREATE TRIGGER IF NOT EXISTS jobs_no_delete
BEFORE DELETE ON jobs
BEGIN
    SELECT RAISE(ABORT, 'jobs are retained until an explicit archive');
END;
"""


class StoreLockedError(RuntimeError):
    """Another API instance holds the data directory."""


def data_dir_problem(data_dir: Path) -> str | None:
    """Pre-start check of the data directory: None, `STORE_LOCKED` or `DATA_DIR_UNUSABLE`.

    Creates nothing. `JobStore.open` still takes the lock itself; this only lets the command line
    refuse with a fixed code instead of failing inside the ASGI lifespan.
    """
    if data_dir.exists() and not data_dir.is_dir():
        return "DATA_DIR_UNUSABLE"
    try:
        fd = os.open(data_dir / "server.lock", os.O_RDWR)
    except FileNotFoundError:
        return None
    except OSError:
        return "DATA_DIR_UNUSABLE"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "STORE_LOCKED"
    except OSError:
        return "DATA_DIR_UNUSABLE"
    finally:
        os.close(fd)  # closing the descriptor also releases a lock taken by this probe
    return None


def data_dir_locked(data_dir: Path) -> bool:
    """True when another API instance holds the lifetime lock."""
    return data_dir_problem(data_dir) == "STORE_LOCKED"


class AdmissionStatus(enum.Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    BUSY = "busy"
    NOT_READY = "not_ready"
    STORAGE_CAPACITY = "storage_capacity"


@dataclass(frozen=True)
class NewJob:
    assessment_id: str
    request_id: str
    effective_sha256: str
    input_sha256: str
    config_sha256: str
    verified_package_sha256: str | None
    package_id: str
    case_json: str
    case_meta: CaseMeta
    accepted_at: str
    accepted_trace_id: str


@dataclass(frozen=True)
class JobRecord:
    assessment_id: str
    request_id: str
    effective_sha256: str
    input_sha256: str
    config_sha256: str
    verified_package_sha256: str | None
    package_id: str
    case_meta: CaseMeta
    case_json: str
    job_status: str
    result_origin: str | None
    result: dict[str, Any] | None
    result_sha256: str | None
    error_code: str | None
    raw_output_sha256: str | None
    raw_output_bytes: int | None
    raw_artifact: str | None
    diagnostics: dict[str, Any]
    accepted_trace_id: str
    accepted_at: str
    started_at: str | None
    finished_at: str | None
    run_id: str | None
    worker_pid: int | None
    worker_pgid: int | None
    worker_start_token: str | None

    def case(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.case_json)
        return value


@dataclass(frozen=True)
class Admission:
    status: AdmissionStatus
    job: JobRecord | None


@dataclass(frozen=True)
class RecoveryReport:
    recovered: list[str]
    pending: list[str]


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        assessment_id=row["assessment_id"],
        request_id=row["request_id"],
        effective_sha256=row["effective_sha256"],
        input_sha256=row["input_sha256"],
        config_sha256=row["config_sha256"],
        verified_package_sha256=row["verified_package_sha256"],
        package_id=row["package_id"],
        case_meta=CaseMeta(
            schema_version=row["schema_version"],
            checklist_version=row["checklist_version"],
            case_id=row["case_id"],
            assessment_target=json.loads(row["target_json"]),
        ),
        case_json=row["case_json"],
        job_status=row["job_status"],
        result_origin=row["result_origin"],
        result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
        result_sha256=row["result_sha256"],
        error_code=row["error_code"],
        raw_output_sha256=row["raw_output_sha256"],
        raw_output_bytes=row["raw_output_bytes"],
        raw_artifact=row["raw_artifact"],
        diagnostics=json.loads(row["diagnostics_json"]),
        accepted_trace_id=row["accepted_trace_id"],
        accepted_at=row["accepted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        run_id=row["run_id"],
        worker_pid=row["worker_pid"],
        worker_pgid=row["worker_pgid"],
        worker_start_token=row["worker_start_token"],
    )


def _json(value: Mapping[str, Any]) -> str:
    return canonical_bytes(value).decode("utf-8")


class JobStore:
    def __init__(self, data_dir: Path, *, max_jobs: int, min_free_disk_bytes: int) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "jobs.sqlite3"
        self.max_jobs = max_jobs
        self.min_free_disk_bytes = min_free_disk_bytes
        self._lock_fd: int | None = None

    # lifecycle -------------------------------------------------------------------------------

    def open(self) -> None:
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.data_dir / "server.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise StoreLockedError("data directory is locked by another API instance") from None
        self._lock_fd = fd
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,)
            )

    def close(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._lock_fd is None:
            raise RuntimeError("store is not open")
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # admission -------------------------------------------------------------------------------

    def _disk_ok(self) -> bool:
        return shutil.disk_usage(self.data_dir).free >= self.min_free_disk_bytes

    def admit(self, job: NewJob, *, ready: bool = True) -> Admission:
        disk_ok = self._disk_ok()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (job.request_id,)
            ).fetchone()
            if row is not None:
                existing = _record(row)
                if existing.effective_sha256 == job.effective_sha256:
                    return Admission(AdmissionStatus.DUPLICATE, existing)
                return Admission(AdmissionStatus.CONFLICT, None)
            if not ready:
                return Admission(AdmissionStatus.NOT_READY, None)
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if count >= self.max_jobs or not disk_ok:
                return Admission(AdmissionStatus.STORAGE_CAPACITY, None)
            if conn.execute("SELECT 1 FROM active_slot").fetchone() is not None:
                return Admission(AdmissionStatus.BUSY, None)
            meta = job.case_meta
            conn.execute(
                """INSERT INTO jobs (assessment_id, request_id, effective_sha256, input_sha256,
                config_sha256, verified_package_sha256, package_id, schema_version,
                checklist_version, case_id, target_json, case_json, job_status,
                accepted_trace_id, accepted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (
                    job.assessment_id,
                    job.request_id,
                    job.effective_sha256,
                    job.input_sha256,
                    job.config_sha256,
                    job.verified_package_sha256,
                    job.package_id,
                    meta.schema_version,
                    meta.checklist_version,
                    meta.case_id,
                    _json(meta.assessment_target),
                    job.case_json,
                    job.accepted_trace_id,
                    job.accepted_at,
                ),
            )
            conn.execute(
                "INSERT INTO active_slot (slot, assessment_id) VALUES (1, ?)", (job.assessment_id,)
            )
            created = conn.execute(
                "SELECT * FROM jobs WHERE assessment_id = ?", (job.assessment_id,)
            ).fetchone()
            return Admission(AdmissionStatus.CREATED, _record(created))

    # queries ---------------------------------------------------------------------------------

    def get(self, assessment_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
        return _record(row) if row is not None else None

    def get_by_request(self, request_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE request_id = ?", (request_id,)).fetchone()
        return _record(row) if row is not None else None

    def active(self) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT jobs.* FROM active_slot JOIN jobs USING (assessment_id)"
            ).fetchone()
        return _record(row) if row is not None else None

    def job_count(self) -> int:
        with self._connect() as conn:
            count: int = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        return count

    def interrupted(self) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE job_status IN ('queued', 'running') ORDER BY accepted_at"
            ).fetchall()
        return [_record(r) for r in rows]

    # transitions -----------------------------------------------------------------------------

    def mark_running(
        self,
        assessment_id: str,
        *,
        run_id: str,
        pid: int,
        pgid: int,
        start_token: str | None,
        started_at: str,
    ) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET job_status = 'running', run_id = ?, worker_pid = ?,
                worker_pgid = ?, worker_start_token = ?, started_at = ?
                WHERE assessment_id = ? AND job_status IN ('queued', 'running')""",
                (run_id, pid, pgid, start_token, started_at, assessment_id),
            )
            if cursor.rowcount != 1:
                raise sqlite3.DatabaseError("job is not queued or running")

    def record_run_id(self, assessment_id: str, run_id: str) -> None:
        """Persist the run marker before spawn so restart recovery can find an orphan worker."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE jobs SET run_id = ? WHERE assessment_id = ? AND job_status = 'queued'",
                (run_id, assessment_id),
            )

    def finish(
        self,
        assessment_id: str,
        *,
        origin: str,
        result: Mapping[str, Any],
        result_sha256: str,
        error_code: str | None,
        raw_output_sha256: str | None,
        raw_output_bytes: int | None,
        raw_artifact: str | None,
        finished_at: str,
        diagnostics: Mapping[str, Any],
        release_slot: bool,
    ) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET job_status = 'finished', result_origin = ?, result_json = ?,
                result_sha256 = ?, error_code = ?, raw_output_sha256 = ?, raw_output_bytes = ?,
                raw_artifact = ?, finished_at = ?, diagnostics_json = ?
                WHERE assessment_id = ?""",
                (
                    origin,
                    _json(result),
                    result_sha256,
                    error_code,
                    raw_output_sha256,
                    raw_output_bytes,
                    raw_artifact,
                    finished_at,
                    _json(diagnostics),
                    assessment_id,
                ),
            )
            if cursor.rowcount != 1:
                raise sqlite3.DatabaseError("unknown job")
            if release_slot:
                conn.execute("DELETE FROM active_slot WHERE assessment_id = ?", (assessment_id,))

    def release_slot(self, assessment_id: str) -> None:
        with self._transaction() as conn:
            conn.execute("DELETE FROM active_slot WHERE assessment_id = ?", (assessment_id,))

    def held_by_finished(self) -> JobRecord | None:
        """A finished job that still holds the slot (its processes were not confirmed stopped)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT jobs.* FROM active_slot JOIN jobs USING (assessment_id) "
                "WHERE jobs.job_status = 'finished'"
            ).fetchone()
        return _record(row) if row is not None else None

    def recover_interrupted(
        self,
        cleanup: Callable[[JobRecord], bool],
        *,
        now: str,
        error_code: str = "SERVER_RESTARTED",
    ) -> RecoveryReport:
        """Close queued/running jobs from a previous process without re-running them.

        `cleanup` must terminate and confirm the worker of a job; if it cannot confirm, the job
        and its slot stay as they are and the server keeps reporting RECOVERY_PENDING.
        """
        recovered: list[str] = []
        pending: list[str] = []
        for job in self.interrupted():
            if not cleanup(job):
                pending.append(job.assessment_id)
                continue
            result = build_supervisor_failure(job.case_meta, error_code)
            self.finish(
                job.assessment_id,
                origin="supervisor",
                result=result,
                result_sha256=canonical_sha256(result),
                error_code=error_code,
                raw_output_sha256=job.raw_output_sha256,
                raw_output_bytes=job.raw_output_bytes,
                raw_artifact=job.raw_artifact,
                finished_at=now,
                diagnostics={
                    "phase": "recovered",
                    "previous_status": job.job_status,
                    "warning_codes": [
                        "SERVER_RESTARTED"
                        if error_code == "SERVER_RESTARTED"
                        else "SUPERVISOR_INTERNAL_ERROR"
                    ],
                    "failure": job_failure(error_code),
                },
                release_slot=True,
            )
            recovered.append(job.assessment_id)
        return RecoveryReport(recovered, pending)
