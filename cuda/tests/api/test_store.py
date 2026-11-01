"""SPEC-08 durable admission, duplicates, capacity and restart state: AC-06, AC-07, AC-09, AC-12."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from apctt_contracts.config import Checklist
from apctt_contracts.hashing import canonical_bytes
from apctt_contracts.results import CaseMeta, validate_result
from apctt_server.store import (
    AdmissionStatus,
    JobRecord,
    JobStore,
    NewJob,
    StoreLockedError,
    data_dir_locked,
)

CASE = {
    "schema_version": "apctt-pdf-example-v1",
    "input_profile": "pdf_report",
    "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
    "case_id": "CASE-1",
    "assessment_as_of": None,
    "assessment_target": {"target_id": "T1", "type": "component", "description": "d"},
    "intended_application": "a",
    "intended_environment": "e",
    "title": "t",
    "documents": [
        {
            "document_id": "D1",
            "title": "t",
            "document_date": None,
            "language": "en",
            "format": "pdf",
            "page_count": 1,
            "sha256": "0" * 64,
        }
    ],
}
CHECKLIST = Checklist.model_validate(
    {
        "version": "DEMO-CHECKLIST-NOT-OFFICIAL",
        "criteria": [{"criterion_id": "C1", "description": "d"}],
    }
)


def new_job(request_id: str, effective: str = "e" * 64) -> NewJob:
    return NewJob(
        assessment_id=uuid.uuid4().hex,
        request_id=request_id,
        effective_sha256=effective,
        input_sha256="i" * 64,
        config_sha256="c" * 64,
        verified_package_sha256=None,
        package_id="demo",
        case_json=canonical_bytes(CASE).decode(),
        case_meta=CaseMeta.from_case(CASE),
        accepted_at="2026-09-29T00:00:00Z",
        accepted_trace_id=uuid.uuid4().hex,
    )


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    s = JobStore(tmp_path / "data", max_jobs=100, min_free_disk_bytes=0)
    s.open()
    yield s
    s.close()


def finish(store: JobStore, job: JobRecord) -> None:
    store.finish(
        job.assessment_id,
        origin="supervisor",
        result={"assessment_status": "failed"},
        result_sha256="r" * 64,
        error_code="WORKER_CRASH",
        raw_output_sha256=None,
        raw_output_bytes=None,
        raw_artifact=None,
        finished_at="2026-09-29T00:00:01Z",
        diagnostics={},
        release_slot=True,
    )


def test_ac06_atomic_duplicate(store: JobStore) -> None:
    threads = 16
    barrier = threading.Barrier(threads)
    outcomes: list[tuple[AdmissionStatus, str]] = []
    lock = threading.Lock()

    def submit() -> None:
        job = new_job("REQ-DUP")
        barrier.wait()
        admission = store.admit(job)
        assert admission.job is not None
        with lock:
            outcomes.append((admission.status, admission.job.assessment_id))

    workers = [threading.Thread(target=submit) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    statuses = [s for s, _ in outcomes]
    assert statuses.count(AdmissionStatus.CREATED) == 1
    assert statuses.count(AdmissionStatus.DUPLICATE) == threads - 1
    assert len({a for _, a in outcomes}) == 1
    assert store.job_count() == 1


def test_ac07_conflict_precedes_busy(store: JobStore) -> None:
    first = store.admit(new_job("REQ-A", effective="1" * 64))
    assert first.status is AdmissionStatus.CREATED and first.job is not None
    # Same ID, different effective input: conflict even though the slot is busy.
    assert store.admit(new_job("REQ-A", effective="2" * 64)).status is AdmissionStatus.CONFLICT
    # New ID while busy: no job is created.
    assert store.admit(new_job("REQ-B")).status is AdmissionStatus.BUSY
    assert store.get_by_request("REQ-B") is None
    # Duplicate while busy returns the existing job; duplicate precedes readiness.
    dup = store.admit(new_job("REQ-A", effective="1" * 64), ready=False)
    assert dup.status is AdmissionStatus.DUPLICATE and dup.job is not None
    assert dup.job.assessment_id == first.job.assessment_id
    assert store.admit(new_job("REQ-C"), ready=False).status is AdmissionStatus.NOT_READY
    finish(store, first.job)
    assert store.admit(new_job("REQ-A", effective="2" * 64)).status is AdmissionStatus.CONFLICT
    again = store.admit(new_job("REQ-A", effective="1" * 64))
    assert again.status is AdmissionStatus.DUPLICATE and again.job is not None
    assert again.job.job_status == "finished"
    assert store.job_count() == 1


def test_ac09_single_reservation(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "data", max_jobs=2, min_free_disk_bytes=0)
    store.open()
    try:
        barrier = threading.Barrier(2)
        results: list[AdmissionStatus] = []

        def submit(request_id: str) -> None:
            job = new_job(request_id)
            barrier.wait()
            results.append(store.admit(job).status)

        workers = [threading.Thread(target=submit, args=(f"REQ-{i}",)) for i in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert sorted(r.value for r in results) == ["busy", "created"]
        active = store.active()
        assert active is not None and active.job_status == "queued"
        # The singleton reservation is a database constraint, not only a process lock.
        with sqlite3.connect(store.db_path) as raw, pytest.raises(sqlite3.IntegrityError):
            raw.execute("INSERT INTO active_slot (slot, assessment_id) VALUES (1, 'other')")
        finish(store, active)
        assert store.active() is None
        # Terminal state is immutable at the database level.
        with pytest.raises(sqlite3.DatabaseError):
            finish(store, active)
        with sqlite3.connect(store.db_path) as raw, pytest.raises(sqlite3.DatabaseError):
            raw.execute(
                "UPDATE jobs SET error_code = 'X' WHERE assessment_id = ?", (active.assessment_id,)
            )
        second = store.admit(new_job("REQ-NEXT"))
        assert second.status is AdmissionStatus.CREATED and second.job is not None
        finish(store, second.job)
        # Capacity: max_jobs reached → STORAGE_CAPACITY and no job row.
        assert store.admit(new_job("REQ-FULL")).status is AdmissionStatus.STORAGE_CAPACITY
        assert store.get_by_request("REQ-FULL") is None
    finally:
        store.close()
    tight = JobStore(tmp_path / "data2", max_jobs=10, min_free_disk_bytes=1 << 62)
    tight.open()
    try:
        assert tight.admit(new_job("REQ-DISK")).status is AdmissionStatus.STORAGE_CAPACITY
    finally:
        tight.close()


def test_ac12_restart_state_no_reexecution(tmp_path: Path) -> None:
    data = tmp_path / "data"
    first = JobStore(data, max_jobs=100, min_free_disk_bytes=0)
    first.open()
    admitted = first.admit(new_job("REQ-R"))
    assert admitted.job is not None
    job_id = admitted.job.assessment_id
    first.mark_running(
        job_id, run_id="run1", pid=999999, pgid=999999, start_token="tok", started_at="t"
    )
    # A second instance on the same data directory must fail to start.
    with pytest.raises(StoreLockedError):
        JobStore(data, max_jobs=100, min_free_disk_bytes=0).open()
    first.close()

    restarted = JobStore(data, max_jobs=100, min_free_disk_bytes=0)
    restarted.open()
    try:
        interrupted = restarted.interrupted()
        assert [j.assessment_id for j in interrupted] == [job_id]
        assert interrupted[0].worker_pgid == 999999 and interrupted[0].run_id == "run1"
        calls: list[str] = []

        def cannot_confirm(job: JobRecord) -> bool:
            calls.append(job.assessment_id)
            return False

        report = restarted.recover_interrupted(cannot_confirm, now="t2")
        assert report.pending == [job_id] and report.recovered == []
        still = restarted.get(job_id)
        assert still is not None and still.job_status == "running"
        assert restarted.admit(new_job("REQ-OTHER")).status is AdmissionStatus.BUSY

        report = restarted.recover_interrupted(lambda job: True, now="t3")
        assert report.recovered == [job_id] and report.pending == []
        done = restarted.get(job_id)
        assert done is not None and done.job_status == "finished"
        assert done.result_origin == "supervisor" and done.error_code == "SERVER_RESTARTED"
        assert done.result is not None and done.result["assessment_status"] == "failed"
        assert validate_result(done.result, case=CASE, checklist=CHECKLIST) == []
        assert restarted.active() is None
        # No automatic re-execution: the same request ID returns the terminal record.
        dup = restarted.admit(new_job("REQ-R"))
        assert dup.status is AdmissionStatus.DUPLICATE and dup.job is not None
        assert dup.job.error_code == "SERVER_RESTARTED"
        # A new request ID for the same case is a new evaluation.
        assert restarted.admit(new_job("REQ-R2")).status is AdmissionStatus.CREATED
        assert calls == [job_id]
    finally:
        restarted.close()


def test_second_instance_refused_with_fixed_code(tmp_path: Path) -> None:
    """Regression: the command line refuses a locked data directory with a fixed problem code."""
    from tests.api.helpers import make_config

    loaded = make_config(tmp_path)
    assert not data_dir_locked(loaded.data_dir)  # absent directory: nothing created
    assert not loaded.data_dir.exists()
    holder = JobStore(loaded.data_dir, max_jobs=100, min_free_disk_bytes=0)
    holder.open()
    try:
        assert data_dir_locked(loaded.data_dir)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "apctt_server",
                "--profile",
                "dev",
                "--config",
                str(tmp_path / "config.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 2
        assert "APCTT_START_REFUSED STORE_LOCKED" in proc.stderr
        assert "Traceback" not in proc.stderr
    finally:
        holder.close()
    assert not data_dir_locked(loaded.data_dir)
