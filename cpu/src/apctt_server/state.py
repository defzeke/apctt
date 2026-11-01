"""Server state, startup checks, job execution and recovery (SPEC-04, SPEC-08, SPEC-16.3).

The API owns admission and terminal state; the supervisor owns the worker lifecycle. The event loop
never blocks: SQLite, file hashing and process inspection run in threads.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apctt_contracts.config import ChecklistPayload, LoadedConfig
from apctt_contracts.hashing import (
    PackageDigestError,
    canonical_sha256,
    package_digest,
    package_manifest,
)
from apctt_contracts.ipc import RunHandle, RunLimits, RunOutcome, RunSpec
from apctt_contracts.results import build_supervisor_failure
from apctt_diagnostics.responses import DiagnosticsPolicy, job_diagnostics, job_failure
from apctt_runtime import procinfo
from apctt_runtime.supervisor import ProcessSupervisor
from apctt_server.documents import DocumentStore
from apctt_server.events import EventLog, utc_now
from apctt_server.store import JobRecord, JobStore

RECOVERY_RETRY_S = 2.0
CUDA_PROBE_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class PackageState:
    declared: str | None
    verified: str | None
    status: str  # verified | not_verified | mismatch
    reason: str | None


@dataclass(frozen=True)
class PendingCleanup:
    """A run whose processes have not been seen to stop; its job keeps the slot until they have.

    `error_code` is recorded if the job is still unfinished once cleanup is confirmed.
    """

    run_id: str | None
    pgid: int | None
    start_token: str | None
    error_code: str = "SERVER_RESTARTED"


@dataclass
class ServerState:
    loaded: LoadedConfig
    store: JobStore
    supervisor: ProcessSupervisor
    policy: DiagnosticsPolicy
    events: EventLog
    package: PackageState
    compute_status: dict[str, Any]
    static_reason: str | None
    bearer_token: str | None
    documents: DocumentStore
    recovery_pending: dict[str, PendingCleanup] = field(default_factory=dict)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    handles: dict[str, RunHandle] = field(default_factory=dict)
    recovery_loop: asyncio.Task[None] | None = None

    @property
    def grace_s(self) -> float:
        return self.loaded.config.limits.termination_grace_ms / 1000

    def admission_ready(self) -> bool:
        return self.static_reason is None and not self.recovery_pending

    def spawn(self, coroutine: Any) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def verify_package(loaded: LoadedConfig) -> PackageState:
    declared = loaded.config.module.declared_package_sha256
    roots: list[tuple[str, Path]] = [("package", loaded.package_root)]
    if not loaded.asset_root.is_relative_to(loaded.package_root):
        roots.append(("assets", loaded.asset_root))
    try:
        verified = package_digest(package_manifest(roots))
    except (PackageDigestError, OSError) as exc:
        reason = exc.reason if isinstance(exc, PackageDigestError) else "unreadable"
        return PackageState(declared, None, "not_verified", reason)
    if declared is not None and declared != verified:
        return PackageState(declared, verified, "mismatch", "declared_digest_mismatch")
    return PackageState(declared, verified, "verified", None)


def probe_compute(loaded: LoadedConfig) -> tuple[dict[str, Any], str | None]:
    compute = loaded.config.compute
    summary: dict[str, Any] = {"device": compute.device, "require_cuda": compute.require_cuda}
    if not compute.require_cuda:
        summary["status"] = "NOT_REQUIRED"
        return summary, None
    try:
        run_probe = importlib.import_module("apctt_runtime.cudacheck").run_probe
    except (ImportError, AttributeError):
        summary["status"] = "CUDA_UNAVAILABLE"
        return summary, "CUDA_UNAVAILABLE"
    report = run_probe(
        loaded.worker_python,
        compute.device,
        timeout_s=CUDA_PROBE_TIMEOUT_S,
        visible_devices=compute.visible_devices,
    )
    summary.update(report.summary())
    return summary, None if report.status == "PASS" else report.status


def build_state(loaded: LoadedConfig) -> ServerState:
    config = loaded.config
    store = JobStore(
        loaded.data_dir,
        max_jobs=config.limits.max_jobs,
        min_free_disk_bytes=config.limits.min_free_disk_bytes,
    )
    store.open()
    events = EventLog(loaded.data_dir / "events.jsonl")
    package = verify_package(loaded)
    compute_status, compute_reason = probe_compute(loaded)
    static_reason = compute_reason
    if package.status == "mismatch":
        static_reason = "CONFIG_INVALID"
    token = None
    if loaded.bearer_token_file is not None:
        token = loaded.bearer_token_file.read_text(encoding="utf-8").strip() or None
    policy = DiagnosticsPolicy(
        profile=config.profile,
        max_bytes=config.limits.max_diagnostics_bytes,
        max_field_errors=config.limits.max_field_errors,
        max_stack_frames=config.limits.max_stack_frames,
        allowed_expected=frozenset({config.schema_version, loaded.checklist.version}),
    )
    events.emit(
        "server_start",
        profile=config.profile,
        config_sha256=loaded.config_sha256,
        package_status=package.status,
        compute=compute_status.get("status"),
        ready_reason=static_reason,
    )
    return ServerState(
        loaded=loaded,
        store=store,
        documents=DocumentStore(loaded.data_dir),
        supervisor=ProcessSupervisor(),
        policy=policy,
        events=events,
        package=package,
        compute_status=compute_status,
        static_reason=static_reason,
        bearer_token=token,
    )


# recovery ------------------------------------------------------------------------------------


def _cleanup(state: ServerState, job: JobRecord) -> bool:
    if job.run_id is None:
        return True  # the run marker is stored before spawn: no worker was started
    return procinfo.recover_worker(
        job.run_id, job.worker_pgid, job.worker_start_token, grace_s=state.grace_s, timeout_s=10
    )


def register_pending(state: ServerState, assessment_id: str, pending: PendingCleanup) -> None:
    """Keep the slot for this job and retry cleanup until it is confirmed (RECOVERY_PENDING)."""
    state.recovery_pending[assessment_id] = pending
    state.events.emit("recovery_pending", assessment_id=assessment_id)
    if state.recovery_loop is None or state.recovery_loop.done():
        state.recovery_loop = asyncio.create_task(retry_recovery(state))
        state.tasks.add(state.recovery_loop)
        state.recovery_loop.add_done_callback(state.tasks.discard)


async def startup_recovery(state: ServerState) -> None:
    """Close interrupted jobs and free a slot left held by a finished job, never re-running."""
    interrupted = await asyncio.to_thread(state.store.interrupted)
    if interrupted:
        report = await asyncio.to_thread(
            state.store.recover_interrupted, lambda job: _cleanup(state, job), now=utc_now()
        )
        state.events.emit("startup_recovery", recovered=report.recovered, pending=report.pending)
        for job in interrupted:
            if job.assessment_id in report.pending:
                register_pending(
                    state,
                    job.assessment_id,
                    PendingCleanup(job.run_id, job.worker_pgid, job.worker_start_token),
                )
    held = await asyncio.to_thread(state.store.held_by_finished)
    if held is not None and held.assessment_id not in state.recovery_pending:
        if await asyncio.to_thread(_cleanup, state, held):
            await asyncio.to_thread(state.store.release_slot, held.assessment_id)
            state.events.emit("slot_released", assessment_id=held.assessment_id)
        else:
            register_pending(
                state,
                held.assessment_id,
                PendingCleanup(held.run_id, held.worker_pgid, held.worker_start_token),
            )


async def retry_recovery(state: ServerState) -> None:
    while state.recovery_pending:
        await asyncio.sleep(RECOVERY_RETRY_S)
        for assessment_id, pending in list(state.recovery_pending.items()):
            confirmed = await asyncio.to_thread(
                procinfo.recover_worker,
                pending.run_id or "",
                pending.pgid,
                pending.start_token,
                grace_s=state.grace_s,
                timeout_s=5,
            )
            if not confirmed:
                continue

            def only_this(record: JobRecord, target: str = assessment_id) -> bool:
                return record.assessment_id == target

            try:
                job = await asyncio.to_thread(state.store.get, assessment_id)
                if job is not None and job.job_status == "finished":
                    await asyncio.to_thread(state.store.release_slot, assessment_id)
                else:
                    await asyncio.to_thread(
                        state.store.recover_interrupted,
                        only_this,
                        now=utc_now(),
                        error_code=pending.error_code,
                    )
            except Exception:  # noqa: BLE001 - storage still failing: keep the slot, retry later
                state.events.emit("recovery_write_failed", assessment_id=assessment_id)
                continue
            state.recovery_pending.pop(assessment_id, None)
            state.events.emit("recovery_confirmed", assessment_id=assessment_id)


# execution -----------------------------------------------------------------------------------


def _outcome_diagnostics(state: ServerState, job: JobRecord, outcome: RunOutcome) -> dict[str, Any]:
    obs = outcome.observation
    timings: dict[str, float | None] = {
        "load": obs.timings_ms.load if obs else None,
        "assess": obs.timings_ms.assess if obs else None,
        "cleanup": obs.timings_ms.cleanup if obs else None,
        "total": outcome.total_ms,
    }
    reasons = dict(obs.unavailable_reasons) if obs else {}
    if obs is None:
        reasons.update(
            load="SIDECAR_UNAVAILABLE", assess="SIDECAR_UNAVAILABLE", cleanup="SIDECAR_UNAVAILABLE"
        )
    return job_diagnostics(
        policy=state.policy,
        phase="finished",
        timings_ms=timings,
        unavailable_reasons=reasons,
        warning_codes=list(outcome.warning_codes),
        output_issues=outcome.output_issues,
        criterion_ids=state.loaded.checklist.ids(),
        extra={
            "loading_policy": "cold_per_case",
            "worker_phase": obs.phase if obs else None,
            "exit_code": outcome.exit_code,
            "exit_signal": outcome.exit_signal,
            "timed_out": outcome.timed_out,
            "cleanup_confirmed": outcome.cleanup_confirmed,
            "raw_output_bytes": outcome.raw_output_bytes,
            "raw_output_sha256": outcome.raw_output_sha256,
            "stdout_bytes": outcome.stdout_bytes,
            "stderr_bytes": outcome.stderr_bytes,
            "stdout_truncated": outcome.stdout_truncated,
            "stderr_truncated": outcome.stderr_truncated,
            "compute": obs.compute.model_dump() if obs else None,
            "accepted_trace_id": job.accepted_trace_id,
            **(
                {"failure": job_failure(outcome.error_code, deadline_ms=_deadline_ms(state))}
                if outcome.result_origin == "supervisor" and outcome.error_code
                else {}
            ),
        },
    )


def _deadline_ms(state: ServerState) -> int:
    return state.loaded.config.limits.deadline_ms


def _run_spec(state: ServerState, job: JobRecord, run_id: str, artifact_dir: Path) -> RunSpec:
    loaded = state.loaded
    config = loaded.config
    return RunSpec(
        assessment_id=job.assessment_id,
        run_id=run_id,
        case=job.case(),
        case_meta=job.case_meta,
        checklist=ChecklistPayload.model_validate(loaded.checklist.payload()),
        module={
            "package_id": config.module.package_id,
            "entrypoint": config.module.entrypoint,
            "module_root": str(loaded.module_root),
            "asset_root": str(loaded.asset_root),
            "settings": config.module.settings,
        },
        compute=config.compute,
        worker_python=loaded.worker_python,
        launcher_argv=tuple(config.worker.launcher_argv),
        extra_env=dict(config.worker.extra_env),
        scratch_root=loaded.worker_tmp_dir,
        artifact_dir=artifact_dir,
        deadline_monotonic=time.monotonic() + config.limits.deadline_ms / 1000,
        limits=RunLimits.from_limits(config.limits),
        document_files=tuple(
            state.documents.path(document["sha256"]) for document in job.case()["documents"]
        ),
    )


async def run_job(state: ServerState, job: JobRecord) -> None:
    run_id = uuid.uuid4().hex
    handle: RunHandle | None = None
    try:
        await asyncio.to_thread(state.store.record_run_id, job.assessment_id, run_id)
        artifact_dir = state.loaded.data_dir / "jobs" / job.assessment_id
        await asyncio.to_thread(os.makedirs, artifact_dir, 0o700)
        spec = _run_spec(state, job, run_id, artifact_dir)
        handle = await state.supervisor.start(spec)
        state.handles[job.assessment_id] = handle
        if handle.pid > 0:
            await asyncio.to_thread(
                state.store.mark_running,
                job.assessment_id,
                run_id=run_id,
                pid=handle.pid,
                pgid=handle.pgid,
                start_token=handle.start_token,
                started_at=utc_now(),
            )
        state.events.emit("run_started", assessment_id=job.assessment_id, run_id=run_id)
        outcome = await state.supervisor.await_result(handle)
    except asyncio.CancelledError:
        raise  # shutdown: the next start recovers the job as SERVER_RESTARTED
    except Exception:  # noqa: BLE001 - convert to a recorded supervisor failure
        state.events.emit("runner_error", assessment_id=job.assessment_id, run_id=run_id)
        await _finish_internal_error(state, job, handle, run_id, None)
        return
    finally:
        state.handles.pop(job.assessment_id, None)
    try:
        await asyncio.to_thread(
            state.store.finish,
            job.assessment_id,
            origin=outcome.result_origin,
            result=outcome.result,
            result_sha256=canonical_sha256(outcome.result),
            error_code=outcome.error_code,
            raw_output_sha256=outcome.raw_output_sha256,
            raw_output_bytes=outcome.raw_output_bytes,
            raw_artifact=outcome.raw_artifact,
            finished_at=utc_now(),
            diagnostics=_outcome_diagnostics(state, job, outcome),
            release_slot=outcome.cleanup_confirmed,
        )
    except Exception:  # noqa: BLE001 - never leave the job running with no record of why
        # The terminal state becomes a supervisor failure that keeps the raw artifact. If that
        # write fails too, the recovery loop retries it until storage works again.
        state.events.emit("finish_failed", assessment_id=job.assessment_id, run_id=run_id)
        await _finish_internal_error(state, job, handle, run_id, outcome)
        return
    state.events.emit(
        "run_finished",
        assessment_id=job.assessment_id,
        run_id=run_id,
        origin=outcome.result_origin,
        error_code=outcome.error_code,
        total_ms=outcome.total_ms,
        cleanup_confirmed=outcome.cleanup_confirmed,
    )
    if not outcome.cleanup_confirmed:
        register_pending(state, job.assessment_id, _pending(run_id, handle))


def _pending(
    run_id: str, handle: RunHandle | None, error_code: str = "INTERNAL_ERROR"
) -> PendingCleanup:
    pgid = handle.pgid if handle is not None and handle.pgid > 1 else None
    token = handle.start_token if handle is not None else None
    return PendingCleanup(run_id, pgid, token, error_code)


async def _finish_internal_error(
    state: ServerState,
    job: JobRecord,
    handle: RunHandle | None,
    run_id: str,
    outcome: RunOutcome | None,
) -> None:
    """Record INTERNAL_ERROR, keeping any raw artifact; never leave the slot without a follow-up."""
    confirmed = outcome.cleanup_confirmed if outcome is not None else handle is None
    if not confirmed and handle is not None and handle.pgid > 1:
        confirmed, _ = await asyncio.to_thread(
            procinfo.terminate_group, handle.pgid, state.grace_s, 10.0
        )
    result = build_supervisor_failure(job.case_meta, "INTERNAL_ERROR")
    try:
        await asyncio.to_thread(
            state.store.finish,
            job.assessment_id,
            origin="supervisor",
            result=result,
            result_sha256=canonical_sha256(result),
            error_code="INTERNAL_ERROR",
            raw_output_sha256=outcome.raw_output_sha256 if outcome is not None else None,
            raw_output_bytes=outcome.raw_output_bytes if outcome is not None else None,
            raw_artifact=outcome.raw_artifact if outcome is not None else None,
            finished_at=utc_now(),
            diagnostics={
                "phase": "finished",
                "warning_codes": ["SUPERVISOR_INTERNAL_ERROR"],
                "failure": job_failure("INTERNAL_ERROR", deadline_ms=_deadline_ms(state)),
            },
            release_slot=confirmed,
        )
    except Exception:  # noqa: BLE001 - storage failing: the recovery loop retries the write
        state.events.emit("finish_failed", assessment_id=job.assessment_id)
        register_pending(state, job.assessment_id, _pending(run_id, handle))
        return
    if not confirmed:
        register_pending(state, job.assessment_id, _pending(run_id, handle))


async def shutdown(state: ServerState) -> None:
    handles = list(state.handles.values())
    for task in list(state.tasks):
        task.cancel()
    await asyncio.gather(*state.tasks, return_exceptions=True)
    for handle in handles:
        if handle.pgid > 1:
            await asyncio.to_thread(procinfo.terminate_group, handle.pgid, state.grace_s, 10.0)
    state.events.emit("server_stop")
    state.events.close()
    state.store.close()
