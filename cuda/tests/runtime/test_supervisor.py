"""SPEC-08/09/16.6 cold-per-case supervisor: AC-10, AC-11, AC-12 (runtime level)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from apctt_contracts.hashing import sha256_hex
from apctt_contracts.ipc import RunHandle, RunOutcome, RunSpec
from apctt_runtime import procinfo, protocol
from apctt_runtime.supervisor import ProcessSupervisor

from .helpers import make_spec


def run_with_handle(spec: RunSpec) -> tuple[RunOutcome, RunHandle]:
    async def main() -> tuple[RunOutcome, RunHandle]:
        supervisor = ProcessSupervisor()
        handle = await supervisor.start(spec)
        return await supervisor.await_result(handle), handle

    return asyncio.run(main())


def run(spec: RunSpec) -> RunOutcome:
    return run_with_handle(spec)[0]


def stdout_values(spec: RunSpec) -> dict:
    merged: dict = {}
    for line in (spec.artifact_dir / "stdout.log").read_text().splitlines():
        if line.startswith("{"):
            merged.update(json.loads(line))
    return merged


def wait_for_stdout(spec: RunSpec, key: str, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = spec.artifact_dir / "stdout.log"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("{") and key in line:
                    return int(json.loads(line)[key])
        time.sleep(0.02)
    raise AssertionError(f"{key} not reported")


def test_ac10_timeout_process_group(tmp_path: Path) -> None:
    # Timeouts in load, assess and cleanup all end as TIMEOUT with a confirmed dead group.
    for entrypoint in ("behaviours:HangLoad", "behaviours:HangAssess", "behaviours:HangClose"):
        spec = make_spec(tmp_path, entrypoint, deadline_s=1.0)
        started = time.monotonic()
        outcome, handle = run_with_handle(spec)
        elapsed = time.monotonic() - started
        assert outcome.error_code == "TIMEOUT" and outcome.timed_out, entrypoint
        assert outcome.result_origin == "supervisor" and outcome.cleanup_confirmed
        assert 0.9 <= elapsed < 8.0, (entrypoint, elapsed)
        assert outcome.total_ms >= 900
        assert handle.pgid == handle.pid and not procinfo.group_members(handle.pgid)
    # Result written before a cleanup hang: still TIMEOUT, raw bytes preserved.
    spec = make_spec(tmp_path, "behaviours:HangClose", deadline_s=1.0)
    outcome = run(spec)
    raw = (spec.artifact_dir / "raw_output.bin").read_bytes()
    assert outcome.error_code == "TIMEOUT" and outcome.raw_output_sha256 == sha256_hex(raw)

    # A grandchild in the owned process group is terminated with the worker.
    spec = make_spec(tmp_path, "behaviours:SpawnAndHang", deadline_s=1.5)
    outcome = run(spec)
    grandchild = stdout_values(spec)["grandchild"]
    assert outcome.error_code == "TIMEOUT" and outcome.cleanup_confirmed
    assert not procinfo.is_alive(grandchild)

    # SIGTERM ignored: SIGKILL after the grace period.
    spec = make_spec(tmp_path, "behaviours:IgnoreTerm", deadline_s=1.0)
    started = time.monotonic()
    outcome = run(spec)
    assert outcome.error_code == "TIMEOUT" and outcome.cleanup_confirmed
    assert "TERMINATED_SIGKILL" in outcome.warning_codes
    assert time.monotonic() - started >= 1.0 + spec.limits.termination_grace_ms / 1000 - 0.1

    # Normal exit that leaves a background process: result kept, group cleaned before return.
    spec = make_spec(tmp_path, "behaviours:LeaveBackground")
    outcome = run(spec)
    assert outcome.result_origin == "module" and outcome.cleanup_confirmed
    assert "DESCENDANTS_TERMINATED" in outcome.warning_codes
    assert not procinfo.is_alive(stdout_values(spec)["grandchild"])


def test_ac11_worker_failure_preserves_raw(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, "behaviours:CrashAfterResult")
    outcome = run(spec)
    raw = (spec.artifact_dir / "raw_output.bin").read_bytes()
    assert outcome.error_code == "WORKER_CRASH" and outcome.result_origin == "supervisor"
    assert outcome.exit_code == 7 and outcome.raw_output_sha256 == sha256_hex(raw)
    assert json.loads(raw)["assessment_status"] == "abstained"  # original bytes untouched

    spec = make_spec(tmp_path, "behaviours:Valid")
    ok = run(spec)
    assert ok.result_origin == "module" and ok.cleanup_confirmed and ok.exit_code == 0
    assert ok.observation is not None and ok.observation.phase == "finished"

    # Worker interpreter missing: supervisor failure, no exception escapes.
    spec = make_spec(tmp_path, "behaviours:Valid")
    broken = RunSpec(**{**spec.__dict__, "worker_python": tmp_path / "missing-python"})
    failed = run(broken)
    assert failed.error_code == "WORKER_CRASH" and "WORKER_SPAWN_FAILED" in failed.warning_codes

    # Oversized logs are capped and flagged; the result is still the module result.
    spec = make_spec(tmp_path, "behaviours:EnvDump", max_log_bytes=64)
    logged = run(spec)
    assert logged.result_origin == "module" and logged.stdout_truncated
    assert (spec.artifact_dir / "stdout.log").stat().st_size == 64


def test_ac12_recovery_waits_for_cleanup(tmp_path: Path, monkeypatch) -> None:
    # 1) An orphan worker left by a crashed API process is found by its run marker and stopped.
    #    It is launched outside asyncio: an asyncio transport kills its child when closed.
    spec = make_spec(tmp_path, "behaviours:HangAssess", deadline_s=120)
    scratch = protocol.prepare_scratch(spec.scratch_root, spec.run_id)
    with open(spec.artifact_dir / "stdout.log", "wb") as out:
        orphan = subprocess.Popen(
            protocol.worker_argv(spec),
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=subprocess.DEVNULL,
            env=protocol.worker_env(spec, scratch.path),
            cwd=scratch.path,
            start_new_session=True,
        )
    assert orphan.stdin is not None
    orphan.stdin.write(protocol.build_envelope(spec, scratch.path, protocol.time_budget_ms(spec)))
    orphan.stdin.close()
    try:
        worker_pid = wait_for_stdout(spec, "pid")
        assert worker_pid == orphan.pid and procinfo.is_alive(orphan.pid)
        token = procinfo.start_token(orphan.pid)
        # The marker alone finds it even without a recorded PGID.
        assert procinfo.recover_worker(spec.run_id, None, None, grace_s=0.5, timeout_s=5)
        orphan.wait(timeout=5)
        assert not procinfo.is_alive(orphan.pid)
        assert token is not None
    finally:
        if orphan.poll() is None:
            os.killpg(orphan.pid, signal.SIGKILL)
            orphan.wait()
        scratch.close(remove=True)

    # 2) A recorded PGID now owned by another process (PID reuse) is never signalled.
    foreign = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
    try:
        assert procinfo.recover_worker(
            "0" * 32, foreign.pid, "token-of-a-different-process", grace_s=0.2, timeout_s=1
        )
        assert foreign.poll() is None and procinfo.is_alive(foreign.pid)
    finally:
        os.killpg(foreign.pid, signal.SIGKILL)
        foreign.wait()

    # 3) If cleanup cannot be confirmed, recovery reports failure instead of freeing the slot.
    stubborn = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
    try:
        monkeypatch.setattr(procinfo, "_signal_group", lambda pgid, sig: None)
        token = procinfo.start_token(stubborn.pid)
        started = time.monotonic()
        assert not procinfo.recover_worker(
            "1" * 32, stubborn.pid, token, grace_s=0.2, timeout_s=0.6
        )
        assert time.monotonic() - started >= 0.6
    finally:
        monkeypatch.undo()
        os.killpg(stubborn.pid, signal.SIGKILL)
        stubborn.wait()
