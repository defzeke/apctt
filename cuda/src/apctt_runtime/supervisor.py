"""SPEC-09/16.6/17.3 cold-per-case supervisor.

One fresh worker process per case, in its own session/process group. The watchdog deadline covers
load, assess, cleanup and exit. On expiry: SIGTERM, grace, SIGKILL on the owned group, then the
group must be confirmed empty. After a normal exit, remaining group members are terminated too.
`total_ms` runs from before spawn until the group is confirmed gone (monotonic clock).
Blocking process inspection and file work run in threads, never on the event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field

from apctt_contracts.ipc import RunHandle, RunObservation, RunOutcome, RunSpec, TerminationOutcome
from apctt_runtime import procinfo
from apctt_runtime.protocol import (
    Scratch,
    StreamSink,
    build_envelope,
    collect_outcome,
    prepare_scratch,
    read_observation,
    stage_documents,
    time_budget_ms,
    worker_argv,
    worker_env,
)


@dataclass
class _ActiveRun:
    spec: RunSpec
    handle: RunHandle
    scratch: Scratch
    stdout: StreamSink
    stderr: StreamSink
    process: asyncio.subprocess.Process | None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)


async def _wait_exit(process: asyncio.subprocess.Process, timeout_s: float) -> bool:
    """Wait for the leader to exit. `Process.wait()` also waits for every pipe to close, which a
    surviving descendant can hold open; the return code is available as soon as the leader exits."""
    deadline = time.monotonic() + timeout_s
    while process.returncode is None:
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


class ProcessSupervisor:
    """Implements apctt_contracts.ipc.Supervisor for local subprocess workers."""

    def __init__(self, *, confirm_timeout_s: float = 10.0) -> None:
        self.confirm_timeout_s = confirm_timeout_s
        self._runs: dict[str, _ActiveRun] = {}

    async def start(self, run_spec: RunSpec) -> RunHandle:
        spec = run_spec
        started = time.monotonic()
        scratch = await asyncio.to_thread(prepare_scratch, spec.scratch_root, spec.run_id)
        try:
            documents = await asyncio.to_thread(stage_documents, scratch.path, spec.document_files)
        except OSError:
            scratch.close(remove=True)
            raise
        stdout = StreamSink(spec.artifact_dir / "stdout.log", spec.limits.max_log_bytes)
        stderr = StreamSink(spec.artifact_dir / "stderr.log", spec.limits.max_log_bytes)
        try:
            process = await asyncio.create_subprocess_exec(
                *worker_argv(spec),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=scratch.path,
                env=worker_env(spec, scratch.path),
                start_new_session=True,
            )
        except OSError:
            handle = RunHandle(spec.run_id, 0, 0, None, started, scratch.path)
            self._runs[spec.run_id] = _ActiveRun(spec, handle, scratch, stdout, stderr, None)
            return handle
        token = await asyncio.to_thread(procinfo.start_token, process.pid)
        handle = RunHandle(spec.run_id, process.pid, process.pid, token, started, scratch.path)
        run = _ActiveRun(spec, handle, scratch, stdout, stderr, process)
        payload = build_envelope(spec, scratch.path, time_budget_ms(spec), documents)
        assert process.stdin and process.stdout and process.stderr
        run.tasks = [
            asyncio.create_task(self._feed(process.stdin, payload)),
            asyncio.create_task(self._pump(process.stdout, stdout)),
            asyncio.create_task(self._pump(process.stderr, stderr)),
        ]
        self._runs[spec.run_id] = run
        return handle

    @staticmethod
    async def _feed(stream: asyncio.StreamWriter, payload: bytes) -> None:
        try:
            stream.write(payload)
            await stream.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.close()

    @staticmethod
    async def _pump(stream: asyncio.StreamReader, sink: StreamSink) -> None:
        while chunk := await stream.read(65536):
            sink.write(chunk)

    async def observe(self, run_handle: RunHandle) -> RunObservation:
        run = self._runs.get(run_handle.run_id)
        if run is None or run.process is None:
            return RunObservation(run_handle.run_id, False, None, ())
        observation, warnings = await asyncio.to_thread(
            read_observation, run.scratch, run.spec.run_id, run.spec.limits.max_observation_bytes
        )
        alive = run.process.returncode is None
        return RunObservation(run_handle.run_id, alive, observation, tuple(warnings))

    async def terminate(self, run_handle: RunHandle, reason: str) -> TerminationOutcome:
        run = self._runs.get(run_handle.run_id)
        started = time.monotonic()
        if run is None or run.process is None or run_handle.pgid <= 1:
            return TerminationOutcome(reason, (), True, 0.0)
        grace = run.spec.limits.termination_grace_ms / 1000
        confirmed, signals = await asyncio.to_thread(
            procinfo.terminate_group, run_handle.pgid, grace, self.confirm_timeout_s
        )
        if not await _wait_exit(run.process, self.confirm_timeout_s):
            confirmed = False
        elapsed = (time.monotonic() - started) * 1000
        return TerminationOutcome(reason, tuple(signals), confirmed, round(elapsed, 3))

    async def await_result(self, run_handle: RunHandle) -> RunOutcome:
        run = self._runs[run_handle.run_id]
        spec = run.spec
        warnings: list[str] = []
        timed_out = False
        confirmed = True
        signals: tuple[str, ...] = ()
        exit_code: int | None = None
        exit_signal: int | None = None
        try:
            if run.process is None:
                warnings.append("WORKER_SPAWN_FAILED")
            else:
                budget = max(0.0, spec.deadline_monotonic - time.monotonic())
                if not await _wait_exit(run.process, budget):
                    timed_out = True
                    termination = await self.terminate(run_handle, "deadline")
                    confirmed, signals = termination.cleanup_confirmed, termination.signals
                else:
                    members = await asyncio.to_thread(procinfo.group_members, run_handle.pgid)
                    if members:
                        warnings.append("DESCENDANTS_TERMINATED")
                        termination = await self.terminate(run_handle, "descendants")
                        confirmed, signals = termination.cleanup_confirmed, termination.signals
                if run.tasks:
                    done, pending = await asyncio.wait(run.tasks, timeout=self.confirm_timeout_s)
                    for task in pending:
                        task.cancel()
                code = run.process.returncode
                if code is not None:
                    exit_code, exit_signal = (code, None) if code >= 0 else (None, -code)
            if "SIGKILL" in signals:
                warnings.append("TERMINATED_SIGKILL")
            if not confirmed:
                warnings.append("CLEANUP_UNCONFIRMED")
            run.stdout.close()
            run.stderr.close()
            total_ms = (time.monotonic() - run_handle.started_monotonic) * 1000
            outcome = await asyncio.to_thread(
                collect_outcome,
                spec,
                run.scratch,
                exit_code=exit_code,
                exit_signal=exit_signal,
                timed_out=timed_out,
                cleanup_confirmed=confirmed,
                total_ms=total_ms,
                stdout=run.stdout.summary(),
                stderr=run.stderr.summary(),
            )
        finally:
            self._runs.pop(run_handle.run_id, None)
            await asyncio.to_thread(run.scratch.close, remove=confirmed)
        merged = tuple(dict.fromkeys((*outcome.warning_codes, *warnings)))
        return dataclasses.replace(outcome, warning_codes=merged)
