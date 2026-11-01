"""Process identity and owned process-group termination (SPEC-08, SPEC-09).

Only process groups created for a worker run are signalled. Identity for restart recovery uses the
run marker in the worker argv (`--run-id <id>`) and the recorded leader start token, so a reused PID
belonging to an unrelated process is never signalled. Zombies count as terminated.
Linux reads /proc; other POSIX systems (macOS) use a fixed `ps` argv. Not an OS sandbox.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PS = "/bin/ps"
_LINUX = sys.platform.startswith("linux") and Path("/proc/self/stat").exists()


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    pgid: int
    state: str
    command: str


def _linux_stat(pid: int) -> tuple[str, int, str] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = data[data.rindex(")") + 2 :].split()
    return rest[0], int(rest[2]), rest[19]


def _linux_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def _ps(columns: str, *extra: str) -> list[str]:
    result = subprocess.run(
        [PS, *extra, "-o", columns], capture_output=True, text=True, check=False, timeout=10
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def list_processes(*, with_command: bool = False) -> list[ProcessInfo]:
    found: list[ProcessInfo] = []
    if _LINUX:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            stat = _linux_stat(pid)
            if stat is None:
                continue
            command = _linux_command(pid) if with_command else ""
            found.append(ProcessInfo(pid, stat[1], stat[0], command))
        return found
    columns = "pid=,pgid=,stat=,command=" if with_command else "pid=,pgid=,stat="
    for line in _ps(columns, "-A", "-ww"):
        parts = line.split(None, 3)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        command = parts[3] if len(parts) > 3 else ""
        found.append(ProcessInfo(int(parts[0]), int(parts[1]), parts[2], command))
    return found


def is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _LINUX:
        stat = _linux_stat(pid)
        return stat is not None and not stat[0].startswith("Z")
    lines = _ps("stat=", "-p", str(pid))
    return bool(lines) and not lines[0].strip().startswith("Z")


def start_token(pid: int) -> str | None:
    """Opaque start identity of a live process, or None if it does not exist."""
    if pid <= 0:
        return None
    if _LINUX:
        stat = _linux_stat(pid)
        return None if stat is None else f"linux-starttime:{stat[2]}"
    lines = _ps("lstart=", "-p", str(pid))
    return f"ps-lstart:{lines[0].strip()}" if lines else None


def group_members(pgid: int) -> list[int]:
    """Non-zombie processes in the group."""
    if pgid <= 1:
        return []
    return [p.pid for p in list_processes() if p.pgid == pgid and not p.state.startswith("Z")]


def find_worker_processes(run_id: str) -> list[ProcessInfo]:
    marker = f"--run-id {run_id}"
    return [
        p
        for p in list_processes(with_command=True)
        if marker in p.command and "apctt_runtime" in p.command and "worker.py" in p.command
    ]


def _signal_group(pgid: int, sig: int) -> None:
    if pgid <= 1:  # never signal our own group (0) or init
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        # Gone, or only an unreaped zombie leader (macOS reports EPERM), or not ours.
        pass


def wait_group_gone(pgid: int, timeout_s: float, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if not group_members(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def terminate_group(pgid: int, grace_s: float, timeout_s: float) -> tuple[bool, list[str]]:
    """SIGTERM, grace, SIGKILL, then confirm the owned group is empty."""
    signals: list[str] = []
    if not group_members(pgid):
        return True, signals
    _signal_group(pgid, signal.SIGTERM)
    signals.append("SIGTERM")
    if wait_group_gone(pgid, grace_s):
        return True, signals
    _signal_group(pgid, signal.SIGKILL)
    signals.append("SIGKILL")
    return wait_group_gone(pgid, timeout_s), signals


def recover_worker(
    run_id: str,
    pgid: int | None,
    recorded_start_token: str | None,
    *,
    grace_s: float,
    timeout_s: float,
) -> bool:
    """Stop a worker left by a previous API process. True only when its group is confirmed gone."""
    groups = {p.pgid for p in find_worker_processes(run_id)}
    if pgid is not None and pgid > 1:
        current = start_token(pgid)
        if current is None:
            if group_members(pgid):
                groups.add(pgid)  # leader gone, members of the old owned group remain
        elif recorded_start_token is not None and current == recorded_start_token:
            groups.add(pgid)
        # Otherwise the PID now belongs to another process: the old group no longer exists.
    confirmed = True
    for group in sorted(groups):
        ok, _ = terminate_group(group, grace_s, timeout_s)
        confirmed = confirmed and ok
    return confirmed
