"""Server-side CUDA probe launcher (stdlib only). The API process never imports torch.

The probe runs as a fresh process of the configured worker interpreter with a minimal environment,
the same executable and device exposure the worker will use. No CPU fallback exists: anything but a
verified kernel run is CUDA_UNAVAILABLE or CUDA_RUNTIME_ERROR.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apctt_contracts.strict_json import StrictJSONError, loads_strict

PROBE_SCRIPT = Path(__file__).resolve().parent / "cuda_probe_worker.py"
STATUSES = frozenset({"PASS", "CUDA_UNAVAILABLE", "CUDA_RUNTIME_ERROR"})
REASONS = frozenset(
    {
        "DEVICE_NOT_CUDA",
        "WORKER_PYTHON_NOT_FOUND",
        "TORCH_NOT_INSTALLED",
        "CPU_ONLY_TORCH",
        "CUDA_QUERY_FAILED",
        "CUDA_NOT_AVAILABLE",
        "DEVICE_NOT_VISIBLE",
        "KERNEL_PROBE_FAILED",
        "KERNEL_RESULT_INVALID",
        "PROBE_TIMEOUT",
        "PROBE_PROTOCOL_ERROR",
    }
)
_FIELDS = (
    "device",
    "torch_version",
    "torch_cuda_build",
    "cuda_available",
    "device_name",
    "total_memory_bytes",
    "compute_capability",
    "kernel_probe_passed",
    "elapsed_ms",
    "python_version",
    "platform",
)


@dataclass(frozen=True)
class ProbeReport:
    status: str
    reason: str | None
    fields: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        values = {name: self.fields.get(name) for name in _FIELDS}
        values["kernel_probe_passed"] = self.fields.get("kernel_probe_passed") is True
        return {"status": self.status, "reason": self.reason, **values}


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:200]
    return None


def run_probe(
    worker_python: Path,
    device: str,
    *,
    timeout_s: float,
    visible_devices: str | None = None,
) -> ProbeReport:
    if not re.fullmatch(r"cuda:[0-9]+", device):
        return ProbeReport("CUDA_UNAVAILABLE", "DEVICE_NOT_CUDA", {"device": device})
    if not worker_python.is_absolute() or not os.access(worker_python, os.X_OK):
        return ProbeReport("CUDA_UNAVAILABLE", "WORKER_PYTHON_NOT_FOUND", {"device": device})
    with tempfile.TemporaryDirectory(prefix="apctt-cuda-probe-") as home:
        env = {
            "PATH": os.pathsep.join([str(worker_python.parent), "/usr/bin", "/bin"]),
            "HOME": home,
            "TMPDIR": home,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        if visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        argv = [str(worker_python), "-s", str(PROBE_SCRIPT), "--device", device]
        process = subprocess.Popen(
            argv,
            cwd=home,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.communicate()
            return ProbeReport("CUDA_RUNTIME_ERROR", "PROBE_TIMEOUT", {"device": device})
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        data = loads_strict(lines[-1]) if lines else None
    except StrictJSONError:
        data = None
    if not isinstance(data, dict) or data.get("status") not in STATUSES:
        return ProbeReport("CUDA_RUNTIME_ERROR", "PROBE_PROTOCOL_ERROR", {"device": device})
    status = str(data["status"])
    reason = data.get("reason")
    expected_exit = {"PASS": 0, "CUDA_UNAVAILABLE": 3, "CUDA_RUNTIME_ERROR": 4}[status]
    if process.returncode != expected_exit or (reason is not None and reason not in REASONS):
        return ProbeReport("CUDA_RUNTIME_ERROR", "PROBE_PROTOCOL_ERROR", {"device": device})
    if status == "PASS" and data.get("kernel_probe_passed") is not True:
        return ProbeReport("CUDA_RUNTIME_ERROR", "PROBE_PROTOCOL_ERROR", {"device": device})
    fields = {name: _scalar(data.get(name)) for name in _FIELDS}
    return ProbeReport(status, None if reason is None else str(reason), fields)
