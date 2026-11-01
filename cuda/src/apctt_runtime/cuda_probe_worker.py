"""CUDA readiness probe. Runs in the worker interpreter (stdlib + torch), never in the API process.

Order (SPEC-16.3): torch/CUDA build → device access → bounded tensor → 64x64 matmul → synchronize →
finite/expected result → exit. Prints one JSON line. Exit 0 PASS, 3 unavailable, 4 runtime error.
Raw exception text goes to stderr only; the JSON carries fixed reason codes.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
import traceback
from typing import Any


def _emit(report: dict[str, Any], code: int) -> int:
    print(json.dumps(report), flush=True)
    return code


def main(argv: list[str]) -> int:
    device = argv[1] if len(argv) == 2 and argv[0] == "--device" else ""
    report: dict[str, Any] = {
        "status": None,
        "reason": None,
        "device": device,
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": None,
        "device_name": None,
        "total_memory_bytes": None,
        "compute_capability": None,
        "kernel_probe_passed": False,
        "elapsed_ms": None,
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }
    started = time.monotonic()
    if not re.fullmatch(r"cuda:[0-9]+", device):
        report.update(status="CUDA_UNAVAILABLE", reason="DEVICE_NOT_CUDA")
        return _emit(report, 3)
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        report.update(status="CUDA_UNAVAILABLE", reason="TORCH_NOT_INSTALLED")
        return _emit(report, 3)
    report["torch_version"] = str(getattr(torch, "__version__", None))
    build = getattr(getattr(torch, "version", None), "cuda", None)
    report["torch_cuda_build"] = None if build is None else str(build)
    if build is None:
        report.update(status="CUDA_UNAVAILABLE", reason="CPU_ONLY_TORCH")
        return _emit(report, 3)
    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        report.update(status="CUDA_RUNTIME_ERROR", reason="CUDA_QUERY_FAILED")
        return _emit(report, 4)
    report["cuda_available"] = available
    if not available:
        report.update(status="CUDA_UNAVAILABLE", reason="CUDA_NOT_AVAILABLE")
        return _emit(report, 3)
    index = int(device.split(":")[1])
    try:
        if index >= torch.cuda.device_count():
            report.update(status="CUDA_UNAVAILABLE", reason="DEVICE_NOT_VISIBLE")
            return _emit(report, 3)
        props = torch.cuda.get_device_properties(index)
        report["device_name"] = str(props.name)
        report["total_memory_bytes"] = int(props.total_memory)
        report["compute_capability"] = f"{props.major}.{props.minor}"
        target = torch.device(device)
        a = torch.full((64, 64), 0.5, dtype=torch.float32, device=target)
        b = a @ a
        torch.cuda.synchronize(target)
        finite = bool(torch.isfinite(b).all().item())
        expected = abs(float(b[0, 0].item()) - 16.0) < 1e-4
        on_device = b.device.type == "cuda"
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        report.update(status="CUDA_RUNTIME_ERROR", reason="KERNEL_PROBE_FAILED")
        return _emit(report, 4)
    report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
    if not (finite and expected and on_device):
        report.update(status="CUDA_RUNTIME_ERROR", reason="KERNEL_RESULT_INVALID")
        return _emit(report, 4)
    report.update(status="PASS", reason=None, kernel_probe_passed=True)
    return _emit(report, 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
