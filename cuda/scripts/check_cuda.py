"""CUDA pre-check launcher (SPEC-16.3). Standard library only; torch runs in the worker process.

usage: python scripts/check_cuda.py --worker-python /approved/worker/bin/python \
           --device cuda:0 --output /approved/run/cuda.json

Exit 0 only when a real kernel ran on the requested device. Otherwise exit 3; no CPU fallback.
A PASS is not evidence of model VRAM fit, isolation or TRL performance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from apctt_runtime.cudacheck import run_probe
except ImportError:  # running from a checkout without an installed package
    sys.path.insert(0, str(ROOT / "src"))
    from apctt_runtime.cudacheck import run_probe


def host_gpus() -> dict:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return {"nvidia_smi": None, "reason": "NVIDIA_SMI_NOT_FOUND"}
    result = subprocess.run(
        [tool, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "nvidia_smi": rows,
        "exit_code": result.returncode,
        "note": "driver-reported values only; not a PyTorch compatibility check",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = run_probe(Path(args.worker_python), args.device, timeout_s=args.timeout)
    record = {
        "record_version": "apctt-cuda-check-v1",
        "checked_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "host": {"platform": platform.platform(), **host_gpus()},
        "worker_python": args.worker_python,
        "device": args.device,
        "probe": report.summary(),
        "cuda_verified": report.status == "PASS",
        "scope": "small kernel probe only; not model VRAM fitness, isolation or TRL performance",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": report.status, "reason": report.reason, "output": str(args.output)})
    )
    return 0 if report.status == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
