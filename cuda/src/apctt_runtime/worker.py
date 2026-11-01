"""APCTT worker wrapper. Runs inside the participant worker interpreter; standard library only.

Launched by the supervisor with a fixed argv and a reviewed environment:

    <worker-python> -s worker.py --run-id <run_id>

It reads exactly one JSON envelope from stdin, then runs load -> assess -> reset_case -> close in
this fresh process, writes result.json via an atomic rename, keeps a small observation sidecar
and exits. The wrapper never parses stdout/stderr and never unpickles anything.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL = "apctt-worker-v2"
EXIT_OK = 0
EXIT_MODULE_FAILURE = 3
EXIT_PROTOCOL = 4
MODULE_CODES = frozenset(
    {"MODULE_ERROR", "OUT_OF_MEMORY", "CUDA_UNAVAILABLE", "CUDA_RUNTIME_ERROR"}
)
_RECEIVED_AT = time.monotonic()


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Exactly what a module may know about the run (SPEC-09): no job store, labels or peers."""

    run_id: str
    seed: int
    time_budget_ms: int
    scratch_dir: Path

    def remaining_ms(self) -> float:
        """Budget left according to this process's own monotonic clock."""
        return self.time_budget_ms - (time.monotonic() - _RECEIVED_AT) * 1000.0


def _write_atomic(name: str, data: bytes) -> None:
    temporary = name + ".tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, name)


def _ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000.0, 3)


def _torch() -> Any:
    return sys.modules.get("torch")


def _synchronize() -> None:
    """Close the asynchronous CUDA work window before reading a timer, if CUDA is in use."""
    torch = _torch()
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001, S110 - timing must not change the module outcome
        pass


class Observer:
    def __init__(self, run_id: str, device: str) -> None:
        self.run_id = run_id
        self.phase = "initializing"
        self.timings: dict[str, float | None] = {"load": None, "assess": None, "cleanup": None}
        self.unavailable: dict[str, str] = {}
        self.warnings: list[str] = []
        self.compute: dict[str, Any] = {
            "device": device,
            "torch_version": None,
            "torch_cuda_build": None,
            "cuda_available": None,
            "kernel_probe_passed": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "measurement_method": None,
        }

    def write(self, phase: str | None = None) -> None:
        if phase is not None:
            self.phase = phase
        reasons = dict(self.unavailable)
        for name, value in self.timings.items():
            if value is None:
                reasons.setdefault(name, "NOT_REACHED")
        document = {
            "protocol_version": PROTOCOL,
            "run_id": self.run_id,
            "phase": self.phase,
            "source": "worker_observed",
            "timings_ms": self.timings,
            "unavailable_reasons": reasons,
            "compute": self.compute,
            "warning_codes": self.warnings,
        }
        try:
            _write_atomic("observation.json", json.dumps(document).encode("utf-8"))
        except OSError:
            pass  # the supervisor reports a missing sidecar; it never blocks the result

    def capture_compute(self) -> None:
        self.unavailable["kernel_probe_passed"] = "PROBE_RUNS_AT_SERVER_START"
        torch = _torch()
        if torch is None:
            self.unavailable["torch"] = "TORCH_NOT_IMPORTED"
            self.unavailable["peak_memory"] = "NOT_MEASURED"
            return
        try:
            self.compute["torch_version"] = str(torch.__version__)
            cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
            self.compute["torch_cuda_build"] = None if cuda_build is None else str(cuda_build)
            self.compute["cuda_available"] = bool(torch.cuda.is_available())
            if str(self.compute["device"]).startswith("cuda") and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                self.compute["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
                self.compute["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
                self.compute["measurement_method"] = (
                    "torch.cuda.max_memory_allocated/max_memory_reserved(device 0); "
                    "process-local caching allocator, not whole-device VRAM"
                )
            else:
                self.unavailable["peak_memory"] = "NOT_MEASURED"
        except Exception:  # noqa: BLE001
            self.unavailable["compute"] = "COMPUTE_QUERY_FAILED"


def _classify(exc: BaseException) -> str:
    code = getattr(exc, "apctt_error_code", None)
    if code in MODULE_CODES:
        return str(code)
    kind = type(exc)
    if kind.__name__ == "OutOfMemoryError" and kind.__module__.startswith("torch"):
        return "OUT_OF_MEMORY"
    return "MODULE_ERROR"


def _fail(run_id: str, phase: str, code: str) -> int:
    document = {"protocol_version": PROTOCOL, "run_id": run_id, "phase": phase, "error_code": code}
    _write_atomic("failure.json", json.dumps(document).encode("utf-8"))
    return EXIT_MODULE_FAILURE


def _plain(value: Any) -> Any:
    """Convert Mapping/sequence output to plain JSON types; reject anything else."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("object keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    raise TypeError("value is not JSON serializable")


def _close_quietly(instance: Any) -> None:
    for name in ("reset_case", "close"):
        try:
            getattr(instance, name)()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def main(argv: list[str]) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    if len(argv) != 2 or argv[0] != "--run-id":
        return EXIT_PROTOCOL
    run_id = argv[1]
    try:
        envelope = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if envelope.get("protocol_version") != PROTOCOL or envelope.get("run_id") != run_id:
            raise ValueError("envelope does not match this run")
        raw_context = envelope["context"]
        scratch = Path(raw_context["scratch_dir"])
        os.chdir(scratch)
        context = RunContext(
            run_id=str(raw_context["run_id"]),
            seed=int(raw_context["seed"]),
            time_budget_ms=int(raw_context["time_budget_ms"]),
            scratch_dir=scratch,
        )
        config = envelope["config"]
        case = envelope["case"]
        module_config = config["module"]
        device = str(config["compute"]["device"])
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return EXIT_PROTOCOL

    observer = Observer(run_id, device)
    observer.write("initializing")
    random.seed(context.seed)
    sys.path.insert(0, str(module_config["module_root"]))

    instance: Any = None
    observer.write("loading")
    start = time.monotonic()
    try:
        module_name, _, attribute = str(module_config["entrypoint"]).partition(":")
        instance = getattr(importlib.import_module(module_name), attribute)()
        instance.load(config, Path(module_config["asset_root"]))
        _synchronize()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        observer.timings["load"] = _ms(start)
        if instance is not None:
            _close_quietly(instance)
        observer.capture_compute()
        observer.write("finished")
        return _fail(run_id, "loading", _classify(exc))
    observer.timings["load"] = _ms(start)

    observer.write("assessing")
    start = time.monotonic()
    try:
        output = instance.assess(case, context)
        _synchronize()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        observer.timings["assess"] = _ms(start)
        _close_quietly(instance)
        observer.capture_compute()
        observer.write("finished")
        return _fail(run_id, "assessing", _classify(exc))
    observer.timings["assess"] = _ms(start)

    try:
        if not isinstance(output, Mapping):
            raise TypeError("assess() must return a mapping")
        data = json.dumps(_plain(output), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        traceback.print_exc()
        _close_quietly(instance)
        observer.capture_compute()
        observer.write("finished")
        return _fail(run_id, "assessing", "INVALID_OUTPUT")
    _write_atomic("result.json", data)

    observer.write("cleaning")
    start = time.monotonic()
    cleanup_failed = False
    for step in (instance.reset_case, instance.close):
        try:
            step()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            cleanup_failed = True
    observer.timings["cleanup"] = _ms(start)
    observer.capture_compute()
    observer.write("finished")
    if cleanup_failed:
        return _fail(run_id, "cleaning", "CLEANUP_ERROR")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
