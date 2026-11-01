"""Fake participant modules for runtime tests. Stdlib only; imported inside the worker process."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def valid_result(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    ids = [c["criterion_id"] for c in config["checklist"]["criteria"]]
    document = case["documents"][0]
    return {
        "schema_version": case["schema_version"],
        "checklist_version": case["checklist_version"],
        "case_id": case["case_id"],
        "assessment_target": case["assessment_target"],
        "assessment_status": "abstained",
        "predicted_trl": None,
        "criterion_status": {cid: "met" for cid in ids},
        "evidence_reference": {
            cid: [{"document_id": document["document_id"], "page": 1}] for cid in ids
        },
        "missing_evidence": {},
        "abstention_reason": "insufficient evidence",
        "error_code": None,
    }


class Valid:
    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        self.config = config
        self.settings = config["module"].get("settings", {})

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        return valid_result(case, self.config)

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        pass


class Crash(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        os._exit(9)


class Raises(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        raise RuntimeError("SENTINEL-module-exception " + os.getcwd())


class LoadRaises(Valid):
    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        raise ValueError("SENTINEL-load-failure")


class CudaMissing(Valid):
    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        error = RuntimeError("no CUDA")
        error.apctt_error_code = "CUDA_UNAVAILABLE"  # type: ignore[attr-defined]
        raise error


class InvalidUtf8(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        Path(context.scratch_dir, "result.json").write_bytes(b'{"a":"\xff\xfe"}')
        os._exit(0)


class PickleOutput(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        Path(context.scratch_dir, "result.json").write_bytes(b"\x80\x04\x95\x05\x00\x00\x00")
        os._exit(0)


class SymlinkResult(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        target = Path(context.scratch_dir, "home", "outside.json")
        target.write_text(json.dumps(valid_result(case, self.config)))
        os.symlink(target, Path(context.scratch_dir, "result.json"))
        os._exit(0)


class DirectoryResult(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        Path(context.scratch_dir, "result.json").mkdir()
        os._exit(0)


class Oversize(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        result = valid_result(case, self.config)
        result["criterion_rationale"] = {cid: "x" * 10_000 for cid in result["criterion_status"]}
        return result


class NoResult(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        os._exit(0)


class ContractViolation(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        result = valid_result(case, self.config)
        del result["missing_evidence"]
        return result


class MetWithoutEvidence(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        result = valid_result(case, self.config)
        result["evidence_reference"] = {cid: [] for cid in result["criterion_status"]}
        return result


class NonSerializable(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        result = valid_result(case, self.config)
        result["confidence"] = float("nan")
        return result


class CleanupFails(Valid):
    def close(self) -> None:
        raise RuntimeError("SENTINEL-cleanup")


class EnvDump(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        print(
            json.dumps(
                {
                    "env": dict(os.environ),
                    "cwd": os.getcwd(),
                    "argv": sys.argv,
                    "sys_path0": sys.path[0],
                    "apctt_server_loaded": "apctt_server" in sys.modules,
                }
            ),
            flush=True,
        )
        return valid_result(case, self.config)


class ContextDump(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        fields = {f.name: getattr(context, f.name) for f in dataclasses.fields(context)}
        fields["scratch_dir"] = str(fields["scratch_dir"])
        print(json.dumps({"context": fields}), flush=True)
        return valid_result(case, self.config)


class Slow(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        time.sleep(float(self.settings.get("sleep_s", 0.2)))
        return valid_result(case, self.config)


def _hang(seconds: float = 300.0) -> None:
    time.sleep(seconds)


class HangAssess(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        print(json.dumps({"pid": os.getpid()}), flush=True)
        _hang()
        return valid_result(case, self.config)


class HangLoad(Valid):
    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        _hang()


class HangClose(Valid):
    def close(self) -> None:
        _hang()


class IgnoreTerm(Valid):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        import signal

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _hang()
        return valid_result(case, self.config)


class SpawnAndHang(Valid):
    """Starts a grandchild in the same process group, reports its PID, then hangs."""

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        import subprocess

        child = subprocess.Popen(["/bin/sleep", "300"])
        print(json.dumps({"grandchild": child.pid}), flush=True)
        _hang()
        return valid_result(case, self.config)


class LeaveBackground(Valid):
    """Returns a valid result but leaves a background process in the group."""

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        import subprocess

        child = subprocess.Popen(["/bin/sleep", "300"])
        print(json.dumps({"grandchild": child.pid}), flush=True)
        return valid_result(case, self.config)


class CrashAfterResult(Valid):
    def reset_case(self) -> None:
        os._exit(7)
