"""Fake participant modules for real-server QA tests. Stdlib only; run inside worker processes."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def result_for(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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


class Counting:
    """Writes one file per real execution (name holds the worker PID), then optionally sleeps."""

    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        self.config = config
        self.settings = config["module"]["settings"]

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        counter = Path(self.settings["counter_dir"])
        (counter / f"{os.getpid()}.{uuid.uuid4().hex}").write_text(case["case_id"])
        time.sleep(float(self.settings.get("sleep_s", 0)))
        return result_for(case, self.config)

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        pass


class Leaky(Counting):
    """Puts a sentinel into stdout, stderr and an exception message, and dumps its environment."""

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        sentinel = self.settings["sentinel"]
        print(json.dumps({"env": dict(os.environ)}), flush=True)
        print(f"stdout {sentinel}", flush=True)
        print(f"stderr {sentinel}", file=sys.stderr, flush=True)
        raise RuntimeError(f"exception {sentinel} {os.getcwd()}")


class SymlinkResult(Counting):
    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        secret = Path(self.settings["outside_file"])
        os.symlink(secret, Path(context.scratch_dir, "result.json"))
        os._exit(0)


class SymlinkSidecar(Counting):
    """Valid result; replaces the observation sidecar with a symlink to an outside file."""

    def close(self) -> None:
        target = Path(self.settings["outside_file"])
        link = Path(os.getcwd(), "observation.json")
        link.unlink(missing_ok=True)
        os.symlink(target, link)
