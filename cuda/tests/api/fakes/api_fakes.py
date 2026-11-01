"""Fake participant modules for API tests. Stdlib only; loaded inside worker processes."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any


def _result(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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
    """Records each real execution as a file in settings.counter_dir, optionally sleeping."""

    def load(self, config: dict[str, Any], asset_root: Path) -> None:
        self.config = config
        self.settings = config["module"]["settings"]

    def assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        counter = Path(self.settings["counter_dir"])
        (counter / f"{uuid.uuid4().hex}.{os.getpid()}").write_text(case["case_id"])
        time.sleep(float(self.settings.get("sleep_s", 0)))
        return _result(case, self.config)

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        pass
