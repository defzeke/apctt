"""CPU format example of the AssessmentModule protocol (SPEC-09). DEMO rules only.

Entry point: participant_example.adapter:AssessmentModule
It runs with the checklist of the server configuration (config/dev.json: the official checklist,
decision D74) and takes the criteria from it; module.settings.criteria_phrases must name exactly
those criteria. A criterion with an empty phrase list is not looked for (insufficient evidence).
Replace this class with your frozen assessment pipeline; keep the four methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from participant_example import demo_rules


class AssessmentModule:
    def __init__(self) -> None:
        self._checklist_version = ""
        self._criterion_ids: list[str] = []
        self._phrases: dict[str, list[str]] = {}
        self._completed_trl: int | None = None

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        checklist = config["checklist"]
        self._checklist_version = checklist["version"]
        self._criterion_ids = [c["criterion_id"] for c in checklist["criteria"]]
        settings = config["module"]["settings"]
        phrases = settings["criteria_phrases"]
        if set(phrases) != set(self._criterion_ids):
            raise ValueError("criteria_phrases must cover exactly the checklist criteria")
        self._phrases = {cid: list(phrases[cid]) for cid in self._criterion_ids}
        trl = settings.get("completed_trl")
        self._completed_trl = trl if isinstance(trl, int) and 1 <= trl <= 9 else None

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        observed = demo_rules.evaluate(case, self._criterion_ids, self._phrases)
        result = demo_rules.base_result(case, self._checklist_version)
        result.update(
            criterion_status=observed["criterion_status"],
            evidence_reference=observed["evidence_reference"],
            missing_evidence=observed["missing_evidence"],
            error_code=None,
        )
        # Only the criteria with configured phrases decide: all of them met gives completed.
        searched = [cid for cid in self._criterion_ids if self._phrases[cid]]
        all_met = bool(searched) and all(
            observed["criterion_status"][cid] == "met" for cid in searched
        )
        if all_met and self._completed_trl is not None:
            result.update(
                assessment_status="completed",
                predicted_trl=self._completed_trl,
                abstention_reason=None,
                trl_rationale=(
                    "DEMO rule: a phrase was found for every criterion with configured phrases; "
                    "not a TRL judgment."
                ),
            )
        else:
            result.update(
                assessment_status="abstained",
                predicted_trl=None,
                abstention_reason="insufficient evidence",
            )
        if observed["criterion_rationale"]:
            result["criterion_rationale"] = observed["criterion_rationale"]
        return result

    def reset_case(self) -> None:
        """No per-case state is kept; a cold worker process ends after close()."""

    def close(self) -> None:
        """Nothing to release in the CPU example."""
