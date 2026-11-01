"""AssessmentModule entry point: rules first, optional SmolLM3 refinement second."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import rules
from .extract import extract
from .llm import LlmRefiner

MET = "met"
NOT_MET = "not met"
INSUFFICIENT = "insufficient evidence"


class AssessmentModule:
    def __init__(self) -> None:
        self._version = ""
        self._criteria: list[Mapping[str, Any]] = []
        self._assignments: dict[str, rules.Rule | None] = {}
        self._settings: dict[str, Any] = {}
        self._asset_root = Path(".")
        self._llm: LlmRefiner | None = None

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        checklist = config["checklist"]
        self._version = str(checklist["version"])
        self._criteria = list(checklist["criteria"])
        self._assignments = rules.match_rules(self._criteria)
        module = config.get("module") or {}
        self._settings = dict(module.get("settings") or {})
        self._asset_root = Path(asset_root)
        if self._settings.get("use_llm", True):
            self._llm = LlmRefiner(self._asset_root)

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        case = dict(case)
        try:
            return self._assess(case, context)
        except Exception as error:
            print(f"assessment fallback: {type(error).__name__}: {error}", file=sys.stderr)
            return self._abstain(case, "The documents could not be assessed.")

    def _assess(self, case: dict[str, Any], context: Any) -> dict[str, Any]:
        remaining_ms = float(context.remaining_ms())
        deadline = time.monotonic() + max(1.0, remaining_ms / 1000.0 - 3.0)
        max_pages = int(self._settings.get("max_pages", 300))
        sentences, pages, documents = extract(case, deadline, max_pages)
        if not sentences:
            return self._abstain(case, "No readable text was found in the documents.")

        findings, predicted = rules.evaluate(
            sentences, self._assignments, case, pages)

        predicted, source = self._refine(case, findings, pages, predicted, context)
        rules.ensure_levels(case, self._assignments, findings, predicted, sentences, pages)

        result = self._result(case, findings, predicted)
        if predicted is not None:
            result["trl_rationale"] = self._rationale(
                predicted, findings, source)
            strength = max(
                (f.strength for f in findings.values() if f.status == MET), default=0.5)
            result["confidence"] = round(min(0.85, max(0.4, strength)), 2)
            result["uncertainty"] = {
                "kind": "trl_interval",
                "lower": max(1, predicted - 1),
                "upper": min(9, predicted + 1),
            }
            result["criterion_rationale"] = {
                cid: finding.rationale
                for cid, finding in findings.items()
                if finding.rationale and finding.status == MET
            }
        print(
            f"assessed {len(documents)} document(s), {len(sentences)} sentences, "
            f"trl={predicted}, source={source}",
            file=sys.stderr,
        )
        return result

    def _refine(
        self,
        case: dict[str, Any],
        findings: dict[str, rules.Finding],
        pages: Mapping[tuple[str, int], str],
        predicted: int | None,
        context: Any,
    ) -> tuple[int | None, str]:
        if self._llm is None or not self._llm.available():
            return predicted, "rules"
        remaining_ms = float(context.remaining_ms())
        minimum = int(self._settings.get("llm_min_remaining_ms", 90000))
        if remaining_ms < minimum:
            return predicted, "rules"
        excerpts = self._excerpts(findings, pages)
        if not excerpts:
            return predicted, "rules"
        deadline = time.monotonic() + max(5.0, remaining_ms / 1000.0 - 20.0)
        value = self._llm.predict(case, excerpts, deadline)
        if value is None:
            return predicted, "rules"
        return value, "rules+adapter"

    def _excerpts(
        self,
        findings: Mapping[str, rules.Finding],
        pages: Mapping[tuple[str, int], str],
    ) -> str:
        ranked: list[tuple[int, str, int]] = []
        for cid, finding in findings.items():
            if finding.status != MET:
                continue
            rule = self._assignments.get(cid)
            level = rule.level if rule is not None and rule.level is not None else 0
            for item in finding.evidence:
                key = (str(item["document_id"]), int(item["page"]))
                text = pages.get(key, "")
                if text:
                    ranked.append((level, item["document_id"], int(item["page"])))
        seen: set[tuple[str, int]] = set()
        parts: list[str] = []
        budget = 2200
        for _level, document_id, page in sorted(ranked, reverse=True):
            if (document_id, page) in seen or budget <= 0:
                continue
            seen.add((document_id, page))
            snippet = pages[(document_id, page)][: min(650, budget)]
            budget -= len(snippet)
            parts.append(f"[page {page}] {snippet}")
            if len(parts) >= 4:
                break
        return "\n\n".join(parts)

    def _result(
        self,
        case: dict[str, Any],
        findings: Mapping[str, rules.Finding],
        predicted: int | None,
    ) -> dict[str, Any]:
        criterion_status: dict[str, str] = {}
        evidence_reference: dict[str, list[dict[str, Any]]] = {}
        missing_evidence: dict[str, str] = {}
        for criterion in self._criteria:
            cid = str(criterion["criterion_id"])
            finding = findings.get(cid) or rules.Finding(
                INSUFFICIENT, missing="No evidence was found for this criterion.")
            criterion_status[cid] = finding.status
            evidence_reference[cid] = [
                item for item in finding.evidence if isinstance(item, dict)
            ]
            if finding.status == INSUFFICIENT:
                missing_evidence[cid] = finding.missing or "No supporting evidence was found."

        return {
            "schema_version": case["schema_version"],
            "checklist_version": self._version,
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
            "assessment_status": "completed" if predicted is not None else "abstained",
            "predicted_trl": predicted,
            "criterion_status": criterion_status,
            "evidence_reference": evidence_reference,
            "missing_evidence": missing_evidence,
            "abstention_reason": None if predicted is not None else "insufficient evidence",
            "error_code": None,
        }

    def _rationale(
        self,
        predicted: int,
        findings: Mapping[str, rules.Finding],
        source: str,
    ) -> str:
        for cid, finding in findings.items():
            rule = self._assignments.get(cid)
            if finding.status != MET or rule is None or rule.level != predicted:
                continue
            span = finding.evidence[0].get("supporting_span", "") if finding.evidence else ""
            if span:
                return f"TRL {predicted}: {span[:220]}"
        return f"TRL {predicted} from the highest met readiness criterion."

    def _abstain(self, case: Mapping[str, Any], reason: str) -> dict[str, Any]:
        criterion_status: dict[str, str] = {}
        evidence_reference: dict[str, list[dict[str, Any]]] = {}
        missing_evidence: dict[str, str] = {}
        for criterion in self._criteria:
            cid = str(criterion["criterion_id"])
            criterion_status[cid] = INSUFFICIENT
            evidence_reference[cid] = []
            missing_evidence[cid] = reason
        return {
            "schema_version": case.get("schema_version"),
            "checklist_version": self._version,
            "case_id": case.get("case_id"),
            "assessment_target": dict(case.get("assessment_target") or {}),
            "assessment_status": "abstained",
            "predicted_trl": None,
            "criterion_status": criterion_status,
            "evidence_reference": evidence_reference,
            "missing_evidence": missing_evidence,
            "abstention_reason": "insufficient evidence",
            "error_code": None,
        }

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
