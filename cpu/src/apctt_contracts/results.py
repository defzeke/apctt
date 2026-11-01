"""SPEC-06 result contract: structural checks and judgment–evidence consistency.

`validate_result` inspects a parsed module result against the current case and trusted checklist.
Issues carry only a path and a fixed rule ID. Module-authored text and module-invented keys are
never copied: unknown keys become `<unknown-field>`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field

from . import UNKNOWN_FIELD
from .config import Checklist
from .types import Identifier, NonBlankText, StrictModel, omittable

CRITERION_STATUSES = ("met", "not met", "insufficient evidence")
ABSTENTION_REASONS = (
    "insufficient evidence",
    "conflicting evidence",
    "ambiguous scope",
    "out of scope",
)
ASSESSMENT_STATUSES = ("completed", "abstained", "failed")
REQUIRED_FIELDS = (
    "schema_version",
    "checklist_version",
    "case_id",
    "assessment_target",
    "assessment_status",
    "predicted_trl",
    "criterion_status",
    "evidence_reference",
    "missing_evidence",
    "abstention_reason",
    "error_code",
)
OPTIONAL_FIELDS = ("confidence", "uncertainty", "trl_rationale", "criterion_rationale")
EVIDENCE_FIELDS = frozenset({"document_id", "page", "supporting_span"})

OUTPUT_RULES = frozenset(
    {
        # SPEC-06 table (0.3.0: PDF documents)
        "result_unknown_field",
        "evidence_keys_match_checklist",
        "evidence_locator_exists",
        "evidence_page_in_document",
        "supporting_span_nonblank",
        "evidence_required_for_met",
        "missing_evidence_required_for_insufficient",
        "rationale_known_criterion",
        "rationale_nonblank_text",
        "trl_rationale_requires_completed",
        # general structure and status rules
        "result_object",
        "result_required_field",
        "result_matches_case",
        "result_matches_checklist",
        "assessment_status_value",
        "strict_trl",
        "status_combination",
        "abstention_reason_value",
        "error_code_required",
        "failed_omits_optional",
        "map_object",
        "criterion_known_id",
        "criterion_status_complete",
        "criterion_status_value",
        "evidence_list",
        "evidence_item_object",
        "evidence_item_required",
        "missing_evidence_text",
        "confidence_range",
        "uncertainty_interval",
        "omit_instead_of_null",
    }
)


@dataclass(frozen=True)
class OutputIssue:
    path: str
    rule: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "rule": self.rule}


@dataclass(frozen=True)
class CaseMeta:
    schema_version: str
    checklist_version: str
    case_id: str
    assessment_target: Mapping[str, Any]

    @classmethod
    def from_case(cls, case: Mapping[str, Any]) -> CaseMeta:
        return cls(
            schema_version=case["schema_version"],
            checklist_version=case["checklist_version"],
            case_id=case["case_id"],
            assessment_target=dict(case["assessment_target"]),
        )


def build_supervisor_failure(meta: CaseMeta, error_code: str) -> dict[str, Any]:
    """A failed result produced by the supervisor; never merged with module output."""
    return {
        "schema_version": meta.schema_version,
        "checklist_version": meta.checklist_version,
        "case_id": meta.case_id,
        "assessment_target": dict(meta.assessment_target),
        "assessment_status": "failed",
        "predicted_trl": None,
        "criterion_status": {},
        "evidence_reference": {},
        "missing_evidence": {},
        "abstention_reason": None,
        "error_code": error_code,
    }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_trl(value: object) -> bool:
    return _is_int(value) and 1 <= value <= 9  # type: ignore[operator]


class _Collector:
    def __init__(self, criterion_ids: tuple[str, ...]) -> None:
        self.ids = criterion_ids
        self.issues: list[OutputIssue] = []

    def add(self, path: str, rule: str) -> None:
        issue = OutputIssue(path, rule)
        if issue not in self.issues:
            self.issues.append(issue)

    def key(self, prefix: str, key: str) -> str:
        return f"{prefix}.{key if key in self.ids else UNKNOWN_FIELD}"


def _page_counts(case: Mapping[str, Any]) -> dict[str, int | None]:
    """document_id -> page_count (None when the case view carries no count, e.g. a raw request)."""
    counts: dict[str, int | None] = {}
    for document in case.get("documents", []):
        pages = document.get("page_count")
        counts[document["document_id"]] = pages if _is_int(pages) else None
    return counts


def _check_evidence(
    out: _Collector,
    evidence: Any,
    status: str,
    page_counts: dict[str, int | None],
) -> None:
    prefix = "result.evidence_reference"
    if not isinstance(evidence, dict):
        out.add(prefix, "map_object")
        return
    complete = status in ("completed", "abstained")
    for key in evidence:
        if key not in out.ids:
            out.add(out.key(prefix, key), "evidence_keys_match_checklist")
    if complete:
        for cid in out.ids:
            if cid not in evidence:
                out.add(f"{prefix}.{cid}", "evidence_keys_match_checklist")
    for key, items in evidence.items():
        if key not in out.ids:
            continue
        path = f"{prefix}.{key}"
        if not isinstance(items, list):
            out.add(path, "evidence_list")
            continue
        for index, item in enumerate(items):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict):
                out.add(item_path, "evidence_item_object")
                continue
            if set(item) - EVIDENCE_FIELDS:
                out.add(f"{item_path}.{UNKNOWN_FIELD}", "result_unknown_field")
            document_id = item.get("document_id")
            if not isinstance(document_id, str) or "page" not in item:
                out.add(item_path, "evidence_item_required")
                continue
            if document_id not in page_counts:
                out.add(item_path, "evidence_locator_exists")
                continue
            page, pages = item["page"], page_counts[document_id]
            if not _is_int(page) or page < 1 or (pages is not None and page > pages):
                out.add(f"{item_path}.page", "evidence_page_in_document")
            if "supporting_span" in item and not _is_text(item["supporting_span"]):
                out.add(f"{item_path}.supporting_span", "supporting_span_nonblank")


def _check_text_map(
    out: _Collector, value: Any, prefix: str, key_rule: str, text_rule: str
) -> None:
    if not isinstance(value, dict):
        out.add(prefix, "map_object")
        return
    for key, text in value.items():
        if key not in out.ids:
            out.add(out.key(prefix, key), key_rule)
        elif not _is_text(text):
            out.add(f"{prefix}.{key}", text_rule)


def _check_optional(out: _Collector, result: Mapping[str, Any], status: str) -> None:
    for name in OPTIONAL_FIELDS:
        if name in result and status == "failed":
            out.add(f"result.{name}", "failed_omits_optional")
    if status == "failed":
        return
    if "confidence" in result:
        value = result["confidence"]
        if value is None:
            out.add("result.confidence", "omit_instead_of_null")
        elif not (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ):
            out.add("result.confidence", "confidence_range")
    if "uncertainty" in result:
        value = result["uncertainty"]
        if value is None:
            out.add("result.uncertainty", "omit_instead_of_null")
        elif not (
            isinstance(value, dict)
            and set(value) == {"kind", "lower", "upper"}
            and value["kind"] == "trl_interval"
            and _is_trl(value["lower"])
            and _is_trl(value["upper"])
            and value["lower"] <= value["upper"]
            and (
                status != "completed"
                or not _is_trl(result.get("predicted_trl"))
                or value["lower"] <= result["predicted_trl"] <= value["upper"]
            )
        ):
            out.add("result.uncertainty", "uncertainty_interval")
    if "trl_rationale" in result:
        if status != "completed":
            out.add("result.trl_rationale", "trl_rationale_requires_completed")
        elif not _is_text(result["trl_rationale"]):
            out.add("result.trl_rationale", "rationale_nonblank_text")
    if "criterion_rationale" in result:
        _check_text_map(
            out,
            result["criterion_rationale"],
            "result.criterion_rationale",
            "rationale_known_criterion",
            "rationale_nonblank_text",
        )


def validate_result(
    value: Any, *, case: Mapping[str, Any], checklist: Checklist
) -> list[OutputIssue]:
    out = _Collector(checklist.ids())
    if not isinstance(value, dict):
        out.add("result", "result_object")
        return out.issues
    allowed = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    if set(value) - allowed:
        out.add(f"result.{UNKNOWN_FIELD}", "result_unknown_field")
    for name in REQUIRED_FIELDS:
        if name not in value:
            out.add(f"result.{name}", "result_required_field")
    if out.issues and any(i.rule == "result_required_field" for i in out.issues):
        return out.issues

    for name in ("schema_version", "case_id"):
        if value[name] != case.get(name):
            out.add(f"result.{name}", "result_matches_case")
    if value["assessment_target"] != case.get("assessment_target"):
        out.add("result.assessment_target", "result_matches_case")
    if value["checklist_version"] != checklist.version or value["checklist_version"] != case.get(
        "checklist_version"
    ):
        out.add("result.checklist_version", "result_matches_checklist")

    status = value["assessment_status"]
    if status not in ASSESSMENT_STATUSES:
        out.add("result.assessment_status", "assessment_status_value")
        return out.issues

    trl, reason, error = value["predicted_trl"], value["abstention_reason"], value["error_code"]
    if status == "completed":
        if not _is_trl(trl):
            out.add("result.predicted_trl", "strict_trl")
        if reason is not None:
            out.add("result.abstention_reason", "status_combination")
        if error is not None:
            out.add("result.error_code", "status_combination")
    elif status == "abstained":
        if trl is not None:
            out.add("result.predicted_trl", "status_combination")
        if reason not in ABSTENTION_REASONS:
            out.add("result.abstention_reason", "abstention_reason_value")
        if error is not None:
            out.add("result.error_code", "status_combination")
    else:
        if trl is not None:
            out.add("result.predicted_trl", "status_combination")
        if reason is not None:
            out.add("result.abstention_reason", "status_combination")
        if not _is_text(error):
            out.add("result.error_code", "error_code_required")

    criteria = value["criterion_status"]
    complete = status in ("completed", "abstained")
    if not isinstance(criteria, dict):
        out.add("result.criterion_status", "map_object")
        criteria = {}
    for key, criterion_value in criteria.items():
        if key not in out.ids:
            out.add(out.key("result.criterion_status", key), "criterion_known_id")
        elif criterion_value not in CRITERION_STATUSES:
            out.add(f"result.criterion_status.{key}", "criterion_status_value")
    if complete:
        for cid in out.ids:
            if cid not in criteria:
                out.add(f"result.criterion_status.{cid}", "criterion_status_complete")

    evidence = value["evidence_reference"]
    _check_evidence(out, evidence, status, _page_counts(case))
    missing = value["missing_evidence"]
    _check_text_map(
        out, missing, "result.missing_evidence", "criterion_known_id", "missing_evidence_text"
    )

    if complete and isinstance(evidence, dict) and isinstance(missing, dict):
        for cid in out.ids:
            state = criteria.get(cid)
            if state == "met" and isinstance(evidence.get(cid), list) and not evidence[cid]:
                out.add(f"result.evidence_reference.{cid}", "evidence_required_for_met")
            if state == "insufficient evidence" and not _is_text(missing.get(cid)):
                out.add(
                    f"result.missing_evidence.{cid}", "missing_evidence_required_for_insufficient"
                )

    _check_optional(out, value, status)
    return out.issues


# Documentation models for OpenAPI. Runtime validation is `validate_result` above.


class EvidenceItem(StrictModel):
    document_id: Identifier
    page: Annotated[int, Field(ge=1)]
    supporting_span: NonBlankText | None = omittable()


class TrlInterval(StrictModel):
    kind: Literal["trl_interval"]
    lower: Annotated[int, Field(ge=1, le=9)]
    upper: Annotated[int, Field(ge=1, le=9)]


class AssessmentTargetOut(StrictModel):
    target_id: Identifier
    type: Literal["component", "subsystem", "system", "technology"]
    description: NonBlankText


class AssessmentResult(StrictModel):
    """SPEC-06 result as returned unchanged from a valid module or built by the supervisor."""

    schema_version: str
    checklist_version: str
    case_id: str
    assessment_target: AssessmentTargetOut
    assessment_status: Literal["completed", "abstained", "failed"]
    predicted_trl: Annotated[int, Field(ge=1, le=9)] | None
    criterion_status: dict[str, Literal["met", "not met", "insufficient evidence"]]
    evidence_reference: dict[str, list[EvidenceItem]]
    missing_evidence: dict[str, NonBlankText]
    abstention_reason: (
        Literal["insufficient evidence", "conflicting evidence", "ambiguous scope", "out of scope"]
        | None
    )
    error_code: str | None
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = omittable()
    uncertainty: TrlInterval | None = omittable()
    trl_rationale: NonBlankText | None = omittable()
    criterion_rationale: dict[str, NonBlankText] | None = omittable()
