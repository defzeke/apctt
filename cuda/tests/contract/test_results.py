"""SPEC-06 result contract: AC-02, AC-03, AC-37, AC-38."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from apctt_contracts.config import Checklist
from apctt_contracts.results import (
    OUTPUT_RULES,
    CaseMeta,
    build_supervisor_failure,
    validate_result,
)

from .fixtures import CHECKLIST, abstained, case, result, stored

CHECK = Checklist.model_validate(CHECKLIST)


def rules(value: Any) -> set[str]:
    return {issue.rule for issue in validate_result(value, case=stored(), checklist=CHECK)}


def issues(value: Any) -> list[tuple[str, str]]:
    return [(i.path, i.rule) for i in validate_result(value, case=stored(), checklist=CHECK)]


def mutate(base: dict[str, Any], change: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    change(base)
    return base


def failed(code: str = "WORKER_CRASH") -> dict[str, Any]:
    return build_supervisor_failure(CaseMeta.from_case(case()), code)


def test_ac02_abstention() -> None:
    assert issues(abstained()) == []
    # Abstention keeps a scientific reason and is not a failure.
    for reason in (
        "insufficient evidence",
        "conflicting evidence",
        "ambiguous scope",
        "out of scope",
    ):
        assert issues(mutate(abstained(), lambda r, x=reason: r.update(abstention_reason=x))) == []
    assert "abstention_reason_value" in rules(
        mutate(abstained(), lambda r: r.update(abstention_reason=None))
    )
    assert "abstention_reason_value" in rules(
        mutate(abstained(), lambda r: r.update(abstention_reason="model unsure"))
    )
    assert "status_combination" in rules(mutate(abstained(), lambda r: r.update(predicted_trl=4)))
    assert "status_combination" in rules(
        mutate(abstained(), lambda r: r.update(error_code="WORKER_CRASH"))
    )
    # A supervisor failure is distinguishable from abstention and is itself valid.
    failure = failed()
    assert failure["assessment_status"] == "failed"
    assert failure["abstention_reason"] is None and failure["error_code"] == "WORKER_CRASH"
    assert failure["predicted_trl"] is None
    assert issues(failure) == []
    assert "status_combination" in rules(
        mutate(failed(), lambda r: r.update(abstention_reason="insufficient evidence"))
    )
    assert "error_code_required" in rules(mutate(failed(), lambda r: r.update(error_code=None)))
    assert "error_code_required" in rules(mutate(failed(), lambda r: r.update(error_code="  ")))
    assert "assessment_status_value" in rules(
        mutate(result(), lambda r: r.update(assessment_status="partial"))
    )


def test_ac03_strict_trl() -> None:
    assert issues(result()) == []
    for bad in (True, False, "5", 0, 10, -1, 5.0, math.nan, math.inf, None, [5]):
        assert "strict_trl" in rules(
            mutate(result(), lambda r, b=bad: r.update(predicted_trl=b))
        ), bad
    for good in range(1, 10):
        assert issues(mutate(result(), lambda r, g=good: r.update(predicted_trl=g))) == []
    assert "status_combination" in rules(
        mutate(result(), lambda r: r.update(abstention_reason="out of scope"))
    )
    assert "status_combination" in rules(mutate(result(), lambda r: r.update(error_code="X")))
    assert "status_combination" in rules(mutate(failed(), lambda r: r.update(predicted_trl=5)))
    # confidence: finite 0..1, bool excluded, explicit null forbidden.
    assert issues(mutate(result(), lambda r: r.update(confidence=0.75))) == []
    assert issues(mutate(result(), lambda r: r.update(confidence=1))) == []
    for bad in (True, 1.5, -0.1, math.nan, math.inf, "0.5"):
        assert "confidence_range" in rules(
            mutate(result(), lambda r, b=bad: r.update(confidence=b))
        ), bad
    assert "omit_instead_of_null" in rules(mutate(result(), lambda r: r.update(confidence=None)))
    # uncertainty interval.
    interval = {"kind": "trl_interval", "lower": 4, "upper": 6}
    assert issues(mutate(result(), lambda r: r.update(uncertainty=dict(interval)))) == []
    assert issues(mutate(abstained(), lambda r: r.update(uncertainty=dict(interval)))) == []
    for bad_interval in (
        {"kind": "trl_interval", "lower": 6, "upper": 4},
        {"kind": "trl_interval", "lower": 6, "upper": 7},  # excludes predicted_trl=5
        {"kind": "trl_interval", "lower": 0, "upper": 5},
        {"kind": "trl_interval", "lower": 4, "upper": True},
        {"kind": "range", "lower": 4, "upper": 6},
        {"kind": "trl_interval", "lower": 4, "upper": 6, "extra": 1},
    ):
        assert "uncertainty_interval" in rules(
            mutate(result(), lambda r, b=bad_interval: r.update(uncertainty=b))
        ), bad_interval
    assert "failed_omits_optional" in rules(mutate(failed(), lambda r: r.update(confidence=0.5)))


def _c1(r: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = r["evidence_reference"]["C1"][0]
    return item


def test_ac37_evidence_consistency() -> None:
    sentinel = "SENTINEL-7f3a9c-module-text"
    cases: list[tuple[str, dict[str, Any], set[str]]] = [
        ("E01 valid completed", result(), set()),
        ("E02 valid abstained", abstained(), set()),
        (
            "E03 failed with empty maps",
            mutate(
                failed("MODULE_ERROR"),
                lambda r: r.update(criterion_status={}, evidence_reference={}, missing_evidence={}),
            ),
            set(),
        ),
        (
            "E04 not met allows empty evidence",
            mutate(result(), lambda r: r["evidence_reference"].update(C2=[])),
            set(),
        ),
        (
            "E06 met requires evidence (completed)",
            mutate(result(), lambda r: r["evidence_reference"].update(C1=[])),
            {"evidence_required_for_met"},
        ),
        (
            "E06 met requires evidence (abstained)",
            mutate(abstained(), lambda r: r["evidence_reference"].update(C1=[])),
            {"evidence_required_for_met"},
        ),
        (
            "E07 missing criterion key",
            mutate(result(), lambda r: r["evidence_reference"].pop("C2")),
            {"evidence_keys_match_checklist"},
        ),
        (
            "E08 unknown criterion key",
            mutate(result(), lambda r: r["evidence_reference"].update(X9=[])),
            {"evidence_keys_match_checklist"},
        ),
        (
            "E09 insufficient evidence requires explanation",
            mutate(result(), lambda r: r.update(missing_evidence={})),
            {"missing_evidence_required_for_insufficient"},
        ),
        (
            "E10 blank explanation",
            mutate(result(), lambda r: r["missing_evidence"].update(C3="   ")),
            {"missing_evidence_required_for_insufficient", "missing_evidence_text"},
        ),
        (
            "E11 empty span",
            mutate(result(), lambda r: _c1(r).update(supporting_span="")),
            {"supporting_span_nonblank"},
        ),
        (
            "E11 whitespace-only span",
            mutate(result(), lambda r: _c1(r).update(supporting_span="   ")),
            {"supporting_span_nonblank"},
        ),
        (
            "E12 quote is kept for judges, not machine-checked",
            mutate(result(), lambda r: _c1(r).update(supporting_span="Words not in the PDF")),
            set(),
        ),
        (
            "E13 page beyond the document",
            mutate(result(), lambda r: _c1(r).update(page=3)),
            {"evidence_page_in_document"},
        ),
        (
            "E13 page zero",
            mutate(result(), lambda r: _c1(r).update(page=0)),
            {"evidence_page_in_document"},
        ),
        (
            "E13 page as string",
            mutate(result(), lambda r: _c1(r).update(page="1")),
            {"evidence_page_in_document"},
        ),
        (
            "E13 page missing",
            mutate(result(), lambda r: _c1(r).pop("page")),
            {"evidence_item_required"},
        ),
        (
            "E14 unknown document",
            mutate(result(), lambda r: _c1(r).update(document_id="D3")),
            {"evidence_locator_exists"},
        ),
        (
            "E15 unknown top-level field",
            mutate(result(), lambda r: r.update(debug={})),
            {"result_unknown_field"},
        ),
        (
            "E16 unknown evidence item field",
            mutate(result(), lambda r: _c1(r).update(score=1)),
            {"result_unknown_field"},
        ),
    ]
    for name, value, expected in cases:
        got = rules(value)
        assert got == expected, (name, got)
    # A raw request view has no page counts: only page >= 1 can be checked there.
    beyond = mutate(result(), lambda r: _c1(r).update(page=3))
    assert validate_result(beyond, case=case(), checklist=CHECK) == []

    # E07/E08 path reporting: trusted checklist IDs are shown, module-invented keys are not.
    missing = issues(mutate(result(), lambda r: r["evidence_reference"].pop("C2")))
    assert ("result.evidence_reference.C2", "evidence_keys_match_checklist") in missing
    unknown = issues(mutate(result(), lambda r: r["evidence_reference"].update(X9=[])))
    assert unknown == [
        ("result.evidence_reference.<unknown-field>", "evidence_keys_match_checklist")
    ]
    met = issues(mutate(result(), lambda r: r["evidence_reference"].update(C1=[])))
    assert met == [("result.evidence_reference.C1", "evidence_required_for_met")]
    extra = issues(mutate(result(), lambda r: r.update(**{"debug-" + sentinel: 1})))
    assert extra == [("result.<unknown-field>", "result_unknown_field")]

    # E20: diagnostics never carry module-authored text.
    leaky = mutate(result(), lambda r: _c1(r).update(supporting_span=sentinel))
    leaky["missing_evidence"]["C3"] = sentinel
    leaky["criterion_rationale"]["C1"] = sentinel
    leaky["evidence_reference"][sentinel] = []
    found = validate_result(leaky, case=stored(), checklist=CHECK)
    assert found and sentinel not in json.dumps([i.as_dict() for i in found])
    assert {i.rule for i in found} <= OUTPUT_RULES

    # Non-object result and wrong identity are rejected.
    assert rules([]) == {"result_object"}
    assert "result_matches_case" in rules(mutate(result(), lambda r: r.update(case_id="OTHER")))
    assert "result_matches_case" in rules(
        mutate(result(), lambda r: r["assessment_target"].update(description="changed"))
    )
    assert "result_matches_checklist" in rules(
        mutate(result(), lambda r: r.update(checklist_version="OTHER"))
    )


def test_ac38_optional_rationale_fields() -> None:
    assert (
        issues(mutate(result(), lambda r: (r.pop("trl_rationale"), r.pop("criterion_rationale"))))
        == []
    )
    assert issues(mutate(result(), lambda r: r.update(criterion_rationale={}))) == []
    cases: list[tuple[str, dict[str, Any], set[str]]] = [
        (
            "E17 unknown rationale key",
            mutate(result(), lambda r: r["criterion_rationale"].update(X9="text")),
            {"rationale_known_criterion"},
        ),
        (
            "E18 blank criterion rationale",
            mutate(result(), lambda r: r["criterion_rationale"].update(C1="  ")),
            {"rationale_nonblank_text"},
        ),
        (
            "E18 null trl rationale",
            mutate(result(), lambda r: r.update(trl_rationale=None)),
            {"rationale_nonblank_text"},
        ),
        (
            "E18 non-string rationale",
            mutate(result(), lambda r: r.update(trl_rationale=5)),
            {"rationale_nonblank_text"},
        ),
        (
            "E19 trl rationale on abstained",
            mutate(abstained(), lambda r: r.update(trl_rationale="x")),
            {"trl_rationale_requires_completed"},
        ),
        (
            "rationale map must be an object",
            mutate(result(), lambda r: r.update(criterion_rationale=["x"])),
            {"map_object"},
        ),
        (
            "failed omits rationale",
            mutate(failed(), lambda r: r.update(criterion_rationale={"C1": "x"})),
            {"failed_omits_optional"},
        ),
    ]
    for name, value, expected in cases:
        got = rules(value)
        assert got == expected, (name, got)
    unknown = issues(mutate(result(), lambda r: r["criterion_rationale"].update(X9="text")))
    assert unknown == [("result.criterion_rationale.<unknown-field>", "rationale_known_criterion")]
