"""Synthetic contract fixtures for the SPEC-06 result rules. Not expert TRL labels."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from apctt_contracts.pdf import stored_case

from .pdfgen import build_pdf

ROOT = Path(__file__).resolve().parents[2]

CHECKLIST: dict[str, Any] = {
    "version": "DEMO-CHECKLIST-NOT-OFFICIAL",
    "official": False,
    "criteria": [
        {"criterion_id": "C1", "description": "Prototype tested in a relevant environment."},
        {"criterion_id": "C2", "description": "System demonstrated in an operational environment."},
        {"criterion_id": "C3", "description": "Subsystem integration verified."},
    ],
}

TARGET: dict[str, Any] = {
    "target_id": "T1",
    "type": "component",
    "description": "Demo heat exchanger",
}

PDF_D1 = build_pdf(
    [
        ["The prototype was tested in a relevant environment at 40 C."],
        ["Field deployment has not been performed."],
    ]
)
PDF_D2 = build_pdf([["Synthetic note 2"], ["Integration tests are planned for next year."]])


def pdf_document(
    document_id: str, title: str, document_date: str | None, data: bytes
) -> dict[str, Any]:
    """A request document: the PDF as strict base64 with the SHA-256 of its bytes."""
    return {
        "document_id": document_id,
        "title": title,
        "document_date": document_date,
        "language": "en",
        "format": "pdf",
        "content_base64": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


CASE: dict[str, Any] = {
    "schema_version": "apctt-pdf-example-v1",
    "input_profile": "pdf_report",
    "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
    "case_id": "CASE-DEMO-001",
    "assessment_as_of": None,
    "assessment_target": TARGET,
    "intended_application": "Synthetic interface test",
    "intended_environment": "Synthetic test environment",
    "title": "DEMO fixture only",
    "documents": [
        pdf_document("D1", "Synthetic note 1", None, PDF_D1),
        pdf_document("D2", "Synthetic note 2", "2026-01-15", PDF_D2),
    ],
}
# What the server stores and validates results against: page counts, no base64 payload.
STORED_CASE: dict[str, Any] = stored_case(CASE, [2, 2])

RESULT: dict[str, Any] = {
    "schema_version": "apctt-pdf-example-v1",
    "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
    "case_id": "CASE-DEMO-001",
    "assessment_target": TARGET,
    "assessment_status": "completed",
    "predicted_trl": 5,
    "criterion_status": {"C1": "met", "C2": "not met", "C3": "insufficient evidence"},
    "evidence_reference": {
        "C1": [
            {
                "document_id": "D1",
                "page": 1,
                "supporting_span": "tested in a relevant environment",
            }
        ],
        "C2": [{"document_id": "D1", "page": 2}],
        "C3": [{"document_id": "D2", "page": 2}],
    },
    "missing_evidence": {"C3": "No completed integration test result is reported."},
    "abstention_reason": None,
    "error_code": None,
    "trl_rationale": (
        "Relevant-environment testing is reported, but operational demonstration "
        "and integration results are not."
    ),
    "criterion_rationale": {"C1": "D1 page 1 reports a prototype test in a relevant environment."},
}


def case() -> dict[str, Any]:
    return copy.deepcopy(CASE)


def stored() -> dict[str, Any]:
    return copy.deepcopy(STORED_CASE)


def result() -> dict[str, Any]:
    return copy.deepcopy(RESULT)


def abstained() -> dict[str, Any]:
    value = result()
    value.update(
        assessment_status="abstained",
        predicted_trl=None,
        abstention_reason="insufficient evidence",
    )
    del value["trl_rationale"]
    return value


def load_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))
