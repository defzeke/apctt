"""Transparent DEMO rules shared by the CPU example and the CUDA dummy.

A criterion is `met` when one of its configured phrases appears on a PDF page; the evidence points
to that document and page and quotes the matched text. Otherwise it is `insufficient evidence` with
an explanation (a criterion with no configured phrase is not looked for). The rules work with any
checklist; the kit's configurations run the official one (decision D74). These are format
demonstrations, not assessment rules. Text is read with pypdf.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

DEMO_CHECKLIST_VERSION = "DEMO-CHECKLIST-NOT-OFFICIAL"
MAX_EVIDENCE = 3


def page_texts(case: Mapping[str, Any]) -> Iterator[tuple[str, int, str]]:
    """(document_id, page number from 1, extracted text) for every page, in packet order."""
    from pypdf import PdfReader

    for document in case["documents"]:
        reader = PdfReader(document["path"])
        for number, page in enumerate(reader.pages, start=1):
            yield document["document_id"], number, page.extract_text() or ""


def evaluate(
    case: Mapping[str, Any],
    criterion_ids: Sequence[str],
    phrases: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    pages = list(page_texts(case))
    status: dict[str, str] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    missing: dict[str, str] = {}
    rationale: dict[str, str] = {}
    for cid in criterion_ids:
        found: list[dict[str, Any]] = []
        for document_id, number, text in pages:
            for phrase in phrases[cid]:
                match = re.search(re.escape(phrase), text, re.IGNORECASE)
                if match is None:
                    continue
                found.append(
                    {"document_id": document_id, "page": number, "supporting_span": match.group(0)}
                )
                break
            if len(found) >= MAX_EVIDENCE:
                break
        evidence[cid] = found
        if not phrases[cid]:
            status[cid] = "insufficient evidence"
            missing[cid] = "DEMO rule: no phrase is configured for this criterion."
        elif found:
            status[cid] = "met"
            first = found[0]
            rationale[cid] = (
                f"DEMO rule: a configured phrase appears in {first['document_id']} "
                f"page {first['page']}."
            )
        else:
            status[cid] = "insufficient evidence"
            missing[cid] = "DEMO rule: no page contains a configured phrase for this criterion."
    return {
        "criterion_status": status,
        "evidence_reference": evidence,
        "missing_evidence": missing,
        "criterion_rationale": rationale,
    }


def base_result(case: Mapping[str, Any], checklist_version: str) -> dict[str, Any]:
    return {
        "schema_version": case["schema_version"],
        "checklist_version": checklist_version,
        "case_id": case["case_id"],
        "assessment_target": dict(case["assessment_target"]),
    }
