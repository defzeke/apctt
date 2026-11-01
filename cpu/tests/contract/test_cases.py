"""SPEC-05/09/16/17 request, case, IPC and compute contracts: AC-04, AC-25."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from apctt_contracts import (
    API_VERSION,
    DEMO_CHECKLIST_VERSION,
    DEV_SCHEMA_VERSION,
    WORKER_PROTOCOL_VERSION,
)
from apctt_contracts.cases import CaseLimits, ProfileVersions, check_request, page_issues
from apctt_contracts.config import Checklist, Compute, load_config
from apctt_contracts.ipc import (
    Observation,
    WorkerContext,
    WorkerDocument,
    WorkerEnvelope,
    worker_protocol_schema,
)
from apctt_contracts.pdf import worker_case
from apctt_contracts.responses import ErrorEnvelope, JobEnvelope
from apctt_contracts.results import validate_result

from .fixtures import CHECKLIST, ROOT, case, load_json, pdf_document, result
from .fixtures import stored as stored_view

LIMITS = CaseLimits(max_documents=20, max_document_bytes=20 * 1024 * 1024)
VERSIONS = ProfileVersions(
    schema_version=DEV_SCHEMA_VERSION, checklist_version=DEMO_CHECKLIST_VERSION
)
OFFICIAL_CHECKLIST_VERSION = "APCTT-2026-CHECKLIST-v1"


def check(body: Any, limits: CaseLimits = LIMITS) -> tuple[str | None, list[tuple[str, str]]]:
    outcome = check_request(body, limits=limits, versions=VERSIONS)
    return outcome.code, [(i.path, i.rule) for i in outcome.issues]


def body(**case_changes: Any) -> dict[str, Any]:
    value = case()
    value.update(case_changes)
    return {"request_id": "REQ-1", "case": value}


def test_ac04_case_and_evidence_ids() -> None:
    # The example request is for the official checklist of config/dev.json (D74).
    example = load_json("docs/EXAMPLE_REQUEST.json")
    official = ProfileVersions(
        schema_version=DEV_SCHEMA_VERSION, checklist_version=OFFICIAL_CHECKLIST_VERSION
    )
    assert check_request(example, limits=LIMITS, versions=official).code is None
    assert check(example)[0] == "VERSION_MISMATCH"
    assert check(body()) == (None, [])

    # Unique document IDs.
    dup_doc = body()
    dup_doc["case"]["documents"][1]["document_id"] = "D1"
    assert check(dup_doc) == (
        "REQUEST_VALIDATION_ERROR",
        [("case.documents[1].document_id", "unique_document_id")],
    )

    # PDF documents: strict base64, PDF header, SHA-256 of the decoded bytes, format, size.
    passed = check_request(body(), limits=LIMITS, versions=VERSIONS)
    assert [data[:5] for data in passed.contents] == [b"%PDF-", b"%PDF-"]
    doc0 = "case.documents[0]"
    for field, value, rule in (
        ("content_base64", "not base64!", "base64_content"),
        ("content_base64", "JVBERi0", "base64_content"),  # missing padding
        ("sha256", "0" * 64, "document_sha256_match"),
        ("sha256", "ABC", "sha256_pattern"),
        ("format", "docx", "allowed_value"),
    ):
        broken = body()
        broken["case"]["documents"][0][field] = value
        assert check(broken)[1] == [(f"{doc0}.{field}", rule)], (field, value)
    not_pdf = body()
    not_pdf["case"]["documents"][0] = pdf_document("D1", "Not a PDF", None, b"plain text")
    assert check(not_pdf)[1] == [(f"{doc0}.content_base64", "pdf_signature")]
    no_content = body()
    del no_content["case"]["documents"][0]["content_base64"]
    assert check(no_content)[1] == [(f"{doc0}.content_base64", "required")]
    small = CaseLimits(max_documents=20, max_document_bytes=100)
    assert check(body(), small)[1] == [
        (f"case.documents[{i}].content_base64", "max_document_bytes") for i in range(2)
    ]
    # After the server counted pages: unreadable files and the total page limit.
    assert [(i.path, i.rule) for i in page_issues([2, None], 100)] == [
        ("case.documents[1].content_base64", "pdf_readable")
    ]
    assert [(i.path, i.rule) for i in page_issues([2, 3], 4)] == [
        ("case.documents", "max_total_pages")
    ]
    assert page_issues([2, 2], 4) == []

    # Missing / unknown / explicit label fields.
    missing = body()
    del missing["case"]["title"]
    assert check(missing) == ("REQUEST_VALIDATION_ERROR", [("case.title", "required")])
    for label in ("trlCurrent", "trlBegin", "trlEnd", "reference_trl", "SECRET-FIELD-NAME"):
        code, found = check(body(**{label: 5}))
        assert (code, found) == (
            "REQUEST_VALIDATION_ERROR",
            [("case.<unknown-field>", "unknown_field")],
        )
    nested = body()
    nested["case"]["documents"][0]["label"] = "TRL 6"
    assert check(nested)[1] == [("case.documents[0].<unknown-field>", "unknown_field")]
    top = {"request_id": "REQ-1", "case": case(), "debug": True}
    assert check(top)[1] == [("<unknown-field>", "unknown_field")]

    # Target, identifiers, dates, SDG values.
    bad_target = body()
    bad_target["case"]["assessment_target"]["type"] = "product"
    assert check(bad_target)[1] == [("case.assessment_target.type", "allowed_value")]
    blank = body()
    blank["case"]["documents"][0]["title"] = "   "
    assert check(blank)[1] == [("case.documents[0].title", "non_blank_string")]
    no_date = body()
    del no_date["case"]["documents"][0]["document_date"]
    assert check(no_date)[1] == [("case.documents[0].document_date", "required")]
    empty_docs = body(documents=[])
    assert check(empty_docs)[1] == [("case.documents", "non_empty_array")]
    for request_id in ("a/b", "", "x" * 129, "é", "a b", "../x"):
        assert check({"request_id": request_id, "case": case()})[1] == [
            ("request_id", "request_id_pattern")
        ], request_id
    assert check({"request_id": "A_b-9" * 25 + "abc", "case": case()}) == (None, [])
    assert check(body(case_id="has space"))[1] == [("case.case_id", "identifier_pattern")]
    assert check(body(assessment_as_of="2026-13-01"))[1] == [("case.assessment_as_of", "iso_date")]
    assert check(body(assessment_as_of="2026-09-29")) == (None, [])
    assert check(body(sdg_targets=[6, 13])) == (None, [])
    assert check(body(sdg_targets=[6, 6]))[1] == [("case.sdg_targets", "unique_items")]
    assert check(body(sdg_targets=[7]))[1] == [("case.sdg_targets[0]", "allowed_value")]
    lang = body()
    lang["case"]["documents"][0]["language"] = "fr"
    assert check(lang)[1] == [("case.documents[0].language", "allowed_value")]
    assert check([])[1] == [("<body>", "object")]

    # Versions, input profile, limits.
    assert check(body(schema_version="apctt-v04"))[0] == "VERSION_MISMATCH"
    assert check(body(checklist_version="OFFICIAL-1"))[0] == "VERSION_MISMATCH"
    assert check(body(input_profile="canonical_report_text")) == (
        "UNSUPPORTED_INPUT_PROFILE",
        [("case.input_profile", "supported_input_profile")],
    )
    many = body()
    many["case"]["documents"] = [
        {**many["case"]["documents"][0], "document_id": f"D{i}"} for i in range(3)
    ]
    assert check(many, CaseLimits(max_documents=2, max_document_bytes=10**6))[1] == [
        ("case.documents", "max_documents")
    ]

    # Result locators must point into the current packet.
    checklist = Checklist.model_validate(CHECKLIST)
    stray = result()
    stray["evidence_reference"]["C1"][0]["document_id"] = "D404"
    assert [i.rule for i in validate_result(stray, case=stored_view(), checklist=checklist)] == [
        "evidence_locator_exists"
    ]


def test_ac25_response_ipc_compute_contract() -> None:
    # Success envelope: finished example and queued shape.
    example = load_json("docs/EXAMPLE_RESPONSE.json")
    envelope = JobEnvelope.model_validate(example)
    assert envelope.meta.api_version == API_VERSION
    queued = dict(example, job_status="queued", result_origin=None, result=None)
    queued["meta"] = dict(example["meta"], started_at=None, finished_at=None)
    JobEnvelope.model_validate(queued)
    for bad in (
        dict(example, result_origin=None),
        dict(example, job_status="queued"),
        dict(example, job_status="completed"),
        dict(example, result_origin="cache"),
    ):
        with pytest.raises(ValidationError):
            JobEnvelope.model_validate(bad)
    ErrorEnvelope.model_validate(
        {
            "error": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "A request field does not satisfy the input contract.",
                "stage": "request_validation",
                "retryable": False,
                "action": "Correct the reported field and submit the request again.",
                "field_errors": [
                    {
                        "path": "case.documents[0].sha256",
                        "rule": "document_sha256_match",
                        "expected": "SHA-256 of the decoded document",
                        "received_type": "string",
                    }
                ],
            },
            "meta": {"trace_id": "t", "api_version": API_VERSION},
            "diagnostics": {"profile": "dev", "duration_ms": 1.2, "truncated": False},
        }
    )

    # Worker envelope: fixed context keys, checklist delivery without labels.
    checklist = Checklist.model_validate(CHECKLIST)
    envelope_value = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "run_id": "run0001",
        "case": worker_case(
            stored_view(),
            [Path("/tmp/run0001/documents/001.pdf"), Path("/tmp/run0001/documents/002.pdf")],
        ),
        "config": {
            "checklist": checklist.payload(),
            "module": {"package_id": "demo", "settings": {}},
            "compute": {
                "device": "cpu",
                "require_cuda": False,
                "allow_cpu_fallback": False,
                "precision": "float32",
                "loading_policy": "cold_per_case",
                "seed": 42,
                "visible_devices": None,
            },
        },
        "context": {
            "run_id": "run0001",
            "seed": 42,
            "time_budget_ms": 120000,
            "scratch_dir": "/tmp/run0001",
        },
    }
    worker = WorkerEnvelope.model_validate(envelope_value)
    for document in envelope_value["case"]["documents"]:  # type: ignore[index]
        WorkerDocument.model_validate(document)  # page_count and path, never content_base64
        assert "content_base64" not in document
    assert set(WorkerContext.model_fields) == {"run_id", "seed", "time_budget_ms", "scratch_dir"}
    assert set(worker.config.checklist.model_dump()) == {"version", "criteria"}
    assert set(worker.config.checklist.criteria[0].model_dump()) == {"criterion_id", "description"}
    with pytest.raises(ValidationError):
        worker.context.seed = 1  # frozen DTO
    for path, value in (
        (("context", "time_budget_ms"), 0),
        (("context", "deadline_monotonic"), 1.0),
        (("config", "checklist", "answers"), {"C1": 5}),
        (("run_id",), "other"),
    ):
        broken = json.loads(json.dumps(envelope_value))
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValidationError):
            WorkerEnvelope.model_validate(broken)

    # Compute: explicit device, no auto, no CPU fallback, CUDA requires require_cuda.
    base_compute = envelope_value["config"]["compute"]  # type: ignore[index]
    Compute.model_validate(dict(base_compute, device="cuda:0", require_cuda=True))
    for change in (
        {"device": "auto"},
        {"device": "cuda:1", "require_cuda": True},
        {"allow_cpu_fallback": True},
        {"device": "cuda:0"},
        {"require_cuda": True},
        {"loading_policy": "persistent"},
        {"seed": -1},
    ):
        with pytest.raises(ValidationError):
            Compute.model_validate(dict(base_compute, **change))

    # Observation sidecar schema.
    observation = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "run_id": "run0001",
        "phase": "finished",
        "source": "worker_observed",
        "timings_ms": {"load": 1.5, "assess": 2.0, "cleanup": None},
        "unavailable_reasons": {"cleanup": "NOT_OBSERVED"},
        "compute": {
            "device": "cpu",
            "torch_version": None,
            "torch_cuda_build": None,
            "cuda_available": None,
            "kernel_probe_passed": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "measurement_method": None,
        },
        "warning_codes": [],
    }
    Observation.model_validate(observation)
    with pytest.raises(ValidationError):
        Observation.model_validate(dict(observation, phase="done"))

    # Frozen machine-readable IPC contract and dev configuration.
    stored = json.loads((ROOT / "contracts/worker-protocol.json").read_text(encoding="utf-8"))
    assert stored == worker_protocol_schema()
    loaded = load_config(ROOT / "config/dev.json", profile="dev")
    assert loaded.config.compute.device == "cpu" and not loaded.config.compute.require_cuda
    assert loaded.checklist.version == OFFICIAL_CHECKLIST_VERSION  # D74: the official default
    assert loaded.checklist.official
    assert loaded.config.limits.max_request_bytes == 83_886_080  # 80 MiB: PDFs as base64
    assert loaded.config.limits.max_documents == 3
    assert loaded.config.limits.max_document_bytes == 52_428_800  # 50 MiB
    assert loaded.config.limits.max_total_pages == 300
    assert len(loaded.config_sha256) == 64
    demo = Checklist.model_validate(load_json("contracts/checklist.demo.json"))
    assert demo.version == DEMO_CHECKLIST_VERSION and demo.official is False
    with pytest.raises(ValidationError):
        Checklist.model_validate(dict(CHECKLIST, criteria=CHECKLIST["criteria"] * 2))


# The official input limits (decision D77, 2026-10-13), the same in every configuration the kit
# and the organizer's templates hold; the 3-minute limit per case (D41) is unchanged.
OFFICIAL_LIMITS = {
    "max_documents": 3,
    "max_document_bytes": 50 * 1024 * 1024,
    "max_request_bytes": 80 * 1024 * 1024,
    "max_total_pages": 300,
    "deadline_ms": 180_000,
}


def test_official_input_limits_in_every_configuration() -> None:
    """Not a required node: D77 limits in the dev and CUDA example configurations and in both
    HQ templates. base64 makes a request about 4/3 of its PDFs, so the request limit caps the PDFs
    of one case at about 59 MiB together, more than the largest public sample PDF (49.3 MiB). The
    participant kit ships config/dev.json, the CUDA variant also config/cuda-example.json, and
    neither ships the HQ templates: in a kit, the files it does not ship are not read."""
    repository = (ROOT / "tools/hq").is_dir()
    for relative in (
        "config/dev.json",
        "config/cuda-example.json",
        "tools/hq/templates/server-config.dev.json",
        "tools/hq/templates/server-config.evaluation.json",
    ):
        if not repository and relative != "config/dev.json" and not (ROOT / relative).exists():
            continue
        limits = json.loads((ROOT / relative).read_text())["limits"]
        assert {k: limits[k] for k in OFFICIAL_LIMITS} == OFFICIAL_LIMITS, relative
    largest = 49.3 * 1024 * 1024
    assert OFFICIAL_LIMITS["max_document_bytes"] > largest
    assert OFFICIAL_LIMITS["max_request_bytes"] * 3 / 4 > largest  # one PDF fits a request
    assert 58 * 1024 * 1024 < OFFICIAL_LIMITS["max_request_bytes"] * 3 / 4 < 61 * 1024 * 1024


def test_config_errors_name_the_field(tmp_path) -> None:
    """Not a required node: a refused configuration says which field is wrong (no values)."""
    from apctt_contracts.config import ConfigError

    config = json.loads((ROOT / "config/dev.json").read_text())
    config["limits"]["deadline_ms"] = 0
    config["compute"]["device"] = "auto"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigError) as caught:
        load_config(path, profile="dev")
    assert caught.value.code == "CONFIG_INVALID"
    assert "limits.deadline_ms: greater_than" in caught.value.details
    assert any(d.startswith("compute.device:") for d in caught.value.details)


def test_versions_and_schema_nullability() -> None:
    """Not a required node: one kit version; omittable fields never advertise null."""
    import tomllib

    from apctt_contracts import KIT_VERSION

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert KIT_VERSION == pyproject["project"]["version"]
    schemas = json.loads((ROOT / "contracts/openapi.json").read_text())["components"]["schemas"]
    for model, fields in (
        ("AssessmentResult", ("confidence", "uncertainty", "trl_rationale", "criterion_rationale")),
        ("EvidenceItem", ("supporting_span",)),
        ("Case", ("description", "benefits", "sdg_targets", "sdg_rationale")),
    ):
        for name in fields:
            text = json.dumps(schemas[model]["properties"][name])
            assert '"null"' not in text and '"default"' not in text, (model, name)
