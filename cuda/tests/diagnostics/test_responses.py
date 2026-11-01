"""SPEC-11 structured, allowlisted diagnostics: AC-14, AC-15, AC-16."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

from apctt_contracts import API_VERSION
from apctt_contracts.cases import FieldIssue
from apctt_contracts.config import ConfigError, load_config
from apctt_contracts.responses import ERROR_CODES, ErrorEnvelope
from apctt_contracts.results import OutputIssue
from apctt_diagnostics.responses import (
    ERROR_SPECS,
    DiagnosticsPolicy,
    error_envelope,
    job_diagnostics,
    json_safe,
)

DEV = DiagnosticsPolicy(profile="dev", max_bytes=32768, max_field_errors=20, max_stack_frames=10)
EVAL = DiagnosticsPolicy(
    profile="evaluation", max_bytes=32768, max_field_errors=20, max_stack_frames=10
)
SENTINEL = "SENTINEL-51ac-diagnostics"
ROOT = Path(__file__).resolve().parents[2]


def _caught() -> BaseException:
    """An exception raised inside product code whose context holds an absolute sentinel path."""
    try:
        load_config(ROOT / f"missing-{SENTINEL}.json")
    except ConfigError as exc:
        return exc
    raise AssertionError("unreachable")


def issue(i: int) -> FieldIssue:
    return FieldIssue(f"case.documents[{i}].title", "non_blank_string", "string", "string")


def test_ac14_diagnostics_schema_and_caps() -> None:
    # Every public error code has a fixed HTTP status, stage, message, action, retryable flag.
    assert set(ERROR_SPECS) == set(ERROR_CODES)
    for code in ERROR_CODES:
        status, body = error_envelope(code, trace_id="t-1", api_version=API_VERSION, policy=DEV)
        ErrorEnvelope.model_validate(body)
        assert body["error"]["code"] == code and status == ERROR_SPECS[code].http_status
        assert body["diagnostics"]["profile"] == "dev"
    assert ERROR_SPECS["BUSY"].retryable and not ERROR_SPECS["REQUEST_VALIDATION_ERROR"].retryable
    assert ERROR_SPECS["REQUEST_TOO_LARGE"].http_status == 413
    assert ERROR_SPECS["UNSUPPORTED_MEDIA_TYPE"].http_status == 415

    # Field errors are capped with an explicit truncation flag; output stays valid JSON.
    issues = [issue(i) for i in range(25)]
    status, body = error_envelope(
        "REQUEST_VALIDATION_ERROR",
        trace_id="t-2",
        api_version=API_VERSION,
        policy=DEV,
        field_issues=issues,
        duration_ms=math.nan,
    )
    assert status == 422
    assert len(body["error"]["field_errors"]) == 20
    assert body["diagnostics"]["truncated"] is True
    assert body["diagnostics"]["field_errors_total"] == 25
    assert body["diagnostics"]["duration_ms"] is None  # NaN is not faked as a number
    json.dumps(body, allow_nan=False)

    # Total diagnostics cap.
    small = DiagnosticsPolicy(
        profile="dev", max_bytes=2048, max_field_errors=20, max_stack_frames=10
    )
    status, body = error_envelope(
        "REQUEST_VALIDATION_ERROR",
        trace_id="t-3",
        api_version=API_VERSION,
        policy=small,
        field_issues=issues,
    )
    assert len(json.dumps(body).encode()) <= 2048
    assert body["diagnostics"]["truncated"] is True

    # Unserializable or non-finite values never break the response.
    assert json_safe({"a": math.inf, "b": b"x", "c": ValueError("x"), "d": [1, math.nan]}) == {
        "a": None,
        "b": None,
        "c": None,
        "d": [1, None],
    }

    # Job diagnostics: worker/supervisor timing sources, unavailable reasons, capped issues.
    diag = job_diagnostics(
        policy=DEV,
        phase="finished",
        timings_ms={"load": 8.0, "assess": 2.5, "cleanup": None, "total": 15.0},
        unavailable_reasons={"cleanup": "NOT_OBSERVED"},
        warning_codes=["SIDECAR_INVALID"],
        output_issues=[OutputIssue(f"result.x{i}", "result_unknown_field") for i in range(30)],
        extra={"exit_code": 0, "nan": math.nan},
    )
    assert diag["timings_ms"] == {"load": 8.0, "assess": 2.5, "cleanup": None, "total": 15.0}
    assert diag["timing_sources"] == {
        "load": "worker_observed",
        "assess": "worker_observed",
        "cleanup": "worker_observed",
        "total": "supervisor",
    }
    assert len(diag["output_issues"]) == 20 and diag["truncated"] is True
    assert diag["nan"] is None
    json.dumps(diag, allow_nan=False)


def test_ac15_allowlisted_errors() -> None:
    exc = _caught()
    status, body = error_envelope(
        "INTERNAL_ERROR", trace_id="t-4", api_version=API_VERSION, policy=DEV, exc=exc
    )
    text = json.dumps(body)
    assert status == 500
    assert SENTINEL not in text and str(ROOT) not in text and "CONFIG_UNREADABLE" not in text
    assert body["error"]["message"] == ERROR_SPECS["INTERNAL_ERROR"].message
    frames = body["diagnostics"]["stack_frames"]
    # Only allowlisted project frames with project-relative paths; tests are not product code.
    assert all(not f["file"].startswith("/") for f in frames)
    assert all(f["file"].startswith("apctt_") for f in frames)

    # Messages do not depend on input; hostile paths/rules/types are replaced.
    hostile = [
        FieldIssue(f"case.{SENTINEL}", "x", "y", "z"),
        FieldIssue("case.title", f"rule-{SENTINEL}", "string", "string"),
        FieldIssue("case.title", "non_blank_string", f"expected {SENTINEL}" * 20, "string"),
        FieldIssue("case.title", "non_blank_string", "string", SENTINEL),
        FieldIssue("/etc/passwd", "required", "field is present", "missing"),
    ]
    _, body = error_envelope(
        "REQUEST_VALIDATION_ERROR",
        trace_id="t-5",
        api_version=API_VERSION,
        policy=DEV,
        field_issues=hostile,
    )
    text = json.dumps(body)
    assert SENTINEL not in text and "/etc/passwd" not in text
    first = body["error"]["field_errors"][0]
    assert first == {
        "path": "<unknown-field>",
        "rule": "invalid_value",
        "expected": "value matching the published API schema",
        "received_type": "unknown",
    }
    # Parse position metadata (numbers only) is allowed; the reason is a fixed code.
    _, body = error_envelope(
        "INVALID_JSON",
        trace_id="t-6",
        api_version=API_VERSION,
        policy=DEV,
        parse_error={"reason": "duplicate_key", "line": 3, "column": 7},
    )
    assert body["diagnostics"]["parse_error"] == {"reason": "duplicate_key", "line": 3, "column": 7}
    _, body = error_envelope(
        "INVALID_JSON",
        trace_id="t-7",
        api_version=API_VERSION,
        policy=DEV,
        parse_error={"reason": SENTINEL, "line": SENTINEL, "column": None},
    )
    assert SENTINEL not in json.dumps(body)


def test_ac16_profile_controls_debug() -> None:
    exc = _caught()
    _, dev = error_envelope(
        "INTERNAL_ERROR", trace_id="t", api_version=API_VERSION, policy=DEV, exc=exc
    )
    _, ev = error_envelope(
        "INTERNAL_ERROR", trace_id="t", api_version=API_VERSION, policy=EVAL, exc=exc
    )
    assert dev["diagnostics"]["stack_frames"]
    assert "stack_frames" not in ev["diagnostics"] and ev["diagnostics"]["profile"] == "evaluation"
    # Nothing in the public builders accepts a caller-controlled debug switch.
    for fn in (error_envelope, job_diagnostics):
        assert not {"debug", "trace", "verbose"} & set(inspect.signature(fn).parameters)
    # Frame count cap.
    capped = DiagnosticsPolicy(
        profile="dev", max_bytes=32768, max_field_errors=20, max_stack_frames=1
    )
    _, body = error_envelope(
        "INTERNAL_ERROR", trace_id="t", api_version=API_VERSION, policy=capped, exc=exc
    )
    assert len(body["diagnostics"]["stack_frames"]) <= 1


def test_job_failure_statements() -> None:
    """Not a required node: one fixed statement per job error code; limits read in seconds."""
    from apctt_contracts.ipc import JOB_ERROR_CODES
    from apctt_diagnostics.responses import JOB_FAILURE_MESSAGES, job_failure

    assert set(JOB_FAILURE_MESSAGES) == set(JOB_ERROR_CODES)
    assert job_failure("TIMEOUT", deadline_ms=180000)["message"] == (
        "The model could not process the documents within the 180-second limit."
    )
    assert "1234.6-second" in job_failure("TIMEOUT", deadline_ms=1234567)["message"]
    assert job_failure("SOMETHING_ELSE")["code"] == "INTERNAL_ERROR"
