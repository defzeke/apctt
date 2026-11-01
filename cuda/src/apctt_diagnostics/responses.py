"""SPEC-11 error envelopes and job diagnostics.

Responses are rebuilt from allowlisted fields. Exception text, request values, Pydantic input or
context, locals, source lines, environment variables and absolute host paths are never copied.
The profile is server configuration; no parameter here lets a client enable debug output.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from apctt_contracts import UNKNOWN_FIELD
from apctt_contracts.cases import KNOWN_FIELDS, FieldIssue
from apctt_contracts.results import OPTIONAL_FIELDS, OUTPUT_RULES, REQUIRED_FIELDS, OutputIssue
from apctt_contracts.strict_json import REASONS as JSON_REASONS


@dataclass(frozen=True)
class ErrorSpec:
    http_status: int
    stage: str
    message: str
    action: str
    retryable: bool


ERROR_SPECS: dict[str, ErrorSpec] = {
    "INVALID_JSON": ErrorSpec(
        400,
        "request_parsing",
        "The request body is not valid strict JSON.",
        "Send UTF-8 JSON without duplicate keys, NaN or Infinity.",
        False,
    ),
    "REQUEST_VALIDATION_ERROR": ErrorSpec(
        422,
        "request_validation",
        "A request field does not satisfy the input contract.",
        "Correct the reported field and submit the request again.",
        False,
    ),
    "VERSION_MISMATCH": ErrorSpec(
        422,
        "request_validation",
        "The case schema or checklist version does not match the server profile.",
        "Use the versions published by GET /v1/package.",
        False,
    ),
    "UNSUPPORTED_INPUT_PROFILE": ErrorSpec(
        422,
        "request_validation",
        "The input profile is not supported by this server.",
        "Submit a pdf_report case.",
        False,
    ),
    "REQUEST_TOO_LARGE": ErrorSpec(
        413,
        "request_size",
        "The request body exceeds the configured size limit.",
        "Reduce the case size; the server never truncates input.",
        False,
    ),
    "UNSUPPORTED_MEDIA_TYPE": ErrorSpec(
        415,
        "request_headers",
        "Only application/json with identity encoding is accepted.",
        "Send Content-Type: application/json without Content-Encoding.",
        False,
    ),
    "REQUEST_ID_CONFLICT": ErrorSpec(
        409,
        "admission",
        "The request_id was already used with a different effective input.",
        "Use a new request_id for a different case or configuration.",
        False,
    ),
    "BUSY": ErrorSpec(
        503,
        "admission",
        "An assessment is already running on this server.",
        "Retry later with the same request_id.",
        True,
    ),
    "NOT_READY": ErrorSpec(
        503,
        "admission",
        "The server is not ready to accept assessments.",
        "Check GET /health/ready and retry after the operator resolves the reason.",
        True,
    ),
    "STORAGE_CAPACITY": ErrorSpec(
        503,
        "admission",
        "The job store has reached its configured capacity.",
        "Ask the operator to archive previous runs before submitting new cases.",
        True,
    ),
    "JOB_NOT_FOUND": ErrorSpec(
        404,
        "lookup",
        "No accessible assessment has this ID.",
        "Use the Location header returned at submission.",
        False,
    ),
    "ROUTE_NOT_FOUND": ErrorSpec(
        404,
        "routing",
        "No API route matches this path.",
        "See GET /openapi.json for the available routes.",
        False,
    ),
    "METHOD_NOT_ALLOWED": ErrorSpec(
        405,
        "routing",
        "This method is not allowed for the route.",
        "See GET /openapi.json for the allowed methods.",
        False,
    ),
    "AUTH_REQUIRED": ErrorSpec(
        401,
        "authentication",
        "Credentials are required.",
        "Send the bearer token issued by the operator.",
        False,
    ),
    "FORBIDDEN": ErrorSpec(
        403,
        "authentication",
        "The credentials are not accepted.",
        "Use the credentials issued by the operator.",
        False,
    ),
    "INTERNAL_ERROR": ErrorSpec(
        500,
        "internal",
        "The server hit an unexpected error.",
        "Report the trace_id to the operator.",
        False,
    ),
}

REQUEST_RULES = frozenset(
    {
        "required",
        "unknown_field",
        "string",
        "strict_integer",
        "strict_positive_integer",
        "array",
        "object",
        "non_empty_array",
        "non_empty_string",
        "non_blank_string",
        "omit_instead_of_null",
        "iso_date",
        "unique_items",
        "invalid_value",
        "allowed_value",
        "request_id_pattern",
        "identifier_pattern",
        "unique_document_id",
        "max_documents",
        "sha256_pattern",
        "base64_content",
        "max_document_bytes",
        "pdf_signature",
        "document_sha256_match",
        "pdf_readable",
        "max_total_pages",
        "supported_input_profile",
        "profile_schema_version",
        "profile_checklist_version",
    }
)
_EXPECTED = [
    re.compile(p)
    for p in (
        r"field is present",
        r"only fields defined by the contract",
        r"string",
        r"integer",
        r"integer >= 1",
        r"array",
        r"object",
        r"JSON object",
        r"array with at least one item",
        r"non-empty string",
        r"string with a non-whitespace character",
        r"omit the optional field instead of null",
        r"YYYY-MM-DD or null",
        r"array without duplicate values",
        r"value matching the published API schema",
        r"component\|subsystem\|system\|technology",
        r"en",
        r"pdf",
        r"one of 6, 11, 13, 14",
        r"documented value",
        r"ASCII \[A-Za-z0-9_-\]\{1,128\}",
        r"ASCII identifier \[A-Za-z0-9\]\[A-Za-z0-9_\.:-\]\{0,127\}",
        r"document_id unique within the case",
        r"at most [0-9]{1,9} documents",
        r"64 lowercase hexadecimal characters",
        r"strict standard base64",
        r"at most [0-9]{1,12} bytes per document",
        r"a PDF file \(%PDF- header\)",
        r"SHA-256 of the decoded document",
        r"a PDF the server can open and count pages of",
        r"at most [0-9]{1,9} pages in total",
        r"pdf_report",
    )
]
GENERIC_EXPECTED = "value matching the published API schema"
RECEIVED_TYPES = frozenset(
    {"null", "boolean", "integer", "number", "string", "array", "object", "missing", "unknown"}
)
_RESULT_NAMES = frozenset(
    {
        "result",
        *REQUIRED_FIELDS,
        *OPTIONAL_FIELDS,
        "document_id",
        "page",
        "supporting_span",
    }
)
_TOKEN = re.compile(r"\[([0-9]{1,6})\]|\.?([^.\[\]]+)")
_PROJECT_PACKAGES = ("apctt_contracts", "apctt_server", "apctt_runtime", "apctt_diagnostics")
_SRC_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DiagnosticsPolicy:
    profile: str
    max_bytes: int
    max_field_errors: int
    max_stack_frames: int
    allowed_expected: frozenset[str] = field(default_factory=frozenset)


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy; unknown or non-finite values become null."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items() if isinstance(k, str)}
    return None


def _safe_path(path: str, names: frozenset[str]) -> str:
    if path in ("<body>", UNKNOWN_FIELD):
        return path
    position = 0
    for match in _TOKEN.finditer(path):
        if match.start() != position:
            return UNKNOWN_FIELD
        position = match.end()
        name = match.group(2)
        if name is not None and name not in names and name != UNKNOWN_FIELD:
            return UNKNOWN_FIELD
    return path if position == len(path) and len(path) <= 256 else UNKNOWN_FIELD


def _safe_expected(expected: str, policy: DiagnosticsPolicy) -> bool:
    return expected in policy.allowed_expected or any(p.fullmatch(expected) for p in _EXPECTED)


def sanitize_field_issue(issue: FieldIssue, policy: DiagnosticsPolicy) -> dict[str, str]:
    path = _safe_path(issue.path, KNOWN_FIELDS)
    rule, expected = issue.rule, issue.expected
    if rule not in REQUEST_RULES:
        rule, expected = "invalid_value", GENERIC_EXPECTED
    elif not _safe_expected(expected, policy):
        expected = GENERIC_EXPECTED
    received = issue.received_type if issue.received_type in RECEIVED_TYPES else "unknown"
    return {"path": path, "rule": rule, "expected": expected, "received_type": received}


def sanitize_output_issue(issue: OutputIssue, criterion_ids: Iterable[str]) -> dict[str, str]:
    names = set(_RESULT_NAMES)
    for cid in criterion_ids:
        names.update(part for part in cid.split(".") if part)
        names.add(cid)
    path = _safe_path(issue.path, frozenset(names))
    rule = issue.rule if issue.rule in OUTPUT_RULES else "result_invalid"
    return {"path": path, "rule": rule}


def stack_frames(exc: BaseException, limit: int) -> list[dict[str, Any]]:
    """Project frames only: package-relative file, line and code-object name."""
    frames: list[dict[str, Any]] = []
    tb: TracebackType | None = exc.__traceback__
    for frame, line in traceback.walk_tb(tb):
        try:
            relative = Path(frame.f_code.co_filename).resolve().relative_to(_SRC_ROOT)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] not in _PROJECT_PACKAGES:
            continue
        frames.append({"file": relative.as_posix(), "line": line, "function": frame.f_code.co_name})
    return frames[-limit:] if limit > 0 else []


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _parse_error(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not value or value.get("reason") not in JSON_REASONS:
        return None
    result: dict[str, Any] = {"reason": value["reason"]}
    for key in ("line", "column"):
        number = value.get(key)
        result[key] = number if isinstance(number, int) and not isinstance(number, bool) else None
    return result


def error_envelope(
    code: str,
    *,
    trace_id: str,
    api_version: str,
    policy: DiagnosticsPolicy,
    field_issues: Sequence[FieldIssue] = (),
    duration_ms: float | None = None,
    exc: BaseException | None = None,
    parse_error: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    spec = ERROR_SPECS.get(code, ERROR_SPECS["INTERNAL_ERROR"])
    if code not in ERROR_SPECS:
        code = "INTERNAL_ERROR"
    errors = [sanitize_field_issue(i, policy) for i in field_issues]
    shown = errors[: policy.max_field_errors]
    diagnostics: dict[str, Any] = {
        "profile": policy.profile,
        "duration_ms": json_safe(duration_ms),
        "truncated": len(shown) < len(errors),
    }
    if errors:
        diagnostics["field_errors_total"] = len(errors)
    parsed = _parse_error(parse_error)
    if parsed is not None:
        diagnostics["parse_error"] = parsed
    if exc is not None and policy.profile == "dev":
        diagnostics["stack_frames"] = stack_frames(exc, policy.max_stack_frames)
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": spec.message,
            "stage": spec.stage,
            "retryable": spec.retryable,
            "action": spec.action,
            "field_errors": shown,
        },
        "meta": {"trace_id": trace_id, "api_version": api_version},
        "diagnostics": diagnostics,
    }
    while _size(body) > policy.max_bytes:
        diagnostics["truncated"] = True
        if len(body["error"]["field_errors"]) > 1:
            body["error"]["field_errors"] = body["error"]["field_errors"][
                : len(body["error"]["field_errors"]) // 2
            ]
        elif diagnostics.get("stack_frames"):
            diagnostics["stack_frames"] = diagnostics["stack_frames"][1:]
        elif body["error"]["field_errors"]:
            body["error"]["field_errors"] = []
        else:
            for key in ("stack_frames", "parse_error", "field_errors_total"):
                diagnostics.pop(key, None)
            break
    return spec.http_status, body


# Fixed statements for failures the server records itself (result_origin=supervisor). They are
# server-defined sentences, never exception or module text (SPEC-11).
JOB_FAILURE_MESSAGES: dict[str, str] = {
    "TIMEOUT": "The model could not process the documents within the {limit}-second limit.",
    "WORKER_CRASH": "The model process could not be started or ended abnormally.",
    "MODULE_ERROR": "The model raised an error while loading or processing the documents.",
    "OUT_OF_MEMORY": "The model ran out of memory while processing the documents.",
    "CUDA_UNAVAILABLE": "The GPU required by the model was not available.",
    "CUDA_RUNTIME_ERROR": "The GPU failed while the model was processing the documents.",
    "CLEANUP_ERROR": "The model failed while releasing its resources after the assessment.",
    "INVALID_OUTPUT": "The model returned a result that does not satisfy the result contract.",
    "WORKER_PROTOCOL_ERROR": (
        "The model process did not return a usable result file (missing, not a regular file, "
        "or larger than the result limit)."
    ),
    "SERVER_RESTARTED": "The server restarted before the assessment finished; it was not re-run.",
    "INTERNAL_ERROR": "The server could not complete or record the assessment.",
}


def job_failure(code: str, *, deadline_ms: int | None = None) -> dict[str, Any]:
    """The diagnostics `failure` block for a supervisor failure result."""
    if code not in JOB_FAILURE_MESSAGES:
        code = "INTERNAL_ERROR"
    if code == "TIMEOUT":
        if deadline_ms is None:
            raise ValueError("TIMEOUT needs the deadline")
        seconds = (
            str(deadline_ms // 1000) if deadline_ms % 1000 == 0 else f"{deadline_ms / 1000:.1f}"
        )
        message = JOB_FAILURE_MESSAGES[code].format(limit=seconds)
        return {"code": code, "message": message, "limit_ms": deadline_ms}
    return {"code": code, "message": JOB_FAILURE_MESSAGES[code]}


def job_diagnostics(
    *,
    policy: DiagnosticsPolicy,
    phase: str,
    timings_ms: Mapping[str, float | None],
    unavailable_reasons: Mapping[str, str],
    warning_codes: Sequence[str],
    output_issues: Sequence[OutputIssue] = (),
    criterion_ids: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issues = [sanitize_output_issue(i, criterion_ids) for i in output_issues]
    shown = issues[: policy.max_field_errors]
    diagnostics: dict[str, Any] = {
        "profile": policy.profile,
        "phase": phase,
        "timings_ms": {k: json_safe(v) for k, v in timings_ms.items()},
        "timing_sources": {
            k: ("supervisor" if k == "total" else "worker_observed") for k in timings_ms
        },
        "unavailable_reasons": {
            str(k): str(v)
            for k, v in unavailable_reasons.items()
            if re.fullmatch(r"[A-Z0-9_]{1,64}", str(v))
        },
        "warning_codes": [c for c in warning_codes if re.fullmatch(r"[A-Z0-9_]{1,64}", c)],
        "truncated": len(shown) < len(issues),
    }
    if issues:
        diagnostics["output_issues"] = shown
        diagnostics["output_issues_total"] = len(issues)
    for key, value in (extra or {}).items():
        if key not in diagnostics:
            diagnostics[key] = json_safe(value)
    while _size(diagnostics) > policy.max_bytes and diagnostics.get("output_issues"):
        diagnostics["truncated"] = True
        diagnostics["output_issues"] = diagnostics["output_issues"][
            : len(diagnostics["output_issues"]) // 2
        ]
    return diagnostics
