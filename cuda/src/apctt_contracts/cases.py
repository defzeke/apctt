"""SPEC-05 request/case schema and allowlisted field issues.

Field issues never echo request values: paths use contract field names or `<unknown-field>`,
array indices stay numeric, and `received_type` is only the JSON type name.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core import ErrorDetails, PydanticCustomError

from . import SUPPORTED_INPUT_PROFILES, UNKNOWN_FIELD
from .pdf import PDF_SIGNATURE, decode_content
from .strict_json import JSONValue
from .types import (
    Identifier,
    IsoDate,
    NonBlankText,
    RequestId,
    Sha256Hex,
    StrictModel,
    omittable,
    reject_null,
)

TargetType = Literal["component", "subsystem", "system", "technology"]
SdgTarget = Literal[6, 11, 13, 14]


class AssessmentTarget(StrictModel):
    target_id: Identifier
    type: TargetType
    description: NonBlankText


class Document(StrictModel):
    """One PDF, carried as strict base64 with the SHA-256 of the decoded bytes."""

    document_id: Identifier
    title: NonBlankText
    document_date: NonBlankText | None
    language: Literal["en"]
    format: Literal["pdf"]
    content_base64: NonBlankText
    sha256: Sha256Hex


class Case(StrictModel):
    schema_version: NonBlankText
    input_profile: NonBlankText
    checklist_version: NonBlankText
    case_id: Identifier
    assessment_as_of: IsoDate | None
    assessment_target: AssessmentTarget
    intended_application: NonBlankText
    intended_environment: NonBlankText
    title: NonBlankText
    documents: Annotated[list[Document], Field(min_length=1)]
    description: NonBlankText | None = omittable()
    benefits: NonBlankText | None = omittable()
    sdg_targets: list[SdgTarget] | None = omittable()
    sdg_rationale: NonBlankText | None = omittable()

    @field_validator("description", "benefits", "sdg_targets", "sdg_rationale", mode="before")
    @classmethod
    def _optional_not_null(cls, value: object) -> object:
        return reject_null(value)

    @field_validator("sdg_targets")
    @classmethod
    def _unique_sdg(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(set(value)) != len(value):
            raise PydanticCustomError("unique_items", "values must not repeat")
        return value


class AssessmentRequest(StrictModel):
    request_id: RequestId
    case: Case


def _field_names(*models: type[BaseModel]) -> frozenset[str]:
    return frozenset(name for model in models for name in model.model_fields)


KNOWN_FIELDS = _field_names(AssessmentRequest, Case, Document, AssessmentTarget)

_RULES: dict[str, tuple[str, str]] = {
    "missing": ("required", "field is present"),
    "extra_forbidden": ("unknown_field", "only fields defined by the contract"),
    "string_type": ("string", "string"),
    "int_type": ("strict_integer", "integer"),
    "greater_than": ("strict_positive_integer", "integer >= 1"),
    "greater_than_equal": ("strict_positive_integer", "integer >= 1"),
    "list_type": ("array", "array"),
    "model_type": ("object", "object"),
    "model_attributes_type": ("object", "object"),
    "dict_type": ("object", "object"),
    "too_short": ("non_empty_array", "array with at least one item"),
    "string_too_short": ("non_empty_string", "non-empty string"),
    "non_blank_string": ("non_blank_string", "string with a non-whitespace character"),
    "null_not_allowed": ("omit_instead_of_null", "omit the optional field instead of null"),
    "iso_date": ("iso_date", "YYYY-MM-DD or null"),
    "unique_items": ("unique_items", "array without duplicate values"),
}
_LITERALS = {
    "type": "component|subsystem|system|technology",
    "language": "en",
    "format": "pdf",
    "sdg_targets": "one of 6, 11, 13, 14",
}
_DEFAULT_RULE = ("invalid_value", "value matching the published API schema")


@dataclass(frozen=True)
class FieldIssue:
    path: str
    rule: str
    expected: str
    received_type: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "rule": self.rule,
            "expected": self.expected,
            "received_type": self.received_type,
        }


def json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def format_path(loc: Sequence[str | int], known: frozenset[str] = KNOWN_FIELDS) -> str:
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            name = part if part in known else UNKNOWN_FIELD
            path += ("." if path else "") + name
    return path or "<body>"


def _last_name(loc: Sequence[str | int]) -> str | None:
    for part in reversed(loc):
        if isinstance(part, str):
            return part
    return None


def issue_from_error(error: ErrorDetails) -> FieldIssue:
    kind = error["type"]
    loc = tuple(error["loc"])
    field = _last_name(loc)
    if kind == "literal_error":
        rule, expected = "allowed_value", _LITERALS.get(field or "", "documented value")
    elif kind == "string_pattern_mismatch":
        if field == "request_id":
            rule, expected = "request_id_pattern", "ASCII [A-Za-z0-9_-]{1,128}"
        elif field == "sha256":
            rule, expected = "sha256_pattern", "64 lowercase hexadecimal characters"
        else:
            rule, expected = (
                "identifier_pattern",
                "ASCII identifier [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}",
            )
    else:
        rule, expected = _RULES.get(kind, _DEFAULT_RULE)
    received = "missing" if kind == "missing" else json_type_name(error.get("input"))
    return FieldIssue(format_path(loc), rule, expected, received)


@dataclass(frozen=True)
class CaseLimits:
    max_documents: int
    max_document_bytes: int


@dataclass(frozen=True)
class ProfileVersions:
    schema_version: str
    checklist_version: str


@dataclass(frozen=True)
class RequestCheck:
    code: str | None
    issues: tuple[FieldIssue, ...]
    request: AssessmentRequest | None = None
    contents: tuple[bytes, ...] = ()  # decoded PDFs, in document order, when the check passed

    @property
    def ok(self) -> bool:
        return self.code is None


def _document_issues(case: Case, limits: CaseLimits) -> tuple[list[FieldIssue], list[bytes]]:
    issues: list[FieldIssue] = []
    contents: list[bytes] = []
    seen_docs: set[str] = set()
    for index, document in enumerate(case.documents):
        base = f"case.documents[{index}]"
        if document.document_id in seen_docs:
            issues.append(
                FieldIssue(
                    f"{base}.document_id",
                    "unique_document_id",
                    "document_id unique within the case",
                    "string",
                )
            )
        seen_docs.add(document.document_id)
        data = decode_content(document.content_base64)
        if data is None:
            issues.append(
                FieldIssue(
                    f"{base}.content_base64", "base64_content", "strict standard base64", "string"
                )
            )
            continue
        if len(data) > limits.max_document_bytes:
            issues.append(
                FieldIssue(
                    f"{base}.content_base64",
                    "max_document_bytes",
                    f"at most {limits.max_document_bytes} bytes per document",
                    "string",
                )
            )
            continue
        if not data.startswith(PDF_SIGNATURE):
            issues.append(
                FieldIssue(
                    f"{base}.content_base64", "pdf_signature", "a PDF file (%PDF- header)", "string"
                )
            )
            continue
        if hashlib.sha256(data).hexdigest() != document.sha256:
            issues.append(
                FieldIssue(
                    f"{base}.sha256",
                    "document_sha256_match",
                    "SHA-256 of the decoded document",
                    "string",
                )
            )
            continue
        contents.append(data)
    if len(case.documents) > limits.max_documents:
        issues.append(
            FieldIssue(
                "case.documents",
                "max_documents",
                f"at most {limits.max_documents} documents",
                "array",
            )
        )
    return issues, contents


def page_issues(page_counts: Sequence[int | None], max_total_pages: int) -> list[FieldIssue]:
    """Issues after the server counted pages (None = not a readable PDF)."""
    issues = [
        FieldIssue(
            f"case.documents[{index}].content_base64",
            "pdf_readable",
            "a PDF the server can open and count pages of",
            "string",
        )
        for index, pages in enumerate(page_counts)
        if pages is None
    ]
    total = sum(pages for pages in page_counts if pages is not None)
    if not issues and total > max_total_pages:
        issues.append(
            FieldIssue(
                "case.documents",
                "max_total_pages",
                f"at most {max_total_pages} pages in total",
                "array",
            )
        )
    return issues


def check_request(
    value: JSONValue | Any, *, limits: CaseLimits, versions: ProfileVersions
) -> RequestCheck:
    """Validate a parsed request body. Precedence: schema, input profile, version, semantics."""
    if not isinstance(value, dict):
        issue = FieldIssue("<body>", "object", "JSON object", json_type_name(value))
        return RequestCheck("REQUEST_VALIDATION_ERROR", (issue,))
    try:
        request = AssessmentRequest.model_validate(value)
    except ValidationError as exc:
        issues = tuple(
            issue_from_error(e) for e in exc.errors(include_url=False, include_context=False)
        )
        return RequestCheck("REQUEST_VALIDATION_ERROR", issues)
    case = request.case
    if case.input_profile not in SUPPORTED_INPUT_PROFILES:
        issue = FieldIssue(
            "case.input_profile",
            "supported_input_profile",
            "|".join(SUPPORTED_INPUT_PROFILES),
            "string",
        )
        return RequestCheck("UNSUPPORTED_INPUT_PROFILE", (issue,))
    mismatches: list[FieldIssue] = []
    if case.schema_version != versions.schema_version:
        mismatches.append(
            FieldIssue(
                "case.schema_version",
                "profile_schema_version",
                versions.schema_version,
                "string",
            )
        )
    if case.checklist_version != versions.checklist_version:
        mismatches.append(
            FieldIssue(
                "case.checklist_version",
                "profile_checklist_version",
                versions.checklist_version,
                "string",
            )
        )
    if mismatches:
        return RequestCheck("VERSION_MISMATCH", tuple(mismatches))
    semantic, contents = _document_issues(case, limits)
    if semantic:
        return RequestCheck("REQUEST_VALIDATION_ERROR", tuple(semantic))
    return RequestCheck(None, (), request, tuple(contents))
