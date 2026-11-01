"""SPEC-07/11/17 HTTP response envelopes, error codes and readiness reasons."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from .results import AssessmentResult
from .types import NonBlankText, StrictModel

ERROR_CODES = (
    "INVALID_JSON",
    "REQUEST_VALIDATION_ERROR",
    "VERSION_MISMATCH",
    "UNSUPPORTED_INPUT_PROFILE",
    "REQUEST_TOO_LARGE",
    "UNSUPPORTED_MEDIA_TYPE",
    "REQUEST_ID_CONFLICT",
    "BUSY",
    "NOT_READY",
    "STORAGE_CAPACITY",
    "JOB_NOT_FOUND",
    "ROUTE_NOT_FOUND",
    "METHOD_NOT_ALLOWED",
    "AUTH_REQUIRED",
    "FORBIDDEN",
    "INTERNAL_ERROR",
)
ErrorCode = Literal[
    "INVALID_JSON",
    "REQUEST_VALIDATION_ERROR",
    "VERSION_MISMATCH",
    "UNSUPPORTED_INPUT_PROFILE",
    "REQUEST_TOO_LARGE",
    "UNSUPPORTED_MEDIA_TYPE",
    "REQUEST_ID_CONFLICT",
    "BUSY",
    "NOT_READY",
    "STORAGE_CAPACITY",
    "JOB_NOT_FOUND",
    "ROUTE_NOT_FOUND",
    "METHOD_NOT_ALLOWED",
    "AUTH_REQUIRED",
    "FORBIDDEN",
    "INTERNAL_ERROR",
]
READY_REASONS = (
    "READY",
    "BUSY",
    "CONFIG_INVALID",
    "STORAGE_CAPACITY",
    "CUDA_UNAVAILABLE",
    "CUDA_RUNTIME_ERROR",
    "RECOVERY_PENDING",
)
ReadyReason = Literal[
    "READY",
    "BUSY",
    "CONFIG_INVALID",
    "STORAGE_CAPACITY",
    "CUDA_UNAVAILABLE",
    "CUDA_RUNTIME_ERROR",
    "RECOVERY_PENDING",
]
JobStatus = Literal["queued", "running", "finished"]


class FieldError(StrictModel):
    path: str
    rule: str
    expected: str
    received_type: str


class ErrorBody(StrictModel):
    code: ErrorCode
    message: NonBlankText
    stage: NonBlankText
    retryable: bool
    action: NonBlankText
    field_errors: list[FieldError] = []


class ErrorMeta(StrictModel):
    trace_id: NonBlankText
    api_version: NonBlankText


class ErrorEnvelope(StrictModel):
    error: ErrorBody
    meta: ErrorMeta
    diagnostics: dict[str, Any]


class JobMeta(StrictModel):
    trace_id: NonBlankText
    api_version: NonBlankText
    schema_version: NonBlankText
    package_id: NonBlankText
    input_sha256: NonBlankText
    config_sha256: NonBlankText
    verified_package_sha256: str | None
    accepted_at: NonBlankText
    started_at: str | None
    finished_at: str | None


class JobEnvelope(StrictModel):
    assessment_id: NonBlankText
    request_id: NonBlankText
    job_status: JobStatus
    result_origin: Literal["module", "supervisor"] | None
    result: AssessmentResult | None
    meta: JobMeta
    diagnostics: dict[str, Any]

    @model_validator(mode="after")
    def _state(self) -> JobEnvelope:
        finished = self.job_status == "finished"
        if finished != (self.result is not None) or finished != (self.result_origin is not None):
            raise ValueError("finished jobs carry origin and result; queued/running carry neither")
        return self


class HealthLive(StrictModel):
    live: Literal[True]
    api_version: NonBlankText


class HealthReady(StrictModel):
    ready: bool
    reason_code: ReadyReason
    loading_policy: Literal["cold_per_case"]
    active_assessment_id: str | None
    compute_status: dict[str, Any]


class PackageInfo(StrictModel):
    package_id: NonBlankText
    profile: Literal["dev", "evaluation"]
    api_version: NonBlankText
    schema_version: NonBlankText
    checklist_version: NonBlankText
    kit_version: NonBlankText
    contract_version: NonBlankText
    worker_protocol_version: NonBlankText
    declared_package_sha256: str | None
    verified_package_sha256: str | None
    verification_status: Literal["verified", "not_verified", "mismatch"]
    example_only: bool
    loading_policy: Literal["cold_per_case"]
    config_sha256: NonBlankText
    checklist_sha256: NonBlankText
    compute: dict[str, Any]
