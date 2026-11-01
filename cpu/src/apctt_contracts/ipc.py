"""SPEC-09/17 worker IPC contract and the API–supervisor boundary.

The worker receives exactly one bounded JSON envelope on stdin, writes `result.json` via an atomic
rename and reports progress through the `observation.json` sidecar. The supervisor protocol keeps
database connections out of the runtime; the API owns admission and terminal state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from . import WORKER_PROTOCOL_VERSION
from .config import ChecklistPayload, Compute, Limits
from .results import OUTPUT_RULES, AssessmentResult, CaseMeta, OutputIssue
from .strict_json import JSONValue
from .types import Identifier, NonBlankText, Sha256Hex, StrictModel

RunId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
Phase = Literal["initializing", "loading", "assessing", "cleaning", "finished"]
WORKER_FAILURE_CODES = frozenset(
    {
        "MODULE_ERROR",
        "OUT_OF_MEMORY",
        "CUDA_UNAVAILABLE",
        "CUDA_RUNTIME_ERROR",
        "CLEANUP_ERROR",
        "INVALID_OUTPUT",
        "WORKER_PROTOCOL_ERROR",
    }
)
JOB_ERROR_CODES = WORKER_FAILURE_CODES | {
    "TIMEOUT",
    "WORKER_CRASH",
    "SERVER_RESTARTED",
    "INTERNAL_ERROR",
}


class WorkerContext(StrictModel):
    run_id: RunId
    seed: NonNegativeInt
    time_budget_ms: PositiveInt
    scratch_dir: NonBlankText


class WorkerRunConfig(StrictModel):
    checklist: ChecklistPayload
    module: dict[str, Any]
    compute: Compute


class WorkerDocument(StrictModel):
    """A document as a module receives it in `case.documents` (worker protocol v2)."""

    document_id: Identifier
    title: NonBlankText
    document_date: NonBlankText | None
    language: Literal["en"]
    format: Literal["pdf"]
    sha256: Sha256Hex
    page_count: PositiveInt
    path: NonBlankText


class WorkerEnvelope(StrictModel):
    protocol_version: Literal["apctt-worker-v2"]
    run_id: RunId
    case: dict[str, Any]
    config: WorkerRunConfig
    context: WorkerContext

    @model_validator(mode="after")
    def _same_run(self) -> WorkerEnvelope:
        if self.context.run_id != self.run_id:
            raise ValueError("context.run_id must equal run_id")
        return self


class WorkerTimings(StrictModel):
    load: float | None
    assess: float | None
    cleanup: float | None


class ComputeObservation(StrictModel):
    device: str | None
    torch_version: str | None
    torch_cuda_build: str | None
    cuda_available: bool | None
    kernel_probe_passed: bool | None
    peak_allocated_bytes: NonNegativeInt | None
    peak_reserved_bytes: NonNegativeInt | None
    measurement_method: str | None


class Observation(StrictModel):
    protocol_version: Literal["apctt-worker-v2"]
    run_id: RunId
    phase: Phase
    source: Literal["worker_observed"]
    timings_ms: WorkerTimings
    unavailable_reasons: dict[str, NonBlankText]
    compute: ComputeObservation
    warning_codes: list[Annotated[str, Field(pattern=r"^[A-Z0-9_]{1,64}$")]]


class WorkerFailure(StrictModel):
    """Written by the worker wrapper when the module fails in-process. Code is allowlisted."""

    protocol_version: Literal["apctt-worker-v2"]
    run_id: RunId
    phase: Phase
    error_code: Literal[
        "MODULE_ERROR",
        "OUT_OF_MEMORY",
        "CUDA_UNAVAILABLE",
        "CUDA_RUNTIME_ERROR",
        "CLEANUP_ERROR",
        "INVALID_OUTPUT",
        "WORKER_PROTOCOL_ERROR",
    ]


def worker_protocol_schema() -> dict[str, Any]:
    """Machine-readable IPC contract frozen in contracts/worker-protocol.json."""
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "stdin": "exactly one UTF-8 JSON envelope, then EOF",
        "files": {
            "result": "result.json.tmp -> result.json (atomic rename, regular file, no symlink)",
            "observation": "observation.json.tmp -> observation.json (bounded sidecar)",
            "failure": "failure.json.tmp -> failure.json (allowlisted error_code only)",
        },
        "stdout_stderr": "logs only; never parsed as protocol",
        "envelope": WorkerEnvelope.model_json_schema(),
        "case_document": WorkerDocument.model_json_schema(),
        "documents": "each PDF is a read-only copy at case.documents[].path inside the scratch "
        "directory; content_base64 is never passed to the worker",
        "observation": Observation.model_json_schema(),
        "failure": WorkerFailure.model_json_schema(),
        "result": AssessmentResult.model_json_schema(),
        "result_rules": sorted(OUTPUT_RULES),
    }


@dataclass(frozen=True)
class RunLimits:
    max_result_bytes: int
    max_log_bytes: int
    max_observation_bytes: int
    termination_grace_ms: int

    @classmethod
    def from_limits(cls, limits: Limits) -> RunLimits:
        return cls(
            max_result_bytes=limits.max_result_bytes,
            max_log_bytes=limits.max_log_bytes,
            max_observation_bytes=limits.max_observation_bytes,
            termination_grace_ms=limits.termination_grace_ms,
        )


@dataclass(frozen=True)
class RunSpec:
    """Everything the supervisor needs for one cold run. No DB handle, no reference labels."""

    assessment_id: str
    run_id: str
    case: Mapping[str, JSONValue]
    case_meta: CaseMeta
    checklist: ChecklistPayload
    module: Mapping[str, JSONValue]
    compute: Compute
    worker_python: Path
    launcher_argv: tuple[str, ...]
    extra_env: Mapping[str, str]
    scratch_root: Path
    artifact_dir: Path
    deadline_monotonic: float  # supervisor-local; the worker receives time_budget_ms instead
    limits: RunLimits
    document_files: tuple[Path, ...] = ()  # stored PDFs in case.documents order


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    pid: int
    pgid: int
    start_token: str | None
    started_monotonic: float
    scratch_dir: Path


@dataclass(frozen=True)
class RunObservation:
    run_id: str
    alive: bool
    observation: Observation | None
    warning_codes: tuple[str, ...]


@dataclass(frozen=True)
class TerminationOutcome:
    reason: str
    signals: tuple[str, ...]
    cleanup_confirmed: bool
    elapsed_ms: float


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    result_origin: Literal["module", "supervisor"]
    result: Mapping[str, JSONValue]
    error_code: str | None
    raw_output_sha256: str | None
    raw_output_bytes: int | None
    raw_artifact: str | None
    output_issues: tuple[OutputIssue, ...]
    exit_code: int | None
    exit_signal: int | None
    timed_out: bool
    cleanup_confirmed: bool
    total_ms: float
    observation: Observation | None
    warning_codes: tuple[str, ...]
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool


class Supervisor(Protocol):
    async def start(self, run_spec: RunSpec) -> RunHandle: ...

    async def observe(self, run_handle: RunHandle) -> RunObservation: ...

    async def await_result(self, run_handle: RunHandle) -> RunOutcome: ...

    async def terminate(self, run_handle: RunHandle, reason: str) -> TerminationOutcome: ...


class RunContext(Protocol):
    run_id: str
    seed: int
    time_budget_ms: int
    scratch_dir: Path


class AssessmentModule(Protocol):
    """Participant entrypoint, default `participant_module.adapter:AssessmentModule`."""

    def load(self, config: Mapping[str, JSONValue], asset_root: Path) -> None: ...

    def assess(
        self, case: Mapping[str, JSONValue], context: RunContext
    ) -> Mapping[str, JSONValue]: ...

    def reset_case(self) -> None: ...

    def close(self) -> None: ...
