"""SPEC-03/16 server configuration, trusted checklist and startup gates.

The configuration is read once at start, validated strictly and frozen. Relative paths resolve
against the directory of the configuration file. Requests can never change these values.
"""

from __future__ import annotations

import ipaddress
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, NonNegativeInt, PositiveInt, ValidationError, model_validator

from . import DEMO_CHECKLIST_VERSION, DEV_SCHEMA_VERSION, LOADING_POLICY, WORKER_PROTOCOL_VERSION
from .hashing import canonical_sha256
from .strict_json import StrictJSONError, loads_strict
from .types import Identifier, NonBlankText, Sha256Hex, StrictModel

ENV_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]*$"
_SENSITIVE_ENV = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|PRIVATE|AWS_|AZURE_|GOOGLE_|GITHUB_|"
    r"ANTHROPIC|OPENAI|HF_)"
)
RESERVED_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "LANG",
        "LC_ALL",
        "PYTHONUTF8",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "CUDA_VISIBLE_DEVICES",
    }
)
ISOLATION_TESTS = (
    "job_store_access",
    "previous_output_access",
    "credential_access",
    "network_egress",
    "descendant_termination",
)


class Criterion(StrictModel):
    criterion_id: Identifier
    description: NonBlankText


class ChecklistPayload(StrictModel):
    """What the worker receives: version and criteria only. No reference labels."""

    version: NonBlankText
    criteria: Annotated[list[Criterion], Field(min_length=1)]


class Checklist(StrictModel):
    version: NonBlankText
    official: bool = False
    criteria: Annotated[list[Criterion], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_ids(self) -> Checklist:
        ids = [c.criterion_id for c in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion_id values must be unique")
        return self

    def ids(self) -> tuple[str, ...]:
        return tuple(c.criterion_id for c in self.criteria)

    def payload(self) -> dict[str, Any]:
        return {"version": self.version, "criteria": [c.model_dump() for c in self.criteria]}


class Limits(StrictModel):
    max_request_bytes: PositiveInt
    max_documents: PositiveInt
    max_document_bytes: PositiveInt
    max_total_pages: PositiveInt
    max_result_bytes: PositiveInt
    deadline_ms: PositiveInt
    termination_grace_ms: PositiveInt
    max_log_bytes: PositiveInt
    max_observation_bytes: PositiveInt
    max_jobs: PositiveInt
    min_free_disk_bytes: NonNegativeInt
    max_diagnostics_bytes: PositiveInt
    max_field_errors: PositiveInt
    max_stack_frames: PositiveInt


class Compute(StrictModel):
    device: Literal["cpu", "cuda:0"]
    require_cuda: bool
    allow_cpu_fallback: Literal[False]
    precision: NonBlankText
    loading_policy: Literal["cold_per_case"]
    seed: NonNegativeInt
    visible_devices: NonBlankText | None = None

    @model_validator(mode="after")
    def _explicit_device(self) -> Compute:
        if (self.device == "cuda:0") != self.require_cuda:
            raise ValueError("cuda:0 requires require_cuda=true and cpu requires false")
        return self


class ModuleConfig(StrictModel):
    package_id: Identifier
    entrypoint: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")]
    module_root: NonBlankText
    package_root: NonBlankText
    asset_root: NonBlankText
    declared_package_sha256: Sha256Hex | None = None
    example_only: bool
    settings: dict[str, Any]


EnvName = Annotated[str, Field(pattern=ENV_NAME_PATTERN)]


class WorkerConfig(StrictModel):
    python: NonBlankText | None = None
    launcher_argv: list[NonBlankText] = []
    extra_env: dict[EnvName, str] = {}

    @model_validator(mode="after")
    def _reviewed_env(self) -> WorkerConfig:
        for name in self.extra_env:
            if name in RESERVED_ENV or _SENSITIVE_ENV.search(name):
                raise ValueError("extra_env contains a reserved or credential-like variable")
        return self


class IsolationConfig(StrictModel):
    evidence_file: NonBlankText | None = None


class AuthConfig(StrictModel):
    bearer_token_file: NonBlankText | None = None


class ServerConfig(StrictModel):
    config_version: Literal["apctt-config-v1"]
    profile: Literal["dev", "evaluation"]
    bind_host: NonBlankText
    bind_port: Annotated[int, Field(ge=1, le=65535)]
    data_dir: NonBlankText
    worker_tmp_dir: NonBlankText
    schema_version: NonBlankText
    checklist_path: NonBlankText
    retry_after_seconds: PositiveInt
    limits: Limits
    compute: Compute
    module: ModuleConfig
    worker: WorkerConfig
    isolation: IsolationConfig
    auth: AuthConfig


class IsolationRecord(StrictModel):
    """Operator-attested isolation evidence. Format-checked only; not a proof of isolation."""

    record_version: Literal["apctt-isolation-v1"]
    mode: Literal["restricted_account", "vm", "container"]
    verified_at: NonBlankText
    operator: NonBlankText
    negative_tests: dict[str, Literal["passed"]]


class ConfigError(ValueError):
    """A refused configuration: fixed code plus field locations and error types (no values)."""

    def __init__(self, code: str, details: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def _details(exc: ValidationError) -> tuple[str, ...]:
    return tuple(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['type']}"
        for error in exc.errors(include_url=False, include_input=False, include_context=False)
    )


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    config: ServerConfig
    checklist: Checklist
    checklist_sha256: str
    config_sha256: str
    data_dir: Path
    worker_tmp_dir: Path
    module_root: Path
    package_root: Path
    asset_root: Path
    worker_python: Path
    isolation_evidence: Path | None
    bearer_token_file: Path | None

    @property
    def profile(self) -> str:
        return self.config.profile

    @property
    def deployment_scope(self) -> str:
        return f"{self.config.profile}:{self.config.module.package_id}"


def _read_json(path: Path, error_code: str) -> Any:
    try:
        return loads_strict(path.read_bytes())
    except (OSError, StrictJSONError):
        raise ConfigError(error_code) from None


def execution_identity(config: ServerConfig, checklist: Checklist, checklist_sha256: str) -> str:
    """config_sha256: allowed execution settings, checklist hash, profile, seed, loading policy."""
    return canonical_sha256(
        {
            "worker_protocol": WORKER_PROTOCOL_VERSION,
            "profile": config.profile,
            "schema_version": config.schema_version,
            "checklist_version": checklist.version,
            "checklist_sha256": checklist_sha256,
            "limits": config.limits.model_dump(),
            "compute": config.compute.model_dump(),
            "loading_policy": config.compute.loading_policy,
            "seed": config.compute.seed,
            "module": {
                "package_id": config.module.package_id,
                "entrypoint": config.module.entrypoint,
                "example_only": config.module.example_only,
                "settings": config.module.settings,
            },
        }
    )


def load_config(path: Path, *, profile: str = "dev") -> LoadedConfig:
    path = path.resolve()
    raw = _read_json(path, "CONFIG_UNREADABLE")
    try:
        config = ServerConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError("CONFIG_INVALID", _details(exc)) from None
    if config.profile != profile:
        raise ConfigError("PROFILE_MISMATCH")
    base = path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    checklist_path = resolve(config.checklist_path)
    checklist_raw = _read_json(checklist_path, "CHECKLIST_UNREADABLE")
    try:
        checklist = Checklist.model_validate(checklist_raw)
    except ValidationError as exc:
        raise ConfigError("CHECKLIST_INVALID", _details(exc)) from None
    checklist_sha256 = canonical_sha256(checklist_raw)
    # Never resolve symlinks here: a venv interpreter is usually a symlink to the base Python,
    # and resolving it would silently drop the worker environment's site-packages.
    if config.worker.python:
        candidate = Path(config.worker.python)
        python = candidate if candidate.is_absolute() else Path(os.path.abspath(base / candidate))
    else:
        python = Path(sys.executable)
    return LoadedConfig(
        path=path,
        config=config,
        checklist=checklist,
        checklist_sha256=checklist_sha256,
        config_sha256=execution_identity(config, checklist, checklist_sha256),
        data_dir=resolve(config.data_dir),
        worker_tmp_dir=resolve(config.worker_tmp_dir),
        module_root=resolve(config.module.module_root),
        package_root=resolve(config.module.package_root),
        asset_root=resolve(config.module.asset_root),
        worker_python=python,
        isolation_evidence=(
            resolve(config.isolation.evidence_file) if config.isolation.evidence_file else None
        ),
        bearer_token_file=(
            resolve(config.auth.bearer_token_file) if config.auth.bearer_token_file else None
        ),
    )


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _isolation_problem(path: Path | None) -> str | None:
    if path is None:
        return "EVALUATION_ISOLATION_EVIDENCE_MISSING"
    try:
        record = IsolationRecord.model_validate(_read_json(path, "X"))
    except (ConfigError, ValidationError):
        return "EVALUATION_ISOLATION_EVIDENCE_INVALID"
    if set(record.negative_tests) != set(ISOLATION_TESTS):
        return "EVALUATION_ISOLATION_EVIDENCE_INVALID"
    return None


def startup_problems(loaded: LoadedConfig) -> list[str]:
    """Fixed problem codes that must block server start. Empty list means start is allowed."""
    config = loaded.config
    problems: list[str] = []
    if not loaded.worker_python.is_file():
        problems.append("WORKER_PYTHON_NOT_FOUND")
    for root in (loaded.module_root, loaded.package_root, loaded.asset_root):
        if root.is_symlink() or not root.is_dir():
            problems.append("MODULE_PATH_NOT_DIRECTORY")
            break
    if loaded.bearer_token_file is not None and not loaded.bearer_token_file.is_file():
        problems.append("BEARER_TOKEN_FILE_MISSING")
    if config.schema_version.strip() == "":
        problems.append("SCHEMA_VERSION_MISSING")
    if config.profile == "dev":
        if not is_loopback(config.bind_host):
            problems.append("DEV_NON_LOOPBACK_BIND")
        return problems
    # evaluation: approved values only, real isolation boundary documented by the operator.
    if loaded.checklist.version == DEMO_CHECKLIST_VERSION or not loaded.checklist.official:
        problems.append("EVALUATION_CHECKLIST_NOT_OFFICIAL")
    if "DRAFT" in loaded.checklist.version.upper():
        problems.append("EVALUATION_CHECKLIST_DRAFT")
    if config.schema_version == DEV_SCHEMA_VERSION:
        problems.append("EVALUATION_DEV_SCHEMA")
    if config.module.example_only:
        problems.append("EVALUATION_EXAMPLE_MODULE")
    if config.compute.loading_policy != LOADING_POLICY:
        problems.append("EVALUATION_LOADING_POLICY")
    launcher = config.worker.launcher_argv
    if not launcher or not Path(launcher[0]).is_absolute() or not Path(launcher[0]).is_file():
        problems.append("EVALUATION_LAUNCHER_MISSING")
    isolation = _isolation_problem(loaded.isolation_evidence)
    if isolation:
        problems.append(isolation)
    if not is_loopback(config.bind_host) and loaded.bearer_token_file is None:
        problems.append("EVALUATION_NON_LOOPBACK_WITHOUT_AUTH")
    return problems
