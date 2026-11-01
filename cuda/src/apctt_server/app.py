"""SPEC-04/05/07/11/17 FastAPI application: fixed routes, strict admission, polling.

Processing order for POST /v1/assessments: auth → media type/encoding → streamed byte cap →
strict JSON → schema/semantic validation (PDF base64, size, signature, SHA-256) → bounded page
count in a separate process → request ID → readiness/capacity → durable admission → 202.
The participant module is never imported here; it runs in a separate worker process.
"""

from __future__ import annotations

import asyncio
import hmac
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from apctt_contracts import API_VERSION, CONTRACT_VERSION, KIT_VERSION, WORKER_PROTOCOL_VERSION
from apctt_contracts.cases import (
    AssessmentRequest,
    CaseLimits,
    FieldIssue,
    ProfileVersions,
    check_request,
    page_issues,
)
from apctt_contracts.config import LoadedConfig
from apctt_contracts.hashing import canonical_bytes, effective_sha256, input_sha256
from apctt_contracts.pdf import stored_case
from apctt_contracts.responses import (
    ErrorEnvelope,
    HealthLive,
    HealthReady,
    JobEnvelope,
    PackageInfo,
)
from apctt_contracts.results import CaseMeta
from apctt_contracts.strict_json import StrictJSONError, loads_strict
from apctt_diagnostics.responses import error_envelope
from apctt_server.documents import CounterFailed
from apctt_server.events import utc_now
from apctt_server.state import ServerState, build_state, run_job, shutdown, startup_recovery
from apctt_server.store import AdmissionStatus, JobRecord, NewJob

STATIC_DIR = Path(__file__).resolve().parent / "static"
_JOB_ID = re.compile(r"[0-9a-f]{32}")
_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": str(code)}
    for code in (400, 401, 403, 404, 409, 413, 415, 422, 503)
}


def _state(request: Request) -> ServerState:
    state: ServerState = request.app.state.apctt
    return state


def _trace() -> str:
    return uuid.uuid4().hex


def _error(
    state: ServerState,
    code: str,
    trace_id: str,
    *,
    started: float | None = None,
    issues: Sequence[FieldIssue] = (),
    exc: BaseException | None = None,
    parse_error: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    duration = (time.perf_counter() - started) * 1000 if started is not None else None
    status, body = error_envelope(
        code,
        trace_id=trace_id,
        api_version=API_VERSION,
        policy=state.policy,
        field_issues=issues,
        duration_ms=round(duration, 3) if duration is not None else None,
        exc=exc,
        parse_error=parse_error,
    )
    return JSONResponse(body, status_code=status, headers=headers)


def _auth_failure(request: Request, state: ServerState, trace_id: str) -> JSONResponse | None:
    if state.bearer_token is None:
        return None
    header = request.headers.get("authorization")
    if not header:
        return _error(state, "AUTH_REQUIRED", trace_id, headers={"WWW-Authenticate": "Bearer"})
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        token.strip().encode(), state.bearer_token.encode()
    ):
        return _error(state, "FORBIDDEN", trace_id)
    return None


def _json_media(value: str | None) -> bool:
    if not value:
        return False
    media, *params = [part.strip() for part in value.split(";")]
    if media.lower() != "application/json":
        return False
    for param in params:
        key, _, raw = param.partition("=")
        if key.strip().lower() != "charset" or raw.strip().strip('"').lower() not in (
            "utf-8",
            "utf8",
        ):
            return False
    return True


def _envelope(state: ServerState, job: JobRecord, trace_id: str) -> dict[str, Any]:
    finished = job.job_status == "finished"
    diagnostics: dict[str, Any]
    if finished:
        diagnostics = dict(job.diagnostics)
    else:
        diagnostics = {
            "profile": state.policy.profile,
            "phase": job.job_status,
            "accepted_trace_id": job.accepted_trace_id,
            "loading_policy": "cold_per_case",
        }
    return {
        "assessment_id": job.assessment_id,
        "request_id": job.request_id,
        "job_status": job.job_status,
        "result_origin": job.result_origin if finished else None,
        "result": job.result if finished else None,
        "meta": {
            "trace_id": trace_id,
            "api_version": API_VERSION,
            "schema_version": job.case_meta.schema_version,
            "package_id": job.package_id,
            "input_sha256": job.input_sha256,
            "config_sha256": job.config_sha256,
            "verified_package_sha256": job.verified_package_sha256,
            "accepted_at": job.accepted_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        },
        "diagnostics": diagnostics,
    }


async def _read_capped(request: Request, cap: int) -> bytes | None:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > cap:
            return None
    return bytes(body)


async def _readiness(state: ServerState) -> tuple[bool, str, str | None]:
    active = await asyncio.to_thread(state.store.active)
    active_id = active.assessment_id if active is not None else None
    if state.static_reason is not None:
        return False, state.static_reason, active_id
    if state.recovery_pending:
        return False, "RECOVERY_PENDING", active_id
    limits = state.loaded.config.limits
    count = await asyncio.to_thread(state.store.job_count)
    disk_ok = await asyncio.to_thread(state.store._disk_ok)  # noqa: SLF001
    if count >= limits.max_jobs or not disk_ok:
        return False, "STORAGE_CAPACITY", active_id
    if active is not None:
        return False, "BUSY", active_id
    return True, "READY", None


def create_app(loaded: LoadedConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = await asyncio.to_thread(build_state, loaded)
        app.state.apctt = state
        await startup_recovery(state)
        try:
            yield
        finally:
            await shutdown(state)

    app = FastAPI(
        title="APCTT Assessment API",
        version=API_VERSION,
        description=(
            "Organizer-hosted assessment server (development kit). Submit a canonical report "
            "case, then poll the job. HTTP 200 on GET means the lookup succeeded, not that the "
            "assessment succeeded."
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    if STATIC_DIR.is_dir():
        # Vendored Swagger UI assets only; without them /static/* is an ordinary 404.
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "ROUTE_NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(
            exc.status_code, "INTERNAL_ERROR"
        )
        return _error(_state(request), code, _trace())

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(_state(request), "REQUEST_VALIDATION_ERROR", _trace())

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        state = _state(request)
        trace_id = _trace()
        state.events.emit("internal_error", trace_id=trace_id, path=request.url.path)
        return _error(state, "INTERNAL_ERROR", trace_id, exc=exc)

    @app.get("/health/live", response_model=HealthLive, tags=["health"])
    async def live() -> dict[str, Any]:
        return {"live": True, "api_version": API_VERSION}

    @app.get(
        "/health/ready",
        tags=["health"],
        responses={200: {"model": HealthReady}, 503: {"model": HealthReady}},
    )
    async def ready(request: Request) -> JSONResponse:
        state = _state(request)
        denied = _auth_failure(request, state, _trace())
        if denied is not None:
            return denied
        is_ready, reason, active_id = await _readiness(state)
        body = {
            "ready": is_ready,
            "reason_code": reason,
            "loading_policy": "cold_per_case",
            "active_assessment_id": active_id,
            "compute_status": state.compute_status,
        }
        return JSONResponse(body, status_code=200 if is_ready else 503)

    @app.get(
        "/v1/package", tags=["assessments"], responses={200: {"model": PackageInfo}, **_ERRORS}
    )
    async def package(request: Request) -> JSONResponse:
        state = _state(request)
        denied = _auth_failure(request, state, _trace())
        if denied is not None:
            return denied
        config = state.loaded.config
        body = {
            "package_id": config.module.package_id,
            "profile": config.profile,
            "api_version": API_VERSION,
            "schema_version": config.schema_version,
            "checklist_version": state.loaded.checklist.version,
            "kit_version": KIT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "declared_package_sha256": state.package.declared,
            "verified_package_sha256": state.package.verified,
            "verification_status": state.package.status,
            "example_only": config.module.example_only,
            "loading_policy": config.compute.loading_policy,
            "config_sha256": state.loaded.config_sha256,
            "checklist_sha256": state.loaded.checklist_sha256,
            "compute": {**config.compute.model_dump(), "probe": state.compute_status},
        }
        return JSONResponse(body)

    @app.post(
        "/v1/assessments",
        status_code=202,
        tags=["assessments"],
        responses={202: {"model": JobEnvelope, "description": "Accepted or duplicate"}, **_ERRORS},
    )
    async def submit(request: Request) -> JSONResponse:
        started = time.perf_counter()
        trace_id = _trace()
        state = _state(request)
        config = state.loaded.config
        denied = _auth_failure(request, state, trace_id)
        if denied is not None:
            return denied
        encoding = request.headers.get("content-encoding")
        if not _json_media(request.headers.get("content-type")) or (
            encoding is not None and encoding.strip().lower() != "identity"
        ):
            return _error(state, "UNSUPPORTED_MEDIA_TYPE", trace_id, started=started)
        raw = await _read_capped(request, config.limits.max_request_bytes)
        if raw is None:
            return _error(state, "REQUEST_TOO_LARGE", trace_id, started=started)
        # Parsing, base64 decoding and hashing of up to max_request_bytes run off the event loop.
        try:
            value = await asyncio.to_thread(loads_strict, raw)
        except StrictJSONError as exc:
            parse = {"reason": exc.reason, "line": exc.line, "column": exc.column}
            return _error(state, "INVALID_JSON", trace_id, started=started, parse_error=parse)
        check = await asyncio.to_thread(
            check_request,
            value,
            limits=CaseLimits(config.limits.max_documents, config.limits.max_document_bytes),
            versions=ProfileVersions(config.schema_version, state.loaded.checklist.version),
        )
        if check.code is not None:
            return _error(state, check.code, trace_id, started=started, issues=check.issues)
        assert isinstance(value, dict) and check.request is not None
        case_value = cast(dict[str, Any], value["case"])
        digests = [document["sha256"] for document in case_value["documents"]]
        try:
            page_counts = await asyncio.to_thread(
                state.documents.inspect_and_store, check.contents, digests
            )
        except CounterFailed as exc:
            state.events.emit("pdf_counter_failed", trace_id=trace_id, cause=str(exc))
            return _error(state, "INTERNAL_ERROR", trace_id, started=started)
        unreadable = page_issues(page_counts, config.limits.max_total_pages)
        if unreadable:
            return _error(
                state, "REQUEST_VALIDATION_ERROR", trace_id, started=started, issues=unreadable
            )
        case_hash = await asyncio.to_thread(input_sha256, case_value)
        counted = [count for count in page_counts if count is not None]
        case_json = await asyncio.to_thread(
            lambda: canonical_bytes(stored_case(case_value, counted)).decode("utf-8")
        )
        new = NewJob(
            assessment_id=uuid.uuid4().hex,
            request_id=check.request.request_id,
            effective_sha256=effective_sha256(
                input_sha256=case_hash,
                config_sha256=state.loaded.config_sha256,
                verified_package_sha256=state.package.verified,
                deployment_scope=state.loaded.deployment_scope,
            ),
            input_sha256=case_hash,
            config_sha256=state.loaded.config_sha256,
            verified_package_sha256=state.package.verified,
            package_id=config.module.package_id,
            case_json=case_json,
            case_meta=CaseMeta.from_case(case_value),
            accepted_at=utc_now(),
            accepted_trace_id=trace_id,
        )
        admission = await asyncio.to_thread(state.store.admit, new, ready=state.admission_ready())
        status = admission.status
        if status is AdmissionStatus.CONFLICT:
            return _error(state, "REQUEST_ID_CONFLICT", trace_id, started=started)
        retry = {"Retry-After": str(config.retry_after_seconds)}
        if status is AdmissionStatus.BUSY:
            return _error(state, "BUSY", trace_id, started=started, headers=retry)
        if status is AdmissionStatus.NOT_READY:
            return _error(state, "NOT_READY", trace_id, started=started, headers=retry)
        if status is AdmissionStatus.STORAGE_CAPACITY:
            return _error(state, "STORAGE_CAPACITY", trace_id, started=started)
        job = admission.job
        assert job is not None
        if status is AdmissionStatus.CREATED:
            state.events.emit("admitted", assessment_id=job.assessment_id, trace_id=trace_id)
            state.spawn(run_job(state, job))
        location = f"/v1/assessments/{job.assessment_id}"
        return JSONResponse(
            _envelope(state, job, trace_id), status_code=202, headers={"Location": location}
        )

    @app.get(
        "/v1/assessments/{assessment_id}",
        tags=["assessments"],
        responses={200: {"model": JobEnvelope}, **_ERRORS},
    )
    async def get_job(request: Request, assessment_id: str) -> JSONResponse:
        trace_id = _trace()
        state = _state(request)
        denied = _auth_failure(request, state, trace_id)
        if denied is not None:
            return denied
        job = None
        if _JOB_ID.fullmatch(assessment_id):
            job = await asyncio.to_thread(state.store.get, assessment_id)
        if job is None:
            return _error(state, "JOB_NOT_FOUND", trace_id)
        body = _envelope(state, job, trace_id)
        handle = state.handles.get(job.assessment_id)
        if handle is not None and job.job_status != "finished":
            observed = await state.supervisor.observe(handle)
            if observed.observation is not None:
                body["diagnostics"]["worker_phase"] = observed.observation.phase
        return JSONResponse(body)

    @app.get("/docs", include_in_schema=False)
    async def docs() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="APCTT Assessment API",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters={"validatorUrl": None},
        )

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        request_schema = AssessmentRequest.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        nested = request_schema.pop("$defs", {})
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        components.update(nested)
        components["AssessmentRequest"] = request_schema
        schema["paths"]["/v1/assessments"]["post"]["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/AssessmentRequest"}}
            },
        }
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app
