"""Synchronous single-run launcher for protocol tests (the async supervisor is RUN-02)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from apctt_contracts.config import ChecklistPayload, Compute
from apctt_contracts.ipc import RunLimits, RunOutcome, RunSpec
from apctt_contracts.pdf import stored_case
from apctt_contracts.results import CaseMeta
from apctt_runtime import protocol

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).resolve().parent / "fakes"
# Protocol tests run a two-criterion DEMO checklist to keep results short; the example request
# (docs/EXAMPLE_REQUEST.json) is for the official checklist of config/dev.json (decision D74), so
# its case gets the DEMO version here. OFFICIAL is the checklist the shipped CPU example runs.
CHECKLIST = ChecklistPayload.model_validate(
    {
        "version": "DEMO-CHECKLIST-NOT-OFFICIAL",
        "criteria": [
            {"criterion_id": "DEMO-C01", "description": "d1"},
            {"criterion_id": "DEMO-C02", "description": "d2"},
        ],
    }
)
_OFFICIAL = json.loads((ROOT / "contracts/checklist.official.json").read_text())
OFFICIAL = ChecklistPayload.model_validate(
    {"version": _OFFICIAL["version"], "criteria": _OFFICIAL["criteria"]}
)
EXAMPLE_CASE: dict[str, Any] = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())["case"]
REQUEST_CASE: dict[str, Any] = {**EXAMPLE_CASE, "checklist_version": CHECKLIST.version}
EXAMPLE_PDF = ROOT / "docs/EXAMPLE_REPORT.pdf"
# The stored view the supervisor receives: page counts, no base64 payload.
CASE: dict[str, Any] = stored_case(REQUEST_CASE, [2])
CPU = Compute.model_validate(
    {
        "device": "cpu",
        "require_cuda": False,
        "allow_cpu_fallback": False,
        "precision": "float32",
        "loading_policy": "cold_per_case",
        "seed": 42,
    }
)


def make_spec(
    tmp_path: Path,
    entrypoint: str,
    *,
    module_root: Path = FAKES,
    settings: dict[str, Any] | None = None,
    deadline_s: float = 60.0,
    max_result_bytes: int = 65536,
    max_log_bytes: int = 65536,
    case: dict[str, Any] | None = None,
    extra_env: dict[str, str] | None = None,
    launcher: tuple[str, ...] = (),
    document_files: tuple[Path, ...] | None = None,
    checklist: ChecklistPayload | None = None,
) -> RunSpec:
    run_id = uuid.uuid4().hex
    artifact_dir = tmp_path / "artifacts" / run_id
    artifact_dir.mkdir(parents=True)
    value = case or CASE
    if document_files is None:  # every document of the case is the example PDF by default
        document_files = tuple(EXAMPLE_PDF for _ in value["documents"])
    return RunSpec(
        assessment_id=uuid.uuid4().hex,
        run_id=run_id,
        case=value,
        case_meta=CaseMeta.from_case(value),
        checklist=checklist or CHECKLIST,
        module={
            "package_id": "fake",
            "entrypoint": entrypoint,
            "module_root": str(module_root),
            "asset_root": str(module_root),
            "settings": settings or {},
        },
        compute=CPU,
        worker_python=Path(sys.executable),
        launcher_argv=launcher,
        extra_env=extra_env or {},
        scratch_root=tmp_path / "worker_tmp",
        artifact_dir=artifact_dir,
        deadline_monotonic=time.monotonic() + deadline_s,
        limits=RunLimits(
            max_result_bytes=max_result_bytes,
            max_log_bytes=max_log_bytes,
            max_observation_bytes=32768,
            termination_grace_ms=500,
        ),
        document_files=document_files,
    )


def run_sync(spec: RunSpec, *, keep_scratch: bool = False) -> RunOutcome:
    started = time.monotonic()
    scratch = protocol.prepare_scratch(spec.scratch_root, spec.run_id)
    try:
        argv = protocol.worker_argv(spec)
        env = protocol.worker_env(spec, scratch.path)
        documents = protocol.stage_documents(scratch.path, spec.document_files)
        payload = protocol.build_envelope(
            spec, scratch.path, protocol.time_budget_ms(spec), documents
        )
        proc = subprocess.run(
            argv,
            input=payload,
            capture_output=True,
            env=env,
            cwd=scratch.path,
            timeout=60,
            start_new_session=True,
            check=False,
        )
        stdout = protocol.StreamSink(spec.artifact_dir / "stdout.log", spec.limits.max_log_bytes)
        stderr = protocol.StreamSink(spec.artifact_dir / "stderr.log", spec.limits.max_log_bytes)
        stdout.write(proc.stdout)
        stderr.write(proc.stderr)
        stdout.close()
        stderr.close()
        code = proc.returncode
        return protocol.collect_outcome(
            spec,
            scratch,
            exit_code=code if code >= 0 else None,
            exit_signal=-code if code < 0 else None,
            timed_out=False,
            cleanup_confirmed=True,
            total_ms=(time.monotonic() - started) * 1000,
            stdout=stdout.summary(),
            stderr=stderr.summary(),
        )
    finally:
        scratch.close(remove=not keep_scratch)


def stdout_json(spec: RunSpec) -> dict[str, Any]:
    lines = (spec.artifact_dir / "stdout.log").read_text().strip().splitlines()
    value: dict[str, Any] = json.loads(lines[-1])
    return value
