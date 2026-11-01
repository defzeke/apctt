"""SPEC-09/17 worker protocol with real worker processes: AC-11, AC-17, AC-36, AC-37, AC-39."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from apctt_contracts.config import Checklist
from apctt_contracts.hashing import sha256_hex
from apctt_contracts.ipc import WorkerContext, WorkerEnvelope
from apctt_contracts.pdf import stored_case
from apctt_contracts.results import validate_result
from apctt_runtime import protocol

from .helpers import (
    CASE,
    CHECKLIST,
    EXAMPLE_CASE,
    OFFICIAL,
    ROOT,
    make_spec,
    run_sync,
    stdout_json,
)


def _raw(spec_dir: Path) -> bytes:
    return (spec_dir / "raw_output.bin").read_bytes()


def test_ac11_invalid_output_and_raw(tmp_path: Path) -> None:
    # Valid output: module origin, result preserved exactly, raw hash of the file bytes.
    spec = make_spec(tmp_path, "behaviours:Valid")
    ok = run_sync(spec)
    assert ok.result_origin == "module" and ok.error_code is None
    assert ok.result["assessment_status"] == "abstained"
    raw = _raw(spec.artifact_dir)
    assert ok.raw_output_sha256 == sha256_hex(raw) and json.loads(raw) == ok.result
    assert ok.exit_code == 0 and ok.raw_artifact == "raw_output.bin"

    expectations = {
        "behaviours:Crash": ("WORKER_CRASH", False),
        "behaviours:Raises": ("MODULE_ERROR", False),
        "behaviours:LoadRaises": ("MODULE_ERROR", False),
        "behaviours:CudaMissing": ("CUDA_UNAVAILABLE", False),
        "behaviours:InvalidUtf8": ("INVALID_OUTPUT", True),
        "behaviours:PickleOutput": ("INVALID_OUTPUT", True),
        "behaviours:ContractViolation": ("INVALID_OUTPUT", True),
        "behaviours:NonSerializable": ("INVALID_OUTPUT", False),
        "behaviours:NoResult": ("WORKER_PROTOCOL_ERROR", False),
        "behaviours:SymlinkResult": ("WORKER_PROTOCOL_ERROR", False),
        "behaviours:DirectoryResult": ("WORKER_PROTOCOL_ERROR", False),
        "behaviours:CleanupFails": ("CLEANUP_ERROR", True),
    }
    for entrypoint, (code, has_raw) in expectations.items():
        spec = make_spec(tmp_path, entrypoint)
        outcome = run_sync(spec)
        assert (outcome.result_origin, outcome.error_code) == ("supervisor", code), entrypoint
        failure = outcome.result
        assert failure["assessment_status"] == "failed" and failure["error_code"] == code
        assert failure["case_id"] == CASE["case_id"]
        assert (
            validate_result(
                failure, case=CASE, checklist=Checklist.model_validate(CHECKLIST.model_dump())
            )
            == []
        )
        raw_path = spec.artifact_dir / "raw_output.bin"
        assert raw_path.exists() == has_raw, entrypoint
        if has_raw:
            assert outcome.raw_output_sha256 == sha256_hex(raw_path.read_bytes())
    # Invalid UTF-8 bytes are preserved byte-for-byte.
    spec = make_spec(tmp_path, "behaviours:InvalidUtf8")
    run_sync(spec)
    assert _raw(spec.artifact_dir) == b'{"a":"\xff\xfe"}'
    # Oversized result: protocol error, bounded raw prefix preserved and flagged.
    spec = make_spec(tmp_path, "behaviours:Oversize", max_result_bytes=4096)
    big = run_sync(spec)
    assert big.error_code == "WORKER_PROTOCOL_ERROR" and "RESULT_OVERSIZE" in big.warning_codes
    assert len(_raw(spec.artifact_dir)) == 4096 and "RAW_TRUNCATED" in big.warning_codes
    # Exception text from the module stays in the local stderr artifact, not in the outcome.
    spec = make_spec(tmp_path, "behaviours:Raises")
    raised = run_sync(spec)
    assert "SENTINEL" not in json.dumps(raised.result)
    assert "SENTINEL-module-exception" in (spec.artifact_dir / "stderr.log").read_text()


def test_ac17_process_import_and_environment_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APCTT_TEST_SECRET_TOKEN", "SENTINEL-parent-env")
    monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
    spec = make_spec(tmp_path, "behaviours:EnvDump", extra_env={"OMP_NUM_THREADS": "1"})
    argv = protocol.worker_argv(spec)
    assert argv[0] == sys.executable and Path(argv[0]).is_absolute()
    assert argv[1:] == ["-s", str(protocol.WORKER_SCRIPT), "--run-id", spec.run_id]
    outcome = run_sync(spec)
    assert outcome.result_origin == "module"
    seen = stdout_json(spec)
    env = seen["env"]
    assert "APCTT_TEST_SECRET_TOKEN" not in env and "PYTHONPATH" not in env
    assert "SENTINEL-parent-env" not in json.dumps(env)
    allowed = protocol.BASE_ENV_KEYS | {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}
    extra = set(env) - allowed
    # macOS may inject __CF_USER_TEXT_ENCODING into every process; nothing else is allowed.
    assert extra <= {"__CF_USER_TEXT_ENCODING"}, extra
    scratch = Path(seen["cwd"]).resolve()
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME"):
        assert Path(env[key]).resolve().is_relative_to(scratch), key
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # cpu profile hides GPUs from the worker
    assert env["PYTHONHASHSEED"] == "42"
    # The wrapper directory is not importable as the participant's sys.path[0].
    assert Path(seen["sys_path0"]) == spec.module["module_root"] or seen["sys_path0"] == str(
        spec.module["module_root"]
    )
    # The API-side process never imported the participant module.
    assert "behaviours" not in sys.modules and "participant_example.adapter" not in sys.modules
    # Scratch is removed after the run; logs stay in the artifact directory.
    assert not scratch.exists()


def test_ac36_observation_sidecar(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, "behaviours:Valid")
    outcome = run_sync(spec)
    obs = outcome.observation
    assert obs is not None and obs.phase == "finished" and obs.source == "worker_observed"
    for name in ("load", "assess", "cleanup"):
        value = getattr(obs.timings_ms, name)
        assert isinstance(value, float) and value >= 0, name
    assert obs.compute.device == "cpu" and obs.compute.torch_version is None
    assert obs.unavailable_reasons.get("peak_memory") == "NOT_MEASURED"
    assert "SIDECAR_INVALID" not in outcome.warning_codes and outcome.total_ms > 0

    # Crafted scratch states: the sidecar never overrides a valid module result.
    good = json.dumps(protocol_valid_result()).encode()
    cases = {
        "symlink": lambda d: os.symlink(d / "home", d / "observation.json"),
        "oversize": lambda d: (d / "observation.json").write_bytes(b" " * 40_000),
        "bad_json": lambda d: (d / "observation.json").write_bytes(b"{not json"),
        "wrong_run": lambda d: (d / "observation.json").write_text(
            json.dumps(dict(obs.model_dump(), run_id="other"))
        ),
        "missing": lambda d: None,
    }
    for name, craft in cases.items():
        spec = make_spec(tmp_path, "behaviours:Valid")
        scratch = protocol.prepare_scratch(spec.scratch_root, spec.run_id)
        try:
            (scratch.path / "result.json").write_bytes(good)
            craft(scratch.path)
            crafted = protocol.collect_outcome(
                spec,
                scratch,
                exit_code=0,
                exit_signal=None,
                timed_out=False,
                cleanup_confirmed=True,
                total_ms=1.0,
                stdout=protocol.StreamSummary(0, 0, False),
                stderr=protocol.StreamSummary(0, 0, False),
            )
        finally:
            scratch.close(remove=True)
        assert crafted.result_origin == "module", name
        expected = "SIDECAR_MISSING" if name == "missing" else "SIDECAR_INVALID"
        assert expected in crafted.warning_codes and crafted.observation is None, name


def protocol_valid_result() -> dict:
    from .fakes.behaviours import valid_result

    return valid_result(CASE, {"checklist": CHECKLIST.model_dump()})


def test_ac37_inconsistent_evidence_is_supervisor_failure(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, "behaviours:MetWithoutEvidence")
    outcome = run_sync(spec)
    assert outcome.result_origin == "supervisor" and outcome.error_code == "INVALID_OUTPUT"
    assert outcome.result["assessment_status"] == "failed"
    rules = {(i.path, i.rule) for i in outcome.output_issues}
    assert ("result.evidence_reference.DEMO-C01", "evidence_required_for_met") in rules
    raw = _raw(spec.artifact_dir)
    assert outcome.raw_output_sha256 == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["evidence_reference"] == {"DEMO-C01": [], "DEMO-C02": []}


def test_ac39_worker_context_time_budget(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, "behaviours:ContextDump", deadline_s=30)
    budget = protocol.time_budget_ms(spec)
    assert isinstance(budget, int) and 0 < budget <= 30_000
    envelope = WorkerEnvelope.model_validate_json(protocol.build_envelope(spec, tmp_path, budget))
    assert set(envelope.context.model_dump()) == {"run_id", "seed", "time_budget_ms", "scratch_dir"}
    assert set(WorkerContext.model_fields) == {"run_id", "seed", "time_budget_ms", "scratch_dir"}
    outcome = run_sync(spec)
    assert outcome.result_origin == "module"
    context = stdout_json(spec)["context"]
    assert set(context) == {"run_id", "seed", "time_budget_ms", "scratch_dir"}
    assert context["run_id"] == spec.run_id and context["seed"] == 42
    assert isinstance(context["time_budget_ms"], int) and 0 < context["time_budget_ms"] <= 30_000
    # An already expired deadline still yields a positive minimal budget; the watchdog decides.
    expired = make_spec(tmp_path, "behaviours:Valid", deadline_s=-5)
    assert protocol.time_budget_ms(expired) == 1


def test_cpu_example_module_output(tmp_path: Path) -> None:
    """Not a required node: the shipped CPU example, as config/dev.json runs it (the official
    checklist, decision D74), returns a contract-valid DEMO result for the example request."""
    settings = json.loads((ROOT / "config/dev.json").read_text())["module"]["settings"]
    spec = make_spec(
        tmp_path,
        "participant_example.adapter:AssessmentModule",
        module_root=ROOT,
        settings=settings,
        case=stored_case(EXAMPLE_CASE, [2]),
        checklist=OFFICIAL,
    )
    outcome = run_sync(spec)
    assert outcome.result_origin == "module", outcome.output_issues
    result = outcome.result
    assert result["assessment_status"] == "abstained"
    assert result["checklist_version"] == "APCTT-2026-CHECKLIST-v1"
    ids = [f"APCTT-C{n:02d}" for n in range(1, 13)]
    assert result["criterion_status"] == {
        cid: "met" if cid == "APCTT-C04" else "insufficient evidence" for cid in ids
    }
    evidence = result["evidence_reference"]["APCTT-C04"][0]
    assert evidence == {"document_id": "D001", "page": 1, "supporting_span": "tested"}
    assert "no page contains" in result["missing_evidence"]["APCTT-C10"]
    assert "no phrase is configured" in result["missing_evidence"]["APCTT-C01"]
