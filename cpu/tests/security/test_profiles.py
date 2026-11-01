"""Evaluation start gates: AC-21. The approved fixture below is a TEST record, not real evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from tests.integration.server import FAKES, ROOT, Server, base_config, free_port, request

TEST_TESTS = {
    "job_store_access": "passed",
    "previous_output_access": "passed",
    "credential_access": "passed",
    "network_egress": "passed",
    "descendant_termination": "passed",
}


def approved_evaluation_config(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """All gates satisfied with test fixtures. Real deployments need operator evidence."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    checklist = tmp_path / "checklist.official-test.json"
    checklist.write_text(
        json.dumps(
            {
                "version": "TEST-OFFICIAL-CHECKLIST-1",
                "official": True,
                "criteria": [{"criterion_id": "T-C01", "description": "Test criterion only."}],
            }
        )
    )
    evidence = tmp_path / "isolation.test-record.json"
    evidence.write_text(
        json.dumps(
            {
                "record_version": "apctt-isolation-v1",
                "mode": "container",
                "verified_at": "2026-09-29T00:00:00Z",
                "operator": "QA TEST FIXTURE - NOT REAL EVIDENCE",
                "negative_tests": TEST_TESTS,
            }
        )
    )
    counter = tmp_path / "count"
    counter.mkdir(exist_ok=True)
    config = base_config(tmp_path, fake="Counting", settings={"counter_dir": str(counter)})
    config.update(
        profile="evaluation",
        schema_version="apctt-test-eval-schema",
        checklist_path=str(checklist),
    )
    config["module"]["example_only"] = False
    config["worker"]["launcher_argv"] = ["/usr/bin/env"]
    config["isolation"]["evidence_file"] = str(evidence)
    return config, evidence


def refused(tmp_path: Path, config: dict[str, Any], profile: str = "evaluation") -> str:
    path = tmp_path / f"cfg-{free_port()}.json"
    path.write_text(json.dumps(config))
    proc = subprocess.run(
        [sys.executable, "-m", "apctt_server", "--profile", profile, "--config", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    return proc.stderr


def test_ac21_unapproved_evaluation_rejected(tmp_path: Path) -> None:
    approved, evidence = approved_evaluation_config(tmp_path)

    def variant(**changes: Any) -> dict[str, Any]:
        config = json.loads(json.dumps(approved))
        for dotted, value in changes.items():
            target = config
            keys = dotted.split("__")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        return config

    demo = variant(checklist_path=str(ROOT / "contracts/checklist.demo.json"))
    assert "EVALUATION_CHECKLIST_NOT_OFFICIAL" in refused(tmp_path, demo)
    draft = tmp_path / "checklist.draft.json"
    draft_value = json.loads(Path(approved["checklist_path"]).read_text())
    draft_value["version"] = "APCTT-2026-CHECKLIST-v1-DRAFT"
    draft.write_text(json.dumps(draft_value))
    assert "EVALUATION_CHECKLIST_DRAFT" in refused(tmp_path, variant(checklist_path=str(draft)))
    assert "EVALUATION_DEV_SCHEMA" in refused(
        tmp_path, variant(schema_version="apctt-pdf-example-v1")
    )
    assert "EVALUATION_EXAMPLE_MODULE" in refused(tmp_path, variant(module__example_only=True))
    assert "EVALUATION_LAUNCHER_MISSING" in refused(tmp_path, variant(worker__launcher_argv=[]))
    assert "EVALUATION_ISOLATION_EVIDENCE_MISSING" in refused(
        tmp_path, variant(isolation__evidence_file=None)
    )
    partial = tmp_path / "isolation.partial.json"
    record = json.loads(evidence.read_text())
    record["negative_tests"].pop("network_egress")
    partial.write_text(json.dumps(record))
    assert "EVALUATION_ISOLATION_EVIDENCE_INVALID" in refused(
        tmp_path, variant(isolation__evidence_file=str(partial))
    )
    failed = tmp_path / "isolation.failed.json"
    record = json.loads(evidence.read_text())
    record["negative_tests"]["network_egress"] = "failed"
    failed.write_text(json.dumps(record))
    assert "EVALUATION_ISOLATION_EVIDENCE_INVALID" in refused(
        tmp_path, variant(isolation__evidence_file=str(failed))
    )
    assert "EVALUATION_NON_LOOPBACK_WITHOUT_AUTH" in refused(tmp_path, variant(bind_host="0.0.0.0"))
    dev_public = base_config(tmp_path)
    dev_public["bind_host"] = "0.0.0.0"
    assert "DEV_NON_LOOPBACK_BIND" in refused(tmp_path, dev_public, profile="dev")
    assert "PROFILE_MISMATCH" in refused(tmp_path, base_config(tmp_path), profile="evaluation")

    # Fully gated evaluation start: DEMO cases are refused; bearer auth is enforced when configured.
    token = tmp_path / "token.txt"
    token.write_text("test-token-value\n")
    token.chmod(0o600)
    authed = variant(auth__bearer_token_file=str(token), bind_port=free_port())
    with Server(tmp_path, authed, name="eval", profile="evaluation") as server:
        assert httpx.get(server.url + "/health/live").status_code == 200
        assert httpx.get(server.url + "/health/ready").status_code == 401
        wrong = httpx.get(server.url + "/v1/package", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 403 and "test-token-value" not in wrong.text
        headers = {"Authorization": "Bearer test-token-value"}
        package = httpx.get(server.url + "/v1/package", headers=headers).json()
        assert package["profile"] == "evaluation" and package["example_only"] is False
        demo_case = httpx.post(
            server.url + "/v1/assessments", json=request("REQ-DEMO"), headers=headers
        )
        assert (
            demo_case.status_code == 422 and demo_case.json()["error"]["code"] == "VERSION_MISMATCH"
        )
    assert FAKES.exists()
