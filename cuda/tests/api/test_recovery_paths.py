"""Regressions for slot release, terminal-write failures, counter failures and data-dir checks.

Not required acceptance nodes. Each test failed before the fix it guards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from apctt_runtime import procinfo
from apctt_server.app import create_app

from .helpers import ROOT, make_config, request, wait_finished

JSON = {"Content-Type": "application/json"}


def post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/v1/assessments", content=json.dumps(body).encode(), headers=JSON)


def ready_reason(client: TestClient) -> str:
    reason: str = client.get("/health/ready").json()["reason_code"]
    return reason


def wait_reason(client: TestClient, expected: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_reason(client) == expected:
            return
        time.sleep(0.1)
    raise AssertionError(f"readiness never became {expected}: {ready_reason(client)}")


def unconfirmed_cleanup(client: TestClient) -> None:
    """Make every finished run report that its processes could not be confirmed stopped."""
    supervisor = client.app.state.apctt.supervisor  # type: ignore[attr-defined]
    original = supervisor.await_result

    async def patched(handle: Any) -> Any:
        outcome = await original(handle)
        return dataclasses.replace(outcome, cleanup_confirmed=False)

    supervisor.await_result = patched


def test_finished_job_holding_slot_is_released_at_start(tmp_path: Path, monkeypatch) -> None:
    loaded = make_config(tmp_path)
    monkeypatch.setattr(procinfo, "recover_worker", lambda *a, **k: False)
    with TestClient(create_app(loaded)) as client:
        unconfirmed_cleanup(client)
        accepted = post(client, request("REQ-HELD"))
        wait_finished(client, accepted.headers["Location"])
        assert ready_reason(client) == "RECOVERY_PENDING"
    monkeypatch.undo()  # the worker is gone; a restart must find out and free the slot
    with TestClient(create_app(loaded)) as client:
        assert ready_reason(client) == "READY"
        second = post(client, request("REQ-AFTER-RESTART"))
        assert second.status_code == 202, second.text
        wait_finished(client, second.headers["Location"])


def test_unconfirmed_cleanup_holds_slot_until_confirmed(tmp_path: Path, monkeypatch) -> None:
    loaded = make_config(tmp_path)
    confirmed = {"value": False}
    monkeypatch.setattr(procinfo, "recover_worker", lambda *a, **k: confirmed["value"])
    with TestClient(create_app(loaded)) as client:
        unconfirmed_cleanup(client)
        accepted = post(client, request("REQ-UNCONFIRMED"))
        body = wait_finished(client, accepted.headers["Location"])
        assert body["result_origin"] == "module"
        assert ready_reason(client) == "RECOVERY_PENDING"
        assert post(client, request("REQ-BLOCKED")).json()["error"]["code"] == "NOT_READY"
        confirmed["value"] = True
        wait_reason(client, "READY")


def test_failed_terminal_writes_are_retried(tmp_path: Path) -> None:
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded)) as client:
        store = client.app.state.apctt.store  # type: ignore[attr-defined]
        real_finish = store.finish
        calls: list[str] = []

        def failing_twice(assessment_id: str, **kwargs: Any) -> None:
            calls.append(kwargs["error_code"] or "none")
            if len(calls) <= 2:
                raise OSError("simulated storage failure")
            real_finish(assessment_id, **kwargs)

        store.finish = failing_twice
        accepted = post(client, request("REQ-WRITES-FAIL"))
        body = wait_finished(client, accepted.headers["Location"], timeout=20)
        assert body["result_origin"] == "supervisor"
        assert body["result"]["error_code"] == "INTERNAL_ERROR"
        assert body["diagnostics"]["phase"] == "recovered"
        assert len(calls) == 3
        wait_reason(client, "READY")


def test_fallback_write_keeps_raw_artifact(tmp_path: Path) -> None:
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded)) as client:
        store = client.app.state.apctt.store  # type: ignore[attr-defined]
        real_finish = store.finish
        calls: list[int] = []

        def failing_once(assessment_id: str, **kwargs: Any) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise OSError("simulated storage failure")
            real_finish(assessment_id, **kwargs)

        store.finish = failing_once
        accepted = post(client, request("REQ-KEEP-RAW"))
        body = wait_finished(client, accepted.headers["Location"])
        assert body["result"]["error_code"] == "INTERNAL_ERROR"
        record = store.get(body["assessment_id"])
        raw = loaded.data_dir / "jobs" / body["assessment_id"] / "raw_output.bin"
        assert record.raw_artifact == "raw_output.bin"
        assert record.raw_output_sha256 == hashlib.sha256(raw.read_bytes()).hexdigest()


def test_pdf_counter_failure_is_a_server_error(tmp_path: Path) -> None:
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded), raise_server_exceptions=False) as client:
        state = client.app.state.apctt  # type: ignore[attr-defined]
        state.documents.python = str(tmp_path / "missing-python")
        response = post(client, request("REQ-COUNTER"))
        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
        assert state.store.job_count() == 0
    events = (loaded.data_dir / "events.jsonl").read_text()
    assert '"event": "pdf_counter_failed"' in events


def test_unusable_data_dir_is_refused(tmp_path: Path) -> None:
    config = json.loads((ROOT / "config/dev.json").read_text())
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    config.update(
        data_dir=str(blocker),
        worker_tmp_dir=str(tmp_path / "worker_tmp"),
        checklist_path=str(ROOT / "contracts/checklist.official.json"),
    )
    config["module"].update(
        module_root=str(ROOT),
        package_root=str(ROOT / "participant_example"),
        asset_root=str(ROOT / "participant_example"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    proc = subprocess.run(
        [sys.executable, "-m", "apctt_server", "--config", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "APCTT_START_REFUSED DATA_DIR_UNUSABLE" in proc.stderr
    assert "Traceback" not in proc.stderr
