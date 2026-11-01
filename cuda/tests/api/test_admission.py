"""SPEC-04/05/07/08 HTTP admission and polling with real workers.

AC-01, AC-05, AC-07, AC-09, AC-13.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
from fastapi.testclient import TestClient

from apctt_contracts.responses import ErrorEnvelope, HealthReady, JobEnvelope, PackageInfo
from apctt_server.app import create_app

from .helpers import EXAMPLE_REQUEST, executions, make_config, request, wait_finished

JSON = {"Content-Type": "application/json"}
SENTINEL = "SENTINEL-5e1d-request-value"


def post(client: TestClient, body: Any, headers: dict[str, str] | None = None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return client.post("/v1/assessments", content=data, headers=headers or JSON)


def test_ac01_real_worker_submit_poll(tmp_path: Path) -> None:
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded)) as client:
        assert client.get("/health/live").json() == {
            "live": True,
            "api_version": "apctt-api-v0.2-draft",
        }
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        HealthReady.model_validate(ready.json())
        assert ready.json()["reason_code"] == "READY"
        package = PackageInfo.model_validate(client.get("/v1/package").json())
        assert package.verification_status == "verified" and package.example_only
        assert package.verified_package_sha256 and len(package.verified_package_sha256) == 64
        assert package.config_sha256 == loaded.config_sha256

        response = post(client, EXAMPLE_REQUEST)
        assert response.status_code == 202, response.text
        accepted = JobEnvelope.model_validate(response.json())
        location = response.headers["Location"]
        assert location == f"/v1/assessments/{accepted.assessment_id}"
        assert accepted.job_status in ("queued", "running") and accepted.result is None

        body = wait_finished(client, location)
        job = JobEnvelope.model_validate(body)
        assert job.result_origin == "module" and job.result is not None
        result = body["result"]
        assert result["assessment_status"] == "abstained"
        # config/dev.json runs the official checklist (D74); the example finds "tested" (C04).
        assert result["checklist_version"] == "APCTT-2026-CHECKLIST-v1"
        assert result["criterion_status"]["APCTT-C04"] == "met"
        assert set(result["criterion_status"]) == {f"APCTT-C{n:02d}" for n in range(1, 13)}
        assert sum(v == "met" for v in result["criterion_status"].values()) == 1
        example = json.loads(
            (Path(__file__).resolve().parents[2] / "docs/EXAMPLE_RESPONSE.json").read_text()
        )
        for key in ("criterion_status", "evidence_reference", "missing_evidence"):
            assert result[key] == example["result"][key], key
        assert result["assessment_target"] == EXAMPLE_REQUEST["case"]["assessment_target"]
        expected_hash = json.loads(
            (Path(__file__).resolve().parents[2] / "docs/EXAMPLE_RESPONSE.json").read_text()
        )["meta"]["input_sha256"]
        assert job.meta.input_sha256 == expected_hash
        assert job.meta.verified_package_sha256 == package.verified_package_sha256
        assert job.meta.started_at and job.meta.finished_at
        assert job.meta.trace_id != accepted.meta.trace_id  # trace is per HTTP request
        diag = body["diagnostics"]
        assert diag["phase"] == "finished" and diag["timings_ms"]["total"] > 0
        for name in ("load", "assess", "cleanup"):
            assert isinstance(diag["timings_ms"][name], float)
        assert diag["exit_code"] == 0 and diag["raw_output_bytes"] > 0
        # The API process never imports the participant module.
        import sys

        assert "participant_example.adapter" not in sys.modules
        # OpenAPI and offline docs routes exist; redoc is disabled.
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/redoc").status_code == 404


def test_ac05_request_errors_and_stream_cap(tmp_path: Path) -> None:
    loaded = make_config(tmp_path, limits={"max_request_bytes": 8192})
    with TestClient(create_app(loaded)) as client:

        def error(response, status: int, code: str) -> dict[str, Any]:
            assert response.status_code == status, response.text
            body: dict[str, Any] = response.json()
            ErrorEnvelope.model_validate(body)
            assert body["error"]["code"] == code
            assert SENTINEL not in response.text
            return body

        error(post(client, b"{not json"), 400, "INVALID_JSON")
        dup = error(post(client, b'{"request_id":"a","request_id":"b"}'), 400, "INVALID_JSON")
        assert dup["diagnostics"]["parse_error"]["reason"] == "duplicate_key"
        error(post(client, b'{"request_id":NaN}'), 400, "INVALID_JSON")
        error(post(client, b'{"a":"\xff"}'), 400, "INVALID_JSON")

        error(
            post(client, EXAMPLE_REQUEST, {"Content-Type": "text/plain"}),
            415,
            "UNSUPPORTED_MEDIA_TYPE",
        )
        error(
            post(
                client,
                EXAMPLE_REQUEST,
                {"Content-Type": "application/json", "Content-Encoding": "gzip"},
            ),
            415,
            "UNSUPPORTED_MEDIA_TYPE",
        )
        error(
            post(client, EXAMPLE_REQUEST, {"Content-Type": "application/json; charset=latin-1"}),
            415,
            "UNSUPPORTED_MEDIA_TYPE",
        )
        bad = request(title="")
        bad["case"][SENTINEL] = SENTINEL
        body = error(
            post(client, bad, {"Content-Type": "Application/JSON; charset=UTF-8"}),
            422,
            "REQUEST_VALIDATION_ERROR",
        )
        paths = {e["path"] for e in body["error"]["field_errors"]}
        assert paths == {"case.title", "case.<unknown-field>"}
        error(post(client, request(schema_version="apctt-v04")), 422, "VERSION_MISMATCH")
        mismatch = post(client, request(checklist_version="OFFICIAL-1")).json()
        assert mismatch["error"]["field_errors"][0]["expected"] == "APCTT-2026-CHECKLIST-v1"
        error(post(client, request(input_profile="raw_pdf")), 422, "UNSUPPORTED_INPUT_PROFILE")

        # Size cap with an honest Content-Length and with a chunked body that has none.
        big = request()
        big["case"]["description"] = "x" * 9000
        error(post(client, big), 413, "REQUEST_TOO_LARGE")

        def chunks():
            yield b'{"request_id":"REQ-CHUNK","case":{"title":"'
            for _ in range(20):
                yield b"y" * 1024
            yield b'"}}'

        chunked = client.post("/v1/assessments", content=chunks(), headers=JSON)
        error(chunked, 413, "REQUEST_TOO_LARGE")

        error(client.get("/v1/assessments/" + "0" * 32), 404, "JOB_NOT_FOUND")
        error(client.get("/v1/assessments/../../etc/passwd"), 404, "ROUTE_NOT_FOUND")
        error(client.get("/nope"), 404, "ROUTE_NOT_FOUND")
        error(client.delete("/v1/assessments"), 405, "METHOD_NOT_ALLOWED")
        # None of the rejected requests created a job.
        assert client.app.state.apctt.store.job_count() == 0  # type: ignore[attr-defined]


def test_ac07_id_conflict(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    loaded = make_config(
        tmp_path, fake=True, settings={"counter_dir": str(counter), "sleep_s": 1.0}
    )
    with TestClient(create_app(loaded)) as client:
        first = post(client, request("REQ-A"))
        assert first.status_code == 202
        # Same ID, different effective input → 409 even while the slot is busy.
        conflict = post(client, request("REQ-A", title="Another title"))
        assert (
            conflict.status_code == 409
            and conflict.json()["error"]["code"] == "REQUEST_ID_CONFLICT"
        )
        same = post(client, request("REQ-A"))
        assert same.status_code == 202 and same.headers["Location"] == first.headers["Location"]
        wait_finished(client, first.headers["Location"])
        again = post(client, request("REQ-A"))
        assert again.status_code == 202 and again.json()["job_status"] == "finished"
        assert post(client, request("REQ-A", title="x")).status_code == 409
        assert executions(counter) == 1


def test_ac09_busy_does_not_block_get(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    loaded = make_config(
        tmp_path, fake=True, settings={"counter_dir": str(counter), "sleep_s": 1.5}
    )
    with TestClient(create_app(loaded)) as client:
        first = post(client, request("REQ-A"))
        assert first.status_code == 202
        busy = post(client, request("REQ-B"))
        assert busy.status_code == 503 and busy.json()["error"]["code"] == "BUSY"
        assert busy.headers["Retry-After"] == "1"
        assert busy.json()["error"]["retryable"] is True
        polled = client.get(first.headers["Location"])
        assert polled.status_code == 200 and polled.json()["job_status"] in ("queued", "running")
        ready = client.get("/health/ready")
        assert ready.status_code == 503 and ready.json()["reason_code"] == "BUSY"
        assert ready.json()["active_assessment_id"] == first.json()["assessment_id"]
        assert post(client, request("REQ-A")).status_code == 202  # duplicate allowed while busy
        wait_finished(client, first.headers["Location"])
        second = post(client, request("REQ-B"))  # the rejected ID can be retried later
        assert second.status_code == 202
        wait_finished(client, second.headers["Location"])
        assert executions(counter) == 2
        assert client.get("/health/ready").json()["reason_code"] == "READY"


def test_ac13_disconnect_does_not_cancel(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    loaded = make_config(
        tmp_path, fake=True, settings={"counter_dir": str(counter), "sleep_s": 0.5}
    )
    app = create_app(loaded)
    payload = json.dumps(request("REQ-GONE")).encode()

    async def submit_then_disconnect() -> list[str]:
        sent: list[str] = []
        messages = [
            {"type": "http.request", "body": payload, "more_body": False},
            {"type": "http.disconnect"},
        ]

        async def receive() -> dict[str, Any]:
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message["type"])
            raise OSError("client connection lost")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/assessments",
            "raw_path": b"/v1/assessments",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
            "client": ("127.0.0.1", 5000),
            "server": ("127.0.0.1", 8000),
        }
        try:
            await app(scope, receive, send)
        except OSError:
            pass
        return sent

    with TestClient(app) as client:
        assert client.portal is not None
        sent = client.portal.call(submit_then_disconnect)
        assert sent == ["http.response.start"]
        # The admitted job keeps running; the client finds it again with the same request.
        again = post(client, request("REQ-GONE"))
        assert again.status_code == 202
        body = wait_finished(client, again.headers["Location"])
        assert body["result_origin"] == "module"
        assert executions(counter) == 1
    assert anyio  # imported for the portal backend


def test_static_route_without_assets_is_404(tmp_path: Path, monkeypatch) -> None:
    """Regression (found by QA-01 AC-18): a missing static directory must not produce 500."""
    import apctt_server.app as app_module

    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path / "no-static")
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded), raise_server_exceptions=False) as client:
        response = client.get("/static/swagger-ui/swagger-ui-bundle.js")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "ROUTE_NOT_FOUND"


def test_failed_terminal_write_becomes_supervisor_failure(tmp_path: Path) -> None:
    """Regression: a failing terminal write must not leave the job running and the slot held."""
    loaded = make_config(tmp_path)
    with TestClient(create_app(loaded)) as client:
        store = client.app.state.apctt.store  # type: ignore[attr-defined]
        real_finish = store.finish
        calls: list[str] = []

        def failing_once(assessment_id: str, **kwargs: Any) -> None:
            calls.append(kwargs["origin"])
            if len(calls) == 1:
                raise OSError("simulated storage failure")
            real_finish(assessment_id, **kwargs)

        store.finish = failing_once
        accepted = post(client, request("REQ-FINISH-FAIL"))
        assert accepted.status_code == 202
        body = wait_finished(client, accepted.headers["Location"])
        assert calls == ["module", "supervisor"]
        assert body["result_origin"] == "supervisor"
        assert body["result"]["error_code"] == "INTERNAL_ERROR"
        assert client.get("/health/ready").json()["reason_code"] == "READY"


def test_timeout_reports_model_cannot_process(tmp_path: Path) -> None:
    """A case over the time limit fails with TIMEOUT and a fixed, readable failure statement."""
    counter = tmp_path / "count"
    counter.mkdir()
    loaded = make_config(
        tmp_path,
        fake=True,
        settings={"counter_dir": str(counter), "sleep_s": 60},
        limits={"deadline_ms": 1500, "termination_grace_ms": 200},
    )
    with TestClient(create_app(loaded)) as client:
        accepted = post(client, request("REQ-SLOW"))
        assert accepted.status_code == 202
        body = wait_finished(client, accepted.headers["Location"])
        assert body["result_origin"] == "supervisor"
        assert body["result"]["assessment_status"] == "failed"
        assert body["result"]["error_code"] == "TIMEOUT"
        assert body["diagnostics"]["timed_out"] is True
        assert body["diagnostics"]["failure"] == {
            "code": "TIMEOUT",
            "message": "The model could not process the documents within the 1.5-second limit.",
            "limit_ms": 1500,
        }
    # A normal module result carries no failure block.
    (tmp_path / "fast").mkdir()
    fast = make_config(tmp_path / "fast")
    with TestClient(create_app(fast)) as client:
        ok = wait_finished(client, post(client, request("REQ-FAST")).headers["Location"])
        assert ok["result_origin"] == "module" and "failure" not in ok["diagnostics"]


def test_pdf_admission_checks_and_storage(tmp_path: Path) -> None:
    """PDFs are page-counted in a separate process, stored once by hash, never kept as base64."""
    import base64
    import hashlib
    import sqlite3
    import stat

    loaded = make_config(tmp_path, limits={"max_total_pages": 2})
    with TestClient(create_app(loaded)) as client:
        damaged = request("REQ-DAMAGED")
        data = b"%PDF-1.4\nthis is not a real PDF body\n%%EOF\n"
        damaged["case"]["documents"][0].update(
            content_base64=base64.b64encode(data).decode(), sha256=hashlib.sha256(data).hexdigest()
        )
        response = post(client, damaged)
        assert response.status_code == 422
        assert [(e["path"], e["rule"]) for e in response.json()["error"]["field_errors"]] == [
            ("case.documents[0].content_base64", "pdf_readable")
        ]

        two = request("REQ-TWO-DOCS")
        second = dict(two["case"]["documents"][0], document_id="D002")
        two["case"]["documents"].append(second)  # 2 + 2 pages > max_total_pages=2
        response = post(client, two)
        assert response.status_code == 422
        assert response.json()["error"]["field_errors"][0]["rule"] == "max_total_pages"

        accepted = post(client, request("REQ-PDF-OK"))
        assert accepted.status_code == 202
        body = wait_finished(client, accepted.headers["Location"])
        assert body["result_origin"] == "module"
        assert body["result"]["evidence_reference"]["APCTT-C04"][0]["page"] == 1

    digest = EXAMPLE_REQUEST["case"]["documents"][0]["sha256"]
    stored_pdf = loaded.data_dir / "documents" / f"{digest}.pdf"
    assert hashlib.sha256(stored_pdf.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(stored_pdf.stat().st_mode) == 0o400
    assert list((loaded.data_dir / "documents").iterdir()) == [stored_pdf]  # damaged one not kept
    with sqlite3.connect(loaded.data_dir / "jobs.sqlite3") as conn:
        (case_json,) = conn.execute(
            "SELECT case_json FROM jobs WHERE request_id = 'REQ-PDF-OK'"
        ).fetchone()
    stored_document = json.loads(case_json)["documents"][0]
    assert "content_base64" not in stored_document and stored_document["page_count"] == 2
