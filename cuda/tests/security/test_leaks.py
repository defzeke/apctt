"""Real server leak and debug-override checks: AC-15, AC-16."""

from __future__ import annotations

import json
from pathlib import Path

from tests.integration.server import ROOT, Server, base_config, request, wait_finished

SENTINEL = "SENTINEL-9c4e-never-leak"
ENV_SENTINEL = "SENTINEL-env-3b1f"


def test_ac15_sentinels_never_leak(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    config = base_config(
        tmp_path, fake="Leaky", settings={"counter_dir": str(counter), "sentinel": SENTINEL}
    )
    env = {"APCTT_LEAK_SENTINEL": ENV_SENTINEL, "AWS_SECRET_ACCESS_KEY": ENV_SENTINEL}
    with Server(tmp_path, config, env=env) as server, server.client() as client:
        texts: list[str] = []

        def record(response) -> None:
            texts.append(response.text)

        # Request-side sentinels: unknown field names/values, invalid types, invalid JSON.
        bad = request("REQ-BAD")
        bad["case"][SENTINEL] = SENTINEL
        bad["case"]["documents"][0]["sha256"] = SENTINEL
        bad["case"]["documents"][0]["title"] = {SENTINEL: SENTINEL}
        record(client.post("/v1/assessments", json=bad))
        record(
            client.post(
                "/v1/assessments",
                content=f'{{"{SENTINEL}": NaN}}'.encode(),
                headers={"Content-Type": "application/json"},
            )
        )
        record(
            client.post(
                "/v1/assessments", content=SENTINEL.encode(), headers={"Content-Type": "text/plain"}
            )
        )
        record(client.get(f"/v1/assessments/{SENTINEL}"))
        record(client.get(f"/{SENTINEL}"))
        # Module-side sentinels: stdout, stderr, exception message.
        accepted = client.post("/v1/assessments", json=request("REQ-LEAK"))
        record(accepted)
        job = wait_finished(client, accepted.headers["Location"])
        assert (
            job["result_origin"] == "supervisor" and job["result"]["error_code"] == "MODULE_ERROR"
        )
        record(client.get(accepted.headers["Location"]))
        for path in ("/health/live", "/health/ready", "/v1/package", "/openapi.json"):
            record(client.get(path))
        joined = "\n".join(texts)
        assert SENTINEL not in joined and ENV_SENTINEL not in joined
        for absolute in (str(tmp_path), str(ROOT), str(Path.home())):
            assert absolute not in joined, absolute

    # The sentinel text exists only in local, restricted artifacts; the worker never saw server env.
    job_dir = Path(config["data_dir"]) / "jobs" / job["assessment_id"]
    stdout = (job_dir / "stdout.log").read_text()
    assert SENTINEL in stdout and SENTINEL in (job_dir / "stderr.log").read_text()
    worker_env = json.loads(stdout.splitlines()[0])["env"]
    assert ENV_SENTINEL not in json.dumps(worker_env)
    assert "AWS_SECRET_ACCESS_KEY" not in worker_env and "APCTT_LEAK_SENTINEL" not in worker_env
    assert oct(job_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((job_dir / "stdout.log").stat().st_mode & 0o777) == "0o600"
    events = (Path(config["data_dir"]) / "events.jsonl").read_text()
    assert SENTINEL not in events and ENV_SENTINEL not in events


def _strip(body: dict) -> dict:
    body = json.loads(json.dumps(body))
    body.get("meta", {}).pop("trace_id", None)
    body.get("diagnostics", {}).pop("duration_ms", None)
    return body


def test_ac16_client_debug_cannot_override(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    with Server(tmp_path, config) as server, server.client() as client:
        bad = request("REQ-DEBUG", title="")
        plain = client.post("/v1/assessments", json=bad)
        hinted = client.post(
            "/v1/assessments?debug=true&trace=1&profile=dev",
            json=bad,
            headers={"X-Debug": "1", "X-APCTT-Debug": "true", "X-APCTT-Profile": "dev"},
        )
        assert plain.status_code == hinted.status_code == 422
        assert _strip(plain.json()) == _strip(hinted.json())
        assert "stack_frames" not in hinted.json()["diagnostics"]
        body = request("REQ-DEBUG2")
        body["debug"] = True
        body["case"]["profile"] = "evaluation"
        rejected = client.post("/v1/assessments", json=body).json()
        assert {e["path"] for e in rejected["error"]["field_errors"]} == {
            "<unknown-field>",
            "case.<unknown-field>",
        }
        # A normal job cannot be switched to another profile, device, seed or timeout by the client.
        location = client.post(
            "/v1/assessments?device=cuda:0&seed=1&timeout=999",
            json=request("REQ-OK"),
            headers={"X-APCTT-Device": "cuda:0"},
        ).headers["Location"]
        job = wait_finished(client, location)
        assert job["diagnostics"]["profile"] == "dev"
        assert job["diagnostics"]["compute"]["device"] == "cpu"

    # evaluation profile: no stack frames or debug switches (server-side configuration only).
    from .test_profiles import approved_evaluation_config

    evaluation, _ = approved_evaluation_config(tmp_path / "eval")
    with (
        Server(tmp_path / "eval", evaluation, profile="evaluation") as server,
        server.client() as client,
    ):
        error = client.post(
            "/v1/assessments?debug=1",
            content=b"{bad",
            headers={"Content-Type": "application/json", "X-Debug": "1"},
        ).json()
        assert error["diagnostics"]["profile"] == "evaluation"
        assert "stack_frames" not in error["diagnostics"]
