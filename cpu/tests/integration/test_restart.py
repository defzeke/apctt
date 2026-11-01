"""Real API process crash and restart: AC-12."""

from __future__ import annotations

from pathlib import Path

import httpx

from apctt_runtime import procinfo

from .server import Server, base_config, free_port, request, wait_finished, wait_for


def test_ac12_real_restart(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    config = base_config(
        tmp_path, fake="Counting", settings={"counter_dir": str(counter), "sleep_s": 60}
    )
    server = Server(tmp_path, config).start().wait_live()
    worker_pid = -1
    try:
        with server.client() as client:
            accepted = client.post("/v1/assessments", json=request("REQ-CRASH"))
            assert accepted.status_code == 202
            location = accepted.headers["Location"]
            wait_for(lambda: any(counter.iterdir()), timeout=30)
            worker_pid = int(next(counter.iterdir()).name.split(".")[0])
            assert client.get(location).json()["job_status"] == "running"
        server.crash()  # the API dies; its worker runs in its own session and survives
        assert procinfo.is_alive(worker_pid)

        # A second instance on the same data directory must not start while the first holds it.
        restarted = Server(tmp_path, config, name="restarted").start().wait_live()
        try:
            with restarted.client() as client:
                body = wait_finished(client, location, timeout=30)
                assert body["result_origin"] == "supervisor"
                assert body["result"]["error_code"] == "SERVER_RESTARTED"
                assert body["result"]["assessment_status"] == "failed"
                assert body["diagnostics"]["failure"]["code"] == "SERVER_RESTARTED"
                assert not procinfo.is_alive(worker_pid)
                assert len(list(counter.iterdir())) == 1  # no automatic re-execution
                duplicate = client.post("/v1/assessments", json=request("REQ-CRASH"))
                assert duplicate.status_code == 202
                assert duplicate.json()["result"]["error_code"] == "SERVER_RESTARTED"
                assert client.get("/health/ready").json()["reason_code"] == "READY"
                second = Server(tmp_path, dict(config, bind_port=free_port()), name="second")
                second.start()
                assert second.process is not None
                assert second.process.wait(timeout=30) != 0
                try:
                    httpx.get(second.url + "/health/live", timeout=1)
                    raise AssertionError("second instance must not serve")
                except httpx.HTTPError:
                    pass
        finally:
            restarted.stop()
    finally:
        server.stop()
        if procinfo.is_alive(worker_pid):
            procinfo.terminate_group(worker_pid, 0.5, 5)
