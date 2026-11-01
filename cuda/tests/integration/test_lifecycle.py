"""Real server process E2E: AC-06, AC-08, AC-13 (+ CPU example evidence rules)."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

from apctt_contracts.config import Checklist
from apctt_contracts.pdf import stored_case
from apctt_contracts.results import validate_result

from ..contract.fixtures import pdf_document
from ..contract.pdfgen import build_pdf
from .server import ROOT, Server, base_config, request, wait_finished


def executions(counter: Path) -> int:
    return len(list(counter.iterdir()))


def test_ac06_simultaneous_duplicate_one_execution(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    config = base_config(
        tmp_path, fake="Counting", settings={"counter_dir": str(counter), "sleep_s": 1}
    )
    with Server(tmp_path, config) as server, server.client() as client:
        body = request("REQ-SAME")
        barrier = threading.Barrier(4)
        responses: list[tuple[int, str, str]] = []

        def submit() -> None:
            barrier.wait()
            response = client.post("/v1/assessments", json=body)
            responses.append(
                (
                    response.status_code,
                    response.headers.get("Location", ""),
                    response.json().get("assessment_id", ""),
                )
            )

        threads = [threading.Thread(target=submit) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert {r[0] for r in responses} == {202}
        assert len({r[1] for r in responses}) == 1 and len({r[2] for r in responses}) == 1
        wait_finished(client, responses[0][1])
        assert executions(counter) == 1


def test_ac08_new_id_means_new_execution(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    config = base_config(tmp_path, fake="Counting", settings={"counter_dir": str(counter)})
    with Server(tmp_path, config) as server, server.client() as client:
        first = client.post("/v1/assessments", json=request("REQ-1"))
        one = wait_finished(client, first.headers["Location"])
        second = client.post("/v1/assessments", json=request("REQ-2"))
        assert second.status_code == 202
        assert second.json()["assessment_id"] != first.json()["assessment_id"]
        two = wait_finished(client, second.headers["Location"])
        assert executions(counter) == 2  # same case, new request ID: a new real execution
        assert one["meta"]["input_sha256"] == two["meta"]["input_sha256"]
        assert (
            one["meta"]["accepted_at"] != two["meta"]["accepted_at"]
            or one["assessment_id"] != two["assessment_id"]
        )


def test_ac13_disconnect_then_get(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    counter.mkdir()
    config = base_config(
        tmp_path, fake="Counting", settings={"counter_dir": str(counter), "sleep_s": 1}
    )
    with Server(tmp_path, config) as server, server.client() as client:
        payload = json.dumps(request("REQ-DROP")).encode()
        head = (
            "POST /v1/assessments HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
        ).encode()
        with socket.create_connection(("127.0.0.1", config["bind_port"])) as sock:
            sock.sendall(head + payload)
            time.sleep(0.3)  # the client gives up without reading the response
        again = client.post("/v1/assessments", json=request("REQ-DROP"))
        assert again.status_code == 202
        body = wait_finished(client, again.headers["Location"])
        assert body["result_origin"] == "module"
        assert executions(counter) == 1


def test_cpu_example_outputs_satisfy_evidence_rules(tmp_path: Path) -> None:
    """E23 (not a required node): the shipped CPU example passes SPEC-06 through the real API,
    with the official checklist of config/dev.json (D74)."""
    config = base_config(tmp_path)
    checklist = Checklist.model_validate(
        json.loads((ROOT / "contracts/checklist.official.json").read_text())
    )
    with Server(tmp_path, config) as server, server.client() as client:
        for request_id, text in (
            ("REQ-EX1", None),
            ("REQ-EX2", "All acceptance criteria were met in tests."),
        ):
            body = request(request_id)
            pages = [2]
            if text:
                body["case"]["documents"].append(
                    pdf_document("D002", "Test results", None, build_pdf([[text]]))
                )
                pages.append(1)
            location = client.post("/v1/assessments", json=body).headers["Location"]
            job = wait_finished(client, location)
            assert job["result_origin"] == "module"
            view = stored_case(body["case"], pages)
            assert validate_result(job["result"], case=view, checklist=checklist) == []
        assert (
            job["result"]["assessment_status"] == "completed"
            and job["result"]["predicted_trl"] == 4
        )
