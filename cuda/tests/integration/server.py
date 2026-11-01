"""Run the real server (`python -m apctt_server`) as a separate process for QA tests."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).resolve().parent / "fakes"
EXAMPLE_REQUEST: dict[str, Any] = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def base_config(
    tmp_path: Path,
    *,
    fake: str | None = None,
    settings: dict[str, Any] | None = None,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = json.loads((ROOT / "config/dev.json").read_text())
    config.update(
        data_dir=str(tmp_path / "server_data"),
        worker_tmp_dir=str(tmp_path / "worker_tmp"),
        checklist_path=str(ROOT / "contracts/checklist.official.json"),
        bind_port=free_port(),
    )
    config["module"].update(
        module_root=str(ROOT),
        package_root=str(ROOT / "participant_example"),
        asset_root=str(ROOT / "participant_example"),
    )
    if fake is not None:
        config["module"].update(
            package_id="qa-fake",
            entrypoint=f"qa_fakes:{fake}",
            module_root=str(FAKES),
            package_root=str(FAKES),
            asset_root=str(FAKES),
            settings=settings or {},
        )
    config["limits"].update(limits or {})
    return config


class Server:
    def __init__(
        self,
        tmp_path: Path,
        config: dict[str, Any],
        *,
        name: str = "server",
        profile: str = "dev",
        env: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.path = tmp_path / f"{name}.json"
        self.path.write_text(json.dumps(config))
        self.log = tmp_path / f"{name}.log"
        self.env = {**os.environ, **(env or {})}
        self.url = f"http://127.0.0.1:{config['bind_port']}"
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> Server:
        with open(self.log, "ab") as log:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "apctt_server",
                    "--profile",
                    self.profile,
                    "--config",
                    str(self.path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env,
                cwd=ROOT,
                start_new_session=True,
            )
        return self

    def wait_live(self, timeout: float = 30.0) -> Server:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.process is not None
            if self.process.poll() is not None:
                raise AssertionError(f"server exited: {self.log.read_text()[-2000:]}")
            try:
                if httpx.get(self.url + "/health/live", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise AssertionError("server did not become live")

    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=30)

    def crash(self) -> None:
        """SIGKILL the API process only; its worker lives in its own session and may survive."""
        assert self.process is not None
        os.kill(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=10)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __enter__(self) -> Server:
        return self.start().wait_live()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def request(request_id: str, **case_changes: Any) -> dict[str, Any]:
    body = json.loads(json.dumps(EXAMPLE_REQUEST))
    body["request_id"] = request_id
    body["case"].update(case_changes)
    return body


def wait_finished(client: httpx.Client, location: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(location)
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        if body["job_status"] == "finished":
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def wait_for(predicate: Any, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")
