"""Temporary dev configurations and polling for API tests."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from apctt_contracts.config import LoadedConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).resolve().parent / "fakes"
EXAMPLE_REQUEST: dict[str, Any] = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())


def make_config(
    tmp_path: Path,
    *,
    fake: bool = False,
    settings: dict[str, Any] | None = None,
    limits: dict[str, int] | None = None,
    name: str = "config.json",
    extra: dict[str, Any] | None = None,
) -> LoadedConfig:
    base = json.loads((ROOT / "config/dev.json").read_text())
    base["data_dir"] = str(tmp_path / "server_data")
    base["worker_tmp_dir"] = str(tmp_path / "worker_tmp")
    base["checklist_path"] = str(ROOT / "contracts/checklist.official.json")
    module = base["module"]
    if fake:
        module.update(
            package_id="fake-api",
            entrypoint="api_fakes:Counting",
            module_root=str(FAKES),
            package_root=str(FAKES),
            asset_root=str(FAKES),
            settings=settings or {},
        )
    else:
        module.update(
            module_root=str(ROOT),
            package_root=str(ROOT / "participant_example"),
            asset_root=str(ROOT / "participant_example"),
        )
    base["limits"].update(limits or {})
    for key, value in (extra or {}).items():
        base[key] = value
    path = tmp_path / name
    path.write_text(json.dumps(base))
    return load_config(path, profile=base["profile"])


def request(request_id: str = "REQ-1", **case_changes: Any) -> dict[str, Any]:
    body = copy.deepcopy(EXAMPLE_REQUEST)
    body["request_id"] = request_id
    body["case"].update(case_changes)
    return body


def wait_finished(client: TestClient, location: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(location)
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        if body["job_status"] == "finished":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def executions(counter_dir: Path) -> int:
    return len(list(counter_dir.iterdir()))
