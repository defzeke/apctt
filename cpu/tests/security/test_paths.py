"""Path, symlink and URL handling: AC-18."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path

from apctt_contracts.config import load_config, startup_problems
from tests.integration.server import ROOT, Server, base_config, request, wait_finished

SECRET = "SENTINEL-outside-file-7d2a"


def raw_get(port: int, path: str) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, response.read().decode("utf-8", "replace")


def test_ac18_symlink_and_path_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"secret": SECRET}))
    counter = tmp_path / "count"
    counter.mkdir()
    settings = {"counter_dir": str(counter), "outside_file": str(outside)}

    # A result.json symlink is rejected without reading its target.
    config = base_config(tmp_path, fake="SymlinkResult", settings=settings)
    with Server(tmp_path, config) as server, server.client() as client:
        job = wait_finished(
            client, client.post("/v1/assessments", json=request("REQ-LINK")).headers["Location"]
        )
        assert job["result"]["error_code"] == "WORKER_PROTOCOL_ERROR"
        assert "RESULT_FILE_REJECTED" in job["diagnostics"]["warning_codes"]
        assert SECRET not in json.dumps(job)
        artifacts = Path(config["data_dir"]) / "jobs" / job["assessment_id"]
        assert not (artifacts / "raw_output.bin").exists()

        # Path traversal through the API and static routes stays inside the application.
        port = config["bind_port"]
        for path in (
            "/v1/assessments/..%2F..%2Fpyproject.toml",
            "/v1/assessments/../../pyproject.toml",
            "/static/../../pyproject.toml",
            "/static/swagger-ui/../../../../pyproject.toml",
            "/static/%2e%2e/%2e%2e/pyproject.toml",
        ):
            status, text = raw_get(port, path)
            assert status == 404 and "[project]" not in text, path

        # Requests cannot choose module paths, assets, URLs, argv or devices.
        for field in ("module_path", "asset_root", "entrypoint", "url", "argv", "python", "device"):
            body = request(f"REQ-{field}")
            body["case"][field] = "/etc/passwd" if field != "url" else "http://example.invalid/x"
            response = client.post("/v1/assessments", json=body)
            assert response.status_code == 422, field
            assert "/etc/passwd" not in response.text and "example.invalid" not in response.text

    # A module that swaps the sidecar for a symlink cannot make anyone write through or read it:
    # the wrapper replaces the link itself (rename), and the outside file stays untouched.
    # Supervisor-side rejection of symlinked sidecars is covered by RUN-01 AC-36.
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    before = outside.read_bytes()
    config = base_config(sidecar_dir, fake="SymlinkSidecar", settings=settings)
    with Server(sidecar_dir, config) as server, server.client() as client:
        job = wait_finished(
            client, client.post("/v1/assessments", json=request("REQ-SIDE")).headers["Location"]
        )
        assert job["result_origin"] == "module"
        assert job["diagnostics"]["worker_phase"] == "finished"
        assert SECRET not in json.dumps(job)
    assert outside.read_bytes() == before and not outside.is_symlink()

    # Package and module roots: symlinks are refused at start or reported as not verified.
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "qa_fakes.py").write_text((ROOT / "tests/integration/fakes/qa_fakes.py").read_text())
    os.symlink(outside, package / "linked.json")
    (tmp_path / "linked").mkdir()
    linked_config = base_config(tmp_path / "linked", fake="Counting", settings=settings)
    linked_config["module"].update(
        module_root=str(package), package_root=str(package), asset_root=str(package)
    )
    with Server(tmp_path, linked_config, name="linked") as server, server.client() as client:
        info = client.get("/v1/package").json()
        assert info["verification_status"] == "not_verified"
        assert info["verified_package_sha256"] is None
    root_link = tmp_path / "root-link"
    os.symlink(package, root_link)
    bad_root = base_config(tmp_path, fake="Counting", settings=settings)
    bad_root["module"].update(
        module_root=str(root_link), package_root=str(root_link), asset_root=str(root_link)
    )
    path = tmp_path / "bad-root.json"
    path.write_text(json.dumps(bad_root))
    loaded = load_config(path, profile="dev")
    assert (
        "MODULE_PATH_NOT_DIRECTORY" in startup_problems(loaded)
        or loaded.module_root == package.resolve()
    )
