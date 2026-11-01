"""CUDA-01 CPU structure tests: AC-26, AC-27, AC-28.

These tests run without a GPU. Where a torch module is needed, a clearly labelled stub is used to
exercise control flow only. Nothing here is CUDA evidence; real GPU tests are tests/cuda (CUDA-02).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from apctt_contracts.config import Checklist, load_config
from apctt_contracts.pdf import local_worker_case
from apctt_contracts.results import validate_result
from apctt_runtime.cudacheck import run_probe
from apctt_server.app import create_app
from participant_example.cuda_dummy import contract, features

from ..contract.fixtures import pdf_document
from ..contract.pdfgen import build_pdf

ROOT = Path(__file__).resolve().parents[2]
STUB = Path(__file__).resolve().parent / "stub_torch"
CASE = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())["case"]
DEMO = Checklist.model_validate(json.loads((ROOT / "contracts/checklist.demo.json").read_text()))
# config/cuda-example.json runs the official checklist (decision D74), like the example request.
OFFICIAL = Checklist.model_validate(
    json.loads((ROOT / "contracts/checklist.official.json").read_text())
)
PHRASES = json.loads((ROOT / "config/cuda-example.json").read_text())["module"]["settings"][
    "criteria_phrases"
]


def manifest_for(payload: bytes) -> dict:
    """Test-only manifest fixture. Product code never fabricates a CUDA training record."""
    return {
        "manifest_version": contract.MANIFEST_VERSION,
        "cuda_trained": True,
        "example_only": True,
        "device": "cuda:0",
        "steps": 128,
        "loss_first": 2.2,
        "loss_last": 1.1,
        "loss_finite": True,
        "grad_finite": True,
        "parameter_delta_max": 0.1,
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
    }


def stub_python(tmp_path: Path, **env: str) -> Path:
    """A wrapper interpreter that puts the structural torch stub first on sys.path."""
    exports = " ".join(f"{k}={v}" for k, v in env.items())
    script = tmp_path / f"stub-python-{len(list(tmp_path.glob('stub-python-*')))}"
    script.write_text(f'#!/bin/sh\nPYTHONPATH={STUB} {exports} exec "{sys.executable}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_ac26_probe_no_cpu_fallback(tmp_path: Path) -> None:
    # Real interpreter without torch: unavailable, never a CPU success.
    report = run_probe(Path(sys.executable), "cuda:0", timeout_s=60)
    assert report.status == "CUDA_UNAVAILABLE" and report.reason == "TORCH_NOT_INSTALLED"
    assert report.summary()["kernel_probe_passed"] is not True
    missing = run_probe(tmp_path / "no-python", "cuda:0", timeout_s=10)
    assert (missing.status, missing.reason) == ("CUDA_UNAVAILABLE", "WORKER_PYTHON_NOT_FOUND")
    # Structural stubs: CPU-only build, CUDA not available, kernel failure.
    cpu_only = run_probe(stub_python(tmp_path), "cuda:0", timeout_s=60)
    assert (cpu_only.status, cpu_only.reason) == ("CUDA_UNAVAILABLE", "CPU_ONLY_TORCH")
    no_device = run_probe(stub_python(tmp_path, STUB_CUDA_BUILD="12.1"), "cuda:0", timeout_s=60)
    assert (no_device.status, no_device.reason) == ("CUDA_UNAVAILABLE", "CUDA_NOT_AVAILABLE")
    broken = run_probe(
        stub_python(tmp_path, STUB_CUDA_BUILD="12.1", STUB_CUDA_AVAILABLE="1"),
        "cuda:0",
        timeout_s=60,
    )
    assert (broken.status, broken.reason) == ("CUDA_RUNTIME_ERROR", "KERNEL_PROBE_FAILED")
    assert "structural stub" not in json.dumps(broken.summary())  # no raw exception text
    assert run_probe(Path(sys.executable), "cpu", timeout_s=10).reason == "DEVICE_NOT_CUDA"

    # check_cuda.py launcher: exit 3 and a written record, no fallback.
    output = tmp_path / "cuda.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_cuda.py"),
            "--worker-python",
            sys.executable,
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 3, proc.stderr
    record = json.loads(output.read_text())
    assert record["probe"]["status"] == "CUDA_UNAVAILABLE" and record["cuda_verified"] is False

    # A CUDA-required server stays not-ready and refuses admission instead of running on CPU.
    config = json.loads((ROOT / "config/cuda-example.json").read_text())
    config.update(
        data_dir=str(tmp_path / "data"),
        worker_tmp_dir=str(tmp_path / "tmp"),
        checklist_path=str(ROOT / "contracts/checklist.official.json"),
    )
    config["module"].update(
        module_root=str(ROOT),
        package_root=str(ROOT / "participant_example"),
        asset_root=str(tmp_path),
    )
    config["worker"]["python"] = sys.executable
    path = tmp_path / "cuda.config.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path, profile="dev")
    assert loaded.config.compute.require_cuda and not loaded.config.compute.allow_cpu_fallback
    with TestClient(create_app(loaded)) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 503 and ready.json()["reason_code"] == "CUDA_UNAVAILABLE"
        body = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())
        response = client.post("/v1/assessments", json=body)
        assert response.status_code == 503 and response.json()["error"]["code"] == "NOT_READY"
        assert client.app.state.apctt.store.job_count() == 0  # type: ignore[attr-defined]


def test_ac27_training_separate_from_api(tmp_path: Path) -> None:
    # Importing and serving the API never imports torch or the training module.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, apctt_server.app, apctt_server.state, apctt_runtime.supervisor; "
            "print(sorted(m for m in sys.modules "
            "if m == 'torch' or m.endswith('cuda_dummy.train')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "[]"
    loaded = load_config(ROOT / "config/dev.json", profile="dev")
    app = create_app(loaded)
    paths = {getattr(route, "path", "") for route in app.routes}
    assert not any("train" in p for p in paths)
    assert not any("train" in p for p in app.openapi()["paths"])

    # The training command refuses CPU and never writes a checkpoint without CUDA.
    for device, code in (("cpu", 2), ("auto", 2), ("cuda:0", 3)):
        out = tmp_path / f"out-{device.replace(':', '')}"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "participant_example.cuda_dummy.train",
                "--device",
                device,
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
            timeout=120,
        )
        assert proc.returncode == code, (device, proc.stderr)
        assert not out.exists() or not any(out.iterdir())

    # The adapter only loads; it does not import or call the training module.
    tree = ast.parse((ROOT / "participant_example/cuda_dummy/adapter.py").read_text())
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not any("train" in name for name in imported)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "run_training" not in names


def test_ac28_checkpoint_and_demo_contract(tmp_path: Path) -> None:
    # Model configuration and checkpoint shapes.
    assert contract.validate_model_config(contract.MODEL_CONFIG) == []
    for change in (
        {"hidden_dim": 64},
        {"num_classes": 10},
        {"precision": "float16"},
        {"architecture": "resnet"},
        {"extra": 1},
    ):
        assert contract.validate_model_config({**contract.MODEL_CONFIG, **change})
    good = {k: list(v) for k, v in contract.EXPECTED_SHAPES.items()}
    assert contract.validate_state_shapes(good) == []
    assert contract.validate_state_shapes({**good, "0.weight": [32, 17]})
    assert contract.validate_state_shapes({k: v for k, v in good.items() if k != "2.bias"})
    assert contract.validate_state_shapes({**good, "3.weight": [1]})

    # Training manifest: only a real CUDA-trained, hash-matching checkpoint is accepted.
    payload = b"checkpoint-bytes"
    manifest = manifest_for(payload)
    assert contract.validate_training_manifest(manifest, payload) == []
    assert "CHECKPOINT_HASH_MISMATCH" in contract.validate_training_manifest(manifest, b"other")
    for change, problem in (
        ({"cuda_trained": False}, "NOT_CUDA_TRAINED"),
        ({"device": "cpu"}, "NOT_CUDA_TRAINED"),
        ({"steps": 0}, "NO_OPTIMIZER_STEPS"),
        ({"loss_last": math.nan}, "NON_FINITE_TRAINING"),
        ({"parameter_delta_max": 0.0}, "PARAMETERS_UNCHANGED"),
        ({"example_only": False}, "NOT_MARKED_EXAMPLE"),
    ):
        assert problem in contract.validate_training_manifest({**manifest, **change}, payload)

    # Deterministic 16-dim features from the text of the case PDFs only.
    view = local_worker_case(CASE, tmp_path / "docs")
    first = features.extract_features(view)
    assert first == features.extract_features(view) and len(first) == 16
    assert all(math.isfinite(v) and v >= 0 for v in first) and math.isclose(sum(first), 1.0)
    changed = json.loads(json.dumps(CASE))
    other = build_pdf([["Completely different words here."]])
    changed["documents"][0] = pdf_document("D001", "Other", None, other)
    assert features.extract_features(local_worker_case(changed, tmp_path / "other")) != first

    # DEMO results satisfy SPEC-06 with the official checklist of config/cuda-example.json (D74)
    # and depend on the model output.
    ids = OFFICIAL.ids()
    results = [
        contract.build_result(view, OFFICIAL.payload(), PHRASES, class_index=k, confidence=0.5)
        for k in (0, 4, 8)
    ]
    for k, result in zip((0, 4, 8), results, strict=True):
        assert validate_result(result, case=view, checklist=OFFICIAL) == [], result
        assert result["assessment_status"] == "completed" and result["predicted_trl"] == k + 1
    assert results[0]["predicted_trl"] != results[1]["predicted_trl"]
    assert set(results[0]["criterion_status"]) == set(ids)
    assert results[0]["criterion_status"]["APCTT-C04"] == "met"  # "tested" on page 1

    # Example guard: any checklist of the configuration (official, DEMO); non-example schemas are
    # refused. Until kit 0.5.0 the dummy refused every checklist but the DEMO one.
    assert contract.check_demo_inputs(OFFICIAL.payload(), CASE) is None
    assert contract.check_demo_inputs(DEMO.payload(), CASE) is None
    assert contract.check_demo_inputs(
        OFFICIAL.payload(), {**CASE, "schema_version": "apctt-v05"}
    ) == ("CASE_NOT_DEMO_SCHEMA")

    # Synthetic training data: deterministic synthetic classes, not TRL labels.
    xs, ys = contract.synthetic_dataset(seed=42)
    assert (xs, ys) == contract.synthetic_dataset(seed=42)
    assert sorted(set(ys)) == list(range(9)) and all(len(x) == 16 for x in xs)

    # The shipped CUDA example configuration is explicit: cuda:0, required, no fallback.
    loaded = load_config(ROOT / "config/cuda-example.json", profile="dev")
    compute = loaded.config.compute
    assert (compute.device, compute.require_cuda, compute.allow_cpu_fallback) == (
        "cuda:0",
        True,
        False,
    )
    assert (
        loaded.config.module.entrypoint == "participant_example.cuda_dummy.adapter:AssessmentModule"
    )
    assert os.path.basename(str(loaded.asset_root)) == "cuda_dummy"
