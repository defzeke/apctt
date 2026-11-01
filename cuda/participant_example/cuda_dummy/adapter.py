"""CUDA dummy AssessmentModule: loads the trained checkpoint and infers on CUDA only.

Entry point: participant_example.cuda_dummy.adapter:AssessmentModule
The checkpoint is loaded with map_location='cpu', weights_only=True, shape-checked, then moved to
the configured CUDA device. There is no CPU fallback and no training here; a missing checkpoint is
a load failure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from participant_example.cuda_dummy import contract, features
from participant_example.cuda_dummy.model import build_model


class CudaDummyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.apctt_error_code = code


class AssessmentModule:
    def __init__(self) -> None:
        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None
        self._checklist: Mapping[str, Any] = {}
        self._phrases: Mapping[str, Any] = {}

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        compute = config["compute"]
        if not (
            str(compute.get("device", "")).startswith("cuda")
            and compute.get("require_cuda") is True
            and compute.get("allow_cpu_fallback") is False
        ):
            raise CudaDummyError("MODULE_ERROR", "compute must be cuda:N with require_cuda=true")
        problem = contract.check_demo_inputs(config["checklist"], None)
        if problem:
            raise CudaDummyError("MODULE_ERROR", problem)
        self._checklist = config["checklist"]
        self._phrases = config["module"]["settings"]["criteria_phrases"]
        try:
            import torch
        except ImportError:
            raise CudaDummyError("CUDA_UNAVAILABLE", "torch is not installed") from None
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise CudaDummyError("CUDA_UNAVAILABLE", "CUDA is not available to this worker")
        root = Path(asset_root)
        model_config = json.loads((root / contract.MODEL_CONFIG_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((root / contract.MANIFEST_FILE).read_text(encoding="utf-8"))
        checkpoint = root / contract.CHECKPOINT_FILE
        problems = contract.validate_model_config(model_config)
        problems += contract.validate_training_manifest(manifest, checkpoint.read_bytes())
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        problems += contract.validate_state_shapes({k: list(v.shape) for k, v in state.items()})
        if problems:
            raise CudaDummyError("MODULE_ERROR", "checkpoint rejected: " + ",".join(problems))
        device = torch.device(compute["device"])
        model = build_model(torch)
        model.load_state_dict(state, strict=True)
        self._model = model.to(device).eval()
        self._torch, self._device = torch, device

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        problem = contract.check_demo_inputs(self._checklist, case)
        if problem:
            raise CudaDummyError("MODULE_ERROR", problem)
        torch = self._torch
        x = torch.tensor(
            [features.extract_features(case)], dtype=torch.float32, device=self._device
        )
        with torch.inference_mode():
            logits = self._model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, index = probabilities.max(dim=1)
        torch.cuda.synchronize(self._device)
        devices = {next(self._model.parameters()).device.type, x.device.type, logits.device.type}
        if devices != {"cuda"}:
            raise CudaDummyError("CUDA_RUNTIME_ERROR", "inference did not run on CUDA")
        return contract.build_result(
            case,
            self._checklist,
            self._phrases,
            class_index=int(index.item()),
            confidence=float(confidence.item()),
        )

    def reset_case(self) -> None:
        """No per-case state is kept between assess() and close()."""

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()  # housekeeping only; not evidence of released memory
