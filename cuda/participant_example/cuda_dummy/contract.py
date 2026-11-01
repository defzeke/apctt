"""Pure-Python contract for the CUDA dummy: model config, checkpoint checks, DEMO results.

No torch import. Used by train.py and adapter.py, and by CPU structure tests (CUDA-01).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from participant_example import demo_rules
from participant_example.cuda_dummy import features

MODEL_CONFIG: dict[str, Any] = {
    "architecture": "mlp-16-32-9",
    "input_dim": 16,
    "hidden_dim": 32,
    "num_classes": 9,
    "activation": "relu",
    "precision": "float32",
    "feature_extractor": features.FEATURE_EXTRACTOR,
}
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "0.weight": (32, 16),
    "0.bias": (32,),
    "2.weight": (9, 32),
    "2.bias": (9,),
}
MANIFEST_VERSION = "apctt-cuda-dummy-training-v1"
CHECKPOINT_FILE = "model.pt"
MODEL_CONFIG_FILE = "model_config.json"
MANIFEST_FILE = "training_manifest.json"
DEMO_SCHEMA_VERSION = "apctt-pdf-example-v1"
MAX_STEPS = 128
TRL_RATIONALE = (
    "DEMO toy MLP: predicted_trl is the class argmax + 1 over hashed text features; "
    "example_only, not a TRL judgment."
)


def validate_model_config(config: Mapping[str, Any]) -> list[str]:
    problems = [
        f"MODEL_CONFIG_MISMATCH:{k}" for k in MODEL_CONFIG if config.get(k) != MODEL_CONFIG[k]
    ]
    problems += [f"MODEL_CONFIG_UNKNOWN:{k}" for k in config if k not in MODEL_CONFIG]
    return problems


def validate_state_shapes(shapes: Mapping[str, Sequence[int]]) -> list[str]:
    problems = [f"MISSING_TENSOR:{k}" for k in EXPECTED_SHAPES if k not in shapes]
    problems += [f"UNEXPECTED_TENSOR:{k}" for k in shapes if k not in EXPECTED_SHAPES]
    problems += [
        f"SHAPE_MISMATCH:{k}"
        for k, shape in shapes.items()
        if k in EXPECTED_SHAPES and tuple(shape) != EXPECTED_SHAPES[k]
    ]
    return problems


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_training_manifest(manifest: Mapping[str, Any], checkpoint: bytes) -> list[str]:
    """Accept only a record of a real CUDA training run whose checkpoint bytes match."""
    problems: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        problems.append("MANIFEST_VERSION")
    if manifest.get("cuda_trained") is not True or not str(manifest.get("device", "")).startswith(
        "cuda"
    ):
        problems.append("NOT_CUDA_TRAINED")
    steps = manifest.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_STEPS:
        problems.append("NO_OPTIMIZER_STEPS")
    if not (
        _finite(manifest.get("loss_first"))
        and _finite(manifest.get("loss_last"))
        and manifest.get("loss_finite") is True
        and manifest.get("grad_finite") is True
    ):
        problems.append("NON_FINITE_TRAINING")
    delta = manifest.get("parameter_delta_max")
    if not _finite(delta) or delta <= 0:
        problems.append("PARAMETERS_UNCHANGED")
    if manifest.get("example_only") is not True:
        problems.append("NOT_MARKED_EXAMPLE")
    if manifest.get("checkpoint_sha256") != hashlib.sha256(checkpoint).hexdigest():
        problems.append("CHECKPOINT_HASH_MISMATCH")
    return problems


def check_demo_inputs(checklist: Mapping[str, Any], case: Mapping[str, Any] | None) -> str | None:
    """The dummy refuses non-example case schemas. It runs with the checklist of its configuration
    (config/cuda-example.json: the official checklist, decision D74); `example_only=true` marks
    its results as a format example, not an assessment. Until kit 0.5.0 it refused every
    checklist but the DEMO one (CHECKLIST_NOT_DEMO)."""
    if case is not None and case.get("schema_version") != DEMO_SCHEMA_VERSION:
        return "CASE_NOT_DEMO_SCHEMA"
    return None


def synthetic_dataset(seed: int, per_class: int = 16) -> tuple[list[list[float]], list[int]]:
    """Synthetic texts per toy class. Labels are synthetic classes, not expert TRL labels."""
    rng = random.Random(seed)
    common = [f"common{i}" for i in range(12)]
    xs: list[list[float]] = []
    ys: list[int] = []
    for label in range(MODEL_CONFIG["num_classes"]):
        vocabulary = [f"toyclass{label}word{i}" for i in range(6)]
        for _ in range(per_class):
            words = rng.choices(vocabulary, k=14) + rng.choices(common, k=6)
            rng.shuffle(words)
            xs.append(features.features_from_text(" ".join(words)))
            ys.append(label)
    return xs, ys


def build_result(
    case: Mapping[str, Any],
    checklist: Mapping[str, Any],
    phrases: Mapping[str, Sequence[str]],
    *,
    class_index: int,
    confidence: float,
) -> dict[str, Any]:
    ids = [c["criterion_id"] for c in checklist["criteria"]]
    observed = demo_rules.evaluate(case, ids, phrases)
    result = demo_rules.base_result(case, checklist["version"])
    result.update(
        assessment_status="completed",
        predicted_trl=int(class_index) + 1,
        criterion_status=observed["criterion_status"],
        evidence_reference=observed["evidence_reference"],
        missing_evidence=observed["missing_evidence"],
        abstention_reason=None,
        error_code=None,
        confidence=round(min(max(float(confidence), 0.0), 1.0), 6),
        trl_rationale=TRL_RATIONALE,
    )
    if observed["criterion_rationale"]:
        result["criterion_rationale"] = observed["criterion_rationale"]
    return result
