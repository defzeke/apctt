"""Stand-alone CUDA training command for the dummy (never called by the API server).

    <worker-python> -m participant_example.cuda_dummy.train --device cuda:0 --output-dir DIR

Runs real forward/loss/backward/optimizer steps on the GPU, checks finite loss and gradients and a
parameter change, then saves a CPU state_dict plus model_config.json and training_manifest.json.
Exit codes: 0 trained, 2 invalid arguments (including any non-CUDA device), 3 CUDA unavailable,
4 training checks failed. No files are written unless training succeeded on CUDA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

from participant_example.cuda_dummy import contract
from participant_example.cuda_dummy.model import build_model


def run_training(
    torch: Any, device: Any, *, seed: int, steps: int, lr: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Device-agnostic training loop. The CLI only ever passes a CUDA device."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    xs, ys = contract.synthetic_dataset(seed)
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    model = build_model(torch).to(device)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    grad_finite = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                grad_finite = False
        optimizer.step()
        losses.append(float(loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final = model.state_dict()
    delta = max(float((final[k] - initial[k]).abs().max().item()) for k in final)
    state = {k: v.detach().to("cpu").contiguous() for k, v in final.items()}
    stats = {
        "steps": steps,
        "optimizer": "Adam",
        "lr": lr,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_finite": bool(losses) and all(math.isfinite(v) for v in losses),
        "grad_finite": grad_finite,
        "parameter_delta_max": delta,
        "parameter_device": next(model.parameters()).device.type,
    }
    return state, stats


def _refuse(code: int, reason: str) -> int:
    print(json.dumps({"status": "REFUSED" if code == 2 else "FAILED", "reason": reason}))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=contract.MAX_STEPS)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cuda:[0-9]+", args.device):
        return _refuse(2, "CUDA_DEVICE_REQUIRED_NO_CPU_FALLBACK")
    if not 1 <= args.steps <= contract.MAX_STEPS:
        return _refuse(2, "STEPS_OUT_OF_RANGE")
    out: Path = args.output_dir
    if out.exists() and any(out.iterdir()):
        return _refuse(2, "OUTPUT_DIR_NOT_EMPTY")
    try:
        import torch
    except ImportError:
        return _refuse(3, "TORCH_NOT_INSTALLED")
    if torch.version.cuda is None or not torch.cuda.is_available():
        return _refuse(3, "CUDA_UNAVAILABLE")
    device = torch.device(args.device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        return _refuse(3, "DEVICE_NOT_VISIBLE")
    properties = torch.cuda.get_device_properties(device)
    state, stats = run_training(torch, device, seed=args.seed, steps=args.steps, lr=args.lr)
    if not (
        stats["loss_finite"]
        and stats["grad_finite"]
        and stats["parameter_delta_max"] > 0
        and stats["parameter_device"] == "cuda"
    ):
        return _refuse(4, "TRAINING_CHECKS_FAILED")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / contract.CHECKPOINT_FILE
    torch.save(state, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    config_text = json.dumps(contract.MODEL_CONFIG, indent=2, sort_keys=True) + "\n"
    (out / contract.MODEL_CONFIG_FILE).write_text(config_text, encoding="utf-8")
    manifest = {
        "manifest_version": contract.MANIFEST_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cuda_trained": True,
        "example_only": True,
        "device": args.device,
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        **{
            k: stats[k]
            for k in (
                "steps",
                "optimizer",
                "lr",
                "loss_first",
                "loss_last",
                "loss_finite",
                "grad_finite",
                "parameter_delta_max",
            )
        },
        "dataset": {
            "kind": "synthetic",
            "classes": contract.MODEL_CONFIG["num_classes"],
            "per_class": 16,
            "labels": "synthetic toy classes, not expert TRL labels",
        },
        "feature_extractor": contract.MODEL_CONFIG["feature_extractor"],
        "checkpoint_file": contract.CHECKPOINT_FILE,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "checkpoint_bytes": len(checkpoint_bytes),
        "model_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
    }
    (out / contract.MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "TRAINED",
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "loss_first": stats["loss_first"],
                "loss_last": stats["loss_last"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
