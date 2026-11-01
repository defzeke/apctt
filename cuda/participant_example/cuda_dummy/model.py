"""Toy MLP 16 → 32 (ReLU) → 9, FP32. Requires torch in the worker environment."""

from __future__ import annotations

from typing import Any


def build_model(torch: Any) -> Any:
    nn = torch.nn
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 9))
