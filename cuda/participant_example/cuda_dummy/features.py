"""Deterministic 16-dimensional hashed bag-of-words features from the case PDFs (no downloads)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

FEATURE_EXTRACTOR = "hashed-bag-of-words-sha256-v1"
DIM = 16
_TOKEN = re.compile(r"[a-z0-9]+")


def features_from_text(text: str) -> list[float]:
    tokens = _TOKEN.findall(text.lower())
    counts = [0] * DIM
    for token in tokens:
        counts[int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % DIM] += 1
    total = len(tokens)
    return [c / total for c in counts] if total else [0.0] * DIM


def extract_features(case: Mapping[str, Any]) -> list[float]:
    """Uses the text of every PDF page in packet order; nothing is selected or summarized away."""
    from participant_example.demo_rules import page_texts

    return features_from_text(" ".join(text for _, _, text in page_texts(case)))
