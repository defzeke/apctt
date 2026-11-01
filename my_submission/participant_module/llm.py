"""Optional SmolLM3 adapter refinement. Rules run first; this only nudges the TRL.

The base model is loaded from the read-only model store, the LoRA adapter from
``asset_root``. Anything that fails here is ignored, so the rules result stands.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

BASE_REL = "HuggingFaceTB/SmolLM3-3B/a07cc9a04f16550a088caea529712d1d335b0ac1"
BASE_REPO = "HuggingFaceTB/SmolLM3-3B"
ADAPTER_DIR = "smollm3-trl-lora"

SYSTEM = (
    "You are a technology-readiness assessor. Read the report excerpts and reply "
    "with one JSON object only: {\"trl\": <integer 1-9>, \"rationale\": \"<one sentence>\"}."
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _base_path() -> str:
    store = os.environ.get("APCTT_MODEL_STORE")
    if store:
        candidate = Path(store) / BASE_REL
        if candidate.is_dir():
            return str(candidate)
    return BASE_REPO


def build_prompt(case: dict[str, Any], excerpts: str) -> list[dict[str, str]]:
    target = case.get("assessment_target") or {}
    user = (
        f"Assessment target: {target.get('description', '')} "
        f"(type: {target.get('type', '')})\n"
        f"Intended application: {case.get('intended_application', '')}\n"
        f"Report: {case.get('title', '')}\n\n"
        f"Report excerpts:\n{excerpts}\n\n"
        "Which TRL does the evidence support?"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class LlmRefiner:
    def __init__(self, asset_root: Path) -> None:
        self._adapter = Path(asset_root) / ADAPTER_DIR
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def available(self) -> bool:
        return (self._adapter / "adapter_config.json").is_file()

    def load(self) -> bool:
        if self._model is not None:
            return True
        if not self.available():
            return False
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception:
            return False
        if not torch.cuda.is_available():
            return False
        base = _base_path()
        local = Path(base).is_dir()
        try:
            tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=local)
            model = AutoModelForCausalLM.from_pretrained(
                base,
                local_files_only=local,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation="sdpa",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
            model = PeftModel.from_pretrained(model, self._adapter).eval()
        except Exception:
            return False
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        return True

    def predict(self, case: dict[str, Any], excerpts: str,
                deadline: float) -> int | None:
        if not self.load() or time.monotonic() >= deadline:
            return None
        prompt = self._tokenizer.apply_chat_template(
            build_prompt(case, excerpts), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        try:
            import torch
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs, max_new_tokens=96, do_sample=False,
                    temperature=None, top_p=None,
                )
            text = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception:
            return None
        match = _JSON.search(text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0)).get("trl")
        except json.JSONDecodeError:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if 1 <= value <= 9 else None

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None
