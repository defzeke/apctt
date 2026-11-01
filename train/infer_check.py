"""Smoke-test the adapter on one held-out record, offline, before packaging.

    python infer_check.py --adapter ../my_submission/assets/smollm3-trl-lora

Loads the base from APCTT_MODEL_STORE when the folder exists (the MCP layout the
event uses), else from a local --base path or the HF repo id. Prints the generation,
the elapsed time, peak VRAM and whether the JSON parsed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

BASE_REL = "HuggingFaceTB/SmolLM3-3B/a07cc9a04f16550a088caea529712d1d335b0ac1"


def resolve_base(explicit: str | None) -> str:
    store = os.environ.get("APCTT_MODEL_STORE")
    if store and (Path(store) / BASE_REL).is_dir():
        return str(Path(store) / BASE_REL)
    return explicit or "HuggingFaceTB/SmolLM3-3B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/val.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    base = resolve_base(args.base)
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        print(json.dumps({"status": "NO_DATA", "file": str(args.data)}))
        return 2
    record = records[args.index % len(records)]

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=Path(base).is_dir())
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=Path(base).is_dir(), dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    prompt = tokenizer.apply_chat_template(
        record["messages"][:-1], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=False, temperature=None, top_p=None)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

    expected = json.loads(record["messages"][-1]["content"])
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
    print(json.dumps({
        "case_id": record.get("case_id"),
        "expected": expected,
        "generated": text,
        "parsed": parsed,
        "parsed_ok": parsed is not None,
        "trl_match": bool(parsed and parsed.get("trl") == expected.get("trl")),
        "seconds": round(elapsed, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
