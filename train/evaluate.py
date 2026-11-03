"""Score the adapter on a JSONL split: exact TRL, off-by-one, invalid JSON.

    python evaluate.py --adapter ../my_submission/assets/smollm3-trl-lora \
        --data data/val.jsonl

Mirrors the competition's part-A scoring when the experts give a TRL:
exact = 1.0, off by one = 0.5, off by two or more = 0.0. A failed parse counts
as 0 and is reported separately, because a contract-invalid result also loses
part B. Loads the base and adapter once, then loops over the split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
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


def parse_trl(text: str) -> int | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0)).get("trl")
    except json.JSONDecodeError:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_rationale(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return ""
    try:
        value = json.loads(match.group(0)).get("rationale", "")
    except json.JSONDecodeError:
        return ""
    return str(value or "")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def token_f1(expected: str, generated: str) -> float:
    """Bag-of-words F1 between the expected and generated rationale."""
    left, right = _words(expected), _words(generated)
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(right), overlap / len(left)
    return round(2 * precision * recall / (precision + recall), 3)


def number_recall(expected: str, generated: str) -> float:
    wanted = _numbers(expected)
    if not wanted:
        return 1.0
    return round(len(wanted & _numbers(generated)) / len(wanted), 3)


def grounded_share(generated: str, source: str) -> float:
    """Share of generated 5-grams that appear verbatim in the excerpt."""
    words = _words(generated)
    grams = {" ".join(words[index:index + 5]) for index in range(len(words) - 4)}
    if not grams:
        return 0.0
    haystack = " " + " ".join(_words(source)) + " "
    return round(sum(1 for gram in grams if gram in haystack) / len(grams), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/val.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all records")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base = resolve_base(args.base)
    records = [json.loads(line) for line in
               args.data.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(json.dumps({"status": "NO_DATA", "file": str(args.data)}))
        return 2

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=Path(base).is_dir())
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=Path(base).is_dir(), dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    torch.cuda.reset_peak_memory_stats()

    rows, points, started = [], [], time.time()
    for record in records:
        prompt = tokenizer.apply_chat_template(
            record["messages"][:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        begin = time.time()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, temperature=None, top_p=None)
        torch.cuda.synchronize()
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        expected = int(json.loads(record["messages"][-1]["content"])["trl"])
        predicted = parse_trl(text)
        delta = None if predicted is None else abs(predicted - expected)
        score = 1.0 if delta == 0 else (0.5 if delta == 1 else 0.0)
        points.append(score)
        target = json.loads(record["messages"][-1]["content"])
        reason = parse_rationale(text)
        rows.append({"case_id": record.get("case_id"), "expected": expected,
                     "predicted": predicted, "delta": delta, "score": score,
                     "rationale_f1": token_f1(str(target.get("rationale", "")), reason),
                     "number_recall": number_recall(str(target.get("rationale", "")), reason),
                     "grounded_5gram": grounded_share(
                         reason, record["messages"][1]["content"]),
                     "seconds": round(time.time() - begin, 1), "raw": text})
        print(f"{record.get('case_id')} expected={expected} predicted={predicted} "
              f"score={score} f1={rows[-1]['rationale_f1']} "
              f"grounded={rows[-1]['grounded_5gram']}", flush=True)

    def mean(key: str) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 3)

    summary = {
        "records": len(rows),
        "mean_A_score": round(sum(points) / len(points), 4),
        "exact": sum(1 for row in rows if row["delta"] == 0),
        "off_by_one": sum(1 for row in rows if row["delta"] == 1),
        "off_by_two_or_more": sum(1 for row in rows if row["delta"] is not None and row["delta"] >= 2),
        "invalid_output": sum(1 for row in rows if row["predicted"] is None),
        "rationale_token_f1": mean("rationale_f1"),
        "rationale_number_recall": mean("number_recall"),
        "rationale_grounded_5gram": mean("grounded_5gram"),
        "generation_seconds_mean": round(sum(row["seconds"] for row in rows) / len(rows), 1),
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
