"""Fine-tune the catalogue cross-encoder as a criterion-evidence scorer (parts B and C).

Development only; not shipped until it beats the rules. Run in WSL from the repository root:

    ~/.venvs/apctt-train/bin/python train/train_ranker.py   # --base defaults to the store copy

Input is ``train/build_criteria_data.py`` output. The model scores (criterion description,
sentence) pairs: a high score means the sentence supports the criterion. One model serves
part B (which criteria are met, which pages come first) and part C (which sentence to quote).

The evaluation compares the model with the rules on the held-out reports:

- ``pairs``: AUC and accuracy on the silver validation pairs, also per label source;
- ``pool``: for each held-out report, every sentence is scored against the criterion of the
  organizer's TRL; ``hit@3`` is the share of reports whose rationale-matched page is among
  the first 3 distinct pages, the same cut part B uses. ``rules`` ranks by the rule order.

The labels are silver (see build_criteria_data.py); treat the numbers as a direction, not
as the competition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_ID = "ms-marco-minilm-l6-v2"
BASE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
# Where tools/get_models.py --store ~/models/store puts it (APCTT_MODEL_STORE wins if set).
DEFAULT_BASE = (Path(os.environ.get("APCTT_MODEL_STORE", "~/models/store")).expanduser()
                / "cross-encoder/ms-marco-MiniLM-L6-v2" / BASE_REVISION)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney); ties broken by order, fine for a direction."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank + 1 for rank, i in enumerate(order) if labels[i])
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 4)


def hit_at_3(rows: list[dict], key) -> bool:
    """True when a positive row's page is among the first 3 distinct pages ranked by key."""
    gold = {r["page"] for r in rows if r["label"]}
    pages: list[int] = []
    for r in sorted(rows, key=key):
        if r["page"] not in pages:
            pages.append(r["page"])
        if len(pages) == 3:
            break
    return bool(gold & set(pages))


def score(model, tokenizer, rows: list[dict], device: str, batch: int, max_len: int) -> list[float]:
    import torch

    out: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            inputs = tokenizer([r["query"] for r in chunk], [r["text"] for r in chunk],
                               truncation=True, max_length=max_len, padding=True,
                               return_tensors="pt").to(device)
            out.extend(model(**inputs).logits.squeeze(-1).float().cpu().tolist())
    return out


def evaluate(model, tokenizer, data: Path, device: str, batch: int, max_len: int) -> dict:
    report: dict = {}
    val = read_jsonl(data / "val.jsonl")
    if val:
        began = time.time()
        scores = score(model, tokenizer, val, device, batch, max_len)
        seconds = time.time() - began
        labels = [r["label"] for r in val]
        by_source: dict[str, list[int]] = defaultdict(list)
        for index, r in enumerate(val):
            by_source[r["source"]].append(index)
        report["pairs"] = {
            "records": len(val),
            "auc": auc(labels, scores),
            "accuracy_at_0": round(sum((s > 0) == bool(l) for s, l in zip(scores, labels)) / len(val), 4),
            "accuracy_by_source": {
                source: round(sum((scores[i] > 0) == bool(labels[i]) for i in idx) / len(idx), 4)
                for source, idx in sorted(by_source.items())
            },
            "pairs_per_second": round(len(val) / max(seconds, 1e-6), 1),
            "device": device,
        }
    pool = read_jsonl(data / "val_pool.jsonl")
    if pool:
        scores = score(model, tokenizer, pool, device, batch, max_len)
        groups: dict[str, list[dict]] = defaultdict(list)
        for r, s in zip(pool, scores):
            groups[r["group"]].append({**r, "score": s})
        model_hits = [hit_at_3(rows, lambda r: -r["score"]) for rows in groups.values()]
        # The rules only cite their own candidates; rank 999 means "never surfaced".
        rule_hits = [hit_at_3([r for r in rows if r["rule_rank"] < 999], lambda r: r["rule_rank"])
                     for rows in groups.values()]
        report["pool"] = {
            "reports": len(groups),
            "model_hit_at_3": round(sum(model_hits) / len(groups), 4),
            "rules_hit_at_3": round(sum(rule_hits) / len(groups), 4),
        }
    return report


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def self_test() -> int:
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auc([1, 1], [0.1, 0.2]) is None
    rows = [{"page": p, "label": int(p == 7), "s": s} for p, s in
            [(1, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (7, 0.5)]]
    assert not hit_at_3(rows, lambda r: -r["s"])  # page 7 is the 4th distinct page
    rows[-1]["s"] = 0.65
    assert hit_at_3(rows, lambda r: -r["s"])
    print("self-test ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default=str(DEFAULT_BASE),
                        help="local model folder (default: the catalogue cross-encoder in the store)")
    parser.add_argument("--data", type=Path, default=ROOT / "train/out/criteria-data")
    parser.add_argument("--out", type=Path, default=ROOT / "train/out/criteria-ranker")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluate --base as is (no training): the untrained baseline")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.base or not Path(args.base).expanduser().is_dir():
        parser.error(f"--base {args.base!r} is not a folder; download it with "
                     "tools/get_models.py (see train/README.md)")
    base = str(Path(args.base).expanduser())

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, local_files_only=True).to(device)

    if args.eval_only:
        print(json.dumps({"status": "BASELINE", "base": base,
                          **evaluate(model, tokenizer, args.data, device, 64, args.max_len)}, indent=2))
        return 0

    train = read_jsonl(args.data / "train.jsonl")
    if not train:
        raise SystemExit(f"{args.data / 'train.jsonl'} is missing or empty; run build_criteria_data.py")
    steps = math.ceil(len(train) / args.batch) * math.ceil(args.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(json.dumps({"status": "STARTING", "device": device, "pairs": len(train),
                      "positives": sum(r["label"] for r in train), "steps": steps}), flush=True)

    started = time.time()
    step = 0
    for epoch in range(math.ceil(args.epochs)):
        model.train()
        random.shuffle(train)
        total = 0.0
        for start in range(0, len(train), args.batch):
            chunk = train[start:start + args.batch]
            inputs = tokenizer([r["query"] for r in chunk], [r["text"] for r in chunk],
                               truncation=True, max_length=args.max_len, padding=True,
                               return_tensors="pt").to(device)
            labels = torch.tensor([float(r["label"]) for r in chunk], device=device)
            weights = torch.tensor([float(r["weight"]) for r in chunk], device=device)
            logits = model(**inputs).logits.squeeze(-1)
            loss = (loss_fn(logits, labels) * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            total += loss.item()
            step += 1
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": round(loss.item(), 4)}),
                      flush=True)
        print(json.dumps({"epoch": epoch + 1,
                          "mean_loss": round(total / math.ceil(len(train) / args.batch), 4)}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    result = evaluate(model, tokenizer, args.data, device, 64, args.max_len)
    manifest = {
        "manifest_version": "train-manifest-v1",
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "task": "criterion-evidence scorer: (criterion description, sentence) -> support logit",
        "dataset_sha256": file_sha256(args.data / "train.jsonl"),
        "pairs": len(train),
        "epochs": args.epochs, "learning_rate": args.lr, "batch": args.batch,
        "max_len": args.max_len, "seed": args.seed, "device": device,
        "torch": torch.__version__, "python": sys.version.split()[0],
        "seconds": round(time.time() - started, 1),
        "labels": "silver: shipped rules + organizer rationale alignment; not expert gold",
        "eval": result,
    }
    (args.out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({"status": "TRAINED", "out": str(args.out), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
