"""Silver (criterion, sentence) pairs for parts B and C: does this sentence support this criterion?

Development only; not shipped. Run in WSL from the repository root:

    ~/.venvs/apctt-train/bin/python train/build_criteria_data.py

There are no expert criterion labels, so the labels are silver, from two sources:

- the organizer's one-sentence rationale: the report sentence that best matches it is a
  positive for the criterion at the labelled TRL (the only signal that is not our rules);
- the shipped rules (``participant_module/rules.py``): full matches at or below the
  labelled TRL, and for the non-level criteria, are positives; matches two or more levels
  above the label, weak partial matches, bare "TRL n" assertions and random sentences of
  the same report are negatives. Matches one level above the label are left out.

Sentences come from ``participant_module/extract.py``, so training text is exactly the
text the module sees at run time. Criterion text is the official description, never the ID.

Writes ``train.jsonl`` / ``val.jsonl`` (pairs), ``val_pool.jsonl`` (every sentence of each
validation report for its label-level criterion, to measure page retrieval) and
``report.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "my_submission"))

from participant_module import rules  # noqa: E402
from participant_module.extract import extract  # noqa: E402
from pipeline import _NUMBERS, _tokens  # noqa: E402

KIT = ROOT / "cuda" / "contracts"


def rationale_hits(sentences, rationale: str, min_score: int, keep: int = 1) -> list[int]:
    # keep=1: on the smoke cases the 2nd-best match was a table row or map legend.
    """Sentences that best match the organizer rationale (lexical overlap), distinct pages."""
    wanted, numbers = _tokens(rationale), set(_NUMBERS.findall(rationale))
    scored = []
    for index, sentence in enumerate(sentences):
        if ".." in sentence.text:
            continue  # table-of-contents dot leaders, not prose
        score = (2 * len(_tokens(sentence.text) & wanted)
                 + 4 * len(set(_NUMBERS.findall(sentence.text)) & numbers))
        if score >= min_score:
            scored.append((-score, index))
    hits: list[int] = []
    for _score, index in sorted(scored):
        if all(sentences[index].page != sentences[h].page for h in hits):
            hits.append(index)
        if len(hits) == keep:
            break
    return hits


def row(case: dict, cid: str, rule: rules.Rule, query: str, sentence, rank: int,
        label: int, source: str, weight: float = 1.0) -> dict:
    return {
        "group": f"{case['case_id']}|{cid}", "case_id": case["case_id"],
        "sdg": case["sdg"], "criterion_id": cid, "rule": rule.key, "query": query,
        "text": sentence.text, "page": sentence.page, "rule_rank": rank,
        "label": label, "weight": weight, "source": source,
    }


def case_rows(case: dict, record: dict, criteria: list[dict], assignments: dict,
              rng: random.Random, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    sentences, _pages, _docs = extract(case, time.monotonic() + 1e9, 300)
    if not sentences:
        return [], []
    label = int(record["reference_labels"]["estimated_trl"])
    hits = rationale_hits(sentences, record["reference_labels"]["rationale"], args.min_overlap)
    targets = rules._target_terms(case)
    descriptions = {str(c["criterion_id"]): str(c["description"]) for c in criteria}
    pairs: list[dict] = []
    pool: list[dict] = []

    for cid, rule in assignments.items():
        if rule is None or rule.key == "target":
            continue  # C12 compares cited evidence with the target; it keeps its rule
        query = descriptions[cid]
        candidates = rules._candidates(sentences, rule, targets)
        rank_of = {c.index: rank for rank, c in enumerate(candidates)}
        total = len(rule.groups)
        level = rule.level
        used: set[int] = set()

        def add(index: int, value: int, source: str, weight: float = 1.0) -> None:
            if index in used:
                return
            used.add(index)
            pairs.append(row(case, cid, rule, query, sentences[index],
                             rank_of.get(index, 999), value, source, weight))

        if level == label and hits:
            for index in hits:
                add(index, 1, "rationale", args.rationale_weight)
            pool.extend(row(case, cid, rule, query, s, rank_of.get(i, 999),
                            int(i in hits), "pool") for i, s in enumerate(sentences))

        # Same cut as rules._decide: a full match in the sentence or its window is "met".
        full = [c for c in candidates
                if total > 0 and (c.own_groups == total or c.window_groups == total)]
        partial = [c for c in candidates if c.own_groups <= total - 2]
        if level is not None and level >= label + 2:
            for c in full[:args.max_negatives]:
                add(c.index, 0, "above_label")
        elif level is None or level <= label:
            pages: set[int] = set()
            for c in full:
                if c.sentence.page in pages or rules._TRL_ASSERT.search(c.sentence.text):
                    continue
                pages.add(c.sentence.page)
                add(c.index, 1, "rule")
                if len(pages) >= args.max_positives:
                    break
        for c in partial[:args.max_negatives]:
            add(c.index, 0, "rule_partial")
        for index, sentence in enumerate(sentences):
            if rules._TRL_ASSERT.search(sentence.text):
                add(index, 0, "trl_assert")  # scoring rule: a bare TRL claim is never evidence
        others = [i for i in range(len(sentences)) if i not in rank_of and i not in hits]
        for index in rng.sample(others, min(args.random_negatives, len(others))):
            add(index, 0, "random")
    return pairs, pool


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdfs", type=Path, default=ROOT / "train/data/pdfs")
    parser.add_argument("--out", type=Path, default=ROOT / "train/out/criteria-data")
    parser.add_argument("--holdout-sdg", default="14", help="SDG held out for validation, or 'none'")
    parser.add_argument("--only", default="", help="comma-separated case ids (smoke runs)")
    parser.add_argument("--min-overlap", type=int, default=8,
                        help="minimum rationale overlap score (2 per word, 4 per number)")
    parser.add_argument("--rationale-weight", type=float, default=2.0)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument("--random-negatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    manifest = json.loads((KIT / "public-cases.json").read_text(encoding="utf-8"))
    checklist = json.loads((KIT / "checklist.official.json").read_text(encoding="utf-8"))
    criteria = list(checklist["criteria"])
    assignments = rules.match_rules(criteria)
    holdout = None if args.holdout_sdg.lower() == "none" else int(args.holdout_sdg)
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    rng = random.Random(args.seed)

    train, val, pool, missing, no_hit = [], [], [], [], []
    for record in manifest["cases"]:
        if wanted and record["case_id"] not in wanted:
            continue
        pdf = args.pdfs / f"{record['case_id']}.pdf"
        if not pdf.is_file():
            missing.append(record["case_id"])
            continue
        # Only the fields extract() and the rules read; module_eval.build_case needs pydantic.
        case = {"case_id": record["case_id"], "sdg": record["sdg"],
                "assessment_target": record["assessment_target"],
                "documents": [{"document_id": record["document"]["document_id"],
                               "path": str(pdf)}]}
        pairs, case_pool = case_rows(case, record, criteria, assignments, rng, args)
        if not any(p["source"] == "rationale" for p in pairs):
            no_hit.append(record["case_id"])
        is_val = holdout is not None and int(record["sdg"]) == holdout
        (val if is_val else train).extend(pairs)
        if is_val:
            pool.extend(case_pool)
        print(f"{record['case_id']} sdg={record['sdg']} pairs={len(pairs)} "
              f"pos={sum(p['label'] for p in pairs)} {'val' if is_val else 'train'}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "val_pool.jsonl", pool)
    report = {
        "holdout_sdg": holdout,
        "train_pairs": len(train), "val_pairs": len(val), "val_pool_rows": len(pool),
        "train_positive": sum(p["label"] for p in train),
        "val_positive": sum(p["label"] for p in val),
        "train_by_source": dict(Counter(f"{p['source']}:{p['label']}" for p in train)),
        "val_by_source": dict(Counter(f"{p['source']}:{p['label']}" for p in val)),
        "cases_without_rationale_match": no_hit,
        "missing_pdfs": missing,
        "labels": "silver: shipped rules + organizer rationale alignment; not expert gold",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if train else 1


if __name__ == "__main__":
    raise SystemExit(main())
