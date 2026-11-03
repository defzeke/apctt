"""One-case check of the criterion-evidence ranker against the shipped module (parts B/C).

Development only. Run in WSL from the repository root:

    ~/.venvs/apctt-train/bin/python train/ranker_check.py --case USGS-SDG-076
    ~/.venvs/apctt-train/bin/python train/ranker_check.py --case USGS-SDG-012,USGS-SDG-040 --device cpu

For each criterion it prints the module's status and first cited page, then the ranker's
top sentences among the rule candidates (the pool a run-time rerank would score; ``--all``
scores every sentence instead). ``*`` marks the page that best matches the organizer
rationale, the only label the rules did not make. The summary counts, per case, how often
each side puts that page first for the criterion at the organizer's TRL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_criteria_data import rationale_hits  # noqa: E402
from module_eval import build_case  # noqa: E402  (puts my_submission on sys.path)
from participant_module import rules  # noqa: E402
from participant_module.adapter import AssessmentModule  # noqa: E402
from participant_module.extract import extract  # noqa: E402
from train_ranker import score  # noqa: E402

KIT = ROOT / "cuda" / "contracts"


def load_ranker(path: Path, device: str = "auto"):
    """(model, tokenizer, device) for a trained ranker folder."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (Path(path) / "config.json").is_file():
        raise SystemExit(f"{path} has no trained model; run train/train_ranker.py first")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
    return model, tokenizer, device


def rank_criteria(model, tokenizer, device, sentences, assignments, descriptions, targets,
                  use_all: bool = False) -> dict[str, list[tuple[float, int]]]:
    """criterion id -> [(p(support), sentence index)], best first, for rule-backed criteria."""
    ranked: dict[str, list[tuple[float, int]]] = {}
    for cid, rule in assignments.items():
        if rule is None or rule.key == "target":
            continue  # C12 compares cited evidence with the target; the ranker has no say
        pool = (list(range(len(sentences))) if use_all else
                [c.index for c in rules._candidates(sentences, rule, targets)])
        rows = [{"query": descriptions[cid], "text": sentences[i].text} for i in pool]
        logits = score(model, tokenizer, rows, device, 64, 256) if rows else []
        ranked[cid] = [(1 / (1 + math.exp(-logit)), index)
                       for logit, index in sorted(zip(logits, pool), reverse=True)]
    return ranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", required=True, help="case id(s), comma-separated")
    parser.add_argument("--model", type=Path, default=ROOT / "train/out/criteria-ranker")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="score every sentence, not only rule candidates")
    args = parser.parse_args(argv)

    model, tokenizer, device = load_ranker(args.model, args.device)
    manifest = {c["case_id"]: c for c in
                json.loads((KIT / "public-cases.json").read_text(encoding="utf-8"))["cases"]}
    checklist = json.loads((KIT / "checklist.official.json").read_text(encoding="utf-8"))
    descriptions = {c["criterion_id"]: c["description"] for c in checklist["criteria"]}
    module = AssessmentModule()
    module.load({"checklist": checklist, "module": {"settings": {}}}, ROOT / "my_submission/assets")
    assignments = module._assignments
    totals = {"cases": 0, "module_first": 0, "ranker_first": 0}

    for case_id in [part.strip() for part in args.case.split(",") if part.strip()]:
        record = manifest.get(case_id)
        pdf = ROOT / "train/data/pdfs" / f"{case_id}.pdf"
        if record is None or not pdf.is_file():
            print(f"{case_id}: not in the manifest or PDF missing")
            continue
        case = build_case(record, pdf, checklist["version"])
        result = module.assess(case, SimpleNamespace(remaining_ms=lambda: 180000.0))
        sentences, _pages, _docs = extract(case, time.monotonic() + 1e9, 300)
        label = int(record["reference_labels"]["estimated_trl"])
        hits = rationale_hits(sentences, record["reference_labels"]["rationale"], 8)
        gold_page = sentences[hits[0]].page if hits else None
        targets = rules._target_terms(case)

        print(f"\n=== {case_id}  organizer TRL {label}  module TRL {result['predicted_trl']}  "
              f"rationale page {gold_page}")
        print(f"    rationale: {record['reference_labels']['rationale'][:160]}")
        began = time.time()
        all_ranked = rank_criteria(model, tokenizer, device, sentences, assignments,
                                   descriptions, targets, args.all)
        pairs_scored = sum(len(v) for v in all_ranked.values())
        for cid, ranked in all_ranked.items():
            rule = assignments[cid]
            cited = result["evidence_reference"][cid]
            module_page = cited[0]["page"] if cited else None
            status = result["criterion_status"][cid]
            best = ranked[0][0] if ranked else 0.0
            mark = lambda page: "*" if page == gold_page else " "  # noqa: E731
            print(f"  {cid} ({rule.key:<10}) module={status:<21} p{module_page}{mark(module_page)}"
                  f"  ranker max p(support)={best:.2f}")
            for prob, index in ranked[:args.top]:
                s = sentences[index]
                print(f"      {prob:.2f} p{s.page}{mark(s.page)} {s.text[:110]}")
            if rule.level == label and gold_page is not None:
                totals["cases"] += 1
                totals["module_first"] += module_page == gold_page and status == "met"
                totals["ranker_first"] += bool(ranked) and sentences[ranked[0][1]].page == gold_page
        seconds = time.time() - began
        print(f"  ranker: {pairs_scored} pairs in {seconds:.1f} s on {device}")

    print("\n" + json.dumps({"rationale_page_first_at_label_level": totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
