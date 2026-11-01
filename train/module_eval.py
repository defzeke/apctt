"""Run the submission module over the 99 public cases and report part-A accuracy.

Development only; not shipped. Needs the kit sources on PYTHONPATH:

    PYTHONPATH=my_submission:cuda/src ~/.venvs/apctt-train/bin/python \
        train/module_eval.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

KIT = Path(__file__).resolve().parent.parent / "cuda"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "my_submission"))

from apctt_contracts.config import Checklist  # noqa: E402
from apctt_contracts.results import validate_result  # noqa: E402
from participant_module.adapter import AssessmentModule  # noqa: E402


def build_case(record: dict, pdf: Path, checklist_version: str) -> dict:
    document = record["document"]
    return {
        "schema_version": "apctt-pdf-example-v1",
        "input_profile": "pdf_report",
        "checklist_version": checklist_version,
        "case_id": record["case_id"],
        "assessment_as_of": None,
        "assessment_target": record["assessment_target"],
        "intended_application": record["intended_application"],
        "intended_environment": record["intended_environment"],
        "title": record["title"],
        "sdg_targets": record.get("sdg_targets") or [record["sdg"]],
        "documents": [{
            "document_id": document["document_id"],
            "title": document["title"],
            "document_date": document["document_date"],
            "language": "en",
            "format": "pdf",
            "sha256": record["pdf_sha256"],
            "path": str(pdf),
            "page_count": record["page_count"],
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--llm", action="store_true", help="enable the adapter refinement")
    parser.add_argument("--out", type=Path, default=Path("train/out/module-eval.json"))
    args = parser.parse_args()

    manifest = json.loads(
        (KIT / "contracts/public-cases.json").read_text(encoding="utf-8"))
    checklist_raw = json.loads(
        (KIT / "contracts/checklist.official.json").read_text(encoding="utf-8"))
    checklist = Checklist.model_validate(checklist_raw)
    pdf_dir = Path("train/data/pdfs")

    records = manifest["cases"]
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        records = [r for r in records if r["case_id"] in wanted]
    if args.limit:
        records = records[: args.limit]

    config = {
        "checklist": {
            "version": checklist.version,
            "criteria": [c.model_dump() for c in checklist.criteria],
        },
        "module": {"settings": {
            "use_llm": args.llm,
            "llm_min_remaining_ms": 60000,
        }},
        "compute": {"device": "cuda:0" if args.llm else "cpu", "seed": 42},
    }

    module = AssessmentModule()
    module.load(config, Path("my_submission/assets"))

    rows = []
    points = []
    for index, record in enumerate(records, start=1):
        pdf = pdf_dir / f"{record['case_id']}.pdf"
        if not pdf.is_file():
            print(f"missing {pdf}")
            continue
        case = build_case(record, pdf, checklist.version)
        context = SimpleNamespace(
            run_id="dev",
            seed=42,
            time_budget_ms=180000,
            scratch_dir=Path("train/out/scratch"),
            remaining_ms=lambda: 180000.0,
        )
        began = time.time()
        try:
            result = module.assess(case, context)
            issues = validate_result(result, case=case, checklist=checklist)
        except Exception:
            traceback.print_exc()
            continue
        module.reset_case()
        seconds = time.time() - began

        expected = int(record["reference_labels"]["estimated_trl"])
        predicted = result.get("predicted_trl")
        if result["assessment_status"] == "completed" and isinstance(predicted, int):
            delta = abs(predicted - expected)
            score = 1.0 if delta == 0 else (0.5 if delta == 1 else 0.0)
        else:
            delta = None
            score = 0.25
        points.append(score)
        met = sum(1 for value in result["criterion_status"].values() if value == "met")
        spans = sum(
            1 for items in result["evidence_reference"].values()
            for item in items if item.get("supporting_span")
        )
        rows.append({
            "case_id": record["case_id"],
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "met": met,
            "spans": spans,
            "status": result["assessment_status"],
            "issues": [f"{i.path}:{i.rule}" for i in issues],
            "seconds": round(seconds, 1),
        })
        flag = "" if not issues else f" ISSUES={len(issues)} {issues[0].rule}"
        print(f"{index:3d}/{len(records)} {record['case_id']} expected={expected} "
              f"predicted={predicted} score={score} met={met} spans={spans} "
              f"{seconds:.1f}s{flag}", flush=True)

    module.close()
    summary = {
        "records": len(rows),
        "mean_A": round(sum(points) / len(points), 4) if points else None,
        "exact": sum(1 for r in rows if r["score"] == 1.0),
        "off_by_one": sum(1 for r in rows if r["score"] == 0.5),
        "off_by_two_plus": sum(1 for r in rows if r["status"] == "completed" and r["score"] == 0.0),
        "abstained": sum(1 for r in rows if r["status"] != "completed"),
        "invalid": sum(1 for r in rows if r["issues"]),
        "met_mean": round(sum(r["met"] for r in rows) / len(rows), 2) if rows else None,
        "spans_mean": round(sum(r["spans"] for r in rows) / len(rows), 2) if rows else None,
        "seconds_mean": round(sum(r["seconds"] for r in rows) / len(rows), 1) if rows else None,
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
