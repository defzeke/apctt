"""Build SFT records for SmolLM3 from the USGS sheet plus the public PDFs.

    python build_data.py --holdout-sdg 14
    python build_data.py --holdout-sdg none --val-frac 0.2 --seed 42

Inputs
    ../USGS_SDG_open_data_set_corrected.xlsx   the organizer's corrected sheet
    data/pdfs/<CASE_ID>.pdf                    from tools/get_public_cases.py --download

Outputs
    data/train.jsonl, data/val.jsonl           one JSON record per case

The target is the sheet's *estimated* TRL and its one-sentence rationale. Those are
weak labels, not expert gold. Use an SDG holdout to measure extrapolation to an unseen
topic, or `--holdout-sdg none` for a stratified random split that keeps every TRL in
the training set. The sheet has no per-criterion labels, so this file trains TRL
synthesis only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import openpyxl
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
XLSX_DEFAULT = ROOT.parent / "USGS_SDG_open_data_set_corrected.xlsx"

TERMS = [
    "prototype", "demonstrat", "field", "laborator", "validat", "test",
    "deploy", "operat", "calibrat", "monitor", "qualif", "pilot",
    "production", "verif", "acceptance", "requirement", "measure",
    "proof of concept", "breadboard", "bench", "full-scale",
]

SYSTEM = (
    "You are a technology-readiness assessor. Read the report excerpts and reply "
    "with one JSON object only: {\"trl\": <integer 1-9>, \"rationale\": \"<one sentence>\"}."
)


def page_texts(pdf_path: Path) -> list[str]:
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        return [page.get_textpage().get_text_range() for page in doc]


def page_score(text: str) -> int:
    low = text.lower()
    return sum(low.count(term) for term in TERMS)


def pick_pages(texts: list[str], max_chars: int, max_pages: int) -> str:
    ranked = sorted(range(len(texts)), key=lambda i: page_score(texts[i]), reverse=True)
    chosen = sorted(ranked[:max_pages])
    if not chosen:
        return ""
    per_page = max(200, max_chars // len(chosen))
    parts = []
    for index in chosen:
        snippet = " ".join(texts[index].split())[:per_page]
        parts.append(f"[page {index + 1}] {snippet}")
    return "\n\n".join(parts)


def build_user_text(row: dict, excerpts: str) -> str:
    return (
        f"Assessment target: {row['unit']} (type: {row['category']})\n"
        f"Intended application: {row['sdg_target']}\n"
        f"Report: {row['title']} ({row['year']})\n\n"
        f"Report excerpts:\n{excerpts}\n\n"
        "Which TRL does the evidence support?"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=XLSX_DEFAULT)
    parser.add_argument("--pdfs", type=Path, default=ROOT / "data" / "pdfs")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--holdout-sdg", default="14",
                        help="SDG number held out for validation, or 'none'")
    parser.add_argument("--max-chars", type=int, default=2400)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="validation share when --holdout-sdg none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sheet = openpyxl.load_workbook(args.xlsx, data_only=True).active
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    args.out.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "missing_pdf": 0, "excluded": 0}
    holdout = None if str(args.holdout_sdg).lower() == "none" else str(args.holdout_sdg)
    collected = []

    for row_values in rows:
        (no, sdg, sdg_target, category, title, year, _pages, _url, _pdf_url,
         trl, unit, rationale, status) = row_values
        match = re.search(r"USGS-SDG-(\d+)", str(status))
        if not match or trl is None:
            counts["excluded"] += 1
            continue
        case_id = f"USGS-SDG-{match.group(1)}"
        pdf_path = args.pdfs / f"{case_id}.pdf"
        if not pdf_path.exists():
            counts["missing_pdf"] += 1
            continue

        row = {
            "sdg": str(sdg), "sdg_target": str(sdg_target),
            "category": str(category), "title": str(title),
            "year": str(year), "unit": str(unit),
        }
        excerpts = pick_pages(page_texts(pdf_path), args.max_chars, args.max_pages)
        record = {
            "case_id": case_id,
            "sdg": str(sdg),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_text(row, excerpts)},
                {"role": "assistant", "content": json.dumps(
                    {"trl": int(trl), "rationale": str(rationale)[:200]},
                    ensure_ascii=False)},
            ],
        }
        collected.append((str(sdg), int(trl), record))

    if holdout is not None:
        train_records = [item for item in collected if item[0] != holdout]
        val_records = [item for item in collected if item[0] == holdout]
    else:
        train_records, val_records = [], []
        by_trl: dict[int, list] = {}
        for item in collected:
            by_trl.setdefault(item[1], []).append(item)
        rng = random.Random(args.seed)
        for trl in sorted(by_trl):
            items = by_trl[trl]
            rng.shuffle(items)
            if len(items) == 1:
                train_records.extend(items)
                continue
            take = min(max(int(round(len(items) * args.val_frac)), 1), len(items) - 1)
            val_records.extend(items[:take])
            train_records.extend(items[take:])
        train_records.sort(key=lambda item: item[2]["case_id"])
        val_records.sort(key=lambda item: item[2]["case_id"])

    with (args.out / "train.jsonl").open("w", encoding="utf-8") as train_path, \
            (args.out / "val.jsonl").open("w", encoding="utf-8") as val_path:
        for _sdg, _trl, record in train_records:
            train_path.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["train"] += 1
        for _sdg, _trl, record in val_records:
            val_path.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["val"] += 1

    counts["train_by_trl"] = {}
    counts["val_by_trl"] = {}
    for _sdg, trl, _record in train_records:
        key = str(trl)
        counts["train_by_trl"][key] = counts["train_by_trl"].get(key, 0) + 1
    for _sdg, trl, _record in val_records:
        key = str(trl)
        counts["val_by_trl"][key] = counts["val_by_trl"].get(key, 0) + 1

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
