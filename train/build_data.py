"""Build SFT records for SmolLM3 from the USGS sheet plus the public PDFs.

    python build_data.py --holdout-sdg 14

Inputs
    ../USGS_SDG_open_data_set_corrected.xlsx   the organizer's corrected sheet
    data/pdfs/<CASE_ID>.pdf                    from tools/get_public_cases.py --download

Outputs
    data/train.jsonl, data/val.jsonl           one JSON record per case

The target is the sheet's *estimated* TRL and its one-sentence rationale. Those are
weak labels, not expert gold: use them to sanity-check the model, and always evaluate
on a held-out SDG. The sheet has no per-criterion labels, so this file trains TRL
synthesis only.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--max-chars", type=int, default=4200)
    parser.add_argument("--max-pages", type=int, default=5)
    args = parser.parse_args()

    sheet = openpyxl.load_workbook(args.xlsx, data_only=True).active
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    args.out.mkdir(parents=True, exist_ok=True)

    train_path = (args.out / "train.jsonl").open("w", encoding="utf-8")
    val_path = (args.out / "val.jsonl").open("w", encoding="utf-8")
    counts = {"train": 0, "val": 0, "missing_pdf": 0, "excluded": 0}
    holdout = None if str(args.holdout_sdg).lower() == "none" else str(args.holdout_sdg)

    try:
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
                        {"trl": int(trl), "rationale": str(rationale)},
                        ensure_ascii=False)},
                ],
            }
            sdg_number = re.search(r"(\d+)", str(sdg))
            is_val = holdout is not None and sdg_number and sdg_number.group(1) == holdout
            stream = val_path if is_val else train_path
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["val" if is_val else "train"] += 1
    finally:
        train_path.close()
        val_path.close()

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
