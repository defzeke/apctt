"""End-to-end training pipeline: recursive PDFs -> JSONL -> QLoRA adapter.

Development only. The submission never ships this file, the PDFs or the dataset.

    # from the repository root, in the training venv
    ~/.venvs/apctt-train/bin/python train/pipeline.py --dry-run
    ~/.venvs/apctt-train/bin/python train/pipeline.py \
        --adapter my_submission/assets/smollm3-trl-lora

Stage 1 (extract) walks ``--pdfs`` recursively, matches every PDF to a public
case by SHA-256 (falling back to ``USGS-SDG-NNN`` in the file name), extracts
page text with pypdfium2, removes running headers/footers and boilerplate, keeps
the highest-scoring pages and the best sentence window on each page, and writes
chat-format ``train.jsonl`` / ``val.jsonl`` plus ``dataset_report.json``.

Targets default to ``--target extractive``: the answer is the verbatim sentence
that best supports the organiser rationale, so the model learns to quote instead
of inventing a paraphrase. Training excerpts are anchored on that sentence;
validation excerpts stay label-free, and ``val-oracle.jsonl`` repeats validation
with the target forced into the excerpt to separate quoting from retrieval.

Stage 2 (train) runs ``train/train_qlora.py`` twice: a token-budget dry run, then
the QLoRA fine-tune (4-bit NF4, LoRA). The adapter goes to
``train/out/pipeline-adapter`` by default; pass the shipped path only when you
intend to replace the submission adapter, then repack and refresh the hashes
(``SKILLS.md``).

The prompt is the module prompt in ``participant_module/llm.py:build_prompt``,
so a new adapter sees the same text at training and inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "cuda" / "contracts" / "public-cases.json"
DEFAULT_PDFS = ROOT / "train" / "data" / "pdfs"
DEFAULT_OUT = ROOT / "train" / "out" / "pipeline-data"
DEFAULT_ADAPTER = ROOT / "train" / "out" / "pipeline-adapter"
TRAINER = ROOT / "train" / "train_qlora.py"

SYSTEM = (
    "You are a technology-readiness assessor. Read the report excerpts and reply "
    "with one JSON object only: {\"trl\": <integer 1-9>, \"rationale\": \"<one sentence>\"}."
)

# Same vocabulary as build_data.py: pages and sentences that describe testing,
# deployment and results are the ones a TRL label can be read from.
TERMS = [
    "prototype", "demonstrat", "field", "laborator", "validat", "test",
    "deploy", "operat", "calibrat", "monitor", "qualif", "pilot",
    "production", "verif", "acceptance", "requirement", "measure",
    "proof of concept", "breadboard", "bench", "full-scale",
]

# Results and comparison language: the organiser rationales quote numbers,
# agreement and error, so a page without these cannot support the label.
EVIDENCE_TERMS = [
    "agreement", "disagreement", "difference", "accuracy", "accurate",
    "compar", "measur", "survey", "assess", "evaluat", "result", "conclusion",
    "error", "bias", "uncertainty", "resolution", "percent", "match",
    "consistent", "observ", "estimate", "dataset", "sample", "coverage",
]

_NUMBERS = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_WORDS = re.compile(r"[a-z]{5,}|\d+")
# Reference lists steal excerpt slots: pages list DOIs and author-year entries.
_CITATION = re.compile(r"doi\.org|https?://|U\.S\. Geological Survey, \d{4}|\(\d{4}\)")


_SPACES = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'{])")
_PAGE_NUMBER = re.compile(r"^[\divxlc]{1,5}$", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(?:suggested citation|references cited|list of (?:figures|tables)|"
    r"conversion factors|abbreviations|acknowledg|publishing service center|"
    r"manuscript approved|edited by|layout and design|for navigation or regulatory|"
    r"any use of trade|permission to reproduce|public domain)",
    re.IGNORECASE,
)
_CASE_ID = re.compile(r"USGS-SDG-(\d+)", re.IGNORECASE)
# pypdfium2 emits U+FFFE/U+FFFD for unmapped glyphs and keeps stray control
# characters; they break words ("Lexing\ufffeton") when they should just vanish.
_GLYPH_ARTIFACTS = re.compile("[\u0000-\u0008\ufeff\u000b\u000c\u000e-\u001f\ufffd\ufffe]")


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def clean_page(text: str) -> str:
    """Drop running headers/footers, page numbers and licence boilerplate."""
    text = _GLYPH_ARTIFACTS.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 8:
        lines = lines[2:-2]
    kept = [
        line for line in lines
        if not _PAGE_NUMBER.match(line) and not _BOILERPLATE.search(line)
    ]
    return normalize(" ".join(kept))


def _tokens(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower()))


def page_score(text: str) -> int:
    low = text.lower()
    score = 2 * sum(low.count(term) for term in TERMS)
    score += sum(low.count(term) for term in EVIDENCE_TERMS)
    score += min(len(_NUMBERS.findall(low)), 20)  # measured values matter
    if len(_CITATION.findall(text)) >= 3:
        score -= 40  # likely a reference list, not results
    return score


def snippet_score(text: str, anchor: set[str] | None) -> int:
    score = page_score(text)
    if anchor:
        score += 8 * len(_tokens(text) & anchor)
    return score


def best_snippet(text: str, budget: int, anchor: set[str] | None = None) -> str:
    """The highest-scoring sentence plus neighbours, within ``budget`` chars."""
    sentences = [part.strip() for part in _BOUNDARY.split(text) if part.strip()]
    if not sentences:
        return text[:budget]
    scores = [snippet_score(sentence, anchor) for sentence in sentences]
    centre = max(range(len(sentences)), key=lambda index: (scores[index], -index))
    start = end = centre
    size = len(sentences[centre])
    while True:
        options = []
        if start > 0 and size + 1 + len(sentences[start - 1]) <= budget:
            options.append((scores[start - 1], start - 1))
        if end + 1 < len(sentences) and size + 1 + len(sentences[end + 1]) <= budget:
            options.append((scores[end + 1], end + 1))
        if not options:
            break
        _score, index = max(options)
        if index < start:
            start = index
        else:
            end = index
        size += 1 + len(sentences[index])
    snippet = " ".join(sentences[start:end + 1])
    if len(snippet) > budget:
        cut = snippet.rfind(" ", 0, budget)
        snippet = snippet[:cut if cut > 0 else budget].strip()
    return snippet


def select_excerpt(
    pages: list[str],
    max_pages: int,
    max_chars: int,
    anchor: str | None = None,
    must_page: int | None = None,
) -> tuple[str, list[int]]:
    """Pick the most TRL- and evidence-dense pages and the best window on each.

    ``anchor`` and ``must_page`` come from the target rationale and are only
    passed for training records, so the validation split stays label-free.
    """
    anchor_tokens = _tokens(anchor) if anchor else None
    scored = []
    for index, text in enumerate(pages):
        if not text:
            continue
        score = page_score(text)
        if anchor_tokens:
            score += 8 * len(_tokens(text) & anchor_tokens)
        scored.append((score, index))
    if not scored:
        return "", []
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    chosen = [index for _score, index in ranked[:max_pages]]
    if must_page is not None and must_page < len(pages) and pages[must_page]:
        chosen = [must_page] + [index for index in chosen if index != must_page]
        chosen = chosen[:max_pages]
    chosen = sorted(chosen)
    per_page = max(200, max_chars // len(chosen))
    parts: list[str] = []
    used: list[int] = []
    for index in chosen:
        snippet = best_snippet(pages[index], per_page, anchor_tokens)
        if snippet:
            parts.append(f"[page {index + 1}] {snippet}")
            used.append(index + 1)
    return "\n\n".join(parts), used


def extract_pages(path: Path) -> list[str]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "pypdfium2 is required for extraction; run this with the training venv "
            "(see train/setup_wsl.sh)"
        ) from exc
    pages: list[str] = []
    with pdfium.PdfDocument(str(path)) as document:
        for index in range(len(document)):
            try:
                page = document[index]
                text = page.get_textpage().get_text_range()
                page.close()
            except Exception:
                text = ""
            pages.append(clean_page(text))
    return pages


# --------------------------------------------------------------------------- labels


def load_labels(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    by_sha: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        digest = str(case.get("pdf_sha256") or "").lower()
        if digest:
            by_sha[digest] = case
        by_id[str(case["case_id"]).upper()] = case
    return by_sha, by_id


def case_id_from_name(path: Path) -> str | None:
    match = _CASE_ID.search(path.stem)
    if not match:
        return None
    return f"USGS-SDG-{int(match.group(1)):03d}"


def match_case(
    path: Path,
    by_sha: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    allow_stem: bool,
) -> tuple[dict[str, Any] | None, str, str]:
    digest = sha256_file(path)
    case = by_sha.get(digest)
    if case is not None:
        return case, digest, "sha256"
    case_id = case_id_from_name(path)
    candidate = by_id.get(case_id) if case_id else None
    if candidate is None:
        return None, digest, "not_in_manifest"
    expected = str(candidate.get("pdf_sha256") or "").lower()
    if allow_stem or not expected or expected == digest:
        return candidate, digest, "case_id"
    return None, digest, f"hash_mismatch:{case_id}"


# --------------------------------------------------------------------------- dataset


def evidence_sentence(
    pages: list[str], rationale: str
) -> tuple[str, int] | None:
    """The sentence that best supports ``rationale``, and its page index.

    Lexical overlap only (content words plus numbers). Extractive targets keep
    the answer copyable from the excerpt instead of asking a 3B model to invent
    the organiser's paraphrase.
    """
    wanted = _tokens(rationale)
    wanted_numbers = set(_NUMBERS.findall(rationale))
    best: tuple[int, int, str] | None = None
    for page_index, text in enumerate(pages):
        if len(re.findall(r"\.{2,}", text)) >= 3:
            continue  # table of contents / dot leaders, not prose
        for raw in _BOUNDARY.split(text):
            sentence = raw.strip()
            if len(sentence) < 40 or ".." in sentence:
                continue
            letters = sum(ch.isalpha() for ch in sentence)
            if letters / len(sentence) < 0.55:
                continue  # figure captions, table rows, headings
            score = 2 * len(_tokens(sentence) & wanted)
            score += 4 * len(set(_NUMBERS.findall(sentence)) & wanted_numbers)
            if best is None or score > best[0]:
                best = (score, page_index, sentence)
    if best is None or best[0] <= 0:
        return None
    score, page_index, sentence = best
    if len(sentence) < 80:  # too terse on its own: add the following sentence
        sentences = [part.strip() for part in _BOUNDARY.split(pages[page_index])
                     if part.strip()]
        position = sentences.index(sentence) if sentence in sentences else -1
        if 0 <= position < len(sentences) - 1:
            joined = f"{sentence} {sentences[position + 1]}"
            if len(joined) <= 240:
                sentence = joined
    return sentence, page_index


def build_record(
    case: dict[str, Any], excerpts: str, rationale: str | None = None
) -> dict[str, Any]:
    target = case.get("assessment_target") or {}
    label = case.get("reference_labels") or {}
    text = str(rationale if rationale is not None else label.get("rationale", ""))
    if len(text) > 240:  # keep the answer inside the token budget; cut on a word
        cut = text.rfind(" ", 0, 240)
        text = text[:cut if cut > 0 else 240].rstrip()
    user = (
        f"Assessment target: {target.get('description', '')} "
        f"(type: {target.get('type', '')})\n"
        f"Intended application: {case.get('intended_application', '')}\n"
        f"Report: {case.get('title', '')}\n\n"
        f"Report excerpts:\n{excerpts}\n\n"
        "Which TRL does the evidence support?"
    )
    return {
        "case_id": case["case_id"],
        "sdg": str(case.get("sdg", "")),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(
                {
                    "trl": int(label["estimated_trl"]),
                    "rationale": text,
                },
                ensure_ascii=False,
            )},
        ],
    }


def split_records(
    items: list[tuple[str, int, dict[str, Any]]],
    holdout: str | None,
    val_frac: float,
    seed: int,
) -> tuple[list[tuple[str, int, dict[str, Any]]], list[tuple[str, int, dict[str, Any]]]]:
    if holdout is not None:
        train = [item for item in items if item[0] != holdout]
        val = [item for item in items if item[0] == holdout]
        return train, val
    by_trl: dict[int, list[tuple[str, int, dict[str, Any]]]] = {}
    for item in items:
        by_trl.setdefault(item[1], []).append(item)
    rng = random.Random(seed)
    train, val = [], []
    for trl in sorted(by_trl):
        group = by_trl[trl]
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        take = min(max(int(round(len(group) * val_frac)), 1), len(group) - 1)
        val.extend(group[:take])
        train.extend(group[take:])
    return train, val


def histogram(items: list[tuple[str, int, dict[str, Any]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _sdg, trl, _record in items:
        counts[str(trl)] = counts.get(str(trl), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- stages


def rationale_coverage(excerpts: str, rationale: str) -> tuple[float, float]:
    """(share of rationale content words, share of its numbers) in the excerpt."""
    wanted_words, wanted_numbers = _tokens(rationale), set(_NUMBERS.findall(rationale))
    have_words, have_numbers = _tokens(excerpts), set(_NUMBERS.findall(excerpts))
    words = len(wanted_words & have_words) / len(wanted_words) if wanted_words else 1.0
    numbers = (len(wanted_numbers & have_numbers) / len(wanted_numbers)
               if wanted_numbers else 1.0)
    return round(words, 3), round(numbers, 3)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.pdfs).resolve()
    if not root.is_dir():
        raise SystemExit(f"{root}: not a directory")
    labels_path = Path(args.labels).resolve()
    if not labels_path.is_file():
        raise SystemExit(f"{labels_path}: label file not found")
    by_sha, by_id = load_labels(labels_path)

    pdfs = sorted(root.rglob("*.pdf"))
    candidates: list[tuple[str, int, dict[str, Any], Path]] = []
    unmatched: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    empty: list[dict[str, str]] = []
    seen: set[str] = set()

    for path in pdfs:
        case, digest, reason = match_case(path, by_sha, by_id, args.allow_stem_match)
        relative = path.relative_to(root).as_posix()
        if case is None:
            unmatched.append({"path": relative, "reason": reason, "sha256": digest})
            continue
        if digest in seen:
            duplicates.append({"path": relative, "case_id": str(case["case_id"])})
            continue
        seen.add(digest)
        label = case.get("reference_labels") or {}
        if label.get("estimated_trl") is None:
            unmatched.append({"path": relative, "reason": "no_trl_label"})
            continue
        candidates.append((str(case.get("sdg", "")), int(label["estimated_trl"]),
                           case, path))
        if args.limit and len(candidates) >= args.limit:
            break

    holdout = None if str(args.holdout_sdg).lower() == "none" else str(args.holdout_sdg)
    # Split before extraction: label-anchored excerpts are for training records
    # only, so the validation split stays a label-free selection.
    train_meta, val_meta = split_records(
        [(sdg, trl, case) for sdg, trl, case, _path in candidates],
        holdout, args.val_frac, args.seed)
    train_ids = {str(case["case_id"]) for _sdg, _trl, case in train_meta}
    val_ids = {str(case["case_id"]) for _sdg, _trl, case in val_meta}

    items: list[tuple[str, int, dict[str, Any]]] = []
    oracle_items: list[dict[str, Any]] = []
    pages_used: list[int] = []
    chars_used: list[int] = []
    coverage_train: list[tuple[float, float]] = []
    coverage_val: list[tuple[float, float]] = []
    in_excerpt_train: list[bool] = []
    in_excerpt_val: list[bool] = []

    for sdg, trl, case, path in candidates:
        case_id = str(case["case_id"])
        organizer = str((case.get("reference_labels") or {}).get("rationale", ""))
        is_train = case_id in train_ids
        pages = extract_pages(path)

        target_text, target_page = organizer, None
        if args.target == "extractive":
            found = evidence_sentence(pages, organizer)
            if found is not None:
                target_text, target_page = found

        anchored = args.train_select == "label" and is_train
        anchor = target_text if anchored else None
        must_page = target_page if anchored else None
        excerpts, chosen = select_excerpt(
            pages, args.max_pages, args.max_chars, anchor, must_page)
        if not excerpts:
            empty.append({"path": path.relative_to(root).as_posix(),
                          "reason": "no_readable_text"})
            continue
        if args.target == "extractive":
            record = build_record(case, excerpts, target_text)
        else:
            record = build_record(case, excerpts)
        items.append((sdg, trl, record))
        pages_used.append(len(chosen))
        chars_used.append(len(excerpts))
        (coverage_train if is_train else coverage_val).append(
            rationale_coverage(excerpts, organizer))
        flat = normalize(excerpts).lower()
        present = normalize(target_text).lower()[:80] in flat
        (in_excerpt_train if is_train else in_excerpt_val).append(present)

        # Oracle retrieval for validation: same target, but an excerpt that is
        # guaranteed to contain it. Separates "can it quote" from "did the
        # label-free selector find the page".
        if args.oracle_val and not is_train and target_page is not None:
            oracle_excerpts, _ = select_excerpt(
                pages, args.max_pages, args.max_chars, target_text, target_page)
            if oracle_excerpts:
                oracle_items.append(build_record(case, oracle_excerpts, target_text))

    train_items = [item for item in items if item[2]["case_id"] in train_ids]
    val_items = [item for item in items if item[2]["case_id"] in val_ids]
    train_items.sort(key=lambda item: item[2]["case_id"])
    val_items.sort(key=lambda item: item[2]["case_id"])

    out = Path(args.out).resolve()
    write_jsonl(out / "train.jsonl", [item[2] for item in train_items])
    write_jsonl(out / "val.jsonl", [item[2] for item in val_items])
    if args.oracle_val and oracle_items:
        write_jsonl(out / "val-oracle.jsonl", sorted(
            oracle_items, key=lambda record: record["case_id"]))

    def mean_pair(values: list[tuple[float, float]]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return (round(sum(v[0] for v in values) / len(values), 3),
                round(sum(v[1] for v in values) / len(values), 3))

    train_words, train_numbers = mean_pair(coverage_train)
    val_words, val_numbers = mean_pair(coverage_val)

    report = {
        "pdfs_root": str(root),
        "labels": str(labels_path),
        "pdfs_found": len(pdfs),
        "matched": len(items),
        "train": len(train_items),
        "val": len(val_items),
        "train_by_trl": histogram(train_items),
        "val_by_trl": histogram(val_items),
        "holdout_sdg": holdout,
        "train_select": args.train_select,
        "target": args.target,
        "target_in_excerpt_train": (round(sum(in_excerpt_train) / len(in_excerpt_train), 3)
                                    if in_excerpt_train else 0.0),
        "target_in_excerpt_val": (round(sum(in_excerpt_val) / len(in_excerpt_val), 3)
                                  if in_excerpt_val else 0.0),
        "oracle_val_records": len(oracle_items),
        "pages_used_mean": round(sum(pages_used) / len(pages_used), 2) if pages_used else 0,
        "excerpt_chars_mean": round(sum(chars_used) / len(chars_used), 1) if chars_used else 0,
        "rationale_word_recall_train": train_words,
        "rationale_number_recall_train": train_numbers,
        "rationale_word_recall_val": val_words,
        "rationale_number_recall_val": val_numbers,
        "unmatched": unmatched,
        "duplicates": duplicates,
        "empty": empty,
    }
    (out / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_trainer(
    args: argparse.Namespace,
    data: Path,
    adapter: Path,
    dry_run: bool,
) -> tuple[int, dict[str, Any] | None]:
    command = [
        args.trainer_python, str(TRAINER),
        "--data", str(data),
        "--out", str(adapter),
        "--base", str(args.base),
        "--max-len", str(args.max_len),
        "--epochs", str(args.epochs),
        "--r", str(args.r),
        "--alpha", str(args.alpha),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    if dry_run:
        command.append("--dry-run")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, end="")
            return result.returncode, None
        stats: dict[str, Any] | None = None
        for line in result.stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "records" in value:
                stats = value
        return 0, stats
    result = subprocess.run(command, cwd=ROOT)
    return result.returncode, None


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recursive PDFs -> JSONL -> QLoRA adapter (development only).")
    parser.add_argument("--pdfs", default=str(DEFAULT_PDFS),
                        help="directory searched recursively for *.pdf")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS),
                        help="public-cases.json (or another file with a cases[] list)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where train.jsonl, val.jsonl and dataset_report.json go")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                        help="adapter output folder")
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B",
                        help="base model: HF repo id or a local model folder")
    parser.add_argument("--holdout-sdg", default="14",
                        help="SDG number held out for validation, or 'none'")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="validation share when --holdout-sdg none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--oracle-val", action=argparse.BooleanOptionalAction, default=True,
                        help="also write val-oracle.jsonl (excerpt forced to contain "
                             "the target) to measure quoting separately from retrieval")
    parser.add_argument("--train-select", choices=["label", "heuristic"], default="label",
                        help="anchor training excerpts on the label rationale; "
                             "validation always uses the label-free heuristic")
    parser.add_argument("--target", choices=["extractive", "organizer"], default="extractive",
                        help="extractive (default): quote the best supporting sentence; "
                             "organizer: copy the organizer's paraphrase")
    parser.add_argument("--limit", type=int, default=0, help="0 = all matched PDFs")
    parser.add_argument("--min-records", type=int, default=10,
                        help="refuse to train with fewer training records")
    parser.add_argument("--allow-stem-match", action="store_true",
                        help="accept USGS-SDG-NNN file names even when the SHA-256 differs")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the dataset and run only the token-budget check")
    parser.add_argument("--skip-precheck", action="store_true",
                        help="skip the token-budget dry run before training")
    parser.add_argument("--trainer-python", default=sys.executable,
                        help="interpreter that runs train/train_qlora.py")
    parser.add_argument("--max-len", type=int, default=896)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    adapter = Path(args.adapter).resolve()

    if args.train_only:
        if not (out / "train.jsonl").is_file():
            raise SystemExit(f"{out / 'train.jsonl'} is missing; run without --train-only")
    else:
        report = build_dataset(args)
        if report["matched"] == 0:
            raise SystemExit("no PDF matched a labelled public case; nothing to train on")

    records = sum(1 for _ in (out / "train.jsonl").open(encoding="utf-8"))
    if records < args.min_records and not args.extract_only:
        raise SystemExit(
            f"only {records} training records (< --min-records {args.min_records}); "
            "lower the threshold deliberately if this is intended")

    if args.extract_only:
        print(json.dumps({"status": "EXTRACTED", "out": str(out), "train": records}))
        return 0

    if not args.skip_precheck:
        code, stats = run_trainer(args, out / "train.jsonl", adapter, dry_run=True)
        if code != 0:
            return code
        if stats is None:
            print("train_qlora.py dry run printed no stats; refusing to train",
                  file=sys.stderr)
            return 1
        if stats.get("records_truncated_to_no_target"):
            print(json.dumps({"status": "REFUSED", "reason": "TARGET_TRUNCATED",
                              "stats": stats}), file=sys.stderr)
            return 1
        if args.dry_run:
            return 0
    elif args.dry_run:
        raise SystemExit("--dry-run cannot be combined with --skip-precheck")

    code, _stats = run_trainer(args, out / "train.jsonl", adapter, dry_run=False)
    if code == 0:
        print(json.dumps({"status": "TRAINED", "adapter": str(adapter),
                          "train": records}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
