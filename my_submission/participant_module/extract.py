"""PDF text extraction and sentence segmentation for the assessment module.

Pages are read with pypdfium2 (in the common base environment). Running headers,
footers, reference lists and boilerplate are dropped so evidence points at prose.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPACES = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'{])")
_TOO_LONG = re.compile(r"(?<=[;:])\s+")
_ALPHA = re.compile(r"[A-Za-z]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CITATION = re.compile(r"^[A-Z][A-Za-z.\-']+,\s+[A-Z]\.")
_PAGE_NUMBER = re.compile(r"^[\divxlc]{1,5}$", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(?:disclaimer|should not be used for|prepared in cooperation|"
    r"in cooperation with|suggested citation|references cited|"
    r"list of (?:figures|tables)|conversion factors|abbreviations|"
    r"acknowledg|for navigation or regulatory|funding for|provided funding|"
    r"reader is referred|not intended|army corps of engineers)",
    re.IGNORECASE,
)
_CAPTION = re.compile(
    r"^(?:photograph|figure|table|map|graph|diagram|chart|image|photo|"
    r"plate|appendix)\b",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(r"^(?:[A-Z][A-Z0-9\-]{1,6})$")
_MIN_WORDS = 6
_MAX_CHARS = 420


@dataclass(frozen=True)
class Sentence:
    document_id: str
    page: int
    order: int
    text: str


def normalize(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    pieces: list[str] = []
    for part in _BOUNDARY.split(text):
        part = part.strip()
        if len(part) <= _MAX_CHARS:
            pieces.append(part)
            continue
        for chunk in _TOO_LONG.split(part):
            chunk = chunk.strip()
            while len(chunk) > _MAX_CHARS:
                cut = chunk.rfind(" ", 0, _MAX_CHARS)
                if cut < 200:
                    cut = _MAX_CHARS
                pieces.append(chunk[:cut].strip())
                chunk = chunk[cut:].strip()
            if chunk:
                pieces.append(chunk)
    return [p for p in pieces if p]


def clean_page(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 8:
        lines = lines[2:-2]
    lines = [line for line in lines if not _PAGE_NUMBER.match(line)]
    return " ".join(lines)


def useful(text: str) -> bool:
    if len(text) < 40 or len(text) > _MAX_CHARS * 1.2:
        return False
    if len(text.split()) < _MIN_WORDS:
        return False
    low = text.lower()
    if low.startswith(("source:", "credit:", "note:")) or _CAPTION.match(text):
        return False
    if _BOILERPLATE.search(low):
        return False
    if "doi.org" in low or "http://" in low or "https://" in low:
        return False
    if sum(1 for word in text.split() if _ABBREVIATION.match(word)) >= 5:
        return False
    if len(_YEAR.findall(text)) >= 3 or _CITATION.match(text):
        return False
    letters = len(_ALPHA.findall(text))
    return letters >= 0.55 * len(text)


def read_pdf(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(str(path)) as document:
        pages: list[str] = []
        for index in range(len(document)):
            try:
                page = document[index]
                text = page.get_textpage().get_text_range()
                page.close()
            except Exception:
                text = ""
            pages.append(normalize(clean_page(text)))
        return pages


def extract(
    case: Mapping[str, Any],
    deadline: float,
    max_pages_per_document: int = 300,
) -> tuple[list[Sentence], dict[tuple[str, int], str], list[str]]:
    """Return sentences, page text by (document_id, page), and documents read."""
    sentences: list[Sentence] = []
    pages: dict[tuple[str, int], str] = {}
    read: list[str] = []
    order = 0

    for document in case.get("documents") or []:
        if time.monotonic() >= deadline:
            break
        document_id = str(document.get("document_id") or "")
        raw_path = document.get("path")
        if not document_id or not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        try:
            texts = read_pdf(path)
        except Exception:
            continue
        read.append(document_id)
        for page_number, text in enumerate(texts[:max_pages_per_document], start=1):
            pages[(document_id, page_number)] = text
            if "references cited" in text.lower() or len(_YEAR.findall(text)) >= 10:
                continue
            for piece in split_sentences(text):
                if useful(piece):
                    sentences.append(Sentence(document_id, page_number, order, piece))
                    order += 1
        if time.monotonic() >= deadline:
            break
    return sentences, pages, read
