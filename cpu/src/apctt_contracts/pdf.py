"""PDF documents in cases (0.3.0): decoding, identity checks, page counting and case views.

A request carries each PDF as `content_base64` with its `sha256`. The server keeps the files
itself and stores and forwards the case without the base64 payload:

- stored view: every document gains `page_count`; `content_base64` is removed.
- worker view: the stored view plus `path`, the read-only copy inside the run's scratch directory.

Page counting imports pypdf lazily. The server runs it as a separate bounded process
(`python -m apctt_contracts.pdf FILE...`), so pypdf never runs inside the API process.
"""

from __future__ import annotations

import base64
import binascii
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PDF_SIGNATURE = b"%PDF-"


def decode_content(text: str) -> bytes | None:
    """Strict standard base64 (no whitespace, correct padding); None if it is not."""
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None


def count_pages(path: Path) -> int | None:
    """Number of pages, or None if the file cannot be opened as a readable PDF."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path, strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            return None
        pages = len(reader.pages)
    except Exception:  # noqa: BLE001 - any parser failure means "not readable"
        return None
    return pages if pages >= 1 else None


def stored_case(case: Mapping[str, Any], page_counts: Sequence[int]) -> dict[str, Any]:
    """The case as stored and validated: no base64 payload, a page count per document."""
    documents = []
    for document, pages in zip(case["documents"], page_counts, strict=True):
        view = {k: v for k, v in document.items() if k != "content_base64"}
        view["page_count"] = pages
        documents.append(view)
    return {**case, "documents": documents}


def worker_case(stored: Mapping[str, Any], paths: Sequence[Path]) -> dict[str, Any]:
    """The case a module receives: the stored view plus the path of each PDF copy."""
    documents = [
        {**document, "path": str(path)}
        for document, path in zip(stored["documents"], paths, strict=True)
    ]
    return {**stored, "documents": documents}


def document_filename(index: int) -> str:
    """Name of the n-th document inside `documents/`; never derived from request text."""
    return f"{index + 1:03d}.pdf"


def local_worker_case(case: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Build the worker view of a request case on your own machine (for direct module tests).

    Decodes every PDF into `directory`, counts its pages in this process and returns the case
    exactly as the server would pass it to a module.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    counts: list[int] = []
    for index, document in enumerate(case["documents"]):
        data = decode_content(document["content_base64"])
        if data is None:
            raise ValueError(f"documents[{index}].content_base64 is not valid base64")
        path = directory / document_filename(index)
        path.write_bytes(data)
        pages = count_pages(path)
        if pages is None:
            raise ValueError(f"documents[{index}] is not a readable PDF")
        paths.append(path)
        counts.append(pages)
    return worker_case(stored_case(case, counts), paths)


def main(argv: list[str]) -> int:
    """Print a JSON list with the page count (or null) of each file argument."""
    print(json.dumps([count_pages(Path(name)) for name in argv]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
