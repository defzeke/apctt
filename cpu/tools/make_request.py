"""Build a request file from PDF reports: python tools/make_request.py REPORT.pdf [...] -o FILE.

The case fields (target, application, environment, title, schema_version, …) are copied from a
template request (default: docs/EXAMPLE_REQUEST.json); `documents` is replaced by the given PDFs,
each with `format`, `content_base64` and `sha256` filled in, and `checklist_version` is the version
of --checklist: by default the official checklist (contracts/checklist.official.json, decision
D74), which config/dev.json and the configuration of README step 6 run. Edit the template or the
output for your own cases. A request over an input limit of --config (default config/dev.json:
max_documents, max_document_bytes, max_request_bytes and, where pypdf is installed as in the kit's
.venv, max_total_pages, with the pages counted as the server counts them; decision D77) is still
written, with a WARNING line: the server would refuse it (422, or 413 REQUEST_TOO_LARGE). A PDF
the server could not open gets a WARNING too. Standard library, plus the kit's page counter
(apctt_contracts.pdf with pypdf) when it can be imported.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def build_request(
    template: dict[str, Any],
    pdfs: list[Path],
    request_id: str,
    checklist_version: str | None = None,
) -> dict[str, Any]:
    case = {k: v for k, v in template["case"].items() if k != "documents"}
    if checklist_version is not None:
        case["checklist_version"] = checklist_version
    documents = []
    for index, pdf in enumerate(pdfs, start=1):
        data = pdf.read_bytes()
        if not data.startswith(b"%PDF-"):
            raise SystemExit(f"{pdf}: not a PDF file (no %PDF- header)")
        documents.append(
            {
                "document_id": f"D{index:03d}",
                "title": pdf.stem,
                "document_date": None,
                "language": "en",
                "format": "pdf",
                "content_base64": base64.b64encode(data).decode("ascii"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    case["documents"] = documents
    return {"request_id": request_id, "case": case}


def page_counter() -> Callable[[Path], int | None] | None:
    """The kit server's own page counting (apctt_contracts.pdf, with pypdf) when this Python has
    it, as the kit's .venv does; None otherwise (the page limit is then checked by the server only).
    """
    if importlib.util.find_spec("pypdf") is None:
        return None
    if str(ROOT / "src") not in sys.path:
        sys.path.append(str(ROOT / "src"))  # the kit's sources, when the kit is not installed
    try:
        from apctt_contracts.pdf import count_pages  # noqa: PLC0415 - optional, only here
    except ImportError:
        return None
    return count_pages


def limit_warnings(
    body: dict[str, Any],
    sizes: list[int],
    limits: dict[str, Any],
    pages: list[int | None] | None = None,
) -> list[str]:
    """The input limits (D77) this request breaks; the page limit only with the page count of each
    PDF (None: the server cannot open it)."""
    found = []
    documents = body["case"]["documents"]
    most = limits.get("max_documents")
    if isinstance(most, int) and len(documents) > most:
        found.append(f"{len(documents)} documents > max_documents {most} (422)")
    largest = limits.get("max_document_bytes")
    for document, size in zip(documents, sizes, strict=True):
        if isinstance(largest, int) and size > largest:
            found.append(
                f"{document['document_id']} is {size} bytes > max_document_bytes {largest} (422)"
            )
    total = len(json.dumps(body).encode("utf-8"))
    cap = limits.get("max_request_bytes")
    if isinstance(cap, int) and total > cap:
        found.append(
            f"the request is {total} bytes > max_request_bytes {cap} (413 REQUEST_TOO_LARGE; "
            "PDFs travel as base64, about 4/3 of their size)"
        )
    if pages is not None:
        for document, count in zip(documents, pages, strict=True):
            if count is None:
                found.append(
                    f"{document['document_id']}: the server cannot open it or count its pages "
                    "(422 pdf_readable)"
                )
        most_pages = limits.get("max_total_pages")
        total_pages = sum(count or 0 for count in pages)
        if isinstance(most_pages, int) and None not in pages and total_pages > most_pages:
            found.append(f"{total_pages} pages in all PDFs > max_total_pages {most_pages} (422)")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files, in document order")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=ROOT / "docs/EXAMPLE_REQUEST.json")
    parser.add_argument(
        "--checklist",
        type=Path,
        default=ROOT / "contracts/checklist.official.json",
        help="checklist whose version the request carries (default: the official checklist)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/dev.json",
        help="configuration whose input limits are checked (default: config/dev.json)",
    )
    parser.add_argument("--case-id", help="case_id to set (identifier)")
    parser.add_argument("--request-id", default="REQ-LOCAL-001")
    args = parser.parse_args(argv)
    template = json.loads(args.template.read_text(encoding="utf-8"))
    version = json.loads(args.checklist.read_text(encoding="utf-8"))["version"]
    body = build_request(template, args.pdfs, args.request_id, version)
    if args.case_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", args.case_id):
            raise SystemExit("--case-id must be an identifier [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
        body["case"]["case_id"] = args.case_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    pages = ", ".join(str(p) for p in args.pdfs)
    print(f"{args.output}: {len(args.pdfs)} document(s) from {pages}, checklist_version {version}")
    try:
        limits = json.loads(args.config.read_text(encoding="utf-8")).get("limits", {})
    except (OSError, ValueError, AttributeError):
        limits = {}
    sizes = [pdf.stat().st_size for pdf in args.pdfs]
    counter = page_counter()
    counts = [counter(pdf) for pdf in args.pdfs] if counter is not None else None
    found = limit_warnings(body, sizes, limits if isinstance(limits, dict) else {}, counts)
    for warning in found:
        print(f"WARNING the server would refuse this request: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
