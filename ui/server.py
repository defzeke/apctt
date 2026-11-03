"""Local test UI for the assessment module and the B/C ranker. Development only; never shipped.

Run inside WSL from the repository root, then open http://127.0.0.1:3000 in the Windows
browser (WSL2 forwards it; plain "localhost" can hit another app on [::1]:3000):

    ~/.venvs/apctt-train/bin/python ui/server.py [--port 3000]

Standard library HTTP server, single-threaded, bound to 127.0.0.1. It runs the shipped
module exactly as ``train/module_eval.py`` does (rules-only settings), validates the result
with the kit's contract checker, and optionally shows the ranker's top sentences per criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from module_eval import build_case  # noqa: E402  (puts my_submission and cuda/src on sys.path)
from apctt_contracts.config import Checklist  # noqa: E402
from apctt_contracts.results import validate_result  # noqa: E402
from participant_module import rules  # noqa: E402
from participant_module.adapter import AssessmentModule  # noqa: E402
from participant_module.extract import extract  # noqa: E402
from ranker_check import load_ranker, rank_criteria  # noqa: E402

KIT = ROOT / "cuda" / "contracts"
SHEET = ROOT / "USGS_SDG_open_data_set_corrected.xlsx"  # organizer sheet (also on Google Drive)
TARGET_TYPES = ("technology", "system", "subsystem", "component")  # contract enum
PDFS = ROOT / "train" / "data" / "pdfs"
UPLOADS = ROOT / "train" / "out" / "ui-uploads"
RANKER = ROOT / "train" / "out" / "criteria-ranker"
INDEX = Path(__file__).with_name("index.html")
MAX_PDF = 50 * 1024 * 1024  # contract limit per PDF
MAX_JSON = 64 * 1024
UPLOAD_ID = re.compile(r"^UPLOAD-([0-9a-f]{12})$")

MANIFEST = {c["case_id"]: c for c in
            json.loads((KIT / "public-cases.json").read_text(encoding="utf-8"))["cases"]}
CHECKLIST_RAW = json.loads((KIT / "checklist.official.json").read_text(encoding="utf-8"))
CHECKLIST = Checklist.model_validate(CHECKLIST_RAW)
DESCRIPTIONS = {c["criterion_id"]: c["description"] for c in CHECKLIST_RAW["criteria"]}
MODULE = AssessmentModule()
MODULE.load({"checklist": CHECKLIST_RAW, "module": {"settings": {}}}, ROOT / "my_submission/assets")
_ranker = None


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = load_ranker(RANKER)  # lazy: about 2 s, only when the box is ticked
    return _ranker


def pdf_path(case_id: str) -> Path | None:
    if case_id in MANIFEST:
        path = PDFS / f"{case_id}.pdf"
        return path if path.is_file() else None
    match = UPLOAD_ID.match(case_id)
    if match:
        path = UPLOADS / f"{match.group(1)}.pdf"
        return path if path.is_file() else None
    return None


def read_sheet(path: Path) -> list[dict[str, str]]:
    """Rows of the organizer xlsx as {header: text}; stdlib only (an xlsx is zipped XML)."""
    import zipfile
    import xml.etree.ElementTree as ET

    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        rows = ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).find(m + "sheetData")
    table = []
    for row in rows:
        cells = {}
        for cell in row:  # key by column letter: empty cells are simply absent
            column = re.match(r"[A-Z]+", cell.get("r", "")).group(0)
            text = "".join(t.text or "" for t in cell.iter(m + "t")) or (cell.findtext(m + "v") or "")
            cells[column] = text.strip()
        table.append(cells)
    if not table:
        return []
    header = table[0]
    return [{header[col]: value for col, value in cells.items() if col in header}
            for cells in table[1:]]


def uploads() -> list[dict]:
    items = []
    for meta in sorted(UPLOADS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (UPLOADS / f"{meta.stem}.pdf").is_file():
            items.append({"id": f"UPLOAD-{meta.stem}",
                          "label": json.loads(meta.read_text(encoding="utf-8"))["name"]})
    return items


def list_cases() -> dict:
    cases = [{
        "case_id": cid, "sdg": c["sdg"], "title": c["title"],
        "trl": c["reference_labels"]["estimated_trl"],
        "has_pdf": (PDFS / f"{cid}.pdf").is_file(),
        "target": c["assessment_target"]["description"],
        "target_type": c["assessment_target"]["type"],
        "application": c["intended_application"],
        "environment": c["intended_environment"],
    } for cid, c in MANIFEST.items()]
    sheet = read_sheet(SHEET) if SHEET.is_file() else []
    # Applications grouped by SDG target: the case field plus the sheet's short category.
    groups: dict[str, set[str]] = {}
    for c in MANIFEST.values():
        groups.setdefault(c.get("sdg_target") or "Other", set()).update(
            {c["intended_application"], c.get("category") or ""})
    for row in sheet:
        groups.setdefault(row.get("SDG target") or "Other", set()).add(row.get("Category", ""))
    options = {
        "targets": sorted({c["target"] for c in cases}
                          | {row.get("TRL assessment unit", "") for row in sheet} - {""}),
        "target_types": list(TARGET_TYPES),
        "applications": [{"group": group, "values": sorted(values - {""})}
                         for group, values in sorted(groups.items())],
        "environments": sorted({c["environment"] for c in cases} - {""}),
        "pdfs": [{"id": c["case_id"], "label": f"{c['case_id']}.pdf · {c['title'][:60]}"}
                 for c in cases if c["has_pdf"]],
        "uploads": uploads(),
        "sheet_rows": len(sheet),
    }
    return {"cases": cases, "options": options, "ranker": (RANKER / "config.json").is_file(),
            "criteria": CHECKLIST_RAW["criteria"]}


def save_upload(data: bytes, name: str) -> dict:
    if not data.startswith(b"%PDF"):
        raise ValueError("not a PDF file")
    import pypdfium2 as pdfium

    UPLOADS.mkdir(parents=True, exist_ok=True)
    key = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{key}.pdf"
    path.write_bytes(data)
    try:
        with pdfium.PdfDocument(str(path)) as document:
            pages = len(document)
    except Exception as error:
        path.unlink(missing_ok=True)
        raise ValueError(f"unreadable PDF: {error}") from error
    meta = {"name": Path(name).name[:200] or "upload.pdf", "pages": pages,
            "sha256": hashlib.sha256(data).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return {"case_id": f"UPLOAD-{key}", **meta}


def build(request: dict) -> tuple[dict, dict | None]:
    """(case, organizer label or None). ``pdf`` picks the file; ``case_id`` (optional) picks
    a public case whose fields the dropdowns may override; the label is shown only when the
    public case runs on its own PDF."""
    pdf_id = str(request.get("pdf") or request.get("case_id") or "")
    path = pdf_path(pdf_id)
    if path is None:
        raise ValueError(f"unknown or missing PDF: {pdf_id!r}")
    target_type = str(request.get("target_type") or "technology")
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target type must be one of {', '.join(TARGET_TYPES)}")
    case_id = str(request.get("case_id") or "")
    if case_id and case_id not in MANIFEST:
        raise ValueError(f"unknown public case: {case_id!r}")

    if pdf_id in MANIFEST:
        record = dict(MANIFEST[pdf_id])
    else:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        record = {
            "case_id": pdf_id, "sdg": 0, "title": meta["name"],
            "assessment_target": {"target_id": f"{pdf_id}-T1", "type": target_type,
                                  "description": Path(meta["name"]).stem},
            "intended_application": "", "intended_environment": "",
            "pdf_sha256": meta["sha256"], "page_count": meta["pages"],
            "document": {"document_id": "D001", "title": meta["name"], "document_date": ""},
        }
    case = build_case(record, path, CHECKLIST.version)
    target = str(request.get("target") or "").strip()
    case["assessment_target"] = {**case["assessment_target"], "type": target_type,
                                 **({"description": target} if target else {})}
    for key, field in (("application", "intended_application"), ("environment", "intended_environment")):
        if request.get(key):
            case[field] = str(request[key])
    label = MANIFEST[case_id]["reference_labels"] if case_id and case_id == pdf_id else None
    return case, label


def assess(request: dict) -> dict:
    case, label = build(request)
    began = time.time()
    result = MODULE.assess(case, SimpleNamespace(remaining_ms=lambda: 180000.0))
    MODULE.reset_case()
    seconds = time.time() - began
    issues = validate_result(result, case=case, checklist=CHECKLIST)
    reply = {
        "case_id": case["case_id"], "title": case["title"], "label": label,
        "result": result, "issues": [f"{i.path}: {i.rule}" for i in issues],
        "seconds": round(seconds, 2), "ranker": None,
    }
    top = max(0, min(10, int(request.get("ranker") or 0)))  # 0 = rules only
    if top:
        model, tokenizer, device = get_ranker()
        began = time.time()
        sentences, _pages, _docs = extract(case, time.monotonic() + 1e9, 300)
        ranked = rank_criteria(model, tokenizer, device, sentences, MODULE._assignments,
                               DESCRIPTIONS, rules._target_terms(case))
        reply["ranker"] = {cid: [{"p": round(p, 3), "page": sentences[i].page,
                                  "text": sentences[i].text} for p, i in items[:top]]
                           for cid, items in ranked.items()}
        reply["ranker_seconds"] = round(time.time() - began, 2)
        reply["ranker_device"] = device
    return reply


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > limit:
            raise ValueError(f"body must be 1..{limit} bytes, got {length}")
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/cases":
            self._json(200, list_cases())
        elif path.startswith("/pdf/"):
            pdf = pdf_path(path[len("/pdf/"):])
            if pdf is None:
                self._json(404, {"error": "no such PDF"})
            else:
                self._send(200, pdf.read_bytes(), "application/pdf")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/upload":
                name = self.headers.get("X-Filename", "upload.pdf")
                self._json(200, save_upload(self._body(MAX_PDF), name))
            elif self.path == "/api/assess":
                request = json.loads(self._body(MAX_JSON))
                if not isinstance(request, dict):
                    raise ValueError("expected a JSON object")
                self._json(200, assess(request))
            else:
                self._json(404, {"error": "not found"})
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # local dev tool: show the cause in the page
            traceback.print_exc()
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(error).__name__}: {error}"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    # 127.0.0.1, not "localhost": a Windows app on [::1]:<port> (e.g. a Next dev server) wins
    # the "localhost" lookup in the browser, while WSL forwards only the IPv4 address here.
    print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
