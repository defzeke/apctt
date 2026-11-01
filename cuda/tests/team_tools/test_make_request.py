"""tools/make_request.py: a request file from PDFs, for the official checklist by default (D74)."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from tests.team_tools.helpers import ROOT

PDF = ROOT / "docs/EXAMPLE_REPORT.pdf"


def make(tmp_path: Path, *extra: str) -> dict[str, object]:
    out = tmp_path / "case.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools/make_request.py"), str(PDF), "-o", str(out), *extra],
        cwd=tmp_path,  # the defaults are the kit's own files, wherever the tool is run from
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    body: dict[str, object] = json.loads(out.read_text())
    return body


def test_requests_carry_the_official_checklist_version_by_default(tmp_path: Path) -> None:
    template = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())["case"]
    body = make(tmp_path, "--case-id", "MY-001")
    case = body["case"]
    assert isinstance(case, dict)
    assert case["checklist_version"] == "APCTT-2026-CHECKLIST-v1"
    assert case["case_id"] == "MY-001"
    for key in ("schema_version", "assessment_target", "intended_application", "title"):
        assert case[key] == template[key], key
    document = case["documents"][0]
    assert base64.b64decode(document["content_base64"]) == PDF.read_bytes()
    demo = make(tmp_path, "--checklist", str(ROOT / "contracts/checklist.demo.json"))["case"]
    assert isinstance(demo, dict)
    assert demo["checklist_version"] == "DEMO-CHECKLIST-NOT-OFFICIAL"


def run_tool(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/make_request.py"), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_a_request_over_the_input_limits_is_written_with_a_warning(tmp_path: Path) -> None:
    """D77: the server refuses such a request (422 for documents and PDF size, 413
    REQUEST_TOO_LARGE for the request size), so the tool says so when it writes the file."""
    out = tmp_path / "case.json"
    within = run_tool(tmp_path, str(PDF), "-o", str(out))
    assert within.returncode == 0 and "WARNING" not in within.stdout + within.stderr
    four = run_tool(tmp_path, *[str(PDF)] * 4, "-o", str(out))
    assert four.returncode == 0, four.stderr
    assert "WARNING" in four.stderr and "4 documents > max_documents 3" in four.stderr
    assert "422" in four.stderr and len(json.loads(out.read_text())["case"]["documents"]) == 4
    config = json.loads((ROOT / "config/dev.json").read_text())
    size = PDF.stat().st_size
    config["limits"].update(max_document_bytes=size - 1, max_request_bytes=size)
    small = tmp_path / "small.json"
    small.write_text(json.dumps(config))
    over = run_tool(tmp_path, str(PDF), "-o", str(out), "--config", str(small))
    assert over.returncode == 0, over.stderr
    assert f"D001 is {size} bytes > max_document_bytes {size - 1}" in over.stderr
    request_bytes = len(json.dumps(json.loads(out.read_text())).encode("utf-8"))
    assert f"the request is {request_bytes} bytes > max_request_bytes {size}" in over.stderr
    assert "413 REQUEST_TOO_LARGE" in over.stderr


def test_a_request_over_the_page_limit_is_written_with_a_warning(tmp_path: Path) -> None:
    """D77 max_total_pages: in the kit's environment (pypdf installed) the tool counts the pages
    the way the server does and warns when all PDFs together have more than the limit (the server
    answers 422 max_total_pages); a PDF it cannot open gets a warning too (422 pdf_readable)."""
    from tests.contract.pdfgen import build_pdf

    long = tmp_path / "long.pdf"
    long.write_bytes(build_pdf([["page"]] * 151))
    out = tmp_path / "case.json"
    one = run_tool(tmp_path, str(long), "-o", str(out))
    assert one.returncode == 0 and "WARNING" not in one.stdout + one.stderr, one.stderr
    two = run_tool(tmp_path, str(long), str(long), "-o", str(out))
    assert two.returncode == 0, two.stderr
    assert "WARNING" in two.stderr
    assert "302 pages in all PDFs > max_total_pages 300 (422)" in two.stderr
    damaged = tmp_path / "damaged.pdf"
    damaged.write_bytes(b"%PDF-1.4\nnot a real PDF body\n%%EOF\n")
    bad = run_tool(tmp_path, str(damaged), "-o", str(out))
    assert bad.returncode == 0, bad.stderr
    assert "D001: the server cannot open it or count its pages (422 pdf_readable)" in bad.stderr
