"""tools/get_public_cases.py: the public sample cases of decision D74 (contracts/public-cases.json).

The published manifest is checked as it is shipped. Downloads come from a local HTTP fake on
127.0.0.1 that serves small synthetic PDFs for a manifest written per test; nothing here needs
network access.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import ssl
import statistics
import subprocess
import sys
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from apctt_contracts.cases import CaseLimits, ProfileVersions, check_request
from tests.team_tools.helpers import ROOT, load_tool

get_public_cases = load_tool("get_public_cases")
MANIFEST = ROOT / "contracts/public-cases.json"
OFFICIAL = json.loads((ROOT / "contracts/checklist.official.json").read_text())
DEV = json.loads((ROOT / "config/dev.json").read_text())
SMOKE = ["USGS-SDG-012", "USGS-SDG-040", "USGS-SDG-061"]


# ------------------------------------------------------------------------- the manifest


def test_the_published_manifest_is_valid_and_consistent() -> None:
    manifest = get_public_cases.load_manifest()
    data = json.loads(MANIFEST.read_text())
    assert get_public_cases.problems(data) == []
    cases = data["cases"]
    assert len(cases) == 99 and data["statistics"]["cases"] == 99
    assert [c["case_id"] for c in manifest.cases] == [c["case_id"] for c in cases]
    for case in cases:
        assert case["case_id"] == f"USGS-SDG-{case['no']:03d}"
        assert case["assessment_target"]["target_id"] == case["case_id"] + "-T1"
        assert case["document"]["document_id"] == "D001"
        assert case["sdg_targets"] == [case["sdg"]]
        for key in ("pdf_url", "publication_url"):
            assert case[key].startswith("https://pubs.usgs.gov/"), case["case_id"]
        assert "not an expert criterion label" in case["reference_labels"]["label_origin"]
    stats = data["statistics"]
    pages = [c["page_count"] for c in cases]
    mib = [c["pdf_bytes"] / 1048576 for c in cases]
    assert stats["pages"]["median"] == statistics.median(pages)
    assert (stats["pages"]["max"], stats["pages"]["min"]) == (max(pages), min(pages))
    assert stats["pages"]["total"] == sum(pages)
    assert stats["pdf_mib"]["max"] == round(max(mib), 2)
    assert stats["pdf_mib"]["total"] == round(sum(mib), 1)

    def counts(key: Any) -> dict[str, int]:
        found: dict[str, int] = {}
        for case in cases:
            found[str(key(case))] = found.get(str(key(case)), 0) + 1
        return found

    assert stats["by_sdg"] == counts(lambda c: c["sdg"]) == {"6": 25, "11": 25, "13": 24, "14": 25}
    assert stats["by_estimated_trl"] == counts(lambda c: c["reference_labels"]["estimated_trl"])
    assert stats["by_type"] == counts(lambda c: c["assessment_target"]["type"])
    # Exclusions (D74): No. 67 is over the official per-PDF limit (D77) and is not a case.
    assert [e["case_id"] for e in data["excluded"]] == ["USGS-SDG-067"]
    assert data["excluded"][0]["pdf_bytes"] > DEV["limits"]["max_document_bytes"]
    assert "max_document_bytes" in data["excluded"][0]["reason"]
    assert not {e["case_id"] for e in data["excluded"]} & {c["case_id"] for c in cases}
    # The three sheet URLs that answered 404 were resolved from their publication pages.
    fixed = [c["no"] for c in cases if any("returned 404" in n for n in c.get("notes", []))]
    assert fixed == [12, 33, 91]


def test_every_case_is_a_full_report() -> None:
    """D80 (2026-10-13): a case contains the document in which the assessment target is described
    and tested, so the two cases for which the list named only appendix 1 use the full report."""
    data = json.loads(MANIFEST.read_text())
    for case in data["cases"]:
        for text in (case["pdf_url"], case["title"], case["document"]["title"]):
            assert "appendix" not in text.lower(), (case["case_id"], text)
    full = {
        "USGS-SDG-013": (
            "https://pubs.usgs.gov/of/2020/1082/ofr20201082.pdf",
            "5c507208261d4641e0640d75a3a9d2861536e2927bc5f8f15fe6a8b788eed4ea",
            7147147,
            36,
            7,
        ),
        "USGS-SDG-022": (
            "https://pubs.usgs.gov/sir/2014/5021/pdf/sir2014-5021.pdf",
            "0d6901ebe5d2e2541211c7379ddb3ff21456d03205b5a5beebfbb47a060926c5",
            8298621,
            42,
            5,
        ),
    }
    cases = {c["case_id"]: c for c in data["cases"]}
    for case_id, (url, sha256, size, pages, trl) in full.items():
        case = cases[case_id]
        assert (case["pdf_url"], case["pdf_sha256"], case["pdf_bytes"]) == (url, sha256, size)
        assert case["page_count"] == pages and case["text_layer"] is True
        assert case["reference_labels"]["estimated_trl"] == trl  # the organizer's estimate, kept
        assert "appendix" in case["sheet_differences"]["pdf_url"]
        assert any(note.startswith("D80") for note in case["notes"])
    assert "D80" in data["decisions"]


def test_the_manifest_follows_the_official_checklist_and_limits() -> None:
    data = json.loads(MANIFEST.read_text())
    assert data["checklist_version"] == OFFICIAL["version"] == "APCTT-2026-CHECKLIST-v1"
    limits = DEV["limits"]
    for key in ("max_documents", "max_document_bytes", "max_request_bytes", "max_total_pages"):
        assert data["limits_applied"][key] == limits[key], key
    for case in data["cases"]:
        assert case["pdf_bytes"] <= limits["max_document_bytes"], case["case_id"]
        assert case["page_count"] <= limits["max_total_pages"], case["case_id"]
        # base64 makes a request about 4/3 of its PDF; the fields add a few kilobytes
        assert (case["pdf_bytes"] + 2) // 3 * 4 + 65536 <= limits["max_request_bytes"]
    assert data["smoke_cases"] == SMOKE
    smoke = [c for c in data["cases"] if c["case_id"] in SMOKE]
    assert len({c["sdg"] for c in smoke}) == 3
    assert all(c["text_layer"] and c["pdf_bytes"] < 3 * 1048576 for c in smoke)
    assert sorted(c["reference_labels"]["estimated_trl"] for c in smoke) == [4, 6, 7]


def test_manifest_problems_are_reported() -> None:
    data = json.loads(MANIFEST.read_text())
    data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
    data["smoke_cases"] = ["USGS-SDG-012", "USGS-SDG-040", "USGS-SDG-067"]
    data["cases"][2]["sdg"] = 7
    found = "\n".join(get_public_cases.problems(data))
    assert "USGS-SDG-001 is listed twice" in found
    assert "smoke case USGS-SDG-067 is not a case" in found
    assert "$.cases[2].sdg" in found


# ------------------------------------------------------------------------ list and labels


def test_list_shows_every_case_and_the_exclusions(capsys: pytest.CaptureFixture[str]) -> None:
    assert get_public_cases.main(["--list"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("USGS-SDG-")]
    assert len(lines) == 99
    assert sum("smoke" in line for line in lines) == 3
    assert "excluded: USGS-SDG-067" in out and "max_document_bytes" in out
    assert "99 cases" in out
    assert get_public_cases.main(["--list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["cases"]
    first = rows[0]
    assert set(first) == {"case_id", "sdg", "estimated_trl", "type", "pages", "mib", "smoke",
                          "title", "pdf_url"}  # fmt: skip
    assert "reference_labels" not in first


def test_labels_are_the_organizer_estimates(capsys: pytest.CaptureFixture[str]) -> None:
    assert get_public_cases.main(["--labels", "--smoke", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [x["case_id"] for x in data["labels"]] == SMOKE
    assert [x["estimated_trl"] for x in data["labels"]] == [4, 7, 6]
    assert "not expert labels" in data["meaning"]
    assert get_public_cases.main(["--labels", "12"]) == 0
    out = capsys.readouterr().out
    assert "USGS-SDG-012  TRL 4" in out and "not expert labels" in out


# --------------------------------------------------------------------------- downloads


def pdf_bytes(index: int) -> bytes:
    base = (ROOT / "docs/EXAMPLE_REPORT.pdf").read_bytes()
    return base + b"%" + b"x" * (5000 + 997 * index) + b"\n"


class FakeUsgs:
    """Serves /files/<name> with Range support. fail: name -> statuses to answer first;
    corrupt: names served with flipped bytes; extra: names served with a longer body;
    ignore_range: answer 200 with the whole file to a Range request."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.fail: dict[str, list[int]] = {}
        self.corrupt: set[str] = set()
        self.extra: set[str] = set()
        self.ignore_range = False
        self.calls: list[tuple[str, str | None, str | None]] = []

    def body(self, name: str) -> bytes:
        data = self.files[name]
        if name in self.corrupt:
            data = bytes(b ^ 0xFF for b in data[:64]) + data[64:]
        if name in self.extra:
            data += b"more"
        return data


@contextlib.contextmanager
def serve(fake: FakeUsgs) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            name = self.path.rsplit("/", 1)[-1]
            ranged = self.headers.get("Range")
            fake.calls.append((self.path, ranged, self.headers.get("User-Agent")))
            if fake.fail.get(name):
                status = fake.fail[name].pop(0)
                self.send_response(status)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if name not in fake.files:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = fake.body(name)
            start = 0
            if ranged and not fake.ignore_range:
                start = int(ranged.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def write_manifest(path: Path, url: str, chosen: list[str]) -> dict[str, bytes]:
    """A manifest of the chosen published cases whose PDFs are small synthetic files at url."""
    data = json.loads(MANIFEST.read_text())
    files: dict[str, bytes] = {}
    cases = []
    for index, case in enumerate(c for c in data["cases"] if c["case_id"] in chosen):
        content = pdf_bytes(index)
        files[f"{case['case_id']}.pdf"] = content
        case.update(
            pdf_url=f"{url}/files/{case['case_id']}.pdf",
            pdf_bytes=len(content),
            pdf_sha256=hashlib.sha256(content).hexdigest(),
            page_count=2,
        )
        cases.append(case)
    data["cases"] = cases
    path.write_text(json.dumps(data))
    return files


@pytest.fixture
def world(tmp_path: Path) -> Iterator[tuple[FakeUsgs, Path, Path]]:
    fake = FakeUsgs({})
    with serve(fake) as url:
        manifest = tmp_path / "public-cases.json"
        fake.files.update(write_manifest(manifest, url, [*SMOKE, "USGS-SDG-001"]))
        yield fake, manifest, tmp_path / "pdfs"


def run(manifest: Path, *argv: str) -> int:
    code: int = get_public_cases.main(["--manifest", str(manifest), *argv])
    return code


def test_download_verifies_every_pdf_and_reuses_complete_files(
    world: tuple[FakeUsgs, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake, manifest, folder = world
    assert run(manifest, "--download", "--all", "--dir", str(folder), "--delay", "0") == 0
    for name, data in fake.files.items():
        assert (folder / name).read_bytes() == data
    assert not list(folder.glob("*.part"))
    assert {agent for _, _, agent in fake.calls} == {get_public_cases.USER_AGENT}
    assert "4 downloaded" in capsys.readouterr().out
    fake.calls.clear()
    assert run(manifest, "--download", "--all", "--dir", str(folder), "--delay", "0") == 0
    assert fake.calls == []  # verified files are not fetched again
    assert "4 already present" in capsys.readouterr().out


def test_download_resumes_a_partial_file(world: tuple[FakeUsgs, Path, Path]) -> None:
    fake, manifest, folder = world
    data = fake.files["USGS-SDG-012.pdf"]
    folder.mkdir()
    (folder / "USGS-SDG-012.pdf.part").write_bytes(data[:1000])
    assert run(manifest, "--download", "USGS-SDG-012", "--dir", str(folder), "--delay", "0") == 0
    assert fake.calls[0][1] == "bytes=1000-"
    assert (folder / "USGS-SDG-012.pdf").read_bytes() == data
    # A server that ignores Range sends the whole file again: the download starts over.
    fake.ignore_range = True
    (folder / "USGS-SDG-040.pdf.part").write_bytes(fake.files["USGS-SDG-040.pdf"][:500])
    assert run(manifest, "--download", "40", "--dir", str(folder), "--delay", "0") == 0
    assert (folder / "USGS-SDG-040.pdf").read_bytes() == fake.files["USGS-SDG-040.pdf"]


def test_a_wrong_or_damaged_file_is_never_kept(
    world: tuple[FakeUsgs, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake, manifest, folder = world
    fake.corrupt.add("USGS-SDG-012.pdf")
    fake.extra.add("USGS-SDG-040.pdf")
    assert run(manifest, "--download", "--smoke", "--dir", str(folder), "--delay", "0") == 1
    captured = capsys.readouterr()
    assert (
        "USGS-SDG-012: the downloaded file is not the manifest's PDF: its SHA-256" in captured.err
    )
    assert "USGS-SDG-040: the server sent more than the manifest's" in captured.err
    assert sorted(p.name for p in folder.iterdir()) == ["USGS-SDG-061.pdf"]
    # A file of the right name but the wrong content is replaced by a verified download.
    fake.corrupt.clear()
    fake.extra.clear()
    (folder / "USGS-SDG-012.pdf").write_bytes(b"%PDF-1.4 not this report")
    assert run(manifest, "--download", "--smoke", "--dir", str(folder), "--delay", "0") == 0
    assert (folder / "USGS-SDG-012.pdf").read_bytes() == fake.files["USGS-SDG-012.pdf"]


def test_busy_servers_are_retried_and_missing_files_explained(
    world: tuple[FakeUsgs, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake, manifest, folder = world
    fake.fail["USGS-SDG-012.pdf"] = [503, 429]
    assert run(manifest, "--download", "12", "--dir", str(folder), "--delay", "0") == 0
    assert len(fake.calls) == 3
    del fake.files["USGS-SDG-061.pdf"]
    assert run(manifest, "--download", "61", "--dir", str(folder), "--delay", "0") == 1
    err = capsys.readouterr().err
    assert "USGS-SDG-061: HTTP 404" in err
    assert "https://pubs.usgs.gov/publication/" in err and "USGS-SDG-061.pdf" in err


def test_a_certificate_problem_gets_a_clear_message(
    world: tuple[FakeUsgs, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, manifest, folder = world

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "unable to get issuer"))

    monkeypatch.setattr(get_public_cases.urllib.request, "urlopen", refuse)
    assert run(manifest, "--download", "--smoke", "--dir", str(folder), "--delay", "0") == 1
    err = capsys.readouterr().err
    assert "certificate" in err and "SSL_CERT_FILE" in err and "certifi" in err
    assert err.count("certificate could not be verified") == 1  # stops after the first case


# ------------------------------------------------------------------------- request files


def test_make_requests_writes_what_the_server_accepts(
    world: tuple[FakeUsgs, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, manifest, folder = world
    out = tmp_path / "requests"
    assert run(manifest, "--download", "--all", "--dir", str(folder), "--delay", "0") == 0
    assert run(manifest, "--make-requests", "--dir", str(folder), "--out", str(out)) == 0
    assert "4 request file(s)" in capsys.readouterr().out
    data = json.loads(manifest.read_text())
    never = set(data["case_template"]["never_send"])
    limits = CaseLimits(
        max_documents=DEV["limits"]["max_documents"],
        max_document_bytes=DEV["limits"]["max_document_bytes"],
    )
    versions = ProfileVersions(DEV["schema_version"], OFFICIAL["version"])
    for case in data["cases"]:
        body = json.loads((out / f"{case['case_id']}.json").read_text())
        assert body["request_id"] == f"PUBLIC-{case['case_id']}"
        assert check_request(body, limits=limits, versions=versions).code is None
        sent = body["case"]
        assert not never & set(sent) and "reference_labels" not in json.dumps(sent)
        assert sent["checklist_version"] == "APCTT-2026-CHECKLIST-v1"
        assert sent["schema_version"] == DEV["schema_version"]
        assert sent["assessment_target"] == case["assessment_target"]
        assert sent["sdg_targets"] == case["sdg_targets"]
        document = sent["documents"][0]
        assert document["sha256"] == case["pdf_sha256"]
        assert document["title"] == case["document"]["title"]
    # The event schema is set by HQ; only the chosen cases are written.
    other = tmp_path / "event"
    argv = ["--make-requests", "--smoke", "--dir", str(folder), "--out", str(other)]
    assert run(manifest, *argv, "--schema-version", "apctt-2026-v1") == 0
    assert sorted(p.stem for p in other.iterdir()) == SMOKE
    body = json.loads((other / "USGS-SDG-040.json").read_text())
    assert body["case"]["schema_version"] == "apctt-2026-v1"


def test_make_requests_needs_verified_pdfs(
    world: tuple[FakeUsgs, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, manifest, folder = world
    folder.mkdir()
    (folder / "USGS-SDG-012.pdf").write_bytes(fake.files["USGS-SDG-012.pdf"])
    (folder / "USGS-SDG-040.pdf").write_bytes(b"%PDF-1.4 something else")
    out = tmp_path / "requests"
    argv = ["--make-requests", "--smoke", "--dir", str(folder), "--out", str(out)]
    assert run(manifest, *argv) == 1
    err = capsys.readouterr().err
    assert "USGS-SDG-040: USGS-SDG-040.pdf is not the manifest's PDF" in err
    assert "USGS-SDG-061: not downloaded" in err
    assert sorted(p.name for p in out.iterdir()) == ["USGS-SDG-012.json"]


def test_selection_and_usage_errors(
    world: tuple[FakeUsgs, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, manifest, folder = world
    assert run(manifest, "--download", "--dir", str(folder)) == 3  # nothing chosen
    assert run(manifest, "--download", "USGS-SDG-002", "--dir", str(folder)) == 2
    assert "unknown case USGS-SDG-002" in capsys.readouterr().err
    assert get_public_cases.main(["--labels", "67"]) == 2
    assert "excluded" in capsys.readouterr().err
    assert run(manifest, "--list", "--download") == 3  # one action at a time
    broken = manifest.with_name("broken.json")
    broken.write_text("{}")
    assert run(broken, "--list") == 2


def test_the_tool_runs_as_a_script_from_any_folder(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools/get_public_cases.py"), "--list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "USGS-SDG-099" in proc.stdout
