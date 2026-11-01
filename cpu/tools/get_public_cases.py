"""The public sample cases (decision D74): list them, download their PDFs, make request files.

    python tools/get_public_cases.py --list [ID ... | --all | --smoke] [--json]
    python tools/get_public_cases.py --labels [ID ... | --all | --smoke] [--json]
    python tools/get_public_cases.py --download (ID ... | --all | --smoke) --dir DIR [--delay S]
    python tools/get_public_cases.py --make-requests [ID ... | --all | --smoke] --dir DIR --out DIR
                                     [--config FILE] [--checklist FILE] [--schema-version TEXT]

The public sample cases are 99 reports of the U.S. Geological Survey (USGS), listed with their case
fields in contracts/public-cases.json (one PDF per case; the PDFs are not part of the kit). An ID
is a case_id such as USGS-SDG-012 or its number (12); --smoke chooses the 3 cases of the
diagnostic run that comes back with every submission (decision D73).

--download fetches each PDF from pubs.usgs.gov, one at a time with a pause between files (--delay,
default 1 s), into DIR/<case_id>.pdf. Every file is checked against the manifest (size and
SHA-256) before it gets its final name; a wrong file is never kept. A complete file is not fetched
again, and an interrupted download resumes from DIR/<case_id>.pdf.part. If the network or the
certificates of this computer stop the download, fetch the PDF in a browser from its pdf_url and
save it as DIR/<case_id>.pdf: the tool then only checks it.

--make-requests writes OUT/<case_id>.json for each downloaded case, in the server's input format
({"request_id": ..., "case": {...}}, the PDF as base64 with its SHA-256), with the checklist
version of the official checklist (--checklist, default contracts/checklist.official.json) and the
schema_version of the kit configuration (--config, default config/dev.json; --schema-version sets
it directly). The reference labels are never written into a request. --labels prints them: they
are the organizer's estimates (an estimated TRL and a short rationale), not expert labels per
checklist criterion.

Exit codes: 0 done, 1 a download or check failed, 2 invalid manifest or unknown case, 3 usage
error. Standard library only (HTTPS with the system's certificates, or certifi's bundle when it is
installed).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

MANIFEST = ROOT / "contracts" / "public-cases.json"
SCHEMA = ROOT / "contracts" / "public-cases.schema.json"
CHECKLIST = ROOT / "contracts" / "checklist.official.json"
CONFIG = ROOT / "config" / "dev.json"
USER_AGENT = "apctt-get-public-cases/1 (APCTT hackathon participant kit)"
TIMEOUT_S = 60
CHUNK = 1 << 20
ATTEMPTS = 4
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
MAX_WAIT_S = 60.0
NUMBER = re.compile(r"\d{1,3}")
EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_USAGE = 0, 1, 2, 3


class ManifestError(Exception):
    """An invalid manifest or an unknown case (exit code 2)."""


class UsageError(Exception):
    """A bad command line (exit code 3)."""


class CaseError(Exception):
    """One case that could not be downloaded or checked; the others go on. manual: the file
    could not be fetched at all, so the message says how to fetch it by hand."""

    def __init__(self, message: str, manual: bool = False) -> None:
        super().__init__(message)
        self.manual = manual


class CertificateError(Exception):
    """The server's certificate could not be verified: every other download would fail too."""


@dataclass
class Manifest:
    path: Path
    data: dict[str, Any]
    cases: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    excluded: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def smoke(self) -> list[str]:
        return list(self.data["smoke_cases"])


# ------------------------------------------------------------------------------ manifest


def problems(data: object) -> list[str]:
    """Every problem of a manifest: the schema (the kit's subset), then the rules below."""
    from make_env import validate  # noqa: PLC0415 - the kit's schema subset, loaded when needed

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    found = validate(data, schema)
    if not isinstance(data, dict):
        return found

    def listed(key: str) -> list[Any]:
        value = data.get(key)
        return value if isinstance(value, list) else []

    ids = [c.get("case_id") for c in listed("cases") if isinstance(c, dict)]
    for case_id in sorted({i for i in ids if ids.count(i) > 1}, key=str):
        found.append(f"case {case_id} is listed twice")
    for entry in listed("excluded"):
        if isinstance(entry, dict) and entry.get("case_id") in ids:
            found.append(f"case {entry.get('case_id')} is both a case and excluded")
    for case_id in listed("smoke_cases"):
        if case_id not in ids:
            found.append(f"smoke case {case_id} is not a case")
    return found


def load_manifest(path: Path = MANIFEST) -> Manifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"{path}: not found") from None
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: not readable JSON ({exc})") from None
    found = problems(data)
    if found:
        raise ManifestError(f"{path}: invalid public case manifest: " + "; ".join(found[:10]))
    manifest = Manifest(path, data, list(data["cases"]))
    manifest.by_id = {case["case_id"]: case for case in manifest.cases}
    manifest.excluded = {entry["case_id"]: entry for entry in data["excluded"]}
    return manifest


def case_id_of(text: str) -> str:
    return f"USGS-SDG-{int(text):03d}" if NUMBER.fullmatch(text) else text.strip()


def select(
    manifest: Manifest, ids: Sequence[str], every: bool, smoke: bool, default_all: bool
) -> list[dict[str, Any]]:
    """The chosen cases in manifest order (IDs, --all or --smoke; one way at a time)."""
    if sum((bool(ids), every, smoke)) > 1:
        raise UsageError("choose cases by ID, by --all or by --smoke, not by several")
    if every or (default_all and not ids and not smoke):
        return list(manifest.cases)
    wanted = manifest.smoke if smoke else [case_id_of(i) for i in ids]
    if not wanted:
        raise UsageError("choose the cases: case IDs (USGS-SDG-012 or 12), --smoke or --all")
    for case_id in wanted:
        if case_id in manifest.excluded:
            reason = manifest.excluded[case_id]["reason"]
            raise ManifestError(f"case {case_id} is excluded from the public cases: {reason}")
        if case_id not in manifest.by_id:
            raise ManifestError(f"unknown case {case_id} (see --list)")
    return [case for case in manifest.cases if case["case_id"] in wanted]


def mib(size: int) -> float:
    return round(size / 1048576, 2)


# ------------------------------------------------------------------------------- network


def ssl_context() -> ssl.SSLContext:
    """The system's certificates, or certifi's bundle when it is installed (python.org builds
    of Python on macOS have no certificates until certifi is used)."""
    try:
        import certifi  # noqa: PLC0415 - optional

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def open_url(url: str, start: int) -> Any:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise CaseError(f"{url}: only http and https addresses are fetched")
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - scheme checked above
    context = ssl_context() if url.startswith("https:") else None
    return urllib.request.urlopen(request, timeout=TIMEOUT_S, context=context)  # noqa: S310


def manual_hint(case: dict[str, Any], folder: Path) -> str:
    return (
        f" The report's page is {case['publication_url']}; you can download the PDF there in a"
        f" browser and save it as {folder / (case['case_id'] + '.pdf')}: --download then only"
        " checks it."
    )


def certificate_message(case: dict[str, Any], reason: object) -> str:
    host = urllib.parse.urlsplit(case["pdf_url"]).hostname
    return (
        f"{case['case_id']}: the server's certificate could not be verified ({host}: {reason})."
        " This Python has no usable root certificates. Do one of: run the tool with the kit's"
        " .venv/bin/python (it includes certifi); install certifi into this Python; set"
        " SSL_CERT_FILE to a CA bundle (for example the path printed by `python -m certifi`); with"
        " a python.org Python on macOS, run its 'Install Certificates.command'. Or download the"
        " PDFs in a browser from their pdf_url (contracts/public-cases.json, or --list --json)"
        " and save each as DIR/<case_id>.pdf: --download then only checks them."
    )


def retry_wait(headers: Any, attempt: int) -> float:
    value = headers.get("Retry-After") if headers is not None else None
    try:
        wait = float(value) if value is not None else float(2**attempt)
    except ValueError:
        wait = float(2**attempt)
    return min(max(wait, 0.0), MAX_WAIT_S)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, case: dict[str, Any]) -> str | None:
    """None when the file is the case's PDF (size and SHA-256), otherwise why it is not."""
    size = path.stat().st_size
    if size != case["pdf_bytes"]:
        return f"{size:,} bytes, the manifest says {case['pdf_bytes']:,}"
    if sha256_of(path) != case["pdf_sha256"]:
        return "its SHA-256 differs from the manifest"
    return None


def fetch(case: dict[str, Any], target: Path, part: Path) -> None:
    """Download into part (resuming it), check it and move it to target."""
    size, url = case["pdf_bytes"], case["pdf_url"]
    for attempt in range(1, ATTEMPTS + 1):
        start = part.stat().st_size if part.is_file() else 0
        if start > size:
            part.unlink()
            start = 0
        if start == size:
            break
        try:
            with open_url(url, start) as response:
                ranged = response.status == 206 and str(
                    response.headers.get("Content-Range", "")
                ).startswith(f"bytes {start}-")
                written = start if ranged else 0
                with part.open("ab" if ranged else "wb") as sink:
                    while chunk := response.read(CHUNK):
                        written += len(chunk)
                        if written > size:
                            break
                        sink.write(chunk)
                if written > size:
                    part.unlink()
                    raise CaseError(
                        f"the server sent more than the manifest's {size:,} bytes; nothing was"
                        " kept. Please tell the organizer."
                    )
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                part.unlink(missing_ok=True)
                continue
            if exc.code in RETRY_STATUSES and attempt < ATTEMPTS:
                time.sleep(retry_wait(exc.headers, attempt))
                continue
            raise CaseError(f"HTTP {exc.code} from {url}.", manual=True) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise CertificateError(certificate_message(case, exc.reason)) from None
            if attempt < ATTEMPTS:
                time.sleep(retry_wait(None, attempt))
                continue
            raise CaseError(f"cannot reach {url} ({exc.reason}).", manual=True) from None
        except ssl.SSLCertVerificationError as exc:
            raise CertificateError(certificate_message(case, exc)) from None
        except (TimeoutError, http.client.HTTPException, OSError) as exc:
            if attempt < ATTEMPTS:  # keep the part: the next attempt resumes it
                time.sleep(retry_wait(None, attempt))
                continue
            raise CaseError(f"the download from {url} stopped ({exc}).", manual=True) from None
    if not part.is_file() or part.stat().st_size != size:
        raise CaseError(f"the download from {url} did not complete after {ATTEMPTS} attempts"
                        " (run the same command again to resume).", manual=True)  # fmt: skip
    problem = check_file(part, case)
    if problem:
        part.unlink()
        raise CaseError(
            f"the downloaded file is not the manifest's PDF: {problem}; nothing was kept (the"
            f" file at {url} may have changed). Please tell the organizer."
        )
    os.replace(part, target)


def download(
    manifest: Manifest, chosen: list[dict[str, Any]], folder: Path, delay: float
) -> tuple[int, dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dir": str(folder), "downloaded": [], "present": [], "failed": {}}
    fetched_before = False
    for case in chosen:
        case_id = case["case_id"]
        target, part = folder / f"{case_id}.pdf", folder / f"{case_id}.pdf.part"
        if target.is_file():
            problem = check_file(target, case)
            if problem is None:
                summary["present"].append(case_id)
                continue
            print(
                f"{case_id}: {target.name} is not the manifest's PDF ({problem}); fetching it again"
            )
            target.unlink()
        if fetched_before and delay > 0:
            time.sleep(delay)  # one file at a time, with a pause: be polite to pubs.usgs.gov
        fetched_before = True
        try:
            fetch(case, target, part)
        except CertificateError as exc:
            summary["failed"][case_id] = str(exc)
            print(str(exc), file=sys.stderr)
            break
        except CaseError as exc:
            message = str(exc) + (manual_hint(case, folder) if exc.manual else "")
            summary["failed"][case_id] = message
            print(f"{case_id}: {message}", file=sys.stderr)
            continue
        summary["downloaded"].append(case_id)
        print(f"{case_id}: downloaded {case['pdf_bytes']:,} bytes, SHA-256 checked")
    skipped = [c["case_id"] for c in chosen if c["case_id"] not in summary["downloaded"]]
    summary["not_attempted"] = [
        i for i in skipped if i not in summary["present"] and i not in summary["failed"]
    ]
    return (EXIT_FAILED if summary["failed"] else EXIT_OK), summary


# ------------------------------------------------------------------------- request files


def build_request(
    case: dict[str, Any], content: bytes, schema_version: str, checklist_version: str
) -> dict[str, Any]:
    """The server's input format; nothing but the case fields (never the reference labels)."""
    document = dict(case["document"])
    document["content_base64"] = base64.b64encode(content).decode("ascii")
    document["sha256"] = hashlib.sha256(content).hexdigest()
    return {
        "request_id": f"PUBLIC-{case['case_id']}",
        "case": {
            "schema_version": schema_version,
            "input_profile": "pdf_report",
            "checklist_version": checklist_version,
            "case_id": case["case_id"],
            "assessment_as_of": None,
            "assessment_target": dict(case["assessment_target"]),
            "intended_application": case["intended_application"],
            "intended_environment": case["intended_environment"],
            "title": case["title"],
            "documents": [document],
            "sdg_targets": list(case["sdg_targets"]),
        },
    }


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError(f"{what} {path}: not readable JSON ({exc})") from None
    if not isinstance(data, dict):
        raise UsageError(f"{what} {path}: not a JSON object")
    return data


def limit_problems(case: dict[str, Any], size: int, limits: dict[str, Any]) -> list[str]:
    found = []
    if case["pdf_bytes"] > limits.get("max_document_bytes", case["pdf_bytes"]):
        found.append("max_document_bytes")
    if case["page_count"] > limits.get("max_total_pages", case["page_count"]):
        found.append("max_total_pages")
    if size > limits.get("max_request_bytes", size):
        found.append("max_request_bytes (413 REQUEST_TOO_LARGE)")
    return found


def make_requests(
    manifest: Manifest, chosen: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[int, dict[str, Any]]:
    config = read_json(args.config, "--config")
    checklist = read_json(args.checklist, "--checklist")
    schema_version = args.schema_version or config.get("schema_version")
    version = checklist.get("version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise UsageError(f"--config {args.config} has no schema_version; use --schema-version")
    if not isinstance(version, str) or not version.strip():
        raise UsageError(f"--checklist {args.checklist} has no version")
    found = config.get("limits")
    limits: dict[str, Any] = found if isinstance(found, dict) else {}
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"out": str(args.out), "schema_version": schema_version,
                               "checklist_version": version, "written": [], "failed": {},
                               "warnings": {}}  # fmt: skip
    for case in chosen:
        case_id = case["case_id"]
        pdf = args.dir / f"{case_id}.pdf"
        problem = None
        if not pdf.is_file():
            problem = f"not downloaded: {pdf} is missing (run --download first)"
        elif reason := check_file(pdf, case):
            problem = f"{pdf.name} is not the manifest's PDF: {reason}"
        if problem:
            summary["failed"][case_id] = problem
            print(f"{case_id}: {problem}", file=sys.stderr)
            continue
        body = build_request(case, pdf.read_bytes(), schema_version, version)
        text = json.dumps(body, indent=2, ensure_ascii=False) + "\n"
        over = limit_problems(case, len(text.encode("utf-8")), limits)
        if over:
            summary["warnings"][case_id] = over
            print(f"{case_id}: the server of {args.config} would refuse this request: "
                  + ", ".join(over), file=sys.stderr)  # fmt: skip
        target = args.out / f"{case_id}.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
        summary["written"].append(case_id)
    return (EXIT_FAILED if summary["failed"] else EXIT_OK), summary


def downloaded(manifest: Manifest, folder: Path) -> list[dict[str, Any]]:
    """The cases with a PDF file in folder (checked later), for --make-requests without IDs."""
    found = [c for c in manifest.cases if (folder / f"{c['case_id']}.pdf").is_file()]
    if not found:
        raise UsageError(f"no downloaded case PDF (<case_id>.pdf) in {folder}: run --download")
    return found


# ----------------------------------------------------------------------- list and labels


def listing(manifest: Manifest, chosen: list[dict[str, Any]]) -> dict[str, Any]:
    smoke = set(manifest.smoke)
    rows = [
        {
            "case_id": c["case_id"],
            "sdg": c["sdg"],
            "estimated_trl": c["reference_labels"]["estimated_trl"],
            "type": c["assessment_target"]["type"],
            "pages": c["page_count"],
            "mib": mib(c["pdf_bytes"]),
            "smoke": c["case_id"] in smoke,
            "title": c["title"],
            "pdf_url": c["pdf_url"],
        }
        for c in chosen
    ]
    excluded = [
        {"case_id": e["case_id"], "reason": e["reason"]} for e in manifest.excluded.values()
    ]
    return {"cases": rows, "excluded": excluded, "smoke_cases": manifest.smoke,
            "licence_note": manifest.data["licence_note"]}  # fmt: skip


def print_listing(data: dict[str, Any]) -> None:
    rows = data["cases"]
    print(f"{'case_id':<13} {'SDG':>3} {'TRL':>3}  {'type':<10} {'pages':>5} {'MiB':>6}  "
          f"{'':<5}  title")  # fmt: skip
    for row in rows:
        title = row["title"] if len(row["title"]) <= 60 else row["title"][:57] + "..."
        flag = "smoke" if row["smoke"] else ""
        print(f"{row['case_id']:<13} {row['sdg']:>3} {row['estimated_trl']:>3}  {row['type']:<10} "
              f"{row['pages']:>5} {row['mib']:>6.2f}  {flag:<5}  {title}")  # fmt: skip
    pages = sum(r["pages"] for r in rows)
    total = sum(r["mib"] for r in rows)
    print(f"{len(rows)} cases, {pages:,} pages, {total:,.1f} MiB in all. TRL is the organizer's"
          " estimate (--labels), not an expert label. smoke: the 3 cases of the diagnostic run"
          " of every submission.")  # fmt: skip
    for entry in data["excluded"]:
        print(f"excluded: {entry['case_id']} ({entry['reason']})")
    print("The reports are USGS publications: U.S. public domain, credit requested ('Credit: U.S."
          " Geological Survey'); some figures are third-party material. See licence_note in"
          " contracts/public-cases.json.")  # fmt: skip


def labels(manifest: Manifest, chosen: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "meaning": manifest.data["reference_label_meaning"],
        "labels": [
            {
                "case_id": c["case_id"],
                "estimated_trl": c["reference_labels"]["estimated_trl"],
                "rationale": c["reference_labels"]["rationale"],
            }
            for c in chosen
        ],
    }


# ------------------------------------------------------------------------------------ CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def parser() -> Parser:
    p = Parser(
        prog="get_public_cases.py",
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Exit codes: 0 done, 1 a download or check failed, 2 invalid manifest or unknown"
        " case, 3 usage error.",
    )
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list the cases")
    action.add_argument("--labels", action="store_true", help="print the reference labels")
    action.add_argument("--download", action="store_true", help="download and check the PDFs")
    action.add_argument("--make-requests", action="store_true", help="write request files")
    p.add_argument("ids", nargs="*", metavar="ID", help="case IDs (USGS-SDG-012 or 12)")
    p.add_argument("--all", action="store_true", help="every case")
    p.add_argument("--smoke", action="store_true", help="the 3 cases of the diagnostic run")
    p.add_argument("--dir", type=Path, help="folder of the PDFs (<case_id>.pdf)")
    p.add_argument("--out", type=Path, help="folder for the request files (--make-requests)")
    p.add_argument(
        "--delay", type=float, default=1.0, help="seconds between two downloads (default 1)"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
        help="configuration with schema_version and limits (default config/dev.json)",
    )
    p.add_argument(
        "--checklist",
        type=Path,
        default=CHECKLIST,
        help="checklist whose version requests carry (default: the official one)",
    )
    p.add_argument("--schema-version", help="schema_version for the requests (overrides --config)")
    p.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    return p


def execute(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    if args.list or args.labels:
        chosen = select(manifest, args.ids, args.all, args.smoke, default_all=True)
        return EXIT_OK, (listing if args.list else labels)(manifest, chosen)
    if args.dir is None:
        raise UsageError("--dir is needed: the folder of the PDFs")
    if args.download:
        if args.delay < 0:
            raise UsageError("--delay must be 0 or more")
        chosen = select(manifest, args.ids, args.all, args.smoke, default_all=False)
        return download(manifest, chosen, args.dir, args.delay)
    if args.out is None:
        raise UsageError("--out is needed: the folder for the request files")
    if args.ids or args.all or args.smoke:
        chosen = select(manifest, args.ids, args.all, args.smoke, default_all=False)
    else:
        chosen = downloaded(manifest, args.dir)
    return make_requests(manifest, chosen, args)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
    except SystemExit as exc:  # usage errors (exit 3) and --help (exit 0)
        return int(exc.code or 0)
    try:
        code, data = execute(args)
    except UsageError as exc:
        print(f"get_public_cases.py: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ManifestError as exc:
        print(f"get_public_cases.py: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.list:
        print_listing(data)
    elif args.labels:
        print(data["meaning"])
        for entry in data["labels"]:
            print(f"{entry['case_id']}  TRL {entry['estimated_trl']}  {entry['rationale']}")
    elif args.download:
        failed = data["failed"]
        print(f"{len(data['downloaded'])} downloaded, {len(data['present'])} already present, "
              f"{len(failed)} failed, in {data['dir']}")  # fmt: skip
        if data["not_attempted"]:
            print("not attempted: " + ", ".join(data["not_attempted"]))
    else:
        print(f"{len(data['written'])} request file(s) in {data['out']} (schema_version "
              f"{data['schema_version']}, checklist_version {data['checklist_version']}); "
              f"{len(data['failed'])} failed")  # fmt: skip
    return code


if __name__ == "__main__":
    sys.exit(main())
