# APCTT Assessment Server — Participant Kit

kit 0.5.0 | API `apctt-api-v0.2-draft` | worker protocol `apctt-worker-v2` | variant: cpu

This kit is a local copy of the server the organizer will use to run your **Assessment Package**
(your TRL assessment module). With it you can run your module exactly the way the organizer's
server runs it — in a fresh, separate process per case — and see whether its output satisfies the
result contract before you submit.

It is a development kit, not an official event release. It contains the official checklist
(`contracts/checklist.official.json`, `APCTT-2026-CHECKLIST-v1`, 12 criteria) and the list of the
public sample cases (`contracts/public-cases.json`: 99 USGS reports, whose PDFs you download with
`tools/get_public_cases.py`; decision D74). The official checklist is the default of the kit's
configuration (`config/dev.json`) and of the example request. The included example modules look
for a few configured phrases only; they are format demonstrations, not TRL models, baselines or
expert judgments.

**Start with [PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md):** the task, the fixed
environment and the model catalogue, the event machine, up to two submissions (the latest valid one
received by the deadline is your final model), how your package is run and how it is scored,
Stage 2, the licence rules and the FAQ.
This variant does not include the CUDA example. GPU/CUDA tests were run with kit 0.3.0 on an
NVIDIA RTX 3070 Laptop GPU on 2026-09-30 and were not re-run for 0.5.0; the CUDA variant of the kit
describes them.

## Step by step: from unpacking to a submission

Follow the steps in order. Each step says what to run and what you should see. Commands are run from
the kit directory unless a step says otherwise. Details for every step are in the linked guides.

The steps assume this layout, with your own work next to the kit. The environment and the models
live outside `my_submission/`: every team runs in the same common base environment, and the
organizer provides the catalogue models, so you submit neither.

```text
work/
├── apctt-participant-kit-<version>-<variant>/   this kit (unpacked)
├── apctt-env/base/                              the common base environment (step 5, tools/make_env.py)
├── models/                                      catalogue models, as <org>/<name>/<revision>/ (step 5)
└── my_submission/                               what you submit (steps 5 and 10)
    ├── participant_module/
    │   ├── __init__.py
    │   ├── adapter.py
    │   ├── LICENSE                              the MIT text
    │   └── THIRD_PARTY_NOTICES.md
    ├── assets/                                  only weights and rules your team trained or wrote
    ├── submission.json                          (step 10)
    ├── declaration.json                         (step 10)
    └── README.md
```

### Step 1 — Check your machine

You need Linux or WSL2 (macOS works for local development; native Windows does not),
uv (installation: `https://docs.astral.sh/uv/`), Python 3.12 and `curl`. Port 8000 on `127.0.0.1`
must be free. The common base environment of step 5 needs network access once and about 1.2 GB of
disk for the CPU development build, or about 8 GB for the CUDA build on Linux x86_64 with an NVIDIA
GPU; catalogue models need the sizes listed in
[MODEL_CATALOG](docs/participant/MODEL_CATALOG.md).

```bash
uv --version
uv python find 3.12 || uv python install 3.12
```

### Step 2 — Unpack and check the kit

```bash
unzip apctt-participant-kit-<version>-<variant>.zip
cd apctt-participant-kit-<version>-<variant>
sha256sum -c MANIFEST.sha256 --quiet && echo "kit OK"
```

You should see `kit OK`. Any `FAILED` line means the download is damaged; download it again.

### Step 3 — Install the server environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

This environment runs the server, the tests and the kit's tools. Your module does not run here:
it runs in the common base environment of step 5.

### Step 4 — Confirm the kit works on your machine

```bash
.venv/bin/python -m pytest
```

The tests (the server's conformance tests and the tests of the licence checker and of the
environment, model and submission tools) run in about a minute, and all must pass. They start their own servers on free ports, so nothing else needs to be
running. If a test fails, see [TROUBLESHOOTING](docs/participant/TROUBLESHOOTING.md) before going
further.

Then run the included example module once through the server. In a first terminal:

```bash
.venv/bin/python -m apctt_server --profile dev --config config/dev.json
```

In a second terminal, from the kit directory:

```bash
curl -s http://127.0.0.1:8000/health/ready
curl -s -i -H 'Content-Type: application/json' --data @docs/EXAMPLE_REQUEST.json \
     http://127.0.0.1:8000/v1/assessments | grep -i '^location'
curl -s http://127.0.0.1:8000/v1/assessments/<assessment_id from the Location line>
```

`/health/ready` says `"reason_code":"READY"`, the POST answers with a `Location`, and after a second
the GET shows `"job_status":"finished"` and `"result_origin":"module"`, with one entry per criterion
of the official checklist (`APCTT-C01` to `APCTT-C12`). The browser view
`http://127.0.0.1:8000/docs` lets you try the same calls. Walk-through of every field:
[QUICKSTART](docs/participant/QUICKSTART.md). Stop the server with `Ctrl+C`.

### Step 5 — Build the base environment, get models and create your module

Your module runs in the **common base environment**, the same for every team and fixed for the
whole event: Python 3.12, torch 2.11.0, numpy, scipy, scikit-learn, pandas, transformers,
sentence-transformers, bitsandbytes, pypdf, pdfplumber, pypdfium2, an OCR engine and more (the list
is in [PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md) §7.1). Build it with the kit's
tool:

```bash
.venv/bin/python tools/make_env.py --base-dir ../apctt-env/base
```

The first run downloads and installs the base, which takes a few minutes; later runs reuse it. On
Linux x86_64 with an NVIDIA GPU (`nvidia-smi` found) the tool builds the CUDA base, exactly as on
the event machines; elsewhere it builds the CPU development base with the same package versions.
`--flavor cpu` forces the CPU build. If the tool reports that no Python 3.12 was found, run
`uv python install 3.12` or pass `--python /path/to/python3.12`. The last lines show the check and
the Python that runs your module:

```text
check   OK: imports come from the base and every base package is installed
worker python for config/my.local.json ("worker": {"python": ...}):
/…/work/apctt-env/base/bin/python
```

Never install anything into the base: the organizer's machines have exactly the base, so a package
you add by hand would be missing at the event.
`.venv/bin/python tools/make_env.py --check --base-dir ../apctt-env/base` checks it again at any
time.

If your module uses public models, take them from the organizer's **model catalogue**, the models
chosen for the reference GPU (RTX 3070 Laptop GPU, 8 GB VRAM); other public models cannot be used.
List them and download the ones you need into `../models`, the same layout as the organizer's
read-only model store:

```bash
.venv/bin/python tools/get_models.py --list
.venv/bin/python tools/get_models.py all-minilm-l6-v2 --store ../models
```

The download needs network access; the evaluation does not have any. The tool verifies every file,
prints the `APCTT_MODEL_STORE` value for your configuration (step 6) and the loading code for your
module (also `tools/get_models.py --snippet ID`). Sizes, VRAM estimates, CPU options and the 4-bit
options: [MODEL_CATALOG](docs/participant/MODEL_CATALOG.md).

Then create your module from the `work/` directory (one level above the kit):

```bash
mkdir -p my_submission/participant_module my_submission/assets
touch my_submission/participant_module/__init__.py
```

Add three small files now, because the first two are part of your package hash:
`my_submission/participant_module/LICENSE` (the full MIT text with your team in the copyright
line; the kit has it in `licenses/MIT.txt`; your own work is MIT only, decision D72),
`my_submission/participant_module/THIRD_PARTY_NOTICES.md` (every third-party component you use,
with version and licence) and `my_submission/README.md` (what your module does and how it was
built). [LICENSING](docs/participant/LICENSING.md) §2 explains the licence rule.

Save the following as `my_submission/participant_module/adapter.py`. It is the smallest module that
passes every contract check: it always abstains, so it is a starting point, not an assessment.

<!-- example: module participant_module/adapter.py -->
```python
"""participant_module/adapter.py — minimal valid AssessmentModule (always abstains)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class AssessmentModule:
    def __init__(self) -> None:
        self.checklist: Mapping[str, Any] = {}
        self.settings: Mapping[str, Any] = {}

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        self.checklist = config["checklist"]
        self.settings = config["module"]["settings"]
        # Load rules or weights from asset_root here. Never download anything.

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        ids = [criterion["criterion_id"] for criterion in self.checklist["criteria"]]
        return {
            "schema_version": case["schema_version"],
            "checklist_version": self.checklist["version"],
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
            "assessment_status": "abstained",
            "predicted_trl": None,
            "criterion_status": {cid: "insufficient evidence" for cid in ids},
            "evidence_reference": {cid: [] for cid in ids},
            "missing_evidence": {cid: "No evidence rule is implemented yet." for cid in ids},
            "abstention_reason": "insufficient evidence",
            "error_code": None,
        }

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        pass
```

The server creates the class, then calls `load` → `assess` → `reset_case` → `close` in a **new
process for every case**; nothing survives between cases. What each method receives and must
return: [MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §1–4. A fuller example that cites evidence
and returns `completed` is `participant_example/adapter.py`.

### Step 6 — Point a local server configuration at your module

Back in the kit directory, create `config/my.local.json` from the default configuration:

<!-- example: local-config -->
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

config = json.load(open("config/dev.json", encoding="utf-8"))
config["module"].update(
    package_id="my-team-module",
    entrypoint="participant_module.adapter:AssessmentModule",
    module_root="../../my_submission",
    package_root="../../my_submission/participant_module",
    asset_root="../../my_submission/assets",
    example_only=False,
    settings={},
)
config["checklist_path"] = "../contracts/checklist.official.json"
config["worker"]["python"] = "../../apctt-env/base/bin/python"
config["worker"]["extra_env"] = {"APCTT_MODEL_STORE": str(Path("../models").resolve())}
with open("config/my.local.json", "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY
```

Relative paths in a configuration are resolved from the configuration file's own directory
(`config/`), which is why they start with `../../`. `checklist_path` is the official checklist
(`APCTT-2026-CHECKLIST-v1`, criteria `APCTT-C01` to `APCTT-C12`), the one the organizer runs your
package with; it is already the default of `config/dev.json`, and the script sets it so that your
configuration names it explicitly. `worker.python` is the base's Python from step 5, and
`APCTT_MODEL_STORE` tells your module where the catalogue models are, as the organizer's configuration does (an absolute path,
because the worker runs in its own scratch folder). Every field, including the limits, is
described in [CONFIGURATION](docs/participant/CONFIGURATION.md).

Start the server with it and check that it picked up your package:

```bash
.venv/bin/python -m apctt_server --profile dev --config config/my.local.json
curl -s http://127.0.0.1:8000/v1/package     # in the second terminal
```

`package_id` is `my-team-module` and `verification_status` is `verified`. If the server refuses to
start, it prints `APCTT_START_REFUSED` with a code and the field at fault; the codes are listed in
[TROUBLESHOOTING](docs/participant/TROUBLESHOOTING.md).

### Step 7 — Submit cases and read the outcome

Save this helper as `submit.py` in the kit directory. It sends one request file with a **fresh
`request_id`** each time, waits for the result and prints what to check:

<!-- example: submit-script -->
```python
"""submit.py — send one request with a fresh request_id and print what to check."""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
MAX_REQUEST_BYTES = 83886080  # limits.max_request_bytes of your configuration (80 MiB, D77)
TIMEOUT = 120  # seconds; the server may count the pages of large PDFs for up to 60 s


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(BASE + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.headers, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.headers, json.loads(error.read())


path = sys.argv[1] if len(sys.argv) > 1 else "my_cases/example.json"
with open(path, encoding="utf-8") as handle:
    body = json.load(handle)
body["request_id"] = f"dev-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000}"
size = len(json.dumps(body).encode("utf-8"))
if size > MAX_REQUEST_BYTES:
    print("rejected 413 REQUEST_TOO_LARGE", f"(body {size} bytes > {MAX_REQUEST_BYTES}; not sent)")
    sys.exit(1)
try:
    status, headers, job = call("POST", "/v1/assessments", body)
except OSError as error:
    reason = getattr(error, "reason", error)
    if isinstance(reason, ConnectionRefusedError):
        print("cannot reach", BASE, "- is the server running?")
    elif isinstance(reason, (BrokenPipeError, ConnectionResetError)):
        # the server answers 413 before it reads a body over its limit and closes the connection
        print("rejected 413 REQUEST_TOO_LARGE", f"(the server closed the connection while {size}",
              f"bytes were sent: {reason}; its limits.max_request_bytes is lower)")
    elif isinstance(reason, TimeoutError):
        print("no answer from", BASE, f"within {TIMEOUT} s - is the server stuck? see its log")
    else:
        print("connection to", BASE, "failed:", reason)
    sys.exit(1)
if status != 202:
    error = job.get("error", {})
    print("rejected", status, error.get("code"), json.dumps(error.get("field_errors", [])))
    sys.exit(1)
while job["job_status"] != "finished":
    time.sleep(0.5)
    _, _, job = call("GET", headers["Location"])
result, diagnostics = job["result"], job["diagnostics"]
print("request_id    ", body["request_id"])
print("result_origin ", job["result_origin"])
print("status        ", result["assessment_status"], "| predicted_trl", result["predicted_trl"],
      "| error_code", result["error_code"])
print("timings_ms    ", diagnostics.get("timings_ms"))
if "failure" in diagnostics:
    print("failure       ", diagnostics["failure"]["message"])
for issue in diagnostics.get("output_issues", []):
    print("output issue  ", issue["path"], issue["rule"])
print("module logs   ", f"server_data/jobs/{job['assessment_id']}/stderr.log")
```

Cases are PDF reports. To turn PDFs into a request file, use the kit's `tools/make_request.py`.
It copies the case fields (target, application, environment, title, `schema_version`) from
`docs/EXAMPLE_REQUEST.json`, sets `checklist_version` to that of the official checklist (the one
your configuration runs; `--checklist` names another) and embeds each PDF as base64 with its
SHA-256. Start with the kit's example report, then your own PDFs:

```bash
.venv/bin/python tools/make_request.py docs/EXAMPLE_REPORT.pdf -o my_cases/example.json --case-id EXAMPLE-001
.venv/bin/python tools/make_request.py ../reports/report-a.pdf -o my_cases/case-001.json --case-id MY-001
```

Edit `assessment_target`, `intended_application`, `intended_environment` and `title` in a new
file to describe what the report assesses. Then, with the server from step 6 running:

```bash
.venv/bin/python submit.py                          # my_cases/example.json (docs/EXAMPLE_REPORT.pdf)
.venv/bin/python submit.py my_cases/case-001.json   # your own PDFs
```

**The public sample cases.** `contracts/public-cases.json` lists 99 real reports of the U.S.
Geological Survey with their case fields, ready to run (decision D74;
[PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md) §5.2). The PDFs are not in the kit:
`tools/get_public_cases.py` downloads them from pubs.usgs.gov, checks each against its SHA-256 and
writes request files for the official checklist. Start with the 3 cases of the diagnostic run that
comes back with every submission (about 5 MB):

```bash
.venv/bin/python tools/get_public_cases.py --list                       # every case: SDG, TRL, pages, MiB
.venv/bin/python tools/get_public_cases.py --download --smoke --dir ../public_cases/pdf
.venv/bin/python tools/get_public_cases.py --make-requests --smoke --dir ../public_cases/pdf --out ../public_cases/requests
.venv/bin/python submit.py ../public_cases/requests/USGS-SDG-012.json
```

`--download --all` fetches all 99 (about 1.1 GB, one file at a time; run it again to resume).
`--labels` prints the organizer's estimated TRL and its reason for each case: a rough reference,
not expert labels per criterion, and never sent to your module. If the download stops with a
certificate error, the tool prints what to do; you can also save a PDF from its `pdf_url` by hand as
`../public_cases/pdf/<case_id>.pdf`, and the tool then only checks it. The reports are U.S. public
domain (credit "U.S. Geological Survey"); some figures in them are third-party material, so use
the PDFs as they are and do not republish figures separately.

Read the output like this:

| You see | Meaning | What to do |
|---|---|---|
| `result_origin module`, no `output issue` lines | Your result passed every contract check | Continue with your logic |
| `result_origin supervisor`, `error_code MODULE_ERROR` | Your code raised an exception (the `failure` line says so) | Read the traceback in the `module logs` file |
| `output issue <path> <rule>` | Your result broke a contract rule | Look the rule up in [DATA_CONTRACT](docs/participant/DATA_CONTRACT.md) §4 |
| `error_code TIMEOUT` and `failure The model could not process the documents within the 180-second limit.` | The case took longer than 3 minutes (`limits.deadline_ms`, load time included) | Make it faster, or stop early with `context.remaining_ms()`: [MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §5 |
| `rejected 422 …` | The request file itself is invalid, or over an input limit (`max_documents`, `max_document_bytes`, `max_total_pages`) | The printed `field_errors` name the field and rule |
| `rejected 413 REQUEST_TOO_LARGE …` | The request is over `limits.max_request_bytes` (80 MiB: about 59 MiB of PDFs, because they travel as base64) | Use fewer or smaller PDFs; `tools/make_request.py` warns when it writes such a file |
| `rejected 503 BUSY` | Another case is still running | Wait and run again |
| `no answer from … within 120 s` | The server did not answer the request (it may take up to a minute to count the pages of large PDFs) | Read the server's console output; restart the server if it is stuck |

Every input field, and what your module receives for each PDF (`path`, `page_count`), is described
in [DATA_CONTRACT](docs/participant/DATA_CONTRACT.md) §1.

### Step 8 — Develop in a loop

1. Change your code in `my_submission/`.
2. Stop the server (`Ctrl+C`) and start it again with `config/my.local.json`. Each case already
   loads your code fresh, but the server computes your package hash at start-up; restarting keeps
   the recorded hash equal to the files that actually ran.
3. Run `.venv/bin/python submit.py`, which always uses a new `request_id`. Re-sending an old
   `request_id` after a code change returns `409 REQUEST_ID_CONFLICT`.

For breakpoints, call your class directly with the checker in
[MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §6, then confirm through the server again. The
worker only sees the base, and `PYTHONPATH` from your shell is not passed on
([MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §7).

**Packages.** Everything in `constraints-base.txt` is in the base, at that version: import it. There
is nothing else and no way to request more. Your own code, and third-party pure-Python modules that
you copy into `participant_module/` with their licences, are part of your package
([LICENSING](docs/participant/LICENSING.md) §2.2); compiled extensions are refused.

**Models.** Load a catalogue model from `<APCTT_MODEL_STORE>/<org>/<name>/<revision>/` with
`local_files_only=True`, and set `HF_HUB_OFFLINE=1` in your code before importing Hugging Face
libraries ([MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §7). Never put public model weights into
`my_submission/`; weights your team trained go into `assets/`.

### Step 9 — Test the conditions of the real evaluation

The organizer's server will differ from the kit's example setup. Check each point locally:

- **Checklist.** Your configuration already runs the official checklist (12 criteria). Your module
  must still take the criteria from `config["checklist"]` and not hard-code their IDs. To check
  that once, copy `contracts/checklist.demo.json` (the short DEMO checklist, kept for this
  exercise only), change the criterion IDs, descriptions and `version`, set `checklist_path` in a
  copy of `config/my.local.json` to it, make a request with
  `tools/make_request.py --checklist <your copy>`, and run it
  ([CONFIGURATION](docs/participant/CONFIGURATION.md) §4).
- **Real reports.** Run the public sample cases of step 7, at least the 3 of the diagnostic run
  and a few long ones (`--list` shows the pages), and compare your TRLs with `--labels`.
- **Time limit.** The event allows 3 minutes per case, model loading included. Time your largest
  realistic case, lower `limits.deadline_ms` to test the edge, and stop in time by checking
  `context.remaining_ms()` ([MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §5).
- **No network.** Your code and your own weights are inside `participant_module/` or `assets/`,
  packages come from the base, catalogue models from `APCTT_MODEL_STORE`; nothing may be
  downloaded at run time. Disconnect the network and run your cases once.
- **Model loading.** Every case starts a new process and loads its models again, inside the
  3-minute limit. Time `timings_ms.load` with the models you use; the large 4-bit models read
  14.50–16.35 GB from disk per case, and the event machine's 16 GB of RAM cannot keep them in the
  file cache ([MODEL_CATALOG](docs/participant/MODEL_CATALOG.md)).
- **The event machine.** An Asus TUF Dash F15 FX516PR-HN002: Intel Core i7-11370H (4 cores, 8
  threads), 16 GB of RAM, RTX 3070 Laptop GPU with 8 GB, on Ubuntu 24.04. At least 12 GB of RAM is
  available to your worker (decision D80); time a CPU-only module on 4 cores
  ([PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md) §7.2). The kit does not measure
  your worker's RAM; measure its peak yourself, for example by logging
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` to stderr at the end of `assess`
  ([MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §7.4).
- **Clean-up.** `reset_case` and `close` must never raise, or the case fails with `CLEANUP_ERROR`.
- **Input limits.** The official limits (decision D77) are in the kit's configuration: at most 3
  PDFs per case (`limits.max_documents`), 50 MiB per PDF (`limits.max_document_bytes`), 300 pages
  per case (`limits.max_total_pages`) and 80 MiB per request (`limits.max_request_bytes`; PDFs as
  base64, so about 59 MiB of PDFs per case). Try your largest realistic case against them: a
  request over 80 MiB is refused with `413 REQUEST_TOO_LARGE`, which `submit.py` prints.
- **Hard PDFs.** Include scanned pages, tables, multi-column layouts and long reports among your
  test PDFs; reading them well is part of the assessment quality you compete on.

### Step 10 — Pack, check and submit

Create the two remaining files in `my_submission/`:

- `declaration.json`: copy `docs/participant/SUBMISSION_DECLARATION.example.json`, delete
  `example_only`, and fill in your team, package and data sources
  ([SUBMISSION](docs/participant/SUBMISSION.md) §3; use `"data_sources": []` if you used no data).
- `submission.json` (`contracts/submission.schema.json`):

<!-- example: submission-json -->
```json
{
  "submission_version": "apctt-submission-v3",
  "team": "MY-TEAM",
  "package_id": "my-team-module",
  "package_version": "0.1.0",
  "base_id": "apctt-base-9e7fe1324bb6",
  "declared_package_sha256": null,
  "license": "MIT",
  "compute": {"device": "cpu"},
  "settings": {},
  "models": []
}
```

Replace `team`, `package_id` and `package_version` with yours (the same values as in
`declaration.json`), keep `license` as `MIT` (the only licence for your own work, decision D72; the
MIT text is in `participant_module/LICENSE`), `compute.device` with `cpu` or `cuda:0` and `settings` with your
`module.settings`. `base_id` is the common base of this kit (`contracts/base-environment.json`; step
5 prints it). In `models`, list every catalogue model your module loads, for example
`{"catalog_id": "all-minilm-l6-v2"}`, and the weights your team trained, with `"team_trained":
true`, their `path` and, for a fine-tune or adapter, the catalogue id of its `base_model`
([SUBMISSION](docs/participant/SUBMISSION.md) §1.2). Leave `declared_package_sha256` as `null`: the
tool fills it in. Then pack, from the kit directory:

<!-- example: pack-submission -->
```bash
.venv/bin/python tools/make_submission.py ../my_submission
```

The tool writes `../MY-TEAM_submission.zip` (your team name) with only `participant_module/`,
`assets/`, `submission.json`, `declaration.json` and `README.md`, writes the package hash into
`submission.json`, and checks the zip exactly as the organizer's intake does, including the
licence checker `tools/check_licenses.py` ([LICENSING](docs/participant/LICENSING.md) §7). It
prints a `REMINDER` line: third-party code in your package is your team's responsibility (decision
D66). Read the last line:

| Result | Exit | Meaning |
|---|---|---|
| `VALID` | 0 | The zip passes every check. |
| `VALID_WITH_REVIEW` | 2 | It passes, with licence items the organizer decides after the submission (`REVIEW` lines). Make sure you can explain each one. |
| `REFUSED` | 1 | An `ERROR` line names each problem; the zip is deleted. Fix them and pack again. |

`.venv/bin/python tools/make_submission.py --check ../MY-TEAM_submission.zip` checks a zip again.
Check that `declared_package_sha256` in `submission.json` equals `verified_package_sha256` from
`GET /v1/package` of your server (restart the server after your last change); how the hash is
built: [CONFIGURATION](docs/participant/CONFIGURATION.md) §5. Before each upload, work through
[MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) §9 and [SUBMISSION](docs/participant/SUBMISSION.md)
§6.

Upload the zip through the channel published on the official competition website. **Each team may submit its Stage 1
model up to two times. The latest valid submission received by the deadline will be treated as the
final model** (decision D73):

1. **Every submission** is checked at intake; you get back the receipt (valid or refused, with the
   reasons) and a diagnostic run of 3 public sample cases (statuses, error codes, failure
   sentences, timings; no score).
2. **Your final model** is the latest valid submission received by the deadline; if your second
   submission is refused, the first valid one stays. It is run on the hidden cases and scored.
3. **A team without a valid submission received by the deadline** is not scored or ranked.

Nothing is approved in advance: every submission is checked in full with the same published rules
([PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md) §9 and §11). Deadlines and the
submission channel are published on the official competition website; they are not part of this kit.

## What this kit does not do

- It does not score your results or compare them with expert labels. There are no answers here.
- It does not tell you whether your evidence is *sufficient*; it only checks that the result is
  structurally valid and consistent (for example, a `met` criterion must cite at least one page).
- It does not reproduce the event hardware, network isolation or the hidden evaluation cases. It
  has the official checklist and the list of the public sample cases (decision D74), and its
  configuration has the official input limits (decision D77) and the 3-minute limit per case.
- It does not give your module network access. `tools/make_env.py` builds the same common base
  as the organizer's machines and `tools/get_models.py` the same model store, but the organizer's
  own machines are the ones that count.
- It does not read the PDFs for you. The server checks, stores and counts the pages of each PDF;
  extracting text, tables or scanned content is your module's job, with the tools you choose.

## How a case is processed

```text
your client ──HTTP──▶ server (FastAPI, one process)
                        │ 1. checks the request (size, JSON, schema, PDFs)    → 4xx on error
                        │ 2. records the job durably, answers 202 + Location
                        │ 3. copies the PDFs read-only into a private folder and
                        │    starts a NEW worker process for this case only
                        ▼
                     worker process (the common base's Python)
                        load(config, asset_root) → assess(case, context) → reset_case() → close()
                        writes result.json, exits
                        │
server ◀────────────────┘ 4. validates the result against the contract
                          5. stores it unchanged (or a failure result) and the raw bytes
your client ──GET──▶ finished job: result + diagnostics
```

One case runs at a time. Nothing is kept between cases: every case pays its own start-up and
model-load time.

## Guides

| Guide | Read it for |
|---|---|
| [PROBLEM_DESCRIPTION](docs/participant/PROBLEM_DESCRIPTION.md) | The challenge: task, fixed environment, event machine, model catalogue, up to two submissions, how it is run and scored, Stage 2, FAQ |
| [QUICKSTART](docs/participant/QUICKSTART.md) | Every field of a request and a finished job, with the example module |
| [MODULE_GUIDE](docs/participant/MODULE_GUIDE.md) | The four methods, inputs, failures, time limit, logs, worker environment and model store |
| [DATA_CONTRACT](docs/participant/DATA_CONTRACT.md) | Every input and output field and rule, with valid and invalid examples |
| [API_REFERENCE](docs/participant/API_REFERENCE.md) | Every endpoint, header, status code and response; a Python client |
| [CONFIGURATION](docs/participant/CONFIGURATION.md) | Every configuration field and default limit; the package hash |
| [TROUBLESHOOTING](docs/participant/TROUBLESHOOTING.md) | Every error, failure, warning and start-up code |
| [SUBMISSION](docs/participant/SUBMISSION.md) | Up to two submissions and what each returns, the bundle, TechPort use, data declaration, handover |
| [LICENSING](docs/participant/LICENSING.md) | Your licence, allowed and refused third-party components, copied code and models, the licence checker |
| [MODEL_CATALOG](docs/participant/MODEL_CATALOG.md) | The public models you may use, chosen for the reference GPU (RTX 3070 Laptop GPU, 8 GB VRAM): sizes, VRAM estimates, CPU options, 4-bit options |

## Contents

| Path | What it is |
|---|---|
| `src/` | Server (`apctt_server`), worker runtime (`apctt_runtime`), contracts (`apctt_contracts`), diagnostics |
| `participant_example/` | CPU example `AssessmentModule` (DEMO rules, not a TRL model) |
| `config/dev.json` | Local configuration of the example module: loopback, CPU, the official checklist, the official input limits |
| `contracts/openapi.json` | Frozen OpenAPI description of the HTTP API |
| `contracts/worker-protocol.json` | JSON Schemas of the worker input, observation file and result |
| `contracts/checklist.official.json` | The official checklist, `APCTT-2026-CHECKLIST-v1` (12 criteria, decision D74), the default of `config/dev.json` and of step 6 |
| `contracts/checklist.demo.json` | A DEMO checklist (two criteria) for the protocol tests and the changed-checklist check of step 9 |
| `contracts/public-cases.json`, `contracts/public-cases.schema.json`, `tools/get_public_cases.py` | The public sample cases (99 USGS reports; the PDFs are downloaded, not included) and the tool that downloads, checks and turns them into request files (step 7) |
| `docs/EXAMPLE_REQUEST.json` | Synthetic request for the official checklist that you can send as is (embeds `docs/EXAMPLE_REPORT.pdf`) |
| `docs/EXAMPLE_REPORT.pdf` | The 2-page synthetic PDF inside the example request |
| `tools/make_request.py` | Builds a request file from your PDFs, for the official checklist by default (step 7) |
| `tools/make_env.py` | Builds and checks the common base environment; prints the worker Python (step 5) |
| `tools/get_models.py`, `tools/model_catalog.py`, `contracts/model-catalog.json` | The model catalogue, its checks, and the download of catalogue models into a local store (step 5) |
| `tools/make_submission.py`, `contracts/submission.schema.json` | Packs and checks your submission zip; the format of `submission.json` (step 10) |
| `tools/check_licenses.py`, `contracts/license-policy.json` | Licence checker for your submission or web system, and the policy it applies (step 10) |
| `requirements-base.in`, `requirements-base.lock`, `constraints-base.txt`, `overrides-base.txt`, `contracts/base-environment.json` | The common base environment: its input, hashed lock (Linux x86_64, Python 3.12), its versions as plain pins, and its record (`base_id`, import names, sizes) |
| `licenses/` | The MIT text for your `LICENSE` (D72), and the Apache-2.0 text that `tools/get_models.py` adds to a catalogue model whose repository has none |
| `docs/EXAMPLE_RESPONSE.json` | Shape of a finished job (illustrative values) |
| `docs/participant/` | The guides listed above and an example data declaration |
| `tests/` | Server conformance tests and the tests of the kit's tools |
| `requirements-server.lock`, `requirements-dev.lock` | Hashed dependency locks (Python 3.12) |
| `MANIFEST.sha256` | SHA-256 of every file in this kit |

Code comments cite sections such as `SPEC-06`; they refer to the organizer's internal server
specification. For participants, the contract is the documentation in `docs/participant/`,
`contracts/openapi.json` and `contracts/worker-protocol.json`.

## Requirements and status

| Item | Status |
|---|---|
| Server Python | 3.12 (the locks and `pyproject.toml` require `>=3.12,<3.13`) |
| Operating system | Linux or WSL2 intended. Native Windows is not supported (the server uses POSIX process groups and file locks). |
| Verified so far | Ubuntu 24.04 x86_64 with Python 3.12: all conformance tests pass (kit 0.2.1). macOS arm64 with Python 3.12: all conformance tests pass (0.3.0, 2026-10-01). |
| Worker Python | The common base's Python made by `tools/make_env.py` (CPython 3.12). The worker wrapper itself needs Python 3.10 or newer. |
| Common base | `apctt-base-9e7fe1324bb6`, built and checked with `tools/make_env.py` on macOS arm64 (CPU development build, 2026-10-10). The CUDA build on the event machines (Asus TUF Dash F15 FX516PR-HN002, Ubuntu 24.04, RTX 3070 Laptop GPU) is not yet verified. |
| GPU | GPU/CUDA tests (AC-29..AC-34, the CUDA dummy example) were run with kit 0.3.0 on an RTX 3070 Laptop GPU (Ubuntu 24.04.5, driver 595.91.07, torch 2.11.0+cu128, Python 3.12) on 2026-09-30, with PDF input. Since 0.3.0 only the dummy's checklist check and `config/cuda-example.json` changed. Not re-run for 0.5.0; NOT RUN on the final event machines (Asus TUF Dash F15 FX516PR-HN002). The GPU measurement of the model catalogue is not measured yet. |

## Licence

The kit's own code and documents are MIT licensed ([LICENSE](LICENSE)). Bundled Swagger UI keeps
Apache-2.0 and installed dependencies keep their own licences
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Your module, model and data are not covered.

The organizer confirmed the licence rules on 2026-10-10 and made your own work MIT only on
2026-10-13 (decision D72): you license your own work (Stage 1 module, rules and trained weights,
Stage 2 web code) under MIT, and Apache-2.0 is no longer accepted for it; third-party components
and catalogue models keep their own licences, and weights you derive from a catalogue model keep
its LICENSE and NOTICE; strong copyleft (AGPL, GPL
and similar) and non-commercial, research-only or otherwise use-restricted components are refused;
public models come only from the organizer's model catalogue (`contracts/model-catalog.json`),
whose weights you never ship.
Third-party pure-Python modules may go into your package with their licences in
`THIRD_PARTY_NOTICES.md` (compiled extensions are refused); if a licence or security problem with
them arises later, your team bears the legal responsibility and the award is revoked (decision
D66, 2026-10-10). You keep ownership of your work: you submit
it as open source and you do not assert patent, copyright, trade-secret or other
intellectual-property claims over it against the organizer, APCTT or anyone who uses it under that
licence (decision D54, 2026-10-10; the clause is in LICENSING §2.1). Details, permissive
alternatives and how to check your submission with `tools/check_licenses.py` (it flags components
from their declared metadata; HQ decides flagged items after submission):
[LICENSING](docs/participant/LICENSING.md). PyTorch and NVIDIA's CUDA libraries come with their own
terms: when `tools/make_env.py` installs them for development from the official index you accept
those terms yourself (NVIDIA's licence agreements are in the licence files of the NVIDIA CUDA library wheels
and of `cuda-bindings`), and you do not ship them in your submission.
