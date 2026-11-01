# Module guide — write and connect your Assessment Package

kit 0.5.0 | worker protocol `apctt-worker-v2`

Your submission is a Python package with one class that the server calls for each case. This page
describes everything that class receives, what it must return, the environment it runs in, how
failures are reported, and how to connect it to the local server.

## 1. The four methods

The default entry point the organizer expects is `participant_module.adapter:AssessmentModule`
(package `participant_module`, file `adapter.py`, class `AssessmentModule`). The class is created
with no arguments, then called in this order, once per process:

| Step | Method | You do | If it raises |
|---|---|---|---|
| 1 | `load(config, asset_root)` | Read settings, load rules/weights from `asset_root`. | Job fails with `MODULE_ERROR` (or your `apctt_error_code`); `reset_case` and `close` are still attempted. |
| 2 | `assess(case, context)` | Assess the case, return the result mapping. | Job fails with `MODULE_ERROR` (or your code). |
| – | (worker) | Converts your mapping to JSON and writes `result.json`. | Not JSON-serializable, NaN/Infinity, non-string keys or not a mapping → `INVALID_OUTPUT`. |
| 3 | `reset_case()` | Drop per-case state. | Job fails with `CLEANUP_ERROR`, even though a result was written. |
| 4 | `close()` | Release files, memory, GPU. | Job fails with `CLEANUP_ERROR`. |

Then the process exits. The next case starts a **new process** and calls `load` again: nothing in
memory, on disk in your scratch directory, or in environment variables survives between cases
(`loading_policy: cold_per_case`). Load time counts toward each case's time limit.

### Minimal valid module

This module passes every contract check for any case and checklist. It always abstains, so it is a
starting point, not an assessment.

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

Layout on disk (an empty `participant_module/__init__.py` makes it a package):

```text
my_submission/
├── participant_module/
│   ├── __init__.py
│   ├── adapter.py
│   ├── LICENSE                 the MIT text (part of the package hash; D72)
│   └── THIRD_PARTY_NOTICES.md
└── assets/                 weights and rules your team trained or wrote (optional)
```

The folder holds no environment and no public model: packages come from the common base
environment, public models from the organizer's model store (§7). What else the submission
contains: [SUBMISSION.md](SUBMISSION.md) §1.

A fuller example that finds evidence and returns `completed` is `participant_example/adapter.py`
with `participant_example/demo_rules.py`.

## 2. What `load` receives

`config` is a plain dict (JSON types only). It is shown here with the two-criterion DEMO checklist
to keep it short; `config/dev.json` and your configuration of README step 6 pass the official
checklist (`APCTT-2026-CHECKLIST-v1`, twelve criteria) in the same form:

```json
{
  "checklist": {
    "version": "DEMO-CHECKLIST-NOT-OFFICIAL",
    "criteria": [
      {"criterion_id": "DEMO-C01", "description": "DEMO ONLY: the report states that a prototype or model was tested."},
      {"criterion_id": "DEMO-C02", "description": "DEMO ONLY: the report states that measured results met stated acceptance criteria."}
    ]
  },
  "module": {
    "package_id": "participant-example-cpu",
    "entrypoint": "participant_example.adapter:AssessmentModule",
    "module_root": "/absolute/path/to/the/directory/containing/your/package",
    "asset_root": "/absolute/path/to/your/assets",
    "settings": {"any": "JSON you put in module.settings of the server config"}
  },
  "compute": {
    "device": "cpu",
    "require_cuda": false,
    "allow_cpu_fallback": false,
    "precision": "float32",
    "loading_policy": "cold_per_case",
    "seed": 42,
    "visible_devices": null
  }
}
```

- `checklist` is the only source of criterion IDs. Use `criterion_id` exactly; your result needs one
  entry per criterion. The official checklist (`contracts/checklist.official.json`, `APCTT-C01` to
  `APCTT-C12`) has other IDs and descriptions than the DEMO one shown above; do not hard-code
  either. The checklist never contains answers or reference labels.
- `module.settings` is your own configuration (thresholds, file names, feature switches). It is part
  of the server configuration hash, so changing it changes `config_sha256`.
- `asset_root` is a `pathlib.Path` to your assets. Treat it as read-only; its files are part of the
  package hash the server computes.
- `compute.seed` is the run's seed. The worker already calls `random.seed(seed)` and sets
  `PYTHONHASHSEED`; seed NumPy, PyTorch or other libraries yourself from this value.

## 3. What `assess` receives and returns

`case` is the `case` object from the request, as plain JSON types, with every document in the
original order. Each document is a PDF: instead of the base64 payload it carries `page_count` and
`path`, a read-only copy of the file in your case's scratch directory. Read the PDFs with the tools
your model needs (text extraction, layout analysis, OCR); the server does not extract, select,
shorten or reorder anything. Field-by-field description: [DATA_CONTRACT.md](DATA_CONTRACT.md) §1.

```python
for document in case["documents"]:
    pdf_path = document["path"]          # e.g. <scratch>/documents/001.pdf
    pages = document["page_count"]       # counted by the server; cite pages 1..page_count
```

`context` has exactly four attributes and one helper:

| Attribute | Type | Meaning |
|---|---|---|
| `context.run_id` | `str` | Server-generated ID of this run (32 hex characters). |
| `context.seed` | `int` | Same as `config["compute"]["seed"]`. |
| `context.time_budget_ms` | `int` | Time left for the whole case when the worker received it. |
| `context.scratch_dir` | `pathlib.Path` | Private, empty directory for this case; deleted afterwards. |
| `context.remaining_ms()` | `float` | Budget left now, measured by the worker's own clock. |

Return a mapping that follows the result contract ([DATA_CONTRACT.md](DATA_CONTRACT.md) §2–3).
In short:

- copy `schema_version`, `case_id` and `assessment_target` from the case and
  `checklist_version` from `config["checklist"]["version"]`;
- give every criterion a status, an evidence list and, for `insufficient evidence`, an explanation;
- a `met` criterion needs at least one evidence item `{document_id, page}` (plus an optional
  `supporting_span`) whose page exists in that PDF;
- use only JSON types (`dict`, `list`, `str`, `int`, `float`, `bool`, `None`); no NaN or Infinity;
- keep the serialized result under 2 MiB (default `max_result_bytes`).

Optional fields let you explain your judgment: `trl_rationale` (completed results only),
`criterion_rationale`, `confidence` (0 to 1) and `uncertainty` (a TRL interval).

## 4. Reporting failure

| You want to say… | Do this | The job shows |
|---|---|---|
| "I cannot assess this case" for a scientific reason | Return `assessment_status: "abstained"` with an allowed `abstention_reason` | `result_origin: module`, your result |
| "My module failed on this case" with your own code | Return `assessment_status: "failed"`, `error_code: "<YOUR_TEXT>"`, empty maps allowed | `result_origin: module`, your failed result |
| Something unexpected happened | Raise an exception | `result_origin: supervisor`, `error_code: MODULE_ERROR` |
| CUDA is unusable | Raise with `exc.apctt_error_code = "CUDA_UNAVAILABLE"` or `"CUDA_RUNTIME_ERROR"` | that code |
| Out of GPU memory | PyTorch's `torch.cuda.OutOfMemoryError`, or set `apctt_error_code = "OUT_OF_MEMORY"` | `OUT_OF_MEMORY` |

Only those four `apctt_error_code` values are honoured (`MODULE_ERROR`, `CUDA_UNAVAILABLE`,
`CUDA_RUNTIME_ERROR`, `OUT_OF_MEMORY`); anything else becomes `MODULE_ERROR`. Exception messages are
never sent over HTTP; they are in `stderr.log`.

Example of setting a code:

```python
error = RuntimeError("CUDA is not available to this worker")
error.apctt_error_code = "CUDA_UNAVAILABLE"
raise error
```

## 5. Time limit

- The per-case limit is **3 minutes** (`limits.deadline_ms`, 180 000 ms in `config/dev.json`; the event
  uses the same value). It covers everything:
  starting the interpreter, importing your package, `load`, `assess`, `reset_case`, `close` and
  process exit.
- At the limit the server sends SIGTERM to the whole worker process group, waits
  `termination_grace_ms` (2 000 ms), then sends SIGKILL. The job fails with `TIMEOUT`, even if
  `result.json` had already been written (its bytes are kept as the raw artifact), and
  `diagnostics.failure.message` states: "The model could not process the documents within the
  180-second limit." 
- Use `context.remaining_ms()` to stop early and return an `abstained` or `failed` result.
- Child processes you start belong to the same process group and are terminated with the worker.
  Any still running after a normal exit are terminated too (warning `DESCENDANTS_TERMINATED`).
  Do not start daemons or detach processes.
- If your model may not finish on the largest documents, check `context.remaining_ms()` and return
  an `abstained` or `failed` result yourself before the limit instead of being stopped.

## 6. Logs and debugging

- Everything you print goes to `server_data/jobs/<assessment_id>/stdout.log` and `stderr.log`
  (first 1 MiB of each; the rest is counted and dropped, warning `LOG_TRUNCATED`). Uncaught
  exceptions print a traceback to `stderr.log`.
- Logs are never parsed and never returned over HTTP. `diagnostics.stdout_bytes`,
  `stderr_bytes` and `*_truncated` tell you how much was written.
- If the result is rejected, `diagnostics.output_issues` lists each problem as
  `{"path": "...", "rule": "..."}` (up to 20). Rule meanings: [DATA_CONTRACT.md](DATA_CONTRACT.md) §4.
- `raw_output.bin` in the same directory holds your result bytes exactly as written.

To debug with breakpoints, call your class directly and check its output with the same validator
the server uses. Run this from the kit directory with the kit's `.venv`, with your package on
`PYTHONPATH` (for example `PYTHONPATH=/path/to/my_submission .venv/bin/python check.py`). If your
module imports packages of the common base (torch, transformers, …), run it with the base's Python
and the kit's sources instead
(`PYTHONPATH=/path/to/my_submission:src ../apctt-env/base/bin/python check.py`):

<!-- example: direct-check -->
```python
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from apctt_contracts.config import Checklist
from apctt_contracts.pdf import local_worker_case
from apctt_contracts.results import validate_result
from participant_module.adapter import AssessmentModule

request = json.loads(Path("docs/EXAMPLE_REQUEST.json").read_text())
checklist = json.loads(Path("contracts/checklist.official.json").read_text())
config = {
    "checklist": {"version": checklist["version"], "criteria": checklist["criteria"]},
    "module": {"settings": {}},
    "compute": {"device": "cpu", "seed": 42},
}
module = AssessmentModule()
module.load(config, Path("assets"))
with tempfile.TemporaryDirectory() as scratch:
    # The case exactly as the server passes it: PDFs decoded to files, page counts added.
    case = local_worker_case(request["case"], Path(scratch) / "documents")
    context = SimpleNamespace(
        run_id="local",
        seed=42,
        time_budget_ms=180000,
        scratch_dir=Path(scratch),
        remaining_ms=lambda: 180000.0,
    )
    result = module.assess(case, context)
module.reset_case()
module.close()

issues = validate_result(result, case=case, checklist=Checklist.model_validate(checklist))
for issue in issues:
    print(issue.path, issue.rule)
print(f"{len(issues)} issues")
```

A direct call runs in your current process, so it does not reproduce the separate environment,
time limit or process isolation. Always finish with a run through the server (§8).

## 7. The worker environment

### 7.1 The process

Each case runs as:

```text
<worker.python> -s <kit>/src/apctt_runtime/worker.py --run-id <run_id>
```

with the working directory set to `context.scratch_dir` and **only** these environment variables:

| Variable | Value |
|---|---|
| `PATH` | directory of the worker interpreter, `/usr/bin`, `/bin` |
| `HOME` | `<scratch>/home` |
| `TMPDIR`, `TMP`, `TEMP` | `<scratch>/tmp` |
| `XDG_CACHE_HOME` | `<scratch>/cache` |
| `LANG`, `LC_ALL` | `C.UTF-8` |
| `PYTHONUTF8` | `1` |
| `PYTHONHASHSEED` | the run seed |
| `PYTHONDONTWRITEBYTECODE` | `1` |
| `PYTHONNOUSERSITE` | `1` |
| `CUDA_VISIBLE_DEVICES` | `""` (no GPU) for `device: cpu`; the configured value otherwise |
| `APCTT_MODEL_STORE` | the read-only model store (§7.3): at the event set by the organizer, locally by your `worker.extra_env` |
| extra | whatever else the operator lists in `worker.extra_env` (never credentials) |

Consequences for your code:

- Your module runs in **its own interpreter**, `worker.python`: the common base environment's
  Python (§7.2). The server's own environment does not need, and should not get, your
  dependencies. The worker wrapper uses only the standard library and needs Python 3.10 or newer.
- `-s` disables the user site-packages directory; `PYTHONPATH` from your shell is not passed.
  The directory in `module.module_root` is placed first on `sys.path`.
- Server environment variables, tokens and cloud keys are not visible.
- Caches your libraries write under `HOME` or `XDG_CACHE_HOME` land in the scratch directory and
  are deleted after the case. Never rely on a cache: load everything from your package, your
  assets or the model store.
- Do not rely on network access. The evaluation environment blocks it.
- The server never imports your module in its own process and never unpickles anything you
  produce; only the JSON in `result.json` is read.

### 7.2 The common base environment

At the event, `worker.python` is the Python of the **common base environment**, the same for
every team, read-only and fixed for the whole event: Linux x86_64, CPython 3.12, torch 2.11.0
(CUDA 12.8 build), numpy, scipy, scikit-learn, pandas, transformers, tokenizers, safetensors,
huggingface-hub, sentence-transformers, accelerate, peft, bitsandbytes, pypdf, pdfplumber,
pypdfium2, onnxruntime, rapidocr-onnxruntime, opencv-python-headless and the Stage 2 web packages;
exact versions in `constraints-base.txt`, identifier `base_id` in `contracts/base-environment.json`
([PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §7.1).

`tools/make_env.py --base-dir DIR` builds the same base on your computer, with the CPU build of
torch 2.11.0 where there is no NVIDIA GPU, and prints its Python for `worker.python`;
`tools/make_env.py --check --base-dir DIR` runs that Python the way the server does (`python -s`,
the restricted `PATH`) and confirms that every base package is installed at its version. Rules
that follow:

- Use the base versions; you cannot add, upgrade, downgrade or replace a package, and a folder in
  your submission with the import name of a base package (for example a copied `transformers/`) is
  refused.
- Never `pip install` into the base: the organizer's machines have exactly the base. Your own code
  and third-party pure-Python modules you copy into `participant_module/` with their licences
  ([LICENSING.md](LICENSING.md) §2.2) are part of your package; compiled extensions are refused.
- Programs outside Python (a system OCR engine, a browser) are not provided. Use packages of the
  base (for example `rapidocr-onnxruntime` or `pypdfium2`).

### 7.3 Public models: the model store

Public models come only from the organizer's model catalogue (`contracts/model-catalog.json`,
[MODEL_CATALOG.md](MODEL_CATALOG.md)), and they are not in your submission. On every evaluation
machine the organizer puts every catalogue model into a **read-only model store**:

```text
<APCTT_MODEL_STORE>/<org>/<name>/<revision>/      for example
<APCTT_MODEL_STORE>/BAAI/bge-base-en-v1.5/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/
```

The folder holds the files the catalogue lists and the repository's licence files.
`python tools/get_models.py ID --store DIR` builds the same folders on your computer (and
`--snippet ID` prints the loading code). Load from that folder, offline:

```python
import os
from pathlib import Path

MODEL = "BAAI/bge-base-en-v1.5/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


class AssessmentModule:
    def load(self, config, asset_root):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before importing Hugging Face libraries
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        folder = Path(os.environ["APCTT_MODEL_STORE"]) / MODEL
        self.model = SentenceTransformer(str(folder), device="cpu", local_files_only=True)
```

(For `transformers`: `AutoModel.from_pretrained(folder, local_files_only=True)`.) Set
`HF_HUB_OFFLINE` in your code: names that start with `HF_` cannot be passed through
`worker.extra_env`. Locally, point `APCTT_MODEL_STORE` in `worker.extra_env` of your configuration
at your own store, as an absolute path (README step 6). The store is read-only: write nothing
there; use `context.scratch_dir`. List every catalogue model you load in `submission.json`
`models[]` by `catalog_id` ([SUBMISSION.md](SUBMISSION.md) §1.2). Weights your team trained,
including a fine-tune or adapter of a catalogue model, go into `assets/` and are loaded from
`asset_root`.

### 7.4 CPU or GPU, and the VRAM budget

A CPU-only module (`compute.device` `cpu`) is fully allowed and scored by the same rules and the
same time limit. The worker then sees no GPU (`CUDA_VISIBLE_DEVICES=""`), and torch runs on the
CPU. With `cuda:0`, your module uses the machine's NVIDIA GPU; the server checks at start-up that
CUDA works in the worker environment and never falls back to the CPU.

The event machine is an Asus TUF Dash F15 FX516PR-HN002 (decision D76): Intel Core i7-11370H with
4 cores and 8 threads, 16 GB of RAM, and the reference GPU, an RTX 3070 Laptop GPU with 8 GB VRAM.
At least 12 GB of RAM is available to your worker (decision D80; the rest is for the operating
system, the kit server and the file cache), and time a CPU-only module on 4 cores. Every catalogue model fits a budget of
at most 7.0 GiB peak at 8,192 tokens of context (language models) or a batch of 32 × 512 tokens
(encoders); the figures in [MODEL_CATALOG.md](MODEL_CATALOG.md) are estimates until the organizer
measures them on that GPU. Several models at once, longer inputs or larger batches can exceed the
card: an out-of-memory error fails the case (`OUT_OF_MEMORY`). The 4-bit models need a CUDA GPU.

**Measuring your worker's RAM.** The kit reports GPU peaks only (`diagnostics.compute`); on the CPU
`unavailable_reasons.peak_memory` is `NOT_MEASURED`. Measure the peak resident memory yourself, on
your largest realistic cases, and compare it with the 12 GB available to your worker on the event
machine (decision D80):

- Inside your module, at the end of `assess` (the worker process runs exactly one case):

  ```python
  import resource
  import sys

  peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux, bytes on macOS
  print(f"peak RSS {peak / 1024 / 1024:.2f} GiB (Linux)", file=sys.stderr)
  ```

  The line is in `server_data/jobs/<assessment_id>/stderr.log`. `ru_maxrss` covers this process
  only; add `resource.RUSAGE_CHILDREN` if your module starts processes of its own.
- Outside the server, on Linux: run the direct check of §6 with the base's Python under
  `/usr/bin/time -v` and read "Maximum resident set size (kbytes)".

Remove or keep the stderr line as you like: logs are never parsed or scored.

### 7.5 Loading time

Every case starts a new process, so every case loads its models again, inside the 3-minute limit
(§5). Load once in `load()`, never per criterion, and time `timings_ms.load` with the models you
use: the large 4-bit models read 14.50–16.35 GB from disk and quantise it on every case, and the
event machine's 16 GB of RAM cannot keep those weights in the file cache between cases.

## 8. Connect your module to the local server

1. Copy the configuration: `cp config/dev.json config/my.local.json` (`*.local.json` files are
   meant for your own paths). `config/dev.json` runs the official checklist
   (`"checklist_path": "../contracts/checklist.official.json"`, decision D74), so your copy does
   too; README step 6 makes the same file with a script.
2. Edit `config/my.local.json` (relative paths are relative to the config file's directory):

   ```json
   "module": {
     "package_id": "my-team-module",
     "entrypoint": "participant_module.adapter:AssessmentModule",
     "module_root": "/path/to/my_submission",
     "package_root": "/path/to/my_submission/participant_module",
     "asset_root": "/path/to/my_submission/assets",
     "declared_package_sha256": null,
     "example_only": false,
     "settings": {}
   },
   "worker": {
     "python": "/path/to/work/apctt-env/base/bin/python",
     "launcher_argv": [],
     "extra_env": {"APCTT_MODEL_STORE": "/path/to/work/models"}
   }
   ```

   Every field is described in [CONFIGURATION.md](CONFIGURATION.md).
3. Start: `.venv/bin/python -m apctt_server --profile dev --config config/my.local.json`
4. Check `curl -s http://127.0.0.1:8000/v1/package`: `package_id` is yours and
   `verification_status` is `verified`.
5. Send cases (start with `docs/EXAMPLE_REQUEST.json`, which is for the official checklist, then
   the public sample cases of README step 7 and your own PDFs made into requests with
   `tools/make_request.py`) and read `result_origin`, `result` and `diagnostics`.
6. Once, run with a changed checklist: copy `contracts/checklist.demo.json` (two DEMO criteria),
   change its criterion IDs and `version`, point `checklist_path` of a second copy of your
   configuration at it, and make the requests with `tools/make_request.py --checklist <copy>`.
   This is the only reason to use the DEMO checklist: it shows that your module takes the
   criteria from `config["checklist"]` instead of hard-coding the twelve official IDs. The
   organizer runs the official checklist.

**Use a new `request_id` after each code change.** A `request_id` is tied to the exact case, the
configuration and your package hash. Re-sending an old ID after you changed your code returns
`409 REQUEST_ID_CONFLICT`; re-sending it unchanged returns the stored result without running again.
Either use fresh IDs (for example `dev-0001`, `dev-0002`, …) or stop the server and delete
`server_data/`.

## 9. Before you submit

- [ ] `result_origin` is `module` for all your test cases, with no `output_issues`.
- [ ] You ran your module with the official checklist (README step 6) on real reports, at least
      the 3 public sample cases of the diagnostic run (`tools/get_public_cases.py --smoke`).
- [ ] You tested once with a checklist whose IDs differ (§8 step 6), so that nothing depends on
      hard-coded criterion IDs.
- [ ] Every case finishes well within the time limit, including `load`.
- [ ] `reset_case` and `close` never raise.
- [ ] Nothing is downloaded at run time: your own files are inside `asset_root`, catalogue
      models are read from `APCTT_MODEL_STORE` with `local_files_only=True`.
- [ ] Your module runs with the base's Python as `worker.python`, and
      `tools/make_env.py --check --base-dir DIR` passes.
- [ ] Every catalogue model you load is listed in `submission.json` `models[]`, and the cases
      finish in time with those models loaded in every case.
- [ ] `tools/make_submission.py` packs your submission with result `VALID` (or `VALID_WITH_REVIEW`
      with review items you can explain).
- [ ] You filled in the data declaration and handover items in [SUBMISSION.md](SUBMISSION.md).
