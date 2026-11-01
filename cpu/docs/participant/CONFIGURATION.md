# Configuration reference

kit 0.5.0 | config format `apctt-config-v1`

The server reads one JSON file at start-up and never changes it while running. Requests cannot
change any setting (profile, device, seed, time limit, module, paths). To change something, edit
the file and restart the server.

```bash
.venv/bin/python -m apctt_server --profile dev --config config/my.local.json
```

`--profile` defaults to `dev` and must equal the file's `profile` (otherwise `PROFILE_MISMATCH`).
Relative paths in the file are resolved against the **directory of the config file**. Unknown
fields are rejected. If the file is refused, the server prints the code and each offending field:

```text
APCTT_START_REFUSED CONFIG_INVALID
  limits.deadline_ms: greater_than
  module.entrypoint: string_pattern_mismatch
```

All start-up codes: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) §6.

## 1. Top-level fields

| Field | `config/dev.json` | Meaning |
|---|---|---|
| `config_version` | `"apctt-config-v1"` | Format version; must be exactly this. |
| `profile` | `"dev"` | `dev` for local use. `evaluation` is the organizer's locked-down mode (§7). |
| `bind_host` | `"127.0.0.1"` | Listen address. `dev` accepts only loopback (`127.x.x.x`, `::1`, `localhost`). |
| `bind_port` | `8000` | Listen port (1–65535). |
| `data_dir` | `"../server_data"` | Job store, results, raw outputs, logs, event log. |
| `worker_tmp_dir` | `"../worker_tmp"` | Parent of the per-case scratch directories. |
| `schema_version` | `"apctt-pdf-example-v1"` | Every case must carry this `schema_version`. |
| `checklist_path` | `"../contracts/checklist.official.json"` | Checklist file (§4). Every case must carry its `version`. The official checklist (decision D74) is the default of `config/dev.json`, of the CUDA example configuration and of your configuration of README step 6. |
| `retry_after_seconds` | `1` | Value of the `Retry-After` header on `503 BUSY` and `NOT_READY`. |
| `limits` | see §3 | Size and time limits. |
| `compute` | see §2 | Device and determinism settings passed to your module. |
| `module` | see §5 | Which package to run and its settings. |
| `worker` | see §6 | Worker interpreter and environment. |
| `isolation` | `{"evidence_file": null}` | Used only by the `evaluation` profile (§7). |
| `auth` | `{"bearer_token_file": null}` | Optional bearer token (§8). |

To run two servers side by side, give each its own `bind_port`, `data_dir` and `worker_tmp_dir`.
Two servers can never share one `data_dir`; the second one fails to start.

## 2. `compute`

| Field | `config/dev.json` | Meaning |
|---|---|---|
| `device` | `"cpu"` | `cpu` or `cuda:0`. There is no automatic choice. |
| `require_cuda` | `false` | Must be `true` exactly when `device` is `cuda:0`. |
| `allow_cpu_fallback` | `false` | Must be `false`: a CUDA configuration never silently runs on CPU. |
| `precision` | `"float32"` | Passed to your module; the server does not interpret it. |
| `loading_policy` | `"cold_per_case"` | Only value: a new worker process per case. |
| `seed` | `42` | Integer ≥ 0. The worker seeds Python's `random` and sets `PYTHONHASHSEED`; `context.seed` gives it to you. |
| `visible_devices` | not set | Optional string for `CUDA_VISIBLE_DEVICES`. When unset and `device` is `cpu`, the worker gets `CUDA_VISIBLE_DEVICES=""` (no GPU visible). |

With `require_cuda: true` the server runs a small CUDA check with the worker interpreter at start-up
and stays not-ready (`CUDA_UNAVAILABLE` / `CUDA_RUNTIME_ERROR`) until CUDA works. The GPU path is not
verified in this kit version.

## 3. `limits`

The four input limits of `config/dev.json` are the **official input limits** of the event (decision
D77, 2026-10-13); the organizer's servers use the same values. They fit the public sample cases
(`contracts/public-cases.json`, 99 reports: median 7.9 MiB and 44 pages per PDF, 95% under 28.8 MiB
and 126 pages, the largest 49.3 MiB and 208 pages; PROBLEM_DESCRIPTION §5.2).

| Field | `config/dev.json` | What happens at the limit |
|---|---|---|
| `max_request_bytes` | `83886080` (80 MiB, official) | Larger request body → `413 REQUEST_TOO_LARGE`. PDFs travel as base64, which is about 4/3 of the file size, so all PDFs of one case together can be at most about 59 MiB. |
| `max_documents` | `3` (official) | More documents in a case → `422`, rule `max_documents`. |
| `max_document_bytes` | `52428800` (50 MiB, official) | A larger PDF (decoded size) → `422`, rule `max_document_bytes`. |
| `max_total_pages` | `300` (official) | More pages in all PDFs together → `422`, rule `max_total_pages`. |
| `max_result_bytes` | `2097152` (2 MiB) | Larger `result.json` → job fails with `WORKER_PROTOCOL_ERROR`; the first 2 MiB are kept. |
| `deadline_ms` | `180000` (3 min) | Whole-case time limit (start, load, assess, cleanup, exit) → `TIMEOUT`, reported as "The model could not process the documents within the 180-second limit." The event uses the same 3-minute limit. |
| `termination_grace_ms` | `2000` | Time between SIGTERM and SIGKILL when a run is stopped. |
| `max_log_bytes` | `1048576` (1 MiB) | Per stream (stdout, stderr); the rest is counted and dropped (`LOG_TRUNCATED`). |
| `max_observation_bytes` | `32768` | Size limit of the worker's progress file; larger → `SIDECAR_INVALID`. |
| `max_jobs` | `1000` | Stored jobs; beyond it new cases get `503 STORAGE_CAPACITY`. Old jobs are never deleted automatically. |
| `min_free_disk_bytes` | `104857600` (100 MiB) | Free space needed in `data_dir` for new cases; below it `503 STORAGE_CAPACITY`. |
| `max_diagnostics_bytes` | `32768` | Size limit of `diagnostics` blocks; lists are shortened and `truncated` set. |
| `max_field_errors` | `20` | Field errors / output issues listed per response. |
| `max_stack_frames` | `10` | Server stack frames shown for `500` in the dev profile. |

All values must be positive integers (`min_free_disk_bytes` may be 0). `deadline_ms` 180000
(3 minutes per case, model loading included) is the event's limit. The organizer's evaluation servers
use the same values for every field of this table. Lower `deadline_ms` locally to see how
much margin your module has.

## 4. Checklist file

The kit has two checklist files. `contracts/checklist.official.json` is the official checklist
(`APCTT-2026-CHECKLIST-v1`, `"official": true`, criteria `APCTT-C01` to `APCTT-C12`, decision D74):
the organizer runs every package with it, it is the default of `config/dev.json` and of your
configuration of README step 6, and `docs/EXAMPLE_REQUEST.json`, `tools/make_request.py` and
`tools/get_public_cases.py` carry its version. `contracts/checklist.demo.json` is a DEMO checklist
with two criteria, for the protocol tests and for one run with a changed checklist (below); it is
shown here because it is short:

```json
{
  "version": "DEMO-CHECKLIST-NOT-OFFICIAL",
  "official": false,
  "criteria": [
    {"criterion_id": "DEMO-C01", "description": "DEMO ONLY: the report states that a prototype or model was tested."},
    {"criterion_id": "DEMO-C02", "description": "DEMO ONLY: the report states that measured results met stated acceptance criteria."}
  ]
}
```

| Field | Rule |
|---|---|
| `version` | Non-blank string; cases must send it as `checklist_version`. |
| `official` | Optional boolean, default `false`. Only organizer checklists are `true`. |
| `criteria` | At least one; `criterion_id` is an identifier (unique), `description` a non-blank string. |

Your module receives `version` and `criteria` only. To test that your module does not depend on
the criterion IDs, copy a checklist file (the DEMO one is the quickest), change the IDs,
descriptions and `version`, point `checklist_path` of a copy of your configuration at it and send
cases with the new `checklist_version` (`tools/make_request.py --checklist <copy>` writes it).
That is the only use of the DEMO checklist: the organizer runs the official one.

## 5. `module`

| Field | Example | Meaning |
|---|---|---|
| `package_id` | `"my-team-module"` | Identifier shown in `/v1/package` and job `meta`. |
| `entrypoint` | `"participant_module.adapter:AssessmentModule"` | `import.path:ClassName` of your class. |
| `module_root` | `"/path/to/my_submission"` | Directory placed first on the worker's `sys.path` (the parent of your package). |
| `package_root` | `"/path/to/my_submission/participant_module"` | Directory hashed as your package. |
| `asset_root` | `"/path/to/my_submission/assets"` | Passed to `load()`; also hashed if outside `package_root`. |
| `declared_package_sha256` | `null` | Optional 64-hex hash you expect; a difference makes the server refuse new cases (`CONFIG_INVALID`). |
| `example_only` | `false` | `true` for the kit's examples; your module should say `false`. |
| `settings` | `{}` | Any JSON object; passed to `load()` as `config["module"]["settings"]`. |

`module_root`, `package_root` and `asset_root` must be existing directories (`MODULE_PATH_NOT_DIRECTORY` otherwise). A symlinked path is resolved to its target directory; symlinks *inside* the package make the hash unavailable (below).

**Package hash.** At start-up the server lists every regular file under `package_root` (label
`package/…`) and, if `asset_root` is not inside it, under `asset_root` (label `assets/…`), skipping
`__pycache__` directories, `.pyc`/`.pyo` files, `.DS_Store` and tool caches. It hashes the sorted
list of paths, sizes and SHA-256 values. A symlink anywhere inside makes the hash unavailable
(`verification_status: not_verified`). Compute the same value yourself:

<!-- example: package-digest -->
```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from apctt_contracts.hashing import package_digest, package_manifest

roots = [("package", Path("participant_example"))]
# If asset_root is outside package_root, add: roots.append(("assets", Path("path/to/assets")))
print(package_digest(package_manifest(roots)))
PY
```

## 6. `worker`

| Field | `config/dev.json` | Meaning |
|---|---|---|
| `python` | `null` | Interpreter that runs your module. `null` uses the server's own interpreter (fine for the examples). For your module, set it to the common base environment's Python that `tools/make_env.py` prints (`<base-dir>/bin/python`; [MODULE_GUIDE.md](MODULE_GUIDE.md) §7.2): at the event every team runs in that base. Relative paths resolve against the config file's directory, and symlinks are kept (so a virtual environment's `bin/python` works). |
| `launcher_argv` | `[]` | Command prefix placed before the worker command. Used by the organizer to run workers inside an isolated account, VM or container; leave empty. |
| `extra_env` | `{}` | Additional environment variables for the worker, e.g. `{"OMP_NUM_THREADS": "4"}`. At the event the organizer sets `APCTT_MODEL_STORE` here (the read-only model store); set the same locally, as an absolute path to your own copy of the catalogue models ([MODULE_GUIDE.md](MODULE_GUIDE.md) §7.3). Names must match `[A-Z_][A-Z0-9_]*`. Names the server sets itself (`PATH`, `HOME`, `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `LANG`, `LC_ALL`, `PYTHONUTF8`, `PYTHONHASHSEED`, `PYTHONDONTWRITEBYTECODE`, `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `CUDA_VISIBLE_DEVICES`) and credential-like names (any name containing `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `API_KEY`, `APIKEY`, `PRIVATE`, `AWS_`, `AZURE_`, `GOOGLE_`, `GITHUB_`, `HF_`, `ANTHROPIC` or `OPENAI`) are rejected (`CONFIG_INVALID`, detail `worker: value_error`). |

## 7. The `evaluation` profile

`evaluation` is how the organizer runs official assessments. It refuses to start unless all of
these hold: an official checklist (`official: true`, not the DEMO one), an event `schema_version`,
`example_only: false`, a `launcher_argv` whose first item is an existing file given by an absolute path, an
isolation evidence record in `isolation.evidence_file`, and loopback binding or a bearer token.
In `evaluation`, `500` responses never include stack frames. Participants use `dev`.

| Field | `config/dev.json` | Meaning |
|---|---|---|
| `evidence_file` (under `isolation`) | `null` | Path to the operator's isolation evidence record. Ignored by `dev`; required by `evaluation`. |

## 8. `auth`

`bearer_token_file` points to a file whose content (whitespace trimmed) is the token. When set,
`/health/ready`, `/v1/package` and `/v1/assessments*` require `Authorization: Bearer <token>`.
The token is never logged or returned. Local development normally leaves it `null`.

## 9. A complete local configuration

```json
{
  "config_version": "apctt-config-v1",
  "profile": "dev",
  "bind_host": "127.0.0.1",
  "bind_port": 8000,
  "data_dir": "../server_data",
  "worker_tmp_dir": "../worker_tmp",
  "schema_version": "apctt-pdf-example-v1",
  "checklist_path": "../contracts/checklist.official.json",
  "retry_after_seconds": 1,
  "limits": {
    "max_request_bytes": 83886080,
    "max_documents": 3,
    "max_document_bytes": 52428800,
    "max_total_pages": 300,
    "max_result_bytes": 2097152,
    "deadline_ms": 180000,
    "termination_grace_ms": 2000,
    "max_log_bytes": 1048576,
    "max_observation_bytes": 32768,
    "max_jobs": 1000,
    "min_free_disk_bytes": 104857600,
    "max_diagnostics_bytes": 32768,
    "max_field_errors": 20,
    "max_stack_frames": 10
  },
  "compute": {
    "device": "cpu",
    "require_cuda": false,
    "allow_cpu_fallback": false,
    "precision": "float32",
    "loading_policy": "cold_per_case",
    "seed": 42
  },
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
  },
  "isolation": {"evidence_file": null},
  "auth": {"bearer_token_file": null}
}
```
