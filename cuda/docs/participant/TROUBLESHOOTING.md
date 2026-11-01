# Troubleshooting — every code and what to do

kit 0.5.0

## 1. Where to look first

| Symptom | Look at | Section |
|---|---|---|
| `POST` or `GET` returned 4xx/5xx | `error.code`, `error.field_errors`, `diagnostics.parse_error` | §2 |
| `202` then the job finished with `result_origin: "supervisor"` | `result.error_code`, `diagnostics.output_issues`, `diagnostics.warning_codes`, `stderr.log` | §3 |
| A finished job has warnings or `null` timings | `diagnostics.warning_codes`, `diagnostics.unavailable_reasons` | §4, §5 |
| The server exits at start with `APCTT_START_REFUSED` | the code and field lines printed after it | §6 |
| `/health/ready` is 503 | `reason_code` | §7 |
| CUDA configuration never becomes ready | `compute_status.reason` | §8 |
| Something else | Frequent questions | §9 |

Your module's own output is in `server_data/jobs/<assessment_id>/stdout.log` and `stderr.log`
(uncaught exceptions print their traceback to `stderr.log`); its exact result bytes are in
`raw_output.bin` next to them.

## 2. HTTP errors

| Code | HTTP | Typical cause | What to do |
|---|---|---|---|
| `INVALID_JSON` | 400 | Syntax error, duplicate key, `NaN`, not UTF-8 (§2.1). | Fix the JSON; use `diagnostics.parse_error.reason`. |
| `REQUEST_VALIDATION_ERROR` | 422 | A field breaks the case schema or a case rule. | Read each `field_errors[]` entry; rules in [DATA_CONTRACT.md](DATA_CONTRACT.md) §4.1. |
| `VERSION_MISMATCH` | 422 | `schema_version` or `checklist_version` differs from the server's. | Use the values from `GET /v1/package` (`expected` shows them). Requests made by `tools/make_request.py` and `tools/get_public_cases.py`, and `docs/EXAMPLE_REQUEST.json`, carry the official checklist's version, which `config/dev.json` and your configuration of README step 6 run; a request for a changed checklist needs `tools/make_request.py --checklist <copy>`. |
| `UNSUPPORTED_INPUT_PROFILE` | 422 | `input_profile` is not `pdf_report`. | Send `pdf_report`. |
| `REQUEST_TOO_LARGE` | 413 | Body over `max_request_bytes`. | Send a smaller case; the server never truncates. Locally you may raise the limit. |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | Missing/other `Content-Type`, a charset other than UTF-8, or `Content-Encoding: gzip`. | Send `Content-Type: application/json`, uncompressed. |
| `REQUEST_ID_CONFLICT` | 409 | The `request_id` was used before with a different case, configuration or package — typically after you changed your code. | Use a new `request_id`, or stop the server and delete `server_data/`. |
| `BUSY` | 503 | Another case is running. Nothing was stored. | Wait `Retry-After` seconds and send the same request again. |
| `NOT_READY` | 503 | The server cannot accept cases (`/health/ready` says why). | Fix the reason in §7, then retry. |
| `STORAGE_CAPACITY` | 503 | `max_jobs` reached or disk space below `min_free_disk_bytes`. | Stop the server, archive or delete `server_data/`, or raise the limit locally. |
| `JOB_NOT_FOUND` | 404 | Unknown or malformed `assessment_id`, or a different `data_dir`. | Use the `Location` header from the `POST`. |
| `ROUTE_NOT_FOUND` | 404 | Wrong path. | See [API_REFERENCE.md](API_REFERENCE.md). |
| `METHOD_NOT_ALLOWED` | 405 | e.g. `DELETE /v1/assessments`. | There is no cancel or delete endpoint. |
| `AUTH_REQUIRED` | 401 | A bearer token is configured and none was sent. | Send `Authorization: Bearer <token>`. |
| `FORBIDDEN` | 403 | The token is wrong. | Use the configured token. |
| `INTERNAL_ERROR` | 500 | A server bug. | Report `meta.trace_id`; in dev, `diagnostics.stack_frames` shows where. |

### 2.1 `parse_error.reason` values (400)

| Reason | Meaning |
|---|---|
| `syntax` | Not valid JSON; `line` and `column` point to the problem. |
| `duplicate_key` | The same key appears twice in one object. |
| `non_finite_number` | `NaN`, `Infinity`, `-Infinity`, or a number too large for a float (e.g. `1e400`). |
| `number_out_of_range` | An integer literal with more than 4 300 digits. |
| `invalid_utf8` | The body is not valid UTF-8. |
| `byte_order_mark` | The body starts with a UTF-8 byte order mark. |
| `lone_surrogate` | A `\ud800`-style escape that is not part of a valid pair. |
| `too_deep` | Nesting too deep to parse. |

## 3. Job failures (`result_origin: "supervisor"`)

The server replaced your output with a failure result. `result.error_code` says why:

| `error_code` | Cause | How to confirm | Fix |
|---|---|---|---|
| `TIMEOUT` | The case exceeded `deadline_ms` (3 minutes; start-up, import, `load`, `assess`, cleanup and exit all count): the model could not process the documents in time. Even a result already written is discarded. | `diagnostics.timed_out: true`; `worker_phase` shows where it stopped. | Load faster, check `context.remaining_ms()`, return early. |
| `MODULE_ERROR` | Your code raised an exception in `load` or `assess`. | Traceback in `stderr.log`; `worker_phase` is `loading` or `assessing`. | Fix the exception. |
| `CLEANUP_ERROR` | `reset_case()` or `close()` raised. A result had been written but the case still fails. | Traceback in `stderr.log`; `raw_output.bin` holds your result. | Make both methods never raise. |
| `INVALID_OUTPUT` | The result broke the contract, was not valid strict JSON, or `assess` returned something that is not a JSON mapping (NaN, non-string keys, custom objects). | `diagnostics.output_issues` (path + rule); none listed when conversion to JSON already failed (see `stderr.log`). | [DATA_CONTRACT.md](DATA_CONTRACT.md) §4.2 explains each rule. |
| `WORKER_PROTOCOL_ERROR` | No `result.json` after a normal exit (for example `os._exit(0)` inside `assess`), `result.json` is a symlink, directory or not a regular file, the result exceeds `max_result_bytes`, or the worker reported a failure but exited with 0. | `warning_codes` may show `RESULT_OVERSIZE`, `RESULT_FILE_REJECTED`. | Return the result from `assess`; never write or replace `result.json` yourself. |
| `WORKER_CRASH` | The worker process ended abnormally: killed by a signal, a native crash, `sys.exit(n)` / `os._exit(n)` with `n ≠ 0`, or the interpreter could not start. | `diagnostics.exit_code` / `exit_signal`; `WORKER_SPAWN_FAILED` if `worker.python` could not be started. | Check `stderr.log`; do not call `sys.exit`; check `worker.python`. |
| `OUT_OF_MEMORY` | PyTorch raised `torch.cuda.OutOfMemoryError`, or your exception set `apctt_error_code = "OUT_OF_MEMORY"`. A process killed by the system without such an exception is reported as `WORKER_CRASH`. | `stderr.log`, `diagnostics.compute`. | Reduce memory use. |
| `CUDA_UNAVAILABLE` | Your exception set `apctt_error_code = "CUDA_UNAVAILABLE"`. | `stderr.log`. | Check the worker environment's CUDA setup. |
| `CUDA_RUNTIME_ERROR` | Your exception set `apctt_error_code = "CUDA_RUNTIME_ERROR"`. | `stderr.log`. | As above. |
| `SERVER_RESTARTED` | The server stopped (Ctrl+C, crash, reboot) while this job was queued or running. At the next start the job was closed; it is never re-run automatically. | `diagnostics.phase: "recovered"`, `previous_status`. | Submit again with a new `request_id`. |
| `INTERNAL_ERROR` | The server failed while handling the run. | `diagnostics.warning_codes` contains `SUPERVISOR_INTERNAL_ERROR`. | Report `meta.trace_id` and the job ID. |

When several apply, the first match wins: `TIMEOUT`, then a failure your module reported
(`MODULE_ERROR`, `CLEANUP_ERROR`, …), then `WORKER_CRASH`, then `WORKER_PROTOCOL_ERROR`, then
`INVALID_OUTPUT`.

A result your module returns itself with `assessment_status: "failed"` is not in this table: it is
a valid result with `result_origin: "module"` and your own `error_code`.

## 4. `diagnostics.warning_codes`

Warnings never change a valid result by themselves.

| Code | Meaning |
|---|---|
| `SIDECAR_MISSING` | The worker's progress file was not found, so `load`/`assess`/`cleanup` timings are unknown. |
| `SIDECAR_INVALID` | The progress file was unreadable, too large, not valid, from another run, or a symlink; it was ignored. |
| `FAILURE_RECORD_INVALID` | The worker's failure record was unreadable or invalid; it was ignored. |
| `RESULT_OVERSIZE` | `result.json` exceeded `max_result_bytes` (the job fails with `WORKER_PROTOCOL_ERROR`). |
| `RAW_TRUNCATED` | Only the first `max_result_bytes` of your result bytes were kept. |
| `RESULT_FILE_REJECTED` | `result.json` was a symlink, directory, other special file or unreadable; it was not read. |
| `LOG_TRUNCATED` | `stdout` or `stderr` exceeded `max_log_bytes`; the rest was counted but not stored. |
| `WORKER_SPAWN_FAILED` | The worker interpreter could not be started (check `worker.python`). |
| `DESCENDANTS_TERMINATED` | Processes your module started were still running after the worker exited and were stopped. |
| `TERMINATED_SIGKILL` | The run ignored SIGTERM and was killed after `termination_grace_ms`. |
| `CLEANUP_UNCONFIRMED` | The server could not confirm that all processes of the run stopped; it will not accept new cases until it can (`RECOVERY_PENDING`). |
| `SERVER_RESTARTED` | The job was closed at start-up after an interruption (§3). |
| `SUPERVISOR_INTERNAL_ERROR` | The server failed while handling the run (§3, `INTERNAL_ERROR`). |

## 5. `diagnostics.unavailable_reasons`

| Key → value | Meaning |
|---|---|
| `load` / `assess` / `cleanup` → `NOT_REACHED` | The run stopped before that step. |
| `load` / `assess` / `cleanup` → `SIDECAR_UNAVAILABLE` | No valid progress file, so no worker timings (see `SIDECAR_MISSING`/`SIDECAR_INVALID`). |
| `peak_memory` → `NOT_MEASURED` | GPU memory is measured only when your module used PyTorch on CUDA. |
| `torch` → `TORCH_NOT_IMPORTED` | Your module did not import PyTorch, so torch fields are `null`. |
| `kernel_probe_passed` → `PROBE_RUNS_AT_SERVER_START` | The CUDA check runs once at server start, not per case. |
| `compute` → `COMPUTE_QUERY_FAILED` | Reading PyTorch/CUDA information failed; the result is unaffected. |

## 6. Start-up refusals

The server prints `APCTT_START_REFUSED <CODE> …` and exits with status 2.

| Code | Cause | Fix |
|---|---|---|
| `CONFIG_UNREADABLE` | The config file is missing or not strict JSON. | Check the `--config` path and JSON syntax. |
| `CONFIG_INVALID` | A field is missing, unknown or out of range; the lines after the code name each field and error type. | See [CONFIGURATION.md](CONFIGURATION.md). |
| `PROFILE_MISMATCH` | `--profile` differs from the file's `profile`. | Use the same value. |
| `CHECKLIST_UNREADABLE` | `checklist_path` is missing or not strict JSON. | Check the path (relative to the config file). |
| `CHECKLIST_INVALID` | The checklist breaks its format (e.g. duplicate `criterion_id`). | See [CONFIGURATION.md](CONFIGURATION.md) §4. |
| `WORKER_PYTHON_NOT_FOUND` | `worker.python` does not exist. | Give the interpreter path of your worker environment. |
| `MODULE_PATH_NOT_DIRECTORY` | `module_root`, `package_root` or `asset_root` is not a directory. | Fix the paths. |
| `BEARER_TOKEN_FILE_MISSING` | `auth.bearer_token_file` does not exist. | Create it or set `null`. |
| `SCHEMA_VERSION_MISSING` | `schema_version` is blank. | Set it. |
| `DEV_NON_LOOPBACK_BIND` | `dev` with a non-loopback `bind_host`. | Use `127.0.0.1`. |
| `STORE_LOCKED` | Another server already uses this `data_dir`. | Stop the other server or use a different `data_dir`. |
| `DATA_DIR_UNUSABLE` | `data_dir` exists but is not a directory, or its lock file cannot be opened. | Point `data_dir` at a writable directory. |
| `EVALUATION_CHECKLIST_NOT_OFFICIAL` | `evaluation` with the DEMO or a non-official checklist. | Organizer only. |
| `EVALUATION_CHECKLIST_DRAFT` | `evaluation` with a checklist whose `version` contains `DRAFT`. | Organizer only: approve and rename the checklist first. |
| `EVALUATION_DEV_SCHEMA` | `evaluation` with the dev `schema_version`. | Organizer only. |
| `EVALUATION_EXAMPLE_MODULE` | `evaluation` with `example_only: true`. | Organizer only. |
| `EVALUATION_LOADING_POLICY` | `evaluation` with a loading policy other than `cold_per_case`. | Organizer only. |
| `EVALUATION_LAUNCHER_MISSING` | `evaluation` without an isolation launcher. | Organizer only. |
| `EVALUATION_ISOLATION_EVIDENCE_MISSING` | `evaluation` without an isolation evidence record. | Organizer only. |
| `EVALUATION_ISOLATION_EVIDENCE_INVALID` | The isolation record is malformed or not all checks passed. | Organizer only. |
| `EVALUATION_NON_LOOPBACK_WITHOUT_AUTH` | `evaluation` on a network address without a bearer token. | Organizer only. |

A second server on the same `data_dir` is refused before start with `STORE_LOCKED` (exit
status 2). Only if the other server takes the lock in the moment between that check and start-up
does the second one stop later, with exit status 3 and `StoreLockedError` in its log.

## 7. `/health/ready` reasons

`READY`, `BUSY`, `CONFIG_INVALID`, `STORAGE_CAPACITY`, `CUDA_UNAVAILABLE`, `CUDA_RUNTIME_ERROR` and
`RECOVERY_PENDING` are described in [API_REFERENCE.md](API_REFERENCE.md) §3. `CONFIG_INVALID` here
means the computed package hash differs from `declared_package_sha256`; correct or remove the
declaration and restart.

## 8. CUDA start-up check (`compute_status.reason`)

Only for configurations with `require_cuda: true`. `status` is `PASS`, `CUDA_UNAVAILABLE` or
`CUDA_RUNTIME_ERROR`.

| Reason | Meaning |
|---|---|
| `DEVICE_NOT_CUDA` | The configured device is not `cuda:N`. |
| `WORKER_PYTHON_NOT_FOUND` | The worker interpreter is missing or not executable. |
| `TORCH_NOT_INSTALLED` | PyTorch is not installed in the worker environment. |
| `CPU_ONLY_TORCH` | The installed PyTorch has no CUDA support. |
| `CUDA_NOT_AVAILABLE` | PyTorch cannot use CUDA (driver or device missing). |
| `CUDA_QUERY_FAILED` | Asking PyTorch about CUDA raised an error. |
| `DEVICE_NOT_VISIBLE` | The device index is not visible to the worker. |
| `KERNEL_PROBE_FAILED` | A 64×64 matrix multiply on the device raised an error. |
| `KERNEL_RESULT_INVALID` | The check ran but returned a wrong or non-finite value. |
| `PROBE_TIMEOUT` | The check did not finish within 120 seconds. |
| `PROBE_PROTOCOL_ERROR` | The check's output was missing or inconsistent. |

## 9. Frequent questions

**I changed my code and now get 409.** The `request_id` is bound to the case, configuration and
package hash. Use a new ID (e.g. add a counter) or delete `server_data/` while the server is stopped.

**`ModuleNotFoundError` for my package.** `module_root` must be the directory that *contains* your
package, and `entrypoint` must be `package.module:Class`. Your shell's `PYTHONPATH` is not passed.

**My dependencies are not found.** The worker runs `worker.python`; `null` means the server's own
environment. Point it at the common base environment's `bin/python` from `tools/make_env.py`, and
run `tools/make_env.py --check --base-dir DIR`. Only the packages of the base exist
(`constraints-base.txt`); there is no way to add one. Third-party pure-Python code may be copied into
`participant_module/` with its licence ([LICENSING.md](LICENSING.md) §2.2).

**My model is not found, or `local_files_only` fails.** Load catalogue models from
`<APCTT_MODEL_STORE>/<org>/<name>/<revision>/` with `local_files_only=True`, set `HF_HUB_OFFLINE=1`
in your code, and set `APCTT_MODEL_STORE` in `worker.extra_env` to the absolute path of your store
([MODULE_GUIDE.md](MODULE_GUIDE.md) §7.3). `tools/get_models.py --check --store DIR` verifies your
copies. A model outside the catalogue is not available at the event.

**`tools/make_submission.py` says `REFUSED`.** Each `ERROR` line names one problem: a file outside
the five entries of the bundle is left out, but wheels, virtual environments, compiled extensions,
undeclared weight files over 50 MiB, a model outside the catalogue or another `base_id` are refused
([SUBMISSION.md](SUBMISSION.md) §1.3). `... is the weights of catalogue model <id>` means a file
has the bytes of a catalogue model's weights: delete it and name the model by `catalog_id`; the
organizer provides it in the model store.

**My `print` output is missing.** It is in `stdout.log` of the job, never in the HTTP response.

**The result looks right but I get `INVALID_OUTPUT`.** Read `diagnostics.output_issues`. The usual
causes: `assessment_target`, `case_id`, `schema_version` or `checklist_version` not copied exactly;
a checklist criterion missing from `criterion_status` or `evidence_reference`; `met` without
evidence; `insufficient evidence` without `missing_evidence`; `null` for an optional field.

**The keys in my result come back in a different order.** Results are stored as canonical JSON with
sorted keys. Values are unchanged, and `raw_output.bin` keeps your exact bytes.

**`TIMEOUT` although `assess` is fast.** The limit covers interpreter start-up, imports and `load`
for every case. Check `timings_ms.load` of a successful run and heavy imports at module level.

**`/v1/package` says `not_verified`.** A symlink exists inside `package_root` or `asset_root`.
Replace it with a real file or directory.

**`/health/ready` stays `RECOVERY_PENDING`.** Processes from an earlier run could not be confirmed
stopped (for example they belong to another user). Stop them; the server retries every 2 seconds.

**Port 8000 is in use.** Set another `bind_port` in your local config.

**The conformance tests fail on Windows.** Native Windows is not supported; use WSL2.
