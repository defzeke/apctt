# Data contract — request, case and result

kit 0.5.0 | API `apctt-api-v0.2-draft` | schema `apctt-pdf-example-v1` (dev)

This page defines every field the server accepts and every field your module returns, the rules
the server enforces, and the exact rule IDs it reports. Machine-readable schemas are in
`contracts/openapi.json` (HTTP) and `contracts/worker-protocol.json` (worker input and result).

## 1. Request and case

### 1.1 Request body

`POST /v1/assessments` takes exactly two fields:

| Field | Type | Rule |
|---|---|---|
| `request_id` | string | ASCII `[A-Za-z0-9_-]`, 1–128 characters. Your idempotency key (§5). |
| `case` | object | The case below. |

### 1.2 Case

| Field | Required | Type and rule |
|---|---|---|
| `schema_version` | yes | Must equal the server's schema version (`apctt-pdf-example-v1` in dev; see `GET /v1/package`). |
| `input_profile` | yes | Must be `pdf_report` (the only supported profile). |
| `checklist_version` | yes | Must equal the version of the server's checklist: `APCTT-2026-CHECKLIST-v1` for the official checklist (`config/dev.json`, your configuration of README step 6, the organizer's servers; decision D74). |
| `case_id` | yes | Identifier (§1.5). Neutral ID; different from `request_id`; may repeat across requests. |
| `assessment_as_of` | yes | `"YYYY-MM-DD"` (a real calendar date) or `null`. The server never fills in today's date. |
| `assessment_target` | yes | Object: `target_id` (identifier), `type` (`component`, `subsystem`, `system` or `technology`), `description` (non-blank string). |
| `intended_application` | yes | Non-blank string. |
| `intended_environment` | yes | Non-blank string. |
| `title` | yes | Non-blank string. |
| `documents` | yes | Array of 1 to `max_documents` (3, the official limit, decision D77) documents. |
| `description` | no | Non-blank string. |
| `benefits` | no | Non-blank string. |
| `sdg_targets` | no | Array of distinct integers from 6, 11, 13, 14. |
| `sdg_rationale` | no | Non-blank string. |

### 1.3 Document

Every document is a PDF file, sent inside the JSON as base64.

| Field | Required | Type and rule |
|---|---|---|
| `document_id` | yes | Identifier, unique within the case. |
| `title` | yes | Non-blank string. |
| `document_date` | yes | Non-blank string (any format) or `null`. |
| `language` | yes | Must be `en`. |
| `format` | yes | Must be `pdf`. |
| `content_base64` | yes | The PDF file, standard base64 (RFC 4648 alphabet, with padding, no line breaks or spaces). Decoded, it must start with `%PDF-`, be at most `max_document_bytes` (50 MiB, official) and be a PDF the server can open and count pages of. All documents together hold at most `max_total_pages` pages (300, official). |
| `sha256` | yes | SHA-256 of the decoded PDF bytes, 64 lowercase hexadecimal characters. It must match. |

`tools/make_request.py` builds these fields from PDF files for you (see the README, step 7).

### 1.4 What your module receives

The server checks and stores each PDF, counts its pages, and hands your module a **read-only copy**
of every file. In `case.documents` your module sees, instead of `content_base64`:

| Field | Meaning |
|---|---|
| `page_count` | Number of pages the server counted (≥ 1). |
| `path` | Absolute path of the PDF inside your case's private scratch directory, e.g. `…/documents/001.pdf`. |

All other document fields (`document_id`, `title`, `document_date`, `language`, `format`, `sha256`)
are passed as sent. Reading the PDF (text extraction, layout, tables, OCR of scanned pages) is your
module's job; choose the tools that give the best assessment. Pages are numbered from 1 in file order.

### 1.5 General rules

- **Unknown fields are rejected** at every level. This includes any label-like field such as
  `trlCurrent`, `trlBegin`, `trlEnd` or `reference_trl`. Unknown names are reported as
  `<unknown-field>`; the name you sent is never echoed.
- **Optional fields are omitted, not `null`.** `"description": null` is rejected
  (`omit_instead_of_null`). Required nullable fields (`assessment_as_of`, `document_date`) must be
  present, with `null` allowed.
- **Identifier** = ASCII, starts with a letter or digit, then letters, digits, `_ . : -`,
  at most 128 characters in total (`[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`).
- **Non-blank string** = contains at least one non-whitespace character.
- **Types are strict**: no conversion between strings, numbers and booleans.
- **Encoding**: `Content-Type: application/json` (optionally `; charset=utf-8`), UTF-8, no byte
  order mark, no `Content-Encoding` other than `identity`. Duplicate keys, `NaN`, `Infinity`,
  numbers too large for a float, integers longer than 4 300 digits, lone surrogates and very deep
  nesting are rejected as `INVALID_JSON`.
- **Size**: the body may not exceed `max_request_bytes` (80 MiB, official; base64 makes a PDF about
  4/3 of its file size, so all PDFs of a case together can be at most about 59 MiB). The limit is enforced while
  reading, with or without `Content-Length`. Nothing is truncated or summarized; the request is
  rejected (`413 REQUEST_TOO_LARGE`).
- **Order of checks**: size and JSON → schema → input profile → versions → document checks
  (uniqueness, base64, size, `%PDF-` header, SHA-256, count) → page count (the server opens each
  PDF in a separate, time-limited process) → total pages. Only the first failing stage is reported, with all of its field errors (up to 20 shown;
  `diagnostics.field_errors_total` gives the full count).

### 1.6 Request errors, as the server reports them

Each row was produced by changing one thing in `docs/EXAMPLE_REQUEST.json`.

| Change to the example request | HTTP | `error.code` | `path` | `rule` | `expected` |
|---|---|---|---|---|---|
| `request_id` is `"my id"` (space) | 422 | `REQUEST_VALIDATION_ERROR` | `request_id` | `request_id_pattern` | ASCII [A-Za-z0-9_-]{1,128} |
| `request_id` missing | 422 | `REQUEST_VALIDATION_ERROR` | `request_id` | `required` | field is present |
| extra top-level field `"debug": true` | 422 | `REQUEST_VALIDATION_ERROR` | `<unknown-field>` | `unknown_field` | only fields defined by the contract |
| extra case field `"trlCurrent": 6` | 422 | `REQUEST_VALIDATION_ERROR` | `case.<unknown-field>` | `unknown_field` | only fields defined by the contract |
| `title` missing | 422 | `REQUEST_VALIDATION_ERROR` | `case.title` | `required` | field is present |
| `title` is `"   "` | 422 | `REQUEST_VALIDATION_ERROR` | `case.title` | `non_blank_string` | string with a non-whitespace character |
| `case_id` is `"DEMO 001"` | 422 | `REQUEST_VALIDATION_ERROR` | `case.case_id` | `identifier_pattern` | ASCII identifier [A-Za-z0-9][A-Za-z0-9_.:-]{0,127} |
| `assessment_as_of` is `"2026-02-30"` | 422 | `REQUEST_VALIDATION_ERROR` | `case.assessment_as_of` | `iso_date` | YYYY-MM-DD or null |
| `assessment_as_of` key missing | 422 | `REQUEST_VALIDATION_ERROR` | `case.assessment_as_of` | `required` | field is present |
| `assessment_target.type` is `"product"` | 422 | `REQUEST_VALIDATION_ERROR` | `case.assessment_target.type` | `allowed_value` | component\|subsystem\|system\|technology |
| `documents` is `[]` | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents` | `non_empty_array` | array with at least one item |
| document `language` is `"ko"` | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].language` | `allowed_value` | en |
| document `document_date` key missing | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].document_date` | `required` | field is present |
| document `format` is `"docx"` | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].format` | `allowed_value` | pdf |
| `content_base64` is not base64 (or has line breaks or wrong padding) | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].content_base64` | `base64_content` | strict standard base64 |
| `content_base64` decodes to a text file | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].content_base64` | `pdf_signature` | a PDF file (%PDF- header) |
| a PDF larger than 50 MiB | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].content_base64` | `max_document_bytes` | at most 52428800 bytes per document |
| a damaged or password-protected PDF | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].content_base64` | `pdf_readable` | a PDF the server can open and count pages of |
| `sha256` of another file | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].sha256` | `document_sha256_match` | SHA-256 of the decoded document |
| `sha256` in upper case | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[0].sha256` | `sha256_pattern` | 64 lowercase hexadecimal characters |
| two documents with the same `document_id` | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents[1].document_id` | `unique_document_id` | document_id unique within the case |
| `sdg_targets` is `[6, 6]` | 422 | `REQUEST_VALIDATION_ERROR` | `case.sdg_targets` | `unique_items` | array without duplicate values |
| `sdg_targets` is `[7]` | 422 | `REQUEST_VALIDATION_ERROR` | `case.sdg_targets[0]` | `allowed_value` | one of 6, 11, 13, 14 |
| `input_profile` is `"canonical_report_text"` | 422 | `UNSUPPORTED_INPUT_PROFILE` | `case.input_profile` | `supported_input_profile` | pdf_report |
| `schema_version` is `"apctt-v04"` | 422 | `VERSION_MISMATCH` | `case.schema_version` | `profile_schema_version` | apctt-pdf-example-v1 |
| `checklist_version` is `"OFFICIAL-1"` | 422 | `VERSION_MISMATCH` | `case.checklist_version` | `profile_checklist_version` | APCTT-2026-CHECKLIST-v1 |
| 4 documents | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents` | `max_documents` | at most 3 documents |
| more than 300 pages in all PDFs | 422 | `REQUEST_VALIDATION_ERROR` | `case.documents` | `max_total_pages` | at most 300 pages in total |
| body is `[]` | 422 | `REQUEST_VALIDATION_ERROR` | `<body>` | `object` | JSON object |

## 2. Result

### 2.1 Fields

Your module returns one JSON object with exactly these fields. Any other field fails the result.

| Field | Required | Type and rule |
|---|---|---|
| `schema_version` | yes | Copy of `case.schema_version`. |
| `checklist_version` | yes | Copy of the checklist `version` (equals `case.checklist_version`). |
| `case_id` | yes | Copy of `case.case_id`. |
| `assessment_target` | yes | Exact copy of `case.assessment_target`. |
| `assessment_status` | yes | `completed`, `abstained` or `failed`. |
| `predicted_trl` | yes | Integer 1–9 when completed, otherwise `null`. |
| `criterion_status` | yes | Object: criterion ID → `met`, `not met` or `insufficient evidence`. |
| `evidence_reference` | yes | Object: criterion ID → array of evidence items (§2.3). |
| `missing_evidence` | yes | Object: criterion ID → non-blank explanation. May be `{}`. |
| `abstention_reason` | yes | When abstained: `insufficient evidence`, `conflicting evidence`, `ambiguous scope` or `out of scope`; otherwise `null`. |
| `error_code` | yes | When failed: a non-blank string of your choice; otherwise `null`. |
| `confidence` | no | Number 0–1 (integers 0 and 1 allowed; booleans not). Not with `failed`. |
| `uncertainty` | no | `{"kind": "trl_interval", "lower": L, "upper": U}`, integers 1–9, `L ≤ U`; when completed, `L ≤ predicted_trl ≤ U`. Not with `failed`. |
| `trl_rationale` | no | Non-blank string explaining `predicted_trl`. Only with `completed`. |
| `criterion_rationale` | no | Object: criterion ID → non-blank explanation. Any subset of criteria. Not with `failed`. |

Optional fields are omitted when unused; `null` is not accepted for them.

### 2.2 Status combinations

| `assessment_status` | `predicted_trl` | `abstention_reason` | `error_code` | criteria maps |
|---|---|---|---|---|
| `completed` | integer 1–9 | `null` | `null` | every checklist criterion present |
| `abstained` | `null` | one of the four reasons | `null` | every checklist criterion present |
| `failed` | `null` | `null` | non-blank string | may be empty; entries present must be valid |

`abstained` is a scientific answer ("the evidence does not allow a TRL"), not an error.

### 2.3 Evidence items and consistency rules

Each evidence item is an object with only these fields:

| Field | Required | Rule |
|---|---|---|
| `document_id` | yes | A document in this case. |
| `page` | yes | Integer from 1 to that document's `page_count`. |
| `supporting_span` | no | Non-blank string: the words on that page that support the judgment. Recommended for every `met` item. The server does not compare it with the PDF text (extraction differs between tools); judges read it next to the cited page. |

For `completed` and `abstained` results:

1. `criterion_status` and `evidence_reference` each have **exactly one key per checklist criterion**.
2. A `met` criterion has **at least one** evidence item.
3. An `insufficient evidence` criterion has a **non-blank explanation** in `missing_evidence`.
4. A `not met` criterion needs no evidence, but may cite the pages that show it is not met.
5. `missing_evidence` and `criterion_rationale` may only use checklist criterion IDs.

These checks are structural. They do not judge whether your evidence is convincing.

### 2.4 How the server treats your result

- A valid result is stored and returned with the same values (`result_origin: "module"`). It is
  stored as canonical JSON, so object keys come back sorted. The organizer keeps the exact bytes
  you wrote (`raw_output.bin`) and their SHA-256.
- An invalid result is replaced by a failure result built by the server
  (`result_origin: "supervisor"`, `assessment_status: "failed"`, `error_code: "INVALID_OUTPUT"`,
  empty maps). `diagnostics.output_issues` lists up to 20 `{path, rule}` entries (§4.2).
  Module-invented keys appear as `<unknown-field>`; your text is never copied into diagnostics.
- Results larger than `max_result_bytes` (2 MiB in dev) fail with `WORKER_PROTOCOL_ERROR`.
- Every failure result built by the server comes with `diagnostics.failure`: a fixed sentence per
  `error_code` (for example "The model could not process the documents within the 180-second
  limit." for `TIMEOUT`, which counts loading and clean-up too). A `failed` result your module
  returns itself has no such block; its `error_code` is your own text.
- Page ranges are checked against the case as the server stores it, where every document carries
  `page_count`. When you check a result yourself, validate against that view (MODULE_GUIDE §6 builds
  it with `local_worker_case`); against the raw request only `page >= 1` can be checked.

## 3. Result examples

All examples below are checked against the case of `docs/EXAMPLE_REQUEST.json` with the
two-criterion DEMO checklist (`DEMO-CHECKLIST-NOT-OFFICIAL`, `DEMO-C01` and `DEMO-C02`), only to keep
them short; with the official checklist every map has twelve entries, as in
`docs/EXAMPLE_RESPONSE.json`. That case has one PDF document `D001` (`docs/EXAMPLE_REPORT.pdf`, 2 pages). Page 1 says "A synthetic prototype
was tested in a laboratory water tank."; page 2 says "Measured results and acceptance criteria are
not supplied.".

### 3.1 Valid: completed

<!-- example: valid-result completed -->
```json
{
  "schema_version": "apctt-pdf-example-v1",
  "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
  "case_id": "DEMO-001",
  "assessment_target": {
    "target_id": "T001",
    "type": "component",
    "description": "Synthetic sensor fixture; not a validated technology"
  },
  "assessment_status": "completed",
  "predicted_trl": 3,
  "criterion_status": {"DEMO-C01": "met", "DEMO-C02": "not met"},
  "evidence_reference": {
    "DEMO-C01": [
      {
        "document_id": "D001",
        "page": 1,
        "supporting_span": "tested in a laboratory water tank"
      }
    ],
    "DEMO-C02": [{"document_id": "D001", "page": 2}]
  },
  "missing_evidence": {},
  "abstention_reason": null,
  "error_code": null,
  "confidence": 0.6,
  "uncertainty": {"kind": "trl_interval", "lower": 2, "upper": 4},
  "trl_rationale": "A laboratory test is reported; no measured results against criteria.",
  "criterion_rationale": {
    "DEMO-C01": "Page 1 reports a laboratory test of the prototype.",
    "DEMO-C02": "Page 2 states that measured results are not supplied."
  }
}
```

### 3.2 Valid: abstained

<!-- example: valid-result abstained -->
```json
{
  "schema_version": "apctt-pdf-example-v1",
  "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
  "case_id": "DEMO-001",
  "assessment_target": {
    "target_id": "T001",
    "type": "component",
    "description": "Synthetic sensor fixture; not a validated technology"
  },
  "assessment_status": "abstained",
  "predicted_trl": null,
  "criterion_status": {"DEMO-C01": "met", "DEMO-C02": "insufficient evidence"},
  "evidence_reference": {
    "DEMO-C01": [{"document_id": "D001", "page": 1, "supporting_span": "tested"}],
    "DEMO-C02": []
  },
  "missing_evidence": {"DEMO-C02": "No measured results are reported."},
  "abstention_reason": "insufficient evidence",
  "error_code": null
}
```

### 3.3 Valid: failed, reported by your module

<!-- example: valid-result failed -->
```json
{
  "schema_version": "apctt-pdf-example-v1",
  "checklist_version": "DEMO-CHECKLIST-NOT-OFFICIAL",
  "case_id": "DEMO-001",
  "assessment_target": {
    "target_id": "T001",
    "type": "component",
    "description": "Synthetic sensor fixture; not a validated technology"
  },
  "assessment_status": "failed",
  "predicted_trl": null,
  "criterion_status": {},
  "evidence_reference": {},
  "missing_evidence": {},
  "abstention_reason": null,
  "error_code": "TEAM_MODEL_INPUT_TOO_LONG"
}
```

This is returned with `result_origin: "module"`. If your module raises instead, the server builds a
similar result with `error_code: "MODULE_ERROR"` and `result_origin: "supervisor"`.

### 3.4 Invalid results and the rule each one breaks

Each block below replaces the named top-level fields of the **completed** example (§3.1) and shows
the rule the server reports.

`met` without evidence → `evidence_required_for_met` at `result.evidence_reference.DEMO-C01`:

<!-- example: invalid-result rule=evidence_required_for_met -->
```json
{"evidence_reference": {"DEMO-C01": [], "DEMO-C02": []}}
```

`insufficient evidence` without an explanation → `missing_evidence_required_for_insufficient`:

<!-- example: invalid-result rule=missing_evidence_required_for_insufficient -->
```json
{"criterion_status": {"DEMO-C01": "met", "DEMO-C02": "insufficient evidence"}, "missing_evidence": {}}
```

A criterion missing from `evidence_reference` → `evidence_keys_match_checklist`:

<!-- example: invalid-result rule=evidence_keys_match_checklist -->
```json
{"evidence_reference": {"DEMO-C01": [{"document_id": "D001", "page": 1}]}}
```

A blank supporting span → `supporting_span_nonblank`:

<!-- example: invalid-result rule=supporting_span_nonblank -->
```json
{"evidence_reference": {"DEMO-C01": [{"document_id": "D001", "page": 1, "supporting_span": "  "}], "DEMO-C02": []}}
```

A page beyond the end of the document (it has 2 pages) → `evidence_page_in_document`:

<!-- example: invalid-result rule=evidence_page_in_document -->
```json
{"evidence_reference": {"DEMO-C01": [{"document_id": "D001", "page": 5}], "DEMO-C02": []}}
```

An evidence item without `page` → `evidence_item_required`:

<!-- example: invalid-result rule=evidence_item_required -->
```json
{"evidence_reference": {"DEMO-C01": [{"document_id": "D001"}], "DEMO-C02": []}}
```

A document that does not exist in this case → `evidence_locator_exists`:

<!-- example: invalid-result rule=evidence_locator_exists -->
```json
{"evidence_reference": {"DEMO-C01": [{"document_id": "D999", "page": 1}], "DEMO-C02": []}}
```

An extra field → `result_unknown_field`:

<!-- example: invalid-result rule=result_unknown_field -->
```json
{"score": 0.8}
```

TRL as a string → `strict_trl`:

<!-- example: invalid-result rule=strict_trl -->
```json
{"predicted_trl": "3"}
```

TRL outside the uncertainty interval → `uncertainty_interval`:

<!-- example: invalid-result rule=uncertainty_interval -->
```json
{"uncertainty": {"kind": "trl_interval", "lower": 5, "upper": 7}}
```

Explicit `null` for an optional field → `omit_instead_of_null`:

<!-- example: invalid-result rule=omit_instead_of_null -->
```json
{"confidence": null}
```

Wrong target copy → `result_matches_case`:

<!-- example: invalid-result rule=result_matches_case -->
```json
{"assessment_target": {"target_id": "T001", "type": "component", "description": "Edited description"}}
```

## 4. Rule reference

### 4.1 Request rules (`error.field_errors[].rule`)

| Rule | Meaning |
|---|---|
| `required` | A required field is missing. |
| `unknown_field` | A field that the contract does not define. |
| `string` | Must be a string. |
| `strict_integer` | Must be a JSON integer (not a string, float or boolean). |
| `strict_positive_integer` | Must be an integer ≥ 1. |
| `array` | Must be an array. |
| `object` | Must be an object (also used when the whole body is not an object). |
| `non_empty_array` | Array must have at least one item. |
| `non_empty_string` | String must not be empty. |
| `non_blank_string` | String must contain a non-whitespace character. |
| `omit_instead_of_null` | Leave the optional field out instead of sending `null`. |
| `iso_date` | Must be a real date `YYYY-MM-DD` (or `null` where allowed). |
| `unique_items` | Array values must not repeat. |
| `allowed_value` | Must be one of the listed values (`expected` shows them). |
| `request_id_pattern` | `request_id` must match `[A-Za-z0-9_-]{1,128}`. |
| `identifier_pattern` | ID must match `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`. |
| `unique_document_id` | Two documents share a `document_id`. |
| `max_documents` | More documents than `max_documents`. |
| `base64_content` | `content_base64` is not strict standard base64. |
| `max_document_bytes` | A decoded PDF is larger than `max_document_bytes`. |
| `pdf_signature` | The decoded file does not start with `%PDF-`. |
| `document_sha256_match` | `sha256` is not the SHA-256 of the decoded file. |
| `sha256_pattern` | `sha256` is not 64 lowercase hexadecimal characters. |
| `pdf_readable` | The server could not open the PDF and count its pages (damaged, password-protected, or too slow to open). |
| `max_total_pages` | More pages in all PDFs together than `max_total_pages`. |
| `supported_input_profile` | `input_profile` is not `pdf_report`. |
| `profile_schema_version` | `schema_version` differs from the server's. |
| `profile_checklist_version` | `checklist_version` differs from the server's. |
| `invalid_value` | Any other type or value error. |

### 4.2 Result rules (`diagnostics.output_issues[].rule`)

| Rule | Meaning and fix |
|---|---|
| `result_object` | The result is not a JSON object. |
| `result_required_field` | A required field is missing (checking stops here). |
| `result_unknown_field` | A field outside §2.1, at the top level or inside an evidence item. |
| `result_matches_case` | `schema_version`, `case_id` or `assessment_target` differs from the case. |
| `result_matches_checklist` | `checklist_version` differs from the checklist or the case. |
| `assessment_status_value` | `assessment_status` is not `completed`, `abstained` or `failed`. |
| `strict_trl` | `completed` needs an integer TRL 1–9. |
| `status_combination` | A field contradicts the status (§2.2), e.g. a TRL with `abstained`. |
| `abstention_reason_value` | `abstained` needs one of the four reasons. |
| `error_code_required` | `failed` needs a non-blank `error_code`. |
| `failed_omits_optional` | `confidence`, `uncertainty`, `trl_rationale` and `criterion_rationale` are not allowed with `failed`. |
| `map_object` | `criterion_status`, `evidence_reference`, `missing_evidence` or `criterion_rationale` is not an object. |
| `criterion_known_id` | A key in `criterion_status` or `missing_evidence` is not a checklist criterion. |
| `criterion_status_complete` | A checklist criterion has no status. |
| `criterion_status_value` | A status other than `met`, `not met`, `insufficient evidence`. |
| `evidence_keys_match_checklist` | `evidence_reference` keys differ from the checklist criteria (missing or extra). |
| `evidence_list` | An evidence value is not an array. |
| `evidence_item_object` | An evidence item is not an object. |
| `evidence_item_required` | An evidence item lacks a string `document_id` or a `page`. |
| `evidence_locator_exists` | `document_id` is not a document of this case. |
| `evidence_page_in_document` | `page` is not an integer from 1 to the document's `page_count`. |
| `supporting_span_nonblank` | `supporting_span` is empty or only whitespace. |
| `evidence_required_for_met` | A `met` criterion has no evidence. |
| `missing_evidence_text` | A `missing_evidence` value is not a non-blank string. |
| `missing_evidence_required_for_insufficient` | An `insufficient evidence` criterion has no explanation. |
| `confidence_range` | `confidence` is not a finite number 0–1. |
| `uncertainty_interval` | `uncertainty` has the wrong shape, range or does not contain `predicted_trl`. |
| `omit_instead_of_null` | An optional field is `null`; leave it out. |
| `trl_rationale_requires_completed` | `trl_rationale` with a status other than `completed`. |
| `rationale_known_criterion` | A `criterion_rationale` key is not a checklist criterion. |
| `rationale_nonblank_text` | A rationale value is not a non-blank string. |

## 5. Identity, duplicates and hashes

| Value | Computed from |
|---|---|
| `input_sha256` | The `case` object as sent (including `content_base64`) in canonical JSON: keys sorted, no whitespace, UTF-8, arrays and text unchanged (`python-json-sortkeys-v1`). Key order and spacing in your request do not change it; any value change does. |
| `config_sha256` | The server's execution settings: profile, schema version, checklist version and hash, limits, compute settings, seed, loading policy, module `package_id`, `entrypoint`, `example_only` and `settings`. Paths and ports are not included. |
| `verified_package_sha256` | Files under `package_root` (and `asset_root` if it is outside `package_root`), computed by the server at start-up (see [CONFIGURATION.md](CONFIGURATION.md) §5). |
| effective identity | `input_sha256` + `config_sha256` + `verified_package_sha256` + profile and `package_id`. |

Submitting a `request_id`:

- **new** → a new job and a new run (even if the case was sent before under another ID);
- **seen before, same effective identity** → `202` with the existing job, no second run, even
  while another case is running or the server is not ready;
- **seen before, different effective identity** (different case, configuration or package) →
  `409 REQUEST_ID_CONFLICT`.

A `request_id` rejected with `503 BUSY` was not stored and can be sent again later.
