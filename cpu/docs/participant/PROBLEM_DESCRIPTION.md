# Problem description — TRL assessment of technical reports

kit 0.5.0 | event procedure of 2026-10-10 and 2026-10-13 (decisions D55–D67 and D72–D81)

**Status.** This page describes the whole challenge: the task, the fixed environment and the model
catalogue, what you submit, how your package is run and how it is scored. The organizer decided
the procedure on 2026-10-10, answered the open points on 2026-10-13 (decision D80) and settled the
remaining ones the same day (decision D81): every rule on this page is in force. Read this page
first; the other guides explain how to do each step.

## 1. Background

APCTT (the Asian and Pacific Centre for Transfer of Technology) supports the transfer of
technologies between organisations. Before a technology is transferred, funded or deployed,
someone has to judge how mature it is. The common yardstick is the **Technology Readiness Level
(TRL)**, a scale from 1 (basic principles observed) to 9 (actual system proven in operational
use). A TRL is assigned by reading the evidence, usually technical reports: what was built, how it
was tested, in which environment, and whether the results met the stated criteria.

Doing this by hand is slow, and different assessors read the same report differently. This
hackathon asks for a system that reads the technical reports about one technology and proposes a
TRL **together with the evidence for it**: for each criterion of a checklist, whether it is met and
on which page of which report the support is. An assessor can then check the judgement page by
page instead of rereading everything.

The competition has two stages:

- **Stage 1**: you build an *Assessment Package*, a Python module that assesses cases. The
  organizer runs it on hidden cases and scores the results automatically and with expert review.
- **Stage 2**: the teams that advance build a web system around their frozen Stage 1 package, for
  an assessor to use, and demonstrate it offline on the event computers.

## 2. The task

Each **case** is one assessment target (a component, subsystem, system or technology) described by
one or more PDF reports. Your module receives the case and returns one **result**:

1. for every criterion of the checklist: `met`, `not met` or `insufficient evidence`;
2. for every `met` criterion: the evidence, as (document, page) references, each with an optional
   short quotation (`supporting_span`) from that page;
3. either a TRL from 1 to 9 (`completed`), or an abstention with a reason (`abstained`: the
   evidence does not allow a TRL), or a failure (`failed`).

How you read the PDFs (text extraction, layout analysis, OCR, retrieval, a language model, rules or
a combination) is your choice, and it is what you compete on. Training a model is allowed but not
required; a rules-based or retrieval-based module is equally welcome.

## 3. Input: a case

A case reaches your module as a mapping (the server has already checked it):

| Field | Meaning |
|---|---|
| `case_id` | Neutral identifier of the case. |
| `assessment_target` | `target_id`, `type` (`component`, `subsystem`, `system` or `technology`) and `description` of what is assessed. |
| `intended_application`, `intended_environment` | What the technology is meant for and where it must work. A TRL is always relative to these. |
| `title`, `assessment_as_of` | Title of the case, and the date of the assessment (or `null`). |
| `documents` | One or more PDFs, in their original order. Each has `document_id`, `title`, `document_date`, `language` (`en`), `page_count` (counted by the server) and `path` (a read-only copy of the PDF). |
| `schema_version`, `checklist_version`, `input_profile` | Versions you copy into your result. |

**Official input limits** (decision D77): at most **3 PDFs per case**, **50 MiB per PDF**
(52,428,800 bytes), **300 pages** in all PDFs of a case together, and a request of at most
**80 MiB** (83,886,080 bytes). PDFs travel as base64, which makes a request about 4/3 of the size
of its PDFs, so all PDFs of one case together can be at most about 59 MiB. The limits were set from
the public sample reports (§5.2: median 7.9 MiB and 44 pages, 95% under 28.8 MiB and 126 pages, the
largest 49.3 MiB and 208 pages; one report of 208 MiB was left out). The 3-minute time limit per
case (§12) is unchanged. The kit's
configuration uses the same limits ([CONFIGURATION.md](CONFIGURATION.md) §3). The
reports are in English. The organizer removes explicit TRL labels from the case fields, but the
text of a report may still state a TRL: see the evidence rule in §13.4. Every field:
[DATA_CONTRACT.md](DATA_CONTRACT.md) §1.

## 4. Output: the result

Your module returns a JSON object with exactly these fields:

| Field | Rule |
|---|---|
| `schema_version`, `checklist_version`, `case_id`, `assessment_target` | Copied from the case and the checklist. |
| `assessment_status` | `completed`, `abstained` or `failed`. |
| `predicted_trl` | Integer 1–9 when `completed`, otherwise `null`. |
| `criterion_status` | Every checklist criterion → `met`, `not met` or `insufficient evidence`. |
| `evidence_reference` | Every checklist criterion → a list of `{document_id, page, supporting_span?}`. A `met` criterion needs at least one item; the page must exist in that PDF. |
| `missing_evidence` | For every `insufficient evidence` criterion, a short explanation of what is missing. |
| `abstention_reason` | When `abstained`: `insufficient evidence`, `conflicting evidence`, `ambiguous scope` or `out of scope`. |
| `error_code` | When `failed`: your own short code. |
| optional | `confidence` (0–1), `uncertainty` (a TRL interval), `trl_rationale`, `criterion_rationale`. |

The server checks the structure and consistency of every result (for example, a `met` criterion
without evidence, a page beyond the end of the PDF, an unknown field). A result that breaks a rule
is not kept: the case is recorded as `failed` (`INVALID_OUTPUT`), and a failed case scores 0
(§13). The kit's server applies exactly the same checks, so you see every problem before you
submit. The server does not check whether a quotation really appears on the cited page (PDF text
extraction differs between tools); the experts do (§13.4). Every field, every rule, and valid and
invalid examples: [DATA_CONTRACT.md](DATA_CONTRACT.md) §2–4.

## 5. The checklist and the public sample cases

### 5.1 The official checklist

The checklist is the list of criteria your module judges, each with a `criterion_id` and a
`description`. The **official checklist** is `contracts/checklist.official.json`, version
`APCTT-2026-CHECKLIST-v1` (decision D74). It is used for every submission, the hidden evaluation
and Stage 2. It has one criterion per TRL level, after the public NASA TRL definitions, and three
criteria on the quality of the evidence:

| ID | TRL | Criterion (short; the full text is in the file) |
|---|---|---|
| APCTT-C01 | 1 | Basic principles stated |
| APCTT-C02 | 2 | Concept and application formulated |
| APCTT-C03 | 3 | Analytical or laboratory proof of concept |
| APCTT-C04 | 4 | Component or breadboard validated in a laboratory |
| APCTT-C05 | 5 | Validated in a relevant environment |
| APCTT-C06 | 6 | Prototype demonstrated in a relevant environment |
| APCTT-C07 | 7 | Prototype demonstrated in the operational environment |
| APCTT-C08 | 8 | Final system qualified by test and demonstration |
| APCTT-C09 | 9 | Final system proven in actual operation |
| APCTT-C10 | – | Measured results compared with requirements |
| APCTT-C11 | – | Test conditions and test-article fidelity described |
| APCTT-C12 | – | Evidence concerns the assessment target itself |

Most public sample cases are models, software or data products. For them the hardware words of
C04–C09 (component or breadboard, prototype, final system qualified) are read by analogy: an early
or representative version of the product, tested as the criterion says. The experts label the
hidden cases the same way (decision D81).

Your module receives the checklist in `load()` and must take the criterion IDs and descriptions
from there, and it returns a status for every criterion (§4). The configuration you create in
README step 6 runs the official checklist, and so do the kit's configuration `config/dev.json`, its
example request and its example modules. Some format examples in the guides use a DEMO checklist
with two criteria (`DEMO-CHECKLIST-NOT-OFFICIAL`) only to stay short. Do not hard-code criterion
IDs; test your module once with a changed copy of a checklist
([CONFIGURATION.md](CONFIGURATION.md) §4). The checklist never contains answers.

### 5.2 The public sample cases

The organizer publishes **99 public sample cases** (decision D74): reports of the U.S. Geological
Survey (USGS) on technologies for SDG 6 (clean water), 11 (cities), 13 (climate) and 14 (life below
water), one PDF per case, chosen by the organizer. `contracts/public-cases.json` lists every case
with the fields your module receives (target, application, environment, title, `sdg_targets`) and
the address, size and SHA-256 of its PDF. The PDFs are not in the kit: `tools/get_public_cases.py`
downloads them from pubs.usgs.gov, checks them and writes request files for the official checklist
(README step 7).

| | |
|---|---|
| Cases | 99: SDG 6 25, SDG 11 25, SDG 13 24, SDG 14 25; 92 technologies and 7 systems; one PDF each |
| Size | median 7.9 MiB and 44 pages; 95% under 28.8 MiB and 126 pages; the largest 49.3 MiB (USGS-SDG-017) and 208 pages (USGS-SDG-075); 5,416 pages and about 1.1 GB in all |
| Text | every PDF has a text layer; USGS-SDG-030 is a scanned 1982 typescript whose text comes from OCR, with OCR errors |
| Reading time | measured on a development Mac: pypdf 6.19 (in the base) typically needs 0.01 s per page, but single pages took up to 12.7 s and a whole report up to about 60 s (USGS-SDG-017); pypdfium2 5.14 (also in the base) read the same ten sample reports in at most 2.1 s each. Time your text extraction against the 3-minute limit (§12) |
| Reference labels | the organizer's estimated TRL (4: 1 case, 5: 12, 6: 54, 7: 29, 8: 1, 9: 2) with a one-sentence reason (`--labels`) |
| Smoke cases | USGS-SDG-012, USGS-SDG-040 and USGS-SDG-061: the diagnostic run of every submission (§11.1) |
| Left out | USGS-SDG-067: its PDF is 208 MiB, over the 50 MiB limit per PDF (§3) |

**What the reference labels are.** The estimated TRL and its reason come from the organizer's
list. They are **not expert labels**: they have no status per criterion and no evidence page, and
they are not how the hidden cases are labelled or scored (§13). Use them to sanity-check your
module, not as ground truth. They are never sent to your module. The public sample cases are not
the hidden evaluation cases.

**How the cases were made.** The type of each assessment target follows the unit the list assessed:
a model, software, workflow, map library or data product is a `technology`; a monitoring, warning
or data-delivery arrangement deployed in the field is a `system`. The intended application restates
the list's category for its SDG target, and the intended environment is written from each report.
The organizer confirmed these case fields on 2026-10-13 (decision D80): they are derived from each
report's own content, they are the same for every team, and the public cases are not scored.
Three addresses of the list no longer answered (Nos. 12, 33 and 91); the same reports were taken
from their publication pages and checked by report number, title and citation. Where the list's
title or year differs from the PDF (for example, Nos. 66, 72 and 74 name another place), the case
uses what the PDF prints. Every case is a full report. For two reports the list named only
appendix 1; these cases use the full report (decision D80): USGS-SDG-013, the seawaveQ report
(36 pages), and USGS-SDG-022, the report on natural flows in Alabama (42 pages). The reason: every
other case is a full report, and a case must contain the document in which the assessment target
is described and tested. Their reference labels are the organizer's estimates from the list,
unchanged. The corrected sheet `USGS_SDG_open_data_set_corrected.xlsx` (2026-10-13: titles, years,
page counts, five PDF addresses and two assessment units corrected from the PDFs, each change
marked) matches the manifest.

**Copyright.** Information written or produced by the USGS is in the U.S. public domain; credit is
requested ("Credit: U.S. Geological Survey", and the suggested citation of each report). Not
everything in the reports is: 93 of the 100 PDFs (the 99 cases and the one left out) state that they may contain
copyrighted material, and some figures come from third parties (for example Esri, Google Earth or
Maxar imagery). Use the PDFs as they are, do not republish such figures on their own, and do not
use the USGS logo.

## 6. Timeline

The dates are published on the official competition website (decision D81). The order is fixed:

| # | Step | What happens |
|---|---|---|
| 1 | Registration | Teams register and accept the participation terms. The event machines are described in §7.2 (decision D76). |
| 2 | Kit, base and catalogue published | This kit with the official checklist and the public sample cases (§5), the common base environment (§7) and the model catalogue (§8) are published and frozen. Before publication the organizer measures every catalogue model on the reference GPU (§8.2). |
| 3 | **Submissions (up to two)** | Each team may submit its Stage 1 model up to two times. For every submission the organizer returns the intake receipt (valid or refused, with the reasons) and a diagnostic run of 3 public sample cases. Not scored. |
| 4 | **Submission deadline: the final model** | The latest valid submission received by the deadline is your final model. It is run once on the frozen hidden cases. A team without a valid submission received by the deadline is not scored. |
| 5 | Expert review | Experts, recruited after the submission deadline, label the hidden cases, judge sampled evidence and review frequently cited pages (§13). |
| 6 | Results and appeals | Provisional results, a 48-hour appeal window (§13.6), the finalist run on 10 more hidden cases, then the Stage 1 results. |
| 7 | Stage 2 hackathon and demonstration | Advancing teams merge their web system with their frozen package on the event computers and demonstrate it offline (§14). |

One valid submission is enough; a second one is optional. Between your two submissions you may
change your code, your rules and which catalogue models you use; each submission is checked in
full (§11). The dates, the submission deadline, the submission and question channels and when the
reply to a submission comes back are published on the official competition website (decision D81).

## 7. The fixed environment

You do not submit an environment, and there is **no environment proposal**. The organizer runs
every package in one fixed environment that it builds itself, the same for every team, so the
evaluation can run every team on 5 machines without installing anything from a submission.

### 7.1 The common base environment

One **base environment** is the same for every team: Linux x86_64, CPython 3.12, the PyTorch
closure of torch 2.11.0+cu128 (with its CUDA 12.8 libraries) and a set of common, licence-allowed
packages. Its identifier is `base_id` `apctt-base-9e7fe1324bb6`; the exact versions and hashes are
in `requirements-base.lock` (94 packages), `constraints-base.txt` and
`contracts/base-environment.json`. Main packages:

| Area | Packages in the base |
|---|---|
| Numerics, classical ML | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6 |
| Transformer models | torch 2.11.0, transformers 5.19.0, tokenizers 0.23.3, safetensors 0.8.0, huggingface-hub 2.2.0, sentence-transformers 5.7.0, accelerate 1.15.0, peft 0.21.2, bitsandbytes 0.50.2 |
| PDF text, layout, OCR | pypdf 6.19.0, pdfplumber 0.11.10, pdfminer.six 20260107, pypdfium2 5.14.0, onnxruntime 1.31.0, rapidocr-onnxruntime 1.4.4, opencv-python-headless 5.0.0.93, pillow 12.3.0 |
| Stage 2 web systems | fastapi 0.141.1, uvicorn 0.54.0, jinja2 3.1.6, python-multipart 0.0.32, httpx 0.28.1, pydantic 2.13.5 |

The rules of the base:

- **One base, frozen.** The base is frozen when the kit is published. It changes only by an
  organizer announcement that applies to every team (with a new `base_id`); there are no per-team
  packages and no requests for packages.
- **Your module runs in the base's own Python.** `tools/make_env.py` builds the same base on your
  computer and prints that interpreter; you put it into `worker.python` of your local configuration
  (README steps 5 and 6). Never install anything into the base by hand: the organizer's machines
  would not have it.
- **Use the base versions.** A package in your submission with the import name of a base package
  (for example a copied `transformers/`) is refused. Your own pure-Python code, and third-party
  pure-Python modules you copy into `participant_module/` with their licences (decision D66, §15),
  are part of your package; compiled extensions are refused.
- **Nothing outside the base.** Programs outside Python (for example a system OCR engine) are not
  provided; use packages of the base (for example `rapidocr-onnxruntime` or `pypdfium2`).

### 7.2 CPU only is fully allowed; a GPU is optional

A module that uses only the CPU (`compute.device` `cpu` in `submission.json`) is fully allowed and
is scored by exactly the same rules and time limit as a GPU module (decision D57). The base
contains the CUDA build of PyTorch, which runs on the CPU as well. With `cuda:0`, your module runs
on the machine's NVIDIA GPU and must fit its memory. The **reference GPU** is the GPU of the event
machine: an **NVIDIA RTX 3070 Laptop GPU with 8 GB VRAM** (8 GiB; decisions D40, D67 and D76). For
development without an NVIDIA GPU (macOS, or Linux and WSL2 without one), `tools/make_env.py`
builds a CPU development base with the same package versions.

**GPU tests.** GPU/CUDA tests of the kit (CUDA training, checkpoint, CUDA inference through the
server, timeout and crash recovery; the CUDA dummy example) were run with kit 0.3.0 on an RTX 3070
Laptop GPU (8 GiB; Ubuntu 24.04.5, driver 595.91.07, torch 2.11.0+cu128) on 2026-09-30, with PDF
input. Since 0.3.0 only the dummy's checklist check and its configuration (official checklist,
input limits) have changed. The tests were not re-run for 0.5.0 and have not been run on the final
event machines; the GPU measurement of the model catalogue (§8.3) is not measured yet.

**The event machine** (decision D76, 2026-10-13). Every case runs on one of **5 identical Asus TUF
Dash F15 FX516PR-HN002** laptops (decision D55):

| Part | Specification |
|---|---|
| CPU | Intel Core i7-11370H, 4 cores / 8 threads, 3.3–4.8 GHz |
| Memory | 16 GB DDR4, shared by the operating system, the kit server and your worker; at least 12 GB is available to your worker (decision D80) |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB GDDR6 (the reference GPU; the integrated Intel Iris Xe graphics is not used) |
| Storage | 512 GB M.2 NVMe SSD |
| Operating system | Ubuntu 24.04 (decisions D40 and D80) |

What this means for your module:

- **Memory.** At least **12 GB of RAM** is available to your worker on the event machines (the
  organizer's answer of 2026-10-13, decision D80); the rest of the 16 GB is for the operating
  system, the kit server and the file cache. A worker that runs out of memory is stopped and the
  case fails. The kit does not measure RAM on the CPU; how to measure your
  worker's peak (for example `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` at the end of
  `assess`, or `/usr/bin/time -v` on Linux) is in [MODULE_GUIDE.md](MODULE_GUIDE.md) §7.4.
- **CPU only.** A CPU-only module has 4 cores (8 threads). Time your load and assessment on 4 cores
  before you rely on them: the 3-minute limit includes everything (§12).
- **Large models load from the SSD every time.** 16 GB of RAM cannot keep the 7–8B 4-bit catalogue
  options in the file cache between cases: each case streams 14.50–16.35 GB of weights from the
  SSD again (§8.3).

### 7.3 Offline execution

The evaluation and the Stage 2 demonstration run **without network access** (decision D46).
Nothing may be downloaded while a case runs. Your code and your own weights are in your package,
public models are in the read-only model store (§8), and packages come from the base. Set
`HF_HUB_OFFLINE=1` in your own code before you import Hugging Face libraries, and load models with
`local_files_only=True` ([MODULE_GUIDE.md](MODULE_GUIDE.md) §7).

## 8. The model catalogue

### 8.1 What is provided

The organizer selected public models that fit on the 8 GB reference GPU and provides them: the
**model catalogue** `contracts/model-catalog.json`, listed with sizes, VRAM estimates and CPU
options in [MODEL_CATALOG.md](MODEL_CATALOG.md) (`python tools/get_models.py --list` prints the
same list). It is the organizer's recommended public model set for the reference GPU, RTX 3070
Laptop GPU with 8 GB VRAM (decisions D64 and D67). The catalogue has 23 models, all Apache-2.0 or
MIT, each at a pinned revision:

| Kind | Models |
|---|---|
| Language models (bf16) | 6, from 0.5B to 2.5B parameters; the two smallest are CPU-friendly |
| Language models (4-bit, GPU only) | 5 options from 3B to 8B parameters, conditional on the GPU measurement (§8.3) |
| Embedding models | 5, most of them CPU-friendly |
| Rerankers | 3 |
| Zero-shot entailment (NLI) | 2 |
| Encoder backbones for team-trained classifiers | 2 |

On every evaluation machine the organizer puts every catalogue model into a **read-only model
store**, laid out as `<store>/<org>/<name>/<revision>/`; your module finds the store in the
environment variable `APCTT_MODEL_STORE`. For development, `python tools/get_models.py ID --store
DIR` downloads the same files into the same layout, and `python tools/get_models.py --snippet ID`
prints the loading code.

### 8.2 The 8 GB budget: estimates and measurement

Every catalogue entry fits a budget of **at most 7.0 GiB peak** on the 8 GiB card: weights + KV
cache or activations + 1.0 GiB of overhead for the CUDA context. Language models are budgeted at
8,192 tokens of context (batch 1), encoders at a batch of 32 × 512 tokens. **The VRAM figures are
estimates** until the organizer measures them: before the kit is published the organizer runs
`tools/hq/check_models_gpu.py` on the RTX 3070 Laptop GPU and removes any entry that does not load
within the budget. Your own use (longer inputs, larger batches, several models at once) is your
responsibility: an out-of-memory error fails the case (`OUT_OF_MEMORY`).

### 8.3 4-bit options

Five language models are offered in 4 bits (bitsandbytes nf4, which is in the base): they load the
publisher's bf16 weights and quantise them on the GPU while loading. They are marked
`"measured": false` and are **conditional on the GPU measurement**: before the kit is published the
organizer runs `tools/hq/check_models_gpu.py` on the RTX 3070 Laptop GPU and removes any 4-bit
entry that does not load within the budget (peak at most 7.0 GiB, and a load well inside the
3-minute limit of a case). They need a CUDA GPU; none of them runs on the CPU. **Cold-load
warning:** the 7–8B entries read 14.50–16.35 GB of weights from disk and quantise them in every case,
because every case starts a new process (§12), and the event machine's 16 GB of RAM cannot keep
those weights in the file cache, so every case streams them from the SSD again (decision D76);
time your load before you depend on them.

### 8.4 What you may use

- **Catalogue models**, named in `submission.json` `models[]` by `catalog_id`. Their weights are
  never in your zip: the organizer provides them. The catalogue records the size and sha256 of
  every catalogue weight file, and a file with those bytes is refused however it is named or
  declared (also as `"team_trained": true`).
- **Weights your team trained**, shipped in `assets/` and declared in `models[]` as
  `"team_trained": true` with their `path`. A fine-tune or adapter (for example LoRA) of a
  catalogue model is your own work: name its catalogue base in `base_model`.
- **Nothing else.** A public model outside the catalogue cannot be used, whatever its licence.
  The reason (decision D80): every evaluation machine has one shared read-only model store, and it
  cannot hold per-team models for this many teams; and public model weights may not be shipped in a
  submission. There are no exceptions. The kit's check
  refuses catalogue weights by their bytes and undeclared weight files over 50 MiB; an undeclared
  weight file under 50 MiB that looks like a public model is a licence review item, which the
  organizer decides after the submission (§9). The rule itself is the same for small files: a
  public model outside the catalogue may not be used at any size (see §16 for the consequences).

## 9. No advance approval

Nothing is approved in advance, for any team (decision D63). The rules are published, mechanical
and the same for every team:

- **Check yourself with the kit's tools.** `tools/make_env.py --check` (your environment is the
  base), `tools/get_models.py --check` (your model copies), `tools/make_submission.py` (packs your
  zip and checks it exactly as the organizer's intake does, including the licence checker
  `tools/check_licenses.py`).
- **Checks at submission.** Every submission is checked in full at intake with those rules; a
  submission that passes them is valid (§11). The diagnostic run of the public sample cases is
  feedback only.
- **Review items are decided after the submission.** Licence items that the policy marks
  "review" (for example weak copyleft in copied code) are decided by the organizer after the
  submission; there are no advance rulings on your package.
- **Questions.** HQ answers questions about TRL and about the competition rules, and publishes
  every answer to all teams. It does not help teams develop or debug their systems (decision
  D80). The channel for questions is published on the official competition website (decision
  D81).
- **No exceptions.** The 2 GB size cap, the 3-minute limit per case (model loading included), the
  absence of network access and the frozen base apply to every team; there are no per-team
  resources, packages or models.

## 10. What you submit

### 10.1 The minimal bundle

You submit one zip, `<team>_submission.zip`, made by `tools/make_submission.py`. It contains **no
environment**: no packages, no virtual environment, no PyTorch or CUDA, no public model weights.

```text
<team>_submission.zip
├── participant_module/      your code and rules
│   ├── __init__.py
│   ├── adapter.py           entry point participant_module.adapter:AssessmentModule
│   ├── LICENSE              the MIT licence, full text (§15)
│   └── THIRD_PARTY_NOTICES.md
├── assets/                  only weights and rules your team trained or wrote (may be empty)
├── submission.json          team, package_id, package_version, base_id, declared_package_sha256,
│                            license, compute, settings, models
├── declaration.json         the data declaration
└── README.md                what your module does and how it was built
```

`submission.json` follows `contracts/submission.schema.json`: `base_id` is the kit's common base,
`compute.device` is `cpu` or `cuda:0`, `settings` is passed to `load()`, and `models[]` lists the
catalogue models you use (`catalog_id`) and your team-trained weights (`name`, `path`,
`"team_trained": true`, and `base_model` for a fine-tune or adapter). `declared_package_sha256` is
the hash of `participant_module/` and `assets/`, computed and written by `tools/make_submission.py`
with the same function the server uses; the organizer recomputes it, and it must equal the hash the
server reports while it runs your cases. Details: [SUBMISSION.md](SUBMISSION.md) §1.

### 10.2 What is refused

Anything other than the five entries above; symbolic links; compiled extensions and binaries
(`.so`, `.pyd`, `.dylib`, `.dll`, `.exe`), wheels, virtual environments, `site-packages`, installed
package metadata, `.git` and `node_modules` folders; copies of base packages; any file with the
bytes of a catalogue weight file (size and sha256 recorded in `contracts/model-catalog.json`),
whatever its name and even if declared as team-trained; weight files over 50 MiB that `models[]`
does not declare as team-trained; a `models[]` entry outside the catalogue, a catalogue model with
weights in the zip, or a `base_model` outside the catalogue; a `base_id`
other than the kit's base; a package hash that differs from the declared one; and anything the
licence checker refuses. Run the tool yourself: it refuses exactly what the organizer refuses.
Undeclared weight files under 50 MiB are not refused by the tool: the licence checker lists them as
review items, and the organizer decides them after the submission (§8.4).

### 10.3 Size

The zip may be at most **2 GB** (2,000,000,000 bytes), for every team. Catalogue models do not
count: they are in the model store.

## 11. Up to two submissions

The organizer decided on 2026-10-13 (decision D73): **"Each team may submit its Stage 1 model up
to two times. The latest valid submission received by the deadline will be treated as the final
model."**

### 11.1 What every submission returns

Each submission is the same kind of zip, made with `tools/make_submission.py` (§10). For every
submission, the first and the second alike, you get back:

| What | Contents |
|---|---|
| The intake receipt | **VALID** or **REFUSED**: the reasons for a refusal, any licence review items (decided after the submission), the time the submission was received, the hashes of your zip and package, the base and the models |
| A diagnostic run of 3 public sample cases | Per case: status, TRL, error code, the server's fixed failure sentence, output issues (path and rule), warning codes and timings; PASS when every case produced a contract-valid result (`completed` or `abstained`), FAIL otherwise. **No score.** |

**Valid** means that the intake accepted your zip: exactly the checks of
`tools/make_submission.py --check`, which you can run yourself (§9). A valid submission whose
sample cases fail is still valid: the run is feedback, not a condition. The 3 cases are always
USGS-SDG-012, USGS-SDG-040 and USGS-SDG-061 (§5.2), so you can run them yourself before you submit.
The diagnostic runs use the organizer's `evaluation` server profile, the profile of the final
run, which shows no stack traces (decision D80).

### 11.2 Your final model

- The **latest valid submission received by the deadline** is your final model. It is run once on
  the frozen hidden cases (and, if you are a finalist, on 10 more) and scored (§13).
- If your second submission is refused, your **first valid submission stays your final model**.
- A submission received after the deadline is not considered.
- **A team without a valid submission received by the deadline is not scored and not ranked.**
- There is no third submission. The time of receipt is the one the organizer's submission channel
  records (decision D80).

What to prepare and the data declaration: [SUBMISSION.md](SUBMISSION.md).

## 12. How your package is run

- **Where.** The organizer copies your package, unchanged, into its pinned reference kit (the
  same server code as this kit) on one of **5 machines of the same specification**. Each machine
  runs one kit server and one case at a time. The base environment and the model store are
  read-only.
- **How.** Every case runs in a **new worker process**: `load` → `assess` → `reset_case` →
  `close`, then the process exits. Nothing is kept between cases, so every case pays its own
  interpreter start and model loading.
- **Time.** **3 minutes per case**, interpreter start and model loading included (decision D41).
  A case that is not finished is stopped and recorded as `failed` with `TIMEOUT`. Check
  `context.remaining_ms()` and return an `abstained` or `failed` result yourself before the limit.
- **First run is official.** Each case is run once and that run counts. A case is run a second
  time only when the failure was caused by the organizer's host (`SERVER_RESTARTED`,
  `INTERNAL_ERROR` from the server, or a host cause in the run log), once, and never after a
  `TIMEOUT`.
- **Offline and isolated.** No network. The worker sees only the environment variables listed in
  [MODULE_GUIDE.md](MODULE_GUIDE.md) §7 and `APCTT_MODEL_STORE`. The official checklist is used.

### 12.1 Fairness of the hidden evaluation

The organizer asked for a fair competition on the hidden cases (decision D80). These rules apply
to every team:

- **What the hidden cases are.** They are disjoint from the public sample cases and are drawn
  from the same kind of reports: technical reports of the USGS type, one PDF per case within the
  limits of §3, with a text layer, spread over the SDGs and TRL levels of the public set (§5.2).
- **What is published.** Their composition is published, not their content: the number of cases
  per SDG and per TRL band (TRL 1–3, 4–6 and 7–9), and the ranges of their page counts and sizes.
- **Frozen before the deadline.** The hidden set is frozen before the submission deadline (§13.6)
  and is not changed afterwards.
- **The same run for everyone.** Every team's final model runs once on the same cases, on
  identical machines assigned at random, in the same `evaluation` profile and with the same
  3-minute limit. A case is run again only for a fault of the organizer's host (see above).
- **No early view.** No team gets any output on a hidden case before the results.
- **Independent labels.** Two experts label each hidden case independently, and a third expert
  decides where they disagree. The experts do not see any team's output while they label. Gold
  pages added by the page review and by accepted appeals (§13.6) apply to every team.
- **No conflicts of interest.** The people who prepare the hidden cases or their labels, and the
  experts, do not take part in the competition and declare their conflicts of interest.

## 13. Scoring

### 13.1 Overview

The Stage 1 score of a team combines automatic scoring and expert review (decision D58), on the
results of its final model only (§11.2). The organizer confirmed this section, with all its
weights and constants, on 2026-10-13 (decision D75):

```text
total = 0.45 × A  +  0.40 × B  +  0.15 × C        (each of A, B, C between 0 and 1)
```

| Part | Weight | What it measures | How |
|---|---|---|---|
| A — TRL decision | 45% | Whether your TRL (or abstention) matches the experts | automatic, per case |
| B — criterion evidence | 40% | Whether your criterion judgements and evidence pages agree with the experts | automatic, against expert labels |
| C — expert review | 15% | Whether your quoted evidence really supports the criterion | blinded expert judgement of sampled claims |

A **failed** result (any cause: your error, `INVALID_OUTPUT`, `TIMEOUT`, a crash, no result)
scores 0 in A and in B for that case.

### 13.2 A — TRL decision (45%)

Per case, then the mean over all scored cases:

| Your result | Experts give a TRL | Experts find no TRL determinable |
|---|---|---|
| `completed`, exact TRL | 1.0 | — |
| `completed`, off by 1 | 0.5 | — |
| `completed`, off by 2 or more | 0 | — |
| `completed`, inside the experts' interval | — | 0.5 |
| `completed`, outside the interval | — | 0 |
| `abstained` | 0.25 | 1.0 with a reason the experts accept, 0.75 with another reason |
| `failed` | 0 | 0 |

So when a TRL can be determined the order is **exact > off by 1 > abstain > off by 2 or more =
failed**: abstaining is better than a wrong guess but worse than a near miss. When the experts find
no TRL determinable, an abstention with the right reason earns full points.

### 13.3 B — criterion evidence (40%)

The experts label every criterion of every hidden case `met`, `not met` or `insufficient
evidence`, and for each `met` criterion they list the **gold pages** that support it. For scoring,
the labels collapse to two kinds:

- **met part**: a criterion the experts label `met` counts as a hit when you also say `met`
  **and** at least one of the **first 3 distinct (document_id, page) items** in your evidence
  list is a gold page. Pages after the first three distinct ones are not looked at, so put your
  best evidence first.
- **not-established part**: a criterion the experts label `not met` or `insufficient evidence`
  counts as a hit when you do **not** say `met`.

```text
B = (2 × mean(met part) + 1 × mean(not-established part)) / 3
```

Both means are taken over all labelled criteria of all your scored cases together; if one part has
no criteria, B is the other part.

### 13.4 C — expert review (15%)

From your scored results, 6 `met` claims are sampled at random (the first evidence item of a `met`
criterion: document, page and `supporting_span`). Experts judge them **blinded** (they do not see
the team), reading the cited page:

| Judgement | Points |
|---|---|
| SUPPORTS — the quoted text is on the cited page and supports the criterion | 1 |
| PARTIAL, NOT_SUPPORTS | 0 |
| No `supporting_span` given | 0 |
| FABRICATED — the quoted text is not on the cited page | −1 |

```text
C = max(0, sum of points / max(6, number of judged claims))
```

A team with fewer than 6 `met` claims gets 0 for the missing ones. Rules for evidence:

- The `supporting_span` must be text that is **on the cited page** (as you extracted it; small
  extraction differences are fine). Text that is not there is fabrication.
- **A sentence that only asserts a TRL number** ("the system has reached TRL 6") **is never
  evidence** for a criterion: it is judged NOT_SUPPORTS. Evidence describes what was built,
  tested or demonstrated, where, and with which result.
- Each fabricated claim costs 1 point (−1), within the C formula above, and C never goes below 0.

### 13.5 Baseline, standard errors, ranking and finalists

- The organizer publishes the score of the **always-abstain baseline** (a module that abstains on
  every case with `insufficient evidence` and claims no criterion), scored by the same rules.
- Each team's score is published with its **standard error** (a paired bootstrap over the cases,
  1,000 resamples, C held fixed).
- Ties are broken by B, then A, then C, then fewer failed cases.
- **Finalists**: the top 40 teams, plus every team whose total is within its standard error of the
  40th total (exact ties always included), are run on **10 more hidden cases** with the same
  final model. The Stage 1 ranking and prizes follow the scores on all **20 cases**.

### 13.6 Expert review after the submission deadline, and appeals

- The hidden cases are fixed before the submission deadline. The expert labels may be written
  after the deadline, on those fixed cases.
- After automatic scoring, the experts review every page that **at least 5 teams** cited for a
  `met` criterion (within their first 3 distinct pages) but that is not a gold page. Pages the
  experts accept are added to the gold set **for every team**, and all teams are re-scored.
- After the provisional results, a **48-hour appeal window** opens for one type of appeal only:
  "page X of document D in case K supports criterion C". An accepted appeal adds the page to the
  gold set for every team. Other appeals (for example against a TRL label) are not accepted.

The organizer's scoring tools are not part of this kit; the kit never scores and holds no answers.

## 14. Stage 2 — web system and demonstration

**What the organizer prepares.** For every advancing team, the organizer sets up an event computer
with the pinned kit, the common base environment, the model store and the team's **final Stage 1
package, unchanged** (the same package hash as its final model, §11.2). The kit server runs the package
as in Stage 1. The computers have no network.

**What you build, and where.** A web system for an assessor that uses your frozen Stage 1 package
through the kit server's HTTP API (`POST /v1/assessments`, then `GET` the job): the user gives
technical reports and gets the TRL and, for every criterion, its status with the evidence pages
shown next to the page of the original PDF. You develop the web system **on site, during the
Stage 2 event**, and your team merges it with its frozen Stage 1 package **on the event computer
itself** (decision D80). There is no advance upload of web code. You may not retrain or replace
the package for Stage 2.

**Environment.** Your web system runs in the same common base environment: its Python code may use
only base packages (fastapi, uvicorn, jinja2, python-multipart, httpx and pydantic are in the
base); there are no additional packages for Stage 2. **Node.js is available on the event
computers** (decision D81), so you may use a Node.js-based front-end build on site. The event
computer has no network: no package can be downloaded during the event, and there is no CDN and
no external fonts or analytics. Bring every front-end dependency with you as files, vendored in
your web directory (for example `node_modules/` with `package.json` and `package-lock.json`, or
the library files with their licence files), and check them with
`tools/check_licenses.py --web`.

**Gate.** A Stage 2 system is scored only when it passes all of these:

1. it runs offline on the event computer at the end of the session;
2. it assesses through the kit server with the team's frozen Stage 1 package (the job's
   `verified_package_sha256` equals the one of the final model);
3. it shows the TRL and every criterion's evidence page next to the original PDF page;
4. the licence checker finds no refused component (`tools/check_licenses.py --web`).

**Rubric, advancing teams and final score.** The Stage 2 rubric, the number of advancing teams
and the weights of the final score: APCTT decides and announces them before Stage 2 starts
(decision D81).

## 15. Licences and intellectual property

The organizer confirmed the licence rules on 2026-10-10 (decision D53) and made the team licence
MIT only on 2026-10-13 (decision D72):

- Your own work (Stage 1 code, rules and trained weights, adapters and fine-tunes included; Stage 2
  web code) is licensed under **MIT**, with the full MIT text in `participant_module/LICENSE`
  (Stage 2: `LICENSE` at the root) and `"license": "MIT"` in `submission.json`. Apache-2.0, which
  earlier versions of the kit allowed, is no longer accepted for your own work. MIT keeps your work usable by
  APCTT later, combined with other work and redistributed, without a licence problem.
- Third-party components and catalogue models keep their own licences (Apache-2.0 and MIT
  components stay allowed). Weights you derive from a catalogue model (a fine-tune or an adapter)
  also stay subject to that model's licence: keep its `LICENSE` and `NOTICE` files with them.
- Third-party components otherwise: strong copyleft (AGPL, GPL, SSPL, EUPL, OSL, RPL), non-commercial,
  research-only, revenue-capped and commercial-only licences are refused; weak copyleft is
  reviewed after the submission; MPL-2.0 is allowed.
- Public models: only the catalogue (§8.4). PyTorch, triton and NVIDIA's CUDA libraries are part
  of the base, installed by the organizer and never shipped by you.
- **Third-party pure-Python modules** may be included in your package (copied into
  `participant_module/`, each listed with its licence in `participant_module/THIRD_PARTY_NOTICES.md`;
  compiled extensions are refused). **If a licence problem or a security problem with code you
  included arises later, your team bears the legal responsibility and the award is revoked**
  (decision D66, 2026-10-10). `tools/make_submission.py` prints this rule as a reminder every time
  it packs or checks your zip.

On 2026-10-10 the organizer also decided (D54) that you submit your work as **open source** and do
not assert patent, copyright, trade-secret or other intellectual-property claims over it against
the organizer, APCTT or anyone who uses it under that licence; you keep ownership. This also
covers patents, for which the MIT licence has no express grant.
`tools/check_licenses.py` flags components from their declared metadata; HQ decides flagged items
after submission.
Details: [LICENSING.md](LICENSING.md).

## 16. Eligibility and disqualification

The organizer confirmed these rules on 2026-10-13 (decision D80). Each rule is followed by its
reason.

**Eligibility:**

- Teams registered by the registration deadline whose members accept the participation terms
  (including the licences of this kit and of the base environment, §15). *Reason:* every team
  competes under the same terms and licences.
- One team per person. *Reason:* each team's work is independent, and nobody competes twice.
- Organizer staff, the people who prepare the hidden cases or their labels, and the experts do
  not take part. *Reason:* they can see or influence the hidden cases or their labels (§12.1).

**Disqualification:**

- trying to reach the network, the hidden cases or labels, other teams' files or the organizer's
  systems, or interfering with the server, the machine or the time measurement (for example
  detached processes). *Reason:* every team must be assessed on the same unseen cases under the
  same conditions;
- answers obtained from the hidden cases or labels by any route other than your module's own
  assessment of the case it receives. *Reason:* the score must measure your module, not knowledge
  of the answers;
- declaring third-party or public weights as team-trained, shipping refused components, or leaving
  refused licence items unresolved. *Reason:* licence and fairness: your work must be usable under
  its licence (§15), and every team uses the same catalogue (§8.4);
- sharing code with another team: code written by one team in another team's submission
  (third-party code under an allowed licence, §15, is not shared code in this sense). *Reason:*
  independent work;
- submitting another team's work, or work you have no right to submit. *Reason:* independent
  work, and the licence and covenant of §15 need the rights to the work.

**Procedure.** Before a team is disqualified, HQ notifies the team in writing with the reason and
the evidence. The team may respond within 48 hours. A panel of at least three organizer members
who have no conflict of interest decides, and the decision and its reason are recorded.

A failed or invalid result is not a disqualification: it scores 0 for that case (§13). A licence or
security problem in third-party code you included that arises later leads to the revocation of the
award (decision D66, §15).

## 17. FAQ

**Can we use only the CPU?** Yes. `compute.device` `cpu` is fully allowed and scored by the same
rules and the same 3-minute limit. The catalogue marks the models that are practical on a CPU.

**Do we need a GPU to develop?** No. `tools/make_env.py` builds a CPU development base with the
same package versions on macOS and on Linux or WSL2 without an NVIDIA GPU. The 4-bit catalogue
models need a CUDA GPU.

**Can we use other public models?** No. Only the models of the catalogue (§8) are available at the
event, offline, and public model weights may not be shipped in your zip. There is no request
procedure for adding a model.

**Can we fine-tune a catalogue model or train adapters?** Yes. Ship the weights you trained (for
example LoRA adapters) in `assets/`, declare them in `submission.json` `models[]` with
`"team_trained": true`, their `path` and the catalogue id of the model you started from in
`base_model`. Your module loads the base model from `APCTT_MODEL_STORE` and your weights from
`asset_root`. License your weights under MIT and keep the `LICENSE` and `NOTICE` files of the
catalogue model you started from with them; they count towards the 2 GB cap.

**Can we add Python packages?** Not as packages: the base is fixed for everyone. You may copy
third-party pure-Python modules into `participant_module/` with their licences in
`THIRD_PARTY_NOTICES.md`; your team bears the responsibility for them (decision D66, §15).
Compiled extensions, wheels and virtual environments are refused.

**How large may the submission be?** 2 GB, for every team. Weight files over 50 MiB must be
declared as team-trained in `models[]`.

**Can we get an exception or an advance check of our package?** No. Nothing is approved in
advance and there are no exceptions (§9). Check your package yourself with
`tools/make_submission.py`, which runs the organizer's intake checks, and use the diagnostic run of
the public sample cases that comes back with each submission.

**Do we have to submit twice?** No. One valid submission received by the deadline is enough. If
you submit twice, the later valid one received by the deadline is your final model; if the second
one is refused, the first valid one stays. A team without a valid submission received by the
deadline is not scored and not ranked.

**Can we still license our work under Apache-2.0?** No. From this kit on, your own work is MIT only
(decision D72); `tools/make_submission.py` refuses a `"license"` of Apache-2.0 and says so.
Third-party code and catalogue models under Apache-2.0 remain allowed.

**How much memory may our module use?** The event machine has 16 GB of RAM, and at least 12 GB of
it is available to your worker (§7.2, decision D80). Measure its peak as
[MODULE_GUIDE.md](MODULE_GUIDE.md) §7.4 shows.

**What if a case takes longer than 3 minutes?** It is stopped and scores 0 in A and B. It is not
re-run. Stop early with `context.remaining_ms()`, and time the load of large models (§8.3).

**Do abstentions help?** When the experts find no TRL determinable, an abstention with an accepted
reason scores 1.0. When a TRL is determinable, an abstention scores 0.25 (better than a TRL off by
two or more). The always-abstain baseline is published so you can compare.

**Can we use native Windows?** No; use WSL2. macOS works for development.

**Are the public sample cases the hidden cases?** No. They are published for development and for
the diagnostic run of your submissions. Their estimated TRLs are a rough reference, not expert
labels (§5.2). The hidden cases are disjoint from them and drawn from the same kind of reports
(§12.1).

**Will HQ help us with our system?** No. HQ answers questions about TRL and about the competition
rules and publishes every answer to all teams; it does not help teams develop or debug their
systems (§9).

## 18. Where to go next

| Guide | For |
|---|---|
| [README.md](../../README.md) | Step by step: kit, base environment, models, module, submission |
| [MODEL_CATALOG.md](MODEL_CATALOG.md) | Every catalogue model: size, VRAM estimate, CPU option, use |
| [MODULE_GUIDE.md](MODULE_GUIDE.md) | Writing the module; the worker environment, model store and time limit |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Every input and output field and rule |
| [SUBMISSION.md](SUBMISSION.md) | The two submissions, the bundle, the data declaration and handover |
| [LICENSING.md](LICENSING.md) | Licences of your work, of copied code and of models; the licence checker |
| [CONFIGURATION.md](CONFIGURATION.md) | The local server configuration |
