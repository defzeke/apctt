# AGENTS.md — APCTT Stage 1 project context

Read this before touching the fine-tune, the training data, the assessment module or the
submission package. It is the state of the project as of 2026-11-04. Deeper detail lives in
`train/HANDOFF.md` (training) and `train/USING_THE_ADAPTER.md` (adapter use and transfer).

## 1. Objective

Build a Stage 1 APCTT submission that estimates the TRL of a USGS report and assesses it
against the 12 official checklist criteria with page evidence. The submission is a zip of
`participant_module/`, `assets/`, `submission.json`, `declaration.json` and `README.md`,
run in the organizer's common base environment.

Score: `total = 0.45*A + 0.40*B + 0.15*C`.

- A (TRL): exact 1.0, off-by-one 0.5, abstain 0.25, off-by-two-or-more 0.
- B (criteria): `(2*met_part + not_established)/3`. A `met` hit requires status `met` and
  one of the first 3 distinct evidence pages matching a gold page.
- C (support): 6 sampled `met` claims; exact supporting span 1, no span 0, fabricated -1.

Public labels are the organizer's *estimates* (54 of 99 are TRL 6), not expert gold and not
the hidden-case labels. Do not tune to them as truth.

## 2. Repository layout

| path | role |
|---|---|
| `cuda/` | competition kit: contracts, docs, tools, scoring reference. Do not edit. |
| `cpu/` | kit variant not used by this team. |
| `train/` | development workspace; never shipped. Data build, training, evaluation, handoff. |
| `my_submission/` | the only directory packed into the submission zip. |
| `ui/` | local test page (`server.py` + `index.html`, stdlib only): runs the module, the contract check and the B/C ranker on a public case or an uploaded PDF. Dev only, never shipped. |
| `train/data/pdfs/` | 99 public USGS PDFs, hash-checked, 1.1 GB; ignored by git; fetch from `cuda/` with `python tools/get_public_cases.py --all --download --dir ../train/data/pdfs`. |
| `train/out/` | adapters, eval JSON, packed zips; ignored by git, so a fresh clone has none. Rebuild with §7. |
| `USGS_SDG_open_data_set_corrected.xlsx` | organizer sheet used to build SFT labels. |

## 3. Base model and adapter

- Base: `smollm3-3b-nf4` = `HuggingFaceTB/SmolLM3-3B`, revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`, Apache-2.0. Only catalogue models are allowed.
- Adapter: `my_submission/assets/smollm3-trl-lora/`, 60,523,464 bytes
  (`sha256 5660a04995386760ef98be35a3419c8d558dfa1367ed4c982adcbbdd30819f68`), MIT. This is
  the 2026-11-04 pipeline adapter (copied from `train/out/pipeline-adapter/`). It replaced
  the 2026-11-03 adapter (`17b781a9…`), which scored 0.68 inside the module against 0.70.
- Training: QLoRA r=8, alpha=32, dropout 0.05, targets q/k/v/o/gate/up/down, 74 records,
  3 epochs, max length 896, lr 2e-4, batch 1, grad-accum 4, `adamw_torch`, seed 42, bf16,
  231.1 s on the RTX 4060 Laptop. Full record: `assets/smollm3-trl-lora/training_manifest.json`.
- Data: `train/pipeline.py` → `train/out/pipeline-data/train.jsonl` / `val.jsonl` = 74 / 25
  split with SDG 14 held out (`sha256 0070eafd…`, byte-identical to `train/out/extractive-data/`).
  The prompt is the module's own `llm.py:build_prompt`. The rationale target is extractive:
  the report sentence that best matches the organizer's reason. The older `build_data.py`
  split stays in `train/data/` (frozen `train/data/sdg14/`, random 79/20 `train/data/random/`).
- Labels: organizer-estimated TRL + 200-character rationale. No per-criterion labels.

### Measured results

| evaluation | result |
|---|---|
| **shipped adapter (11-04)**, `train/module_eval.py --llm`, SDG 14 holdout (25) | 0.70 mean A, 13 exact, 9 off-by-one, 3 off-by-two+, 0 invalid; predicts 6 for all 25 |
| shipped adapter (11-04), `train/evaluate.py` on `pipeline-data/val.jsonl` | 0.70 mean A, 13 exact, 0 invalid; rationale grounded 5-gram 0.998 |
| old adapter (11-03), `train/module_eval.py --llm`, SDG 14 holdout (25), rerun 2026-11-04 | 0.68 mean A, 12 exact, 0 invalid |
| old adapter (11-03), `train/evaluate.py` on `train/data/val.jsonl` | 0.72 mean A, 14 exact, 0 invalid |
| rules only, `train/module_eval.py`, SDG 14 holdout (25) | 0.70 mean A, 13 exact (2026-11-04) |
| rules only, `train/module_eval.py`, 99 public cases | 0.7525 mean A, 54 exact, 41 off-by-one, 4 off-by-two+, 0 invalid |
| always predict 6 | 0.753 on 99 public, 0.70 on SDG 14 |
| adapter cost | about 33 s cold load, 2.06 GiB peak VRAM, about 14 s/case integrated, 6.75 GiB RSS |
| rules cost | 0.9 s/case (2026-11-04, RTX 4060 laptop; 2.0 s on the old one), 0.28 GiB RSS |
| B/C ranker, `train/train_ranker.py`, SDG 14 holdout (25), silver labels | pair AUC 0.91 (0.59 untrained); rationale page in top 3: 0.48 ranker vs 0.44 rules |
| B/C ranker, `train/ranker_check.py`, SDG 14 holdout (25) | rationale page cited first at the label's criterion: 7/25 ranker vs 1/25 module |
| B/C ranker cost | 79 pairs/s on CPU: 4-6 s/case to rerank the rule candidates (330-440 pairs) |

The ranker is `ms-marco-minilm-l6-v2` fine-tuned on silver (criterion, sentence) pairs:
rule matches plus the sentence that best matches the organizer rationale. It lives in
`train/out/criteria-ranker/` (88 MB, git-ignored) and is **not** in the submission. The
"rationale page" is the only label the rules did not make, and it covers one criterion per
case, so read these numbers as a direction, not as part-B accuracy.

`evaluate.py` and `module_eval.py` differ: `evaluate.py` feeds the full training prompt from
`val.jsonl`; `module_eval.py` feeds excerpts from the PDF and uses the module path. Quote the
metric with the tool name. The old adapter was trained on `build_data.py:build_user_text`,
which is not the module prompt, and it lost 0.04 inside the module (0.72 → 0.68). The
shipped adapter is trained on the module prompt itself and scores 0.70 with both tools.
Excerpt selection still differs: rule-selected pages in the module, `pipeline.py` pages in
`evaluate.py`.

To test an adapter inside the module before shipping it, use
`module_eval.py --llm --assets <dir holding smollm3-trl-lora/>` with
`APCTT_MODEL_STORE=~/models/store`. The SmolLM3 base is symlinked into the store layout
there: `~/models/store/HuggingFaceTB/SmolLM3-3B/a07cc9a…` → `~/models/SmolLM3-3B`.

The public-case JSON (`train/out/module-eval.json`) is from the 2026-11-04 run on the training
laptop (after the evidence-filter fix) and is not in git; regenerate it after a fresh clone.

## 4. Current shipped behaviour

Decision: the submission is **rules-only, CPU**. The adapter is bundled and declared but
disabled (`settings.use_llm = false`) because it does not beat the rules prior on the held-out
split: the 2026-11-04 adapter ties it (0.70 vs 0.70) by predicting 6 everywhere, and costs
about 6.7 GiB RSS and a 27-38 s cold first case. Do not re-enable it without new evidence.

`participant_module`:

- `adapter.py` — entry point `AssessmentModule`; rules first, optional adapter refine, fallback to abstain.
- `extract.py` — pypdfium2 page text; strips headers, footers, page numbers, captions,
  reference lists, URLs, disclaimers, abbreviation lists, credit lines, map legends,
  place-name lists, contents dot leaders, "… showing" figure-list captions, table headnotes,
  report-series header lines and acknowledgements. That last set (2026-11-04) removed 108
  junk spans (68 in first position, the ones part C samples) from the 99 cases.
- `rules.py` — rules matched to criteria by official description (not by ID), candidate
  scoring, evidence selection, fill the TRL ladder up to the predicted level, target check.
- `llm.py` — lazy 4-bit base + adapter loader; only used when `use_llm` is true. The code
  default is `false` too, so a config without `settings` stays rules-only.
- TRL policy: predict 6 when any criterion is met, because the public prior is 6; evidence is
  used for B/C, not for a raw TRL ladder. Levels 7-9 require independent matches.

The module config is `config["checklist"]` plus `config["module"]["settings"]`; in
`submission.json` the key is top-level `settings`.

## 5. Submission package

- Team `CICA`, package `cica-usgs-trl-assessor` 0.1.0, base `apctt-base-9e7fe1324bb6`,
  licence MIT, device `cpu`.
- `models`: catalogue `smollm3-3b-nf4` plus team-trained `assets/smollm3-trl-lora`
  (`base_model: smollm3-3b-nf4`).
- Current artifact: `train/out/CICA_submission.zip`, 60.6 MB,
  zip `sha256 efa68e1ec5e9df0f11467fa8f4256b584ed44d6cd16d51d8f3b026cf95894d4d`,
  package `sha256 9a5b98347c307857c09a23b8013972ab2dbb759c9fe1bd3099e4dec353391d19`
  (packed 2026-11-04 with the 2026-11-04 adapter, from LF files; the pack is deterministic, so a repack of the same
  files gives the same hashes).
- `make_submission.py` writes `declared_package_sha256` into `submission.json` and
  `declaration.json`. The package hash covers only `participant_module/` and `assets/`, as
  raw bytes: CRLF line endings change it (see §8).
- `--check` result: `VALID_WITH_REVIEW`. The only review items are the two dataset licence
  strings (`U.S. public domain (USGS); ...`), which the licence checker cannot map to an SPDX
  id. That is expected.
- Adapter-only handoff bundle: `train/out/CICA_smollm3-trl-lora.zip`, built by
  `python train/pack_adapter.py` (deterministic), 55,911,107 bytes,
  `sha256 a5c245ba3be72cf81974fd1cd934779345acb6cbdbec5726a68c7e67faf6e4d7`. Its
  `train/data/val.jsonl` entry is `train/out/pipeline-data/val.jsonl`, the shipped adapter's
  prompt format, so rebuild `pipeline-data` before packing after a fresh clone.

## 6. Environment

- Training laptop: Windows host + WSL2 Ubuntu 26.04 (distro name `Ubuntu`); RTX 4060 Laptop
  8 GB; 16 GB RAM; repo at `/mnt/c/Users/user/Documents/dev/apctt` there. Base weights live at
  `/home/zeke/models/SmolLM3-3B` (bf16 safetensors, 5.8 GB), not in the HF cache.
- Training venv: `~/.venvs/apctt-train` with Python 3.14.4, torch 2.15.0+cu132,
  transformers 5.22.1 (resolved from `>=5.19`), peft 0.21.2, bitsandbytes 0.50.2,
  datasets 5.1.0, pyarrow 26.0.0. The event base environment pins transformers 5.19.0
  (`cuda/constraints-base.txt`); test the adapter path against that before enabling it.
- `pydantic 2.13.5` (the base version) is installed in the training venv only for local
  `apctt_contracts` validation (`module_eval.py`, `ranker_check.py`).
- Catalogue models other than SmolLM3 live in a kit-layout store at `~/models/store`
  (`cuda/tools/get_models.py ID --store ~/models/store`); `train_ranker.py` defaults to the
  cross-encoder there, or under `$APCTT_MODEL_STORE` when that is set.
- `wsl.exe` expands `$VAR` on the Windows side, so a `$VAR` inside the `bash -lc '...'`
  wrapper arrives empty. Run multi-step commands from an interactive WSL shell, or build the
  value in Git Bash first.
- Run GPU work inside WSL, not the Windows Python.
- Packing (`make_submission.py`, `train/pack_adapter.py`) needs only the standard library and
  runs with any Windows or WSL Python. `module_eval.py` needs `pypdfium2` and `pydantic`, and
  imports the Unix-only `resource` module, so it runs in WSL.
- Do not hard-code the clone path. Commands below run from the repository root.

## 7. Commands

```bash
# WSL calls from Git Bash, started in the repository root
MSYS_NO_PATHCONV=1 wsl -d Ubuntu --cd "$(cygpath -w "$PWD")" -- bash -lc '<cmd>'

# rebuild data (SDG 14 holdout, the shipped split)
~/.venvs/apctt-train/bin/python train/build_data.py --holdout-sdg 14

# token-budget check, then train
~/.venvs/apctt-train/bin/python train/train_qlora.py --data train/data/train.jsonl --max-len 896 --dry-run
~/.venvs/apctt-train/bin/python train/train_qlora.py --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora

# recursive PDFs -> JSONL -> QLoRA (dev only; adapter defaults to train/out/)
~/.venvs/apctt-train/bin/python train/pipeline.py --dry-run

# adapter scoring and one-case smoke test
~/.venvs/apctt-train/bin/python train/evaluate.py --adapter my_submission/assets/smollm3-trl-lora \
  --data train/data/val.jsonl --out train/out/eval.json
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora --index 0

# module over the public cases
~/.venvs/apctt-train/bin/python train/module_eval.py [--llm] [--only IDS] \
  --out train/out/module-eval.json

# adapter test prompts (six labelled cases, see train/TEST_PROMPTS.md)
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/test_prompts.jsonl --index 0

# pack and verify the submission
PYTHONPATH=cuda/src ~/.venvs/apctt-train/bin/python cuda/tools/make_submission.py \
  my_submission --output train/out
PYTHONPATH=cuda/src ~/.venvs/apctt-train/bin/python cuda/tools/make_submission.py \
  --check train/out/CICA_submission.zip

# adapter-only handoff bundle
python train/pack_adapter.py

# B/C criterion-evidence ranker (dev only; not shipped)
(cd cuda && PYTHONPATH=src ~/.venvs/apctt-train/bin/python tools/get_models.py \
  ms-marco-minilm-l6-v2 --store ~/models/store)          # once, needs network
~/.venvs/apctt-train/bin/python train/build_criteria_data.py
~/.venvs/apctt-train/bin/python train/train_ranker.py --eval-only   # untrained baseline
~/.venvs/apctt-train/bin/python train/train_ranker.py
~/.venvs/apctt-train/bin/python train/ranker_check.py --case USGS-SDG-076 [--device cpu]

# local test UI, then open http://127.0.0.1:3000 in the Windows browser
~/.venvs/apctt-train/bin/python ui/server.py [--port 3000]
```

The UI prints `127.0.0.1`, not `localhost`: a Windows app listening on `[::1]:3000` (for
example a Next.js dev server) wins the browser's `localhost` lookup. Uploads go to the
git-ignored `train/out/ui-uploads/`.

## 8. Hard constraints and gotchas

- Catalogue-only models. Never ship base weights, packages, a venv, PDFs or training data in
  the submission zip. `assets/` holds only team weights and rules.
- Any weight file over 50 MiB must be declared as team-trained in `submission.json`.
- The 180 s per-case budget includes model load. Rules-only is safe; adapter is not.
- Keep training examples at or below about 900 tokens. Longer sequences crash WSL with
  `CUDA driver error: device not ready` (free-RAM pressure, not a clean OOM). Fix with
  `wsl --shutdown` and retry; check free RAM first.
- SmolLM3: `enable_thinking=False`; transformers 5.x uses `dtype=`, not `torch_dtype=`, and
  has no `warmup_ratio`.
- `cuda/contracts/public-cases.json` must be read with `encoding="utf-8"` on Windows.
- Tool calls that write regex through heredocs lose one backslash level (`\b` becomes
  `\x08`). Use the write/edit tools, or `chr(92)`, not shell heredocs, for regex code.
- `module_eval.py` needs `PYTHONPATH=my_submission:cuda/src` (it inserts them itself).
- After changing anything under `assets/`, repack; the package hash and both zips change.
- Line endings: `.gitattributes` forces LF. Without it, Windows git (`core.autocrlf=true`)
  checks files out as CRLF and the package hash changes although no content changed. If
  `git ls-files --eol` shows `w/crlf` for a file, fix that before packing.

## 9. Known limits and next work

- The adapter collapses to TRL 6 and does not beat the prior on held-out data. Do not spend
  more time on part A with the current adapter and 74 weak labels.
- Part B/C have the headroom: evidence page precision (gold pages are often abstract or
  results pages), avoiding filled-ladder claims that no sentence supports, and tightening
  TRL 7-9 rules.
- The organizer's labels carry no evidence pages, so per-criterion rule quality can only be
  reviewed manually on a sample.
- The filled ladder cites one page for C01-C06 (see `ranker_check.py` output), which is why
  the module puts the rationale page first for only 1/25 held-out cases. Next step: rerank
  the rule candidates with the B/C ranker behind a setting that defaults off, then compare
  on the SDG 14 split. Shipping it needs `models[]` to declare the 88 MB weights as
  team-trained (`base_model: ms-marco-minilm-l6-v2`), the Apache-2.0 text with the weights,
  and about 6 s/case on CPU.
- The ranker's scores are poorly calibrated (for example 0.08 for a strong C10 sentence),
  so use its order, not its probability, to decide `met`.
- If the adapter is ever re-enabled, first align `llm.py:build_prompt` with the training
  prompt (report year, excerpt selection), then re-measure on the SDG 14 holdout.
- The earlier package hash `cd3e6aed…` could not be reproduced from the committed files
  (neither LF nor CRLF), so the old zip was packed from an uncommitted state. Commit the
  repack together with its source, as `SKILLS.md` §7 says.
- `cuda/` still tracks `__pycache__/*.pyc`; that is the kit's and is left alone.
