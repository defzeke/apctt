# AGENTS.md — APCTT Stage 1 project context

Read this before touching the fine-tune, the training data, the assessment module or the
submission package. It is the state of the project as of 2026-11-02. Deeper detail lives in
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
| `train/data/pdfs/` | 99 public USGS PDFs, hash-checked, 1.1 GB. |
| `train/out/` | adapters, eval JSON, packed zips; ignored by git. |
| `USGS_SDG_open_data_set_corrected.xlsx` | organizer sheet used to build SFT labels. |

## 3. Base model and adapter

- Base: `smollm3-3b-nf4` = `HuggingFaceTB/SmolLM3-3B`, revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`, Apache-2.0. Only catalogue models are allowed.
- Adapter: `my_submission/assets/smollm3-trl-lora/`, 60,523,464 bytes
  (`sha256 17b781a9ae7c4c8f05f6472399d9b4aca280e61ec62a6335f6bedcbd011cdee2`), MIT.
- Training: QLoRA r=8, alpha=32, dropout 0.05, targets q/k/v/o/gate/up/down, 74 records,
  3 epochs, max length 896, lr 2e-4, batch 1, grad-accum 4, `adamw_torch`, seed 42, bf16,
  1150.6 s. Full record: `assets/smollm3-trl-lora/training_manifest.json`.
- Data: `train/data/train.jsonl` / `val.jsonl` = 74 / 25 split with SDG 14 held out.
  Frozen copies: `train/data/sdg14/`; stratified random 79/20 split: `train/data/random/`.
- Labels: organizer-estimated TRL + 200-character rationale. No per-criterion labels.

### Measured results

| evaluation | result |
|---|---|
| adapter, `train/evaluate.py`, SDG 14 holdout (25) | 0.72 mean A, 14 exact, 0 invalid |
| adapter, `train/module_eval.py`, SDG 14 holdout (25) | 0.68 mean A, 12 exact, 0 invalid |
| rules only, `train/module_eval.py`, SDG 14 holdout (25) | 0.70 mean A, 13 exact |
| rules only, `train/module_eval.py`, 99 public cases | 0.7525 mean A, 54 exact, 41 off-by-one, 4 off-by-two+, 0 invalid |
| always predict 6 | 0.753 on 99 public, 0.70 on SDG 14 |
| adapter cost | about 33 s cold load, 2.06 GiB peak VRAM, about 14 s/case integrated, 6.75 GiB RSS |
| rules cost | 1.9 s/case, 0.26 GiB RSS |

`evaluate.py` and `module_eval.py` differ: `evaluate.py` feeds the full training prompt from
`val.jsonl`; `module_eval.py` feeds excerpts from the PDF and uses the module path. Quote the
metric with the tool name.

## 4. Current shipped behaviour

Decision: the submission is **rules-only, CPU**. The adapter is bundled and declared but
disabled (`settings.use_llm = false`) because it did not beat the rules prior on the held-out
split. Do not re-enable it without new evidence.

`participant_module`:

- `adapter.py` — entry point `AssessmentModule`; rules first, optional adapter refine, fallback to abstain.
- `extract.py` — pypdfium2 page text; strips headers, footers, page numbers, captions,
  reference lists, URLs, disclaimers, abbreviation lists.
- `rules.py` — rules matched to criteria by official description (not by ID), candidate
  scoring, evidence selection, fill the TRL ladder up to the predicted level, target check.
- `llm.py` — lazy 4-bit base + adapter loader; only used when `use_llm` is true.
- TRL policy: predict 6 when any criterion is met, because the public prior is 6; evidence is
  used for B/C, not for a raw TRL ladder. Levels 7-9 require independent matches.

The module config is `config["checklist"]` plus `config["module"]["settings"]`; in
`submission.json` the key is top-level `settings`.

## 5. Submission package

- Team `CICA`, package `cica-usgs-trl-assessor` 0.1.0, base `apctt-base-9e7fe1324bb6`,
  licence MIT, device `cpu`.
- `models`: catalogue `smollm3-3b-nf4` plus team-trained `assets/smollm3-trl-lora`
  (`base_model: smollm3-3b-nf4`).
- Current artifact: `train/out/CICA_submission.zip`, 60.5 MB,
  zip `sha256 72a90d9ca1081f3432aad048165ee515f9434b8716157fdd18f4a84eb18f24b2`,
  package `sha256 d030430f18f1582cd9aa31d2cb202af65786176d6a308bbc05e9d4aad7d75588`.
- `make_submission.py` writes `declared_package_sha256` into `submission.json` and
  `declaration.json`. The package hash covers only `participant_module/` and `assets/`.
- `--check` result: `VALID_WITH_REVIEW`. The only review items are the two dataset licence
  strings (`U.S. public domain (USGS); ...`), which the licence checker cannot map to an SPDX
  id. That is expected.
- Adapter-only handoff bundle: `train/out/CICA_smollm3-trl-lora.zip`, 55,990,894 bytes,
  `sha256 9078fb494e82fd4cffec98efc5634c1e5746c3179a25b90323253cd298368828`.

## 6. Environment

- Windows host + WSL2 Ubuntu 22.04; RTX 3050 6 GB; 16 GB RAM (WSL capped at 7.7 GB + 2 GB swap).
- Training venv: `~/.venvs/apctt-train` with Python 3.12, torch 2.11.0+cu128,
  transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, TRL, datasets.
- `pydantic 2.14.2` was installed only for local `apctt_contracts` validation.
- Run GPU work inside WSL, not the Windows Python.

## 7. Commands

```bash
# all WSL calls from Git Bash need this prefix
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -lc 'cd "/mnt/c/Github Projects/apctt" && <cmd>'

# rebuild data (SDG 14 holdout, the shipped split)
~/.venvs/apctt-train/bin/python train/build_data.py --holdout-sdg 14

# token-budget check, then train
~/.venvs/apctt-train/bin/python train/train_qlora.py --data train/data/train.jsonl --max-len 896 --dry-run
~/.venvs/apctt-train/bin/python train/train_qlora.py --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora

# adapter scoring and one-case smoke test
~/.venvs/apctt-train/bin/python train/evaluate.py --adapter my_submission/assets/smollm3-trl-lora \
  --data train/data/val.jsonl --out train/out/eval.json
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora --index 0

# module over the public cases
~/.venvs/apctt-train/bin/python train/module_eval.py [--llm] [--only IDS] \
  --out train/out/module-eval.json

# pack and verify the submission
PYTHONPATH=cuda/src ~/.venvs/apctt-train/bin/python cuda/tools/make_submission.py \
  my_submission --output train/out
PYTHONPATH=cuda/src ~/.venvs/apctt-train/bin/python cuda/tools/make_submission.py \
  --check train/out/CICA_submission.zip
```

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

## 9. Known limits and next work

- The adapter collapses to TRL 6 and does not beat the prior on held-out data. Do not spend
  more time on part A with the current adapter and 74 weak labels.
- Part B/C have the headroom: evidence page precision (gold pages are often abstract or
  results pages), avoiding filled-ladder claims that no sentence supports, and tightening
  TRL 7-9 rules.
- The organizer's labels carry no evidence pages, so per-criterion rule quality can only be
  reviewed manually on a sample.
- Tracked build artifacts remain in git from an earlier external commit. Suggested cleanup
  (not run): `git rm -r --cached train/out train/__pycache__ train/data/pdfs`.
