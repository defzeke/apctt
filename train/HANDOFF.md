# Handoff: continue training SmolLM3 for APCTT

Last updated 2026-11-04. This document is for the next person on the team. It assumes
no access to the previous chat. All commands run from the repository root; on the training
laptop that is `/mnt/c/Users/user/Documents/dev/apctt` inside WSL.

## 1. What we are doing and why

We are fine-tuning **`smollm3-3b-nf4`**, a model in the organizer's frozen catalogue
(`cuda/contracts/model-catalog.json`), with **QLoRA** to turn retrieved report
excerpts into:

```json
{"trl": 6, "rationale": "One sentence grounded in the excerpts."}
```

Only the **LoRA adapter** is submitted. The base weights are never zipped: the event
machine provides them in `APCTT_MODEL_STORE`, and the submission declares the adapter
as team-trained with `base_model: "smollm3-3b-nf4"`.

Hard rules that constrain this work:

- Only catalogue models may be used. No external model, no merged 7B weights.
- Team work is MIT; the base model keeps Apache-2.0. Ship the Apache-2.0 text and a
  NOTICE with the adapter.
- The zip is capped at 2 GB. An 8B merged model does not fit; a LoRA adapter does.
- Evaluation is offline and every case gets 180 s, including model load.
- The public PDFs' TRLs are organizer estimates, not expert gold. Never treat
  `train.jsonl` accuracy as the competition score.

Scope reminder: this adapter only addresses part A (45%, TRL). Parts B (40%) and C
(15%) need a retrieval/rules/evidence pipeline in `participant_module/adapter.py`.
Do not expect the adapter alone to win.

## 2. Machine and environment (already built)

| Item | Value |
|---|---|
| Host | Windows, 16 GB RAM |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM |
| WSL | Ubuntu 26.04 (distro name `Ubuntu`) |
| Training venv | `~/.venvs/apctt-train` (WSL home, not the repo) |
| Python | 3.14.4 |
| torch | 2.15.0+cu132, CUDA 13.2, bf16 supported |
| transformers / peft / bitsandbytes / datasets | 5.22.1 / 0.21.2 / 0.50.2 / 5.1.0 (training venv; the event base env pins transformers 5.19.0) |
| Base weights | `/home/zeke/models/SmolLM3-3B` (bf16 safetensors) |
| Repo | `/mnt/c/Users/user/Documents/dev/apctt` on this laptop; any clone path works |

Verify in one line:

```bash
wsl -d Ubuntu -- bash -lc '~/.venvs/apctt-train/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"'
```

Expected: `2.15.0+cu132 NVIDIA GeForce RTX 4060 Laptop GPU True`.

Rebuild from scratch with `train/setup_wsl.sh` (installs uv, Python 3.12, torch 2.11 and
`train/requirements-train.txt`; the venv in use here is Python 3.14 with torch 2.15, and
needs `datasets` installed separately).

## 3. What is already on disk

On the training laptop. `train/data/pdfs/` and `train/out/` are git-ignored, so a fresh clone
has neither: fetch the PDFs, then rebuild what you need with the commands below. The `out/`
rows are local artifacts from earlier runs, not part of the repository.

| Path | Contents | Size |
|---|---|---|
| `train/data/pdfs/` | 99 public USGS PDFs, hash-checked | ~1.1 GB |
| `train/data/train.jsonl` | 74 training records (SDG 6, 11, 13) | small |
| `train/data/val.jsonl` | 25 held-out records (SDG 14) | small |
| `train/data/small.jsonl` | 10-record subset used for the smoke run | small |
| `/home/zeke/models/SmolLM3-3B/` | base model (bf16 safetensors, revision pin recorded in the manifest) | 5.8 GB |
| `my_submission/assets/smollm3-trl-lora/` | **current adapter**: the 2026-11-04 pipeline adapter (r=8, 3 epochs, 74-record SDG-14 split, module prompt, extractive targets) | 58 MB |
| `my_submission/assets/smollm3-trl-lora/LICENSE`, `NOTICE` | Apache-2.0 text and attribution | 11 KB |
| `train/data/sdg14/` | frozen SDG-14 split (identical to `train/data/`) | small |
| `train/data/random/` | 79/20 stratified random split used for v2 | small |
| `train/out/v2-adapter/` | adapter trained on a stratified random split (0.75 vs 0.80; not shipped) | 58 MB |
| `train/out/eval*.json` | per-case predictions and summaries | small |
| `train/out/pipeline-data/` | recursive build: `train.jsonl`, `val.jsonl`, `dataset_report.json` | small |
| `train/out/pipeline-adapter/` | adapter from `train/pipeline.py` (module prompt, extractive targets, r=8, 3 epochs; 0.70 on SDG 14 with `evaluate.py` and `module_eval.py --llm`). **Copied into `my_submission/` on 2026-11-04** | 58 MB |
| `train/out/module-eval-{llm-shipped,llm-pipeline,rules-sdg14}.json` | old adapter, new adapter and rules inside the module on SDG 14 (0.68 / 0.70 / 0.70) | small |
| `train/out/pipeline-eval.json` | per-case scores for the pipeline adapter | small |
| `train/out/extractive-data/` | extractive build: `train.jsonl`, `val.jsonl`, `val-oracle.jsonl`, `dataset_report.json` | small |
| `train/out/pipeline-adapter-extractive/` | extractive-target adapter, r=16, 6 epochs (quotes verbatim; 0.70 A on SDG 14) | 121 MB |
| `train/out/eval-extractive-{val,oracle,train}.json` | per-case scores for the extractive adapter | small |
| `train/out/validate/`, `smoke5/` | early smoke adapters (do not ship) | ~60 MB each |
| `~/models/store/cross-encoder/ms-marco-MiniLM-L6-v2/<rev>/` | catalogue cross-encoder, base of the B/C ranker | 92 MB |
| `train/out/criteria-data/` | silver (criterion, sentence) pairs: 9,267 train / 3,116 val, `val_pool.jsonl`, `report.json` | small |
| `train/out/criteria-ranker/` | B/C ranker (2 epochs; AUC 0.91, rationale-page top-3 0.48 vs rules 0.44; not shipped) | 88 MB |
| `train/out/ranker-check-{sdg14,smoke}.txt` | per-criterion ranker vs module output for the held-out and smoke cases | small |
| `train/out/module-eval.json` | rules module over the 99 public cases (2026-11-04) | small |

## 4. First-day verification (5 minutes)

```bash
wsl -d Ubuntu
cd <repository root>
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl --max-len 896 --dry-run
```

Expected:

```json
{"records": 74, "max_len": 896, "tokens_max": 896, "tokens_mean": 778.4,
 "target_tokens_min": 15, "target_tokens_mean": 41.0,
 "records_truncated_to_no_target": 0}
```

If `records_truncated_to_no_target` is not 0, stop and fix data before training (see
§8).

## 5. Full training

```bash
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora \
  | tee train/out/full-train.log
```

Defaults that are known to work on this card:

- `--max-len 896`, `--r 8`, `--alpha 32`, `--epochs 3`, `--batch 1`,
  `--grad-accum 4`, `--lr 2e-4`, `--optim adamw_torch`, bf16.
- Measured: **231.1 s (about 4 minutes)** for 74 records x 3 epochs on the RTX 4060
  Laptop (shipped adapter's `training_manifest.json`); 1150.6 s on the earlier RTX 3050.
  Model load adds about 1.5 minutes.
- Do not use `--r 16 --grad-accum 8` here. That combination died at the first
  optimizer step with `CUDA driver error: device not ready` while Windows had only
  2.6 GB free; the WSL kernel logged `dxgkio_make_resident: Ioctl failed: -12`
  (ENOMEM). `wsl --shutdown` released the model page cache and freed 7.7 GB.
- The script refuses to start if any record's answer is truncated
  (`TARGET_TRUNCATED`) instead of silently training on nothing.
- On success it prints `{"status": "TRAINED", ...}` and writes
  `adapter_config.json`, `adapter_model.safetensors` and `training_manifest.json`.

OOM ladder, in order:

1. `--max-len 768` (and rebuild data with `--max-chars 2000`).
2. `--r 8`.
3. `--grad-accum 4`.
4. Restart WSL (`wsl --shutdown`) to clear leaked memory.

Do **not** raise `--max-len` above ~900 on this laptop. Longer sequences produce
`RuntimeError: CUDA driver error: device not ready`, which is memory pressure inside
WSL, not a bug in the script.

## 6. Evaluate

```bash
~/.venvs/apctt-train/bin/python train/evaluate.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/data/val.jsonl \
  --out train/out/eval.json
```

It loads once and reports:

- `mean_A_score` using the competition's rule: exact 1.0, off-by-one 0.5, else 0
- `exact`, `off_by_one`, `off_by_two_or_more`, `invalid_output`
- mean generation seconds and peak VRAM

For a quick check use `--limit 5`. For a single case with the raw generation:

```bash
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora --index 0
```

Measured baselines (2026-11-01, RTX 3050, 2.08 GiB peak; pipeline rows 2026-11-03, RTX 4060):

| Adapter | Validation | Records | mean_A_score | exact | off-by-one | off-by-two+ | invalid |
|---|---|---|---|---|---|---|---|
| **pipeline r=8, 3 ep, extractive (shipped since 2026-11-04)** | SDG 14 held out | 25 | **0.70** | 13 | 9 | 3 | 0 |
| same, inside the module (`module_eval.py --llm`) | SDG 14 held out | 25 | **0.70** | 13 | 9 | 3 | 0 |
| v1 (shipped until 2026-11-03) | SDG 14 held out | 25 | 0.72 | 14 | 8 | 3 | 0 |
| v1, inside the module (rerun 2026-11-04) | SDG 14 held out | 25 | 0.68 | 12 | 10 | 3 | 0 |
| v1 | stratified random | 20 | 0.80 | 13 | 6 | 1 | 0 |
| v2 | stratified random | 20 | 0.75 | 11 | 8 | 1 | 0 |
| pipeline (organizer targets, 6 ep, r=16) | SDG 14 held out | 25 | 0.70 | 12 | 11 | 2 | 0 |
| pipeline (extractive targets, 6 ep, r=16) | SDG 14 held out | 25 | 0.70 | 14 | 7 | 4 | 0 |

v1 was shipped first because it generalized to an unseen SDG better than v2. On 2026-11-04
the r=8 pipeline adapter replaced it: it is trained on the module's own prompt, and inside
the module it scores 0.70 against v1's 0.68 (case USGS-SDG-077 is the only difference). It
also quotes the report instead of paraphrasing (grounded 5-gram 0.998). The differences are one or two cases on 20-25 records; treat them as
noise. The dominant failure is collapse toward TRL 6: the training set has 41 sixes and
no eights or nines, while SDG 14 contains two nines and one eight. The model answered
6 for all three.

`evaluate.py` also reports three rationale metrics (`rationale_token_f1`,
`rationale_number_recall`, `rationale_grounded_5gram` = share of generated 5-grams
found verbatim in the excerpt). Extractive targets changed what the model does:

| Split | Records | token-F1 | number recall | grounded 5-gram |
|---|---|---|---|---|
| train (target always in excerpt) | 74 | 1.00 | 1.00 | 1.00 |
| val-oracle (target forced into excerpt) | 25 | 0.72 | 0.79 | 0.96 |
| val (label-free retrieval) | 25 | 0.24 | 0.38 | 1.00 |
| val, organizer-target adapter (before) | 25 | 0.18 | 0.29 | 0.005 |

Groundedness 1.0 means every phrase it emits now exists verbatim in the excerpt: the
fabrication problem is gone. The remaining gap is selection and retrieval — the val
excerpt contains the chosen target sentence only 12% of the time, so token-F1 on the
honest split cannot rise much without a better page selector. TRL accuracy is
unchanged (0.70, the always-6 prior): on the train split the model reproduces the label
distribution exactly (73/74, predicted 4:1 5:9 6:42 7:22 vs gold 4:1 5:10 6:41 7:22),
i.e. it memorizes case→label, not a TRL ladder.

Reference single case: `USGS-SDG-076`, expected 6, generated 6, 33.4 s for 160 new
tokens, 2.06 GiB peak (cold adapter loader, WSL). With the extractive adapter it quotes
"Contoured elevation differences between the Northern Gulf of Mexico TBDEM and the 3DEP
1/3 arc-second DEMs indicate that 85 percent…".

## 7. Package for submission

```bash
cp cuda/licenses/Apache-2.0.txt my_submission/assets/smollm3-trl-lora/LICENSE
printf 'SmolLM3-3B (HuggingFaceTB), Apache-2.0, revision a07cc9a04f16550a088caea529712d1d335b0ac1\nAdapter weights: team-trained QLoRA (r=8), MIT.\n' \
  > my_submission/assets/smollm3-trl-lora/NOTICE
```

`LICENSE` next to the adapter is the **Apache-2.0 base licence**, not the adapter's. The
adapter's MIT text is `participant_module/LICENSE`; `train/pack_adapter.py` adds it to the
handoff bundle as `smollm3-trl-lora/LICENSE-MIT`.

In `my_submission/submission.json`, `models` must contain both:

```json
[
  {"catalog_id": "smollm3-3b-nf4"},
  {"name": "smollm3-trl-lora",
   "path": "assets/smollm3-trl-lora",
   "team_trained": true,
   "base_model": "smollm3-3b-nf4",
   "license": "MIT",
   "description": "QLoRA r=8 adapter (3 epochs, 74 USGS reports, seed 42) used by the optional CUDA refinement path; disabled by default."}
]
```

Then:

```bash
python cuda/tools/make_submission.py my_submission --output train/out
```

Exit 0 = VALID. Exit 2 = licence review item. Exit 1 = refused; read each `ERROR`
line. The shipped adapter (about 58 MB) is tracked in git so that a clone can be packed
as is; do not commit other adapters (`train/out/` is ignored).

The module loader must read the base from `APCTT_MODEL_STORE` and the adapter from
`asset_root`, with `HF_HUB_OFFLINE=1` and `local_files_only=True`. The real loader is
`participant_module/llm.py`; the sketch below shows the shape:

```python
def load(self, config, asset_root):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from pathlib import Path
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    base = Path(os.environ["APCTT_MODEL_STORE"]) / (
        "HuggingFaceTB/SmolLM3-3B/a07cc9a04f16550a088caea529712d1d335b0ac1")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4"))
    self.model = PeftModel.from_pretrained(model, Path(asset_root) / "smollm3-trl-lora")
```

Match the `BitsAndBytesConfig` exactly to `train_qlora.py` (`bnb_4bit_quant_type`,
double quant, bf16 compute). The pattern is also visible in
`cuda/participant_example/cuda_dummy/adapter.py`.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA driver error: device not ready` | Sequence or batch too large for 6 GB / WSL RAM | Lower `--max-len`, `--r`, `--grad-accum`; restart WSL |
| `TARGET_TRUNCATED` | Answer fell outside `--max-len` | Lower `build_data.py --max-chars`, or raise `--max-len` within the safe envelope |
| `tokens_max == max_len`, `target_tokens_min == 0` in dry run | Same | Rebuild data with `--max-chars 2000` |
| `TypeError: unexpected keyword argument 'warmup_ratio'` | transformers 5.x | Already fixed (uses `warmup_steps`); do not reintroduce |
| `OSError: ... offline mode is enabled` in `evaluate.py` | Cache missing or revision mismatch | Pass `--base <snapshot path>` or re-export/download the model |
| Load takes > 2 min | Cold read of 6.2 GB from disk | Normal on this laptop; event GPU also cold-loads, so measure it |
| WSL says `Failed to start the systemd user session` | Harmless warning in this setup | Ignore |
| Need more context | 16 GB host, 7.7 GB WSL | Optional: `C:\Users\<you>\.wslconfig` with `memory=10GB` and `swap=4GB`, then `wsl --shutdown`. Leaves less for Windows. |

## 9. What to improve next, in priority order

1. **Do not chase part A with the adapter.** Always predicting 6 scores 0.753 on the
   99 public cases and 0.70 on the held-out SDG 14 split; the adapter scores 0.70
   there. The rules-only module matches the prior (0.7525 public, 0.70 SDG 14) at
   0.9 s and 0.28 GiB per case. Spend effort on parts B and C (evidence pages and
   spans), where there is still headroom.
2. **Tune the data, not the architecture.** Try `--max-chars 3200` with `--max-len 896`
   only if it stays under the token limit; try `--holdout-sdg 6` and `11` to see
   whether the score is stable across splits.
3. **Criterion mode: started as a ranker, not yet wired in.** Silver labels are
   bootstrapped from the rules plus the sentence that best matches the organizer
   rationale (`build_criteria_data.py`). A cross-encoder trained on them
   (`train_ranker.py`) cites the rationale page first for 7/25 held-out cases against
   1/25 for the module (`ranker_check.py`). It is a reranker over verbatim extracted
   sentences, so the span stays on the page and there is no fabrication risk. Next: a
   `use_ranker` setting (default off) that reorders the rule candidates, then re-measure.
   Use its order, not its probability (it scores a strong C10 sentence at 0.08).
4. **Grounding is done; selection is the bottleneck.** `--target extractive` made the
   adapter quote verbatim (grounded 5-gram 1.00 on val, 0.96 with oracle excerpts, up
   from 0.005). The chosen target sentence is in the label-free val excerpt only 12%
   of the time, so the next lever is the page/sentence selector: anchor validation
   retrieval on the evidence terms and measure oracle-vs-honest F1 as the target.
5. **Prompt skew: fixed for the prompt text, not for excerpt selection.** The shipped
   adapter (2026-11-04) is trained on `llm.py:build_prompt` itself, and it scores 0.70 with
   both `evaluate.py` and `module_eval.py --llm` (the old one: 0.72 vs 0.68). The excerpts
   still differ: `pipeline.py` pages in training, rule-selected pages in the module.
6. **Measure the event budget.** On the event machine the module must load base +
   adapter and assess within 180 s. Time a cold load from the model store in WSL and
   keep generation short (`max_new_tokens` <= 200).

## 10. Handoff checklist

- [x] `train/data/{train,val}.jsonl` present (74 / 25)
- [x] `train/train_qlora.py --dry-run` reports `records_truncated_to_no_target: 0`
- [x] Full training completed; `training_manifest.json` present in the adapter folder
- [x] `evaluate.py` returns a recorded `mean_A_score` (0.70 on SDG 14 for the shipped adapter; 0.72 for the old one)
- [x] `module_eval.py --llm` on SDG 14 recorded for both adapters (0.70 shipped, 0.68 old)
- [x] Apache-2.0 and NOTICE copied next to the adapter
- [x] Cold-load + one-case timing recorded (33.4 s, 2.06 GiB)
- [x] Adapter declared in `submission.json` with `base_model: "smollm3-3b-nf4"`
- [x] `make_submission.py` gives `VALID_WITH_REVIEW` (only the two dataset licence
      strings are flagged; they are USGS public domain)
- [x] `participant_module` implements criterion status + evidence (parts B and C)

## 11. Files in this folder

- `setup_wsl.sh` — one-time environment build
- `build_data.py` — xlsx + PDFs -> JSONL
- `pipeline.py` — recursive PDFs -> JSONL -> QLoRA, with a token-budget precheck
- `train_qlora.py` — QLoRA trainer; `--dry-run` validates token budgets
- `evaluate.py` — part-A scoring over a split
- `infer_check.py` — single-case generation, timing, peak VRAM
- `module_eval.py` — the shipped module over the public cases
- `pack_adapter.py` — deterministic adapter-only handoff zip
- `build_criteria_data.py`, `train_ranker.py` — B/C criterion-evidence ranker (see `README.md`)
- `ranker_check.py` — one-case ranker vs module comparison, per criterion
- `requirements-train.txt` — training-only dependencies
- `README.md` — short command reference

## 12. Shipped module (v0.1.0)

`my_submission/` is packed as `train/out/CICA_submission.zip` (60.6 MB, package hash
`9a5b98347c307857c09a23b8013972ab2dbb759c9fe1bd3099e4dec353391d19`, zip hash
`efa68e1ec5e9df0f11467fa8f4256b584ed44d6cd16d51d8f3b026cf95894d4d`, packed 2026-11-04,
`--check` `VALID_WITH_REVIEW`).

- CPU rules-only by default (`settings.use_llm = false`, and `false` in code when the
  setting is absent): pypdfium2 page text,
  header/footer/caption/reference/credit/map-legend filtering, plus contents dot leaders,
  "… showing" captions, table headnotes, report-series headers and acknowledgements
  (2026-11-04: 108 junk cited spans to 0 on the 99 cases), description-matched
  criteria rules, TRL prior 6 with the ladder filled to that level, page-and-span
  evidence (same-page window spans when a single sentence only partially matches).
- Optional CUDA path loads `smollm3-3b-nf4` + the QLoRA adapter and trusts its TRL.
  It is bundled and declared but off. The 2026-11-04 adapter ties the rules (0.70 vs 0.70)
  on the held-out SDG 14 split by predicting 6 everywhere: 27-38 s for the cold first case,
  about 5 s after that, 6.7 GiB RSS.
- Public-case results (rules, `module_eval.py`, 2026-11-04): mean A 0.7525, 54 exact, 41 off-by-one, 4 off-by-two+,
  0 invalid, 7.2 met criteria, 14.4 spans, 0.9 s, 0.28 GiB.
- Module source: `participant_module/{adapter,extract,rules,llm}.py`; no packages
  copied in; runtime deps come from the common base.
- Adapter use and transfer: `train/USING_THE_ADAPTER.md` (standalone loading, prompt
  format, hashes, handoff bundle, limits).
- Any change to the model, data, results or package must follow `SKILLS.md` (documentation
  and hash sync) and `AGENTS.md` (project context).
