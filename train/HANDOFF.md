# Handoff: continue training SmolLM3 for APCTT

Last updated 2026-11-01. This document is for the next person on the team. It assumes
no access to the previous chat.

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
| GPU | NVIDIA GeForce RTX 3050, 6 GB VRAM |
| WSL | Ubuntu 22.04, 7.7 GB RAM, 2 GB swap |
| Training venv | `~/.venvs/apctt-train` (WSL home, not the repo) |
| Python | 3.12 |
| torch | 2.11.0+cu128, CUDA 12.8, bf16 supported |
| transformers / peft / bitsandbytes | 5.22.1 / 0.21.2 / 0.50.2 |
| Repo | `/mnt/c/Github Projects/apctt` |

Verify in one line:

```bash
wsl -d Ubuntu-22.04 -- bash -lc '~/.venvs/apctt-train/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"'
```

Expected: `2.11.0+cu128 NVIDIA GeForce RTX 3050 True`.

Rebuild from scratch with `train/setup_wsl.sh` (installs uv, Python 3.12, torch, and
`train/requirements-train.txt`).

## 3. What is already on disk

| Path | Contents | Size |
|---|---|---|
| `train/data/pdfs/` | 99 public USGS PDFs, hash-checked | ~1.1 GB |
| `train/data/train.jsonl` | 74 training records (SDG 6, 11, 13) | small |
| `train/data/val.jsonl` | 25 held-out records (SDG 14) | small |
| `train/data/small.jsonl` | 10-record subset used for the smoke run | small |
| `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/` | base model at revision `a07cc9a04f16550a088caea529712d1d335b0ac1` | 6.2 GB |
| `my_submission/assets/smollm3-trl-lora/` | **current adapter**, 3 epochs on the 74-record SDG-14 split | 58 MB |
| `my_submission/assets/smollm3-trl-lora/LICENSE`, `NOTICE` | Apache-2.0 text and attribution | 11 KB |
| `train/data/sdg14/` | frozen SDG-14 split (identical to `train/data/`) | small |
| `train/data/random/` | 79/20 stratified random split used for v2 | small |
| `train/out/v2-adapter/` | adapter trained on a stratified random split (0.75 vs 0.80; not shipped) | 58 MB |
| `train/out/eval*.json` | per-case predictions and summaries | small |
| `train/out/validate/`, `smoke5/` | early smoke adapters (do not ship) | ~60 MB each |

## 4. First-day verification (5 minutes)

```bash
wsl -d Ubuntu-22.04
cd "/mnt/c/Github Projects/apctt"
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
cd "/mnt/c/Github Projects/apctt"
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora \
  | tee train/out/full-train.log
```

Defaults that are known to work on this card:

- `--max-len 896`, `--r 8`, `--alpha 32`, `--epochs 3`, `--batch 1`,
  `--grad-accum 4`, `--lr 2e-4`, `--optim adamw_torch`, bf16.
- Measured on this laptop: **18.3 minutes** for 74 records x 3 epochs, plus ~1.5
  minutes of model load.
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

Measured baselines (2026-11-01, RTX 3050, 2.08 GiB peak):

| Adapter | Validation | Records | mean_A_score | exact | off-by-one | off-by-two+ | invalid |
|---|---|---|---|---|---|---|---|
| v1 (shipped) | SDG 14 held out | 25 | **0.72** | 14 | 8 | 3 | 0 |
| v1 (shipped) | stratified random | 20 | **0.80** | 13 | 6 | 1 | 0 |
| v2 | stratified random | 20 | 0.75 | 11 | 8 | 1 | 0 |

v1 generalizes to an unseen SDG better than v2 does on the random split, so v1 is the
shipped adapter. The differences are one or two cases on 20-25 records; treat them as
noise. The dominant failure is collapse toward TRL 6: the training set has 41 sixes and
no eights or nines, while SDG 14 contains two nines and one eight. The model answered
6 for all three.

Reference single case: `USGS-SDG-076`, expected 6, generated 6, 33.4 s for 160 new
tokens, 2.06 GiB peak (cold adapter loader, WSL).

## 7. Package for submission

```bash
cd "/mnt/c/Github Projects/apctt"
cp cuda/licenses/Apache-2.0.txt my_submission/assets/smollm3-trl-lora/LICENSE
printf 'SmolLM3-3B (HuggingFaceTB), Apache-2.0, revision a07cc9a04f16550a088caea529712d1d335b0ac1\nAdapter weights: team-trained QLoRA, MIT.\n' \
  > my_submission/assets/smollm3-trl-lora/NOTICE
```

In `my_submission/submission.json`, `models` must contain both:

```json
[
  {"catalog_id": "smollm3-3b-nf4"},
  {"name": "smollm3-trl-lora",
   "path": "assets/smollm3-trl-lora",
   "team_trained": true,
   "base_model": "smollm3-3b-nf4",
   "license": "MIT",
   "description": "QLoRA r=16 adapter for JSON TRL synthesis"}
]
```

Then:

```bash
python cuda/tools/make_submission.py my_submission
```

Exit 0 = VALID. Exit 2 = licence review item. Exit 1 = refused; read each `ERROR`
line. Do not commit the adapter to git; it is a build artifact.

The module loader must read the base from `APCTT_MODEL_STORE` and the adapter from
`asset_root`, with `HF_HUB_OFFLINE=1` and `local_files_only=True`:

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
   99 public cases and 0.70 on the held-out SDG 14 split; the adapter scores 0.68
   there. The rules-only module matches the prior (0.7525 public, 0.70 SDG 14) at
   ~2 s and 0.26 GiB per case. Spend effort on parts B and C (evidence pages and
   spans), where there is still headroom.
2. **Tune the data, not the architecture.** Try `--max-chars 3200` with `--max-len 896`
   only if it stays under the token limit; try `--holdout-sdg 6` and `11` to see
   whether the score is stable across splits.
3. **Add the criterion mode.** The sheet has no per-criterion labels, so bootstrap
   them with retrieval + rules and train a second prompt
   (`{"status": "met|not met|insufficient evidence", "page": N}`). Cite a page index,
   never free-text quotes, to avoid the -1 fabrication penalty.
4. **Ground the rationale.** Prefer extractive rationales taken from the cited page
   over generated prose. The experts score support, not fluency.
5. **Measure the event budget.** On the event machine the module must load base +
   adapter and assess within 180 s. Time a cold load from the model store in WSL and
   keep generation short (`max_new_tokens` <= 200).

## 10. Handoff checklist

- [x] `train/data/{train,val}.jsonl` present (74 / 25)
- [x] `train/train_qlora.py --dry-run` reports `records_truncated_to_no_target: 0`
- [x] Full training completed; `training_manifest.json` present in the adapter folder
- [x] `evaluate.py` returns a recorded `mean_A_score` (0.72 on SDG 14)
- [x] Apache-2.0 and NOTICE copied next to the adapter
- [x] Cold-load + one-case timing recorded (33.4 s, 2.06 GiB)
- [x] Adapter declared in `submission.json` with `base_model: "smollm3-3b-nf4"`
- [x] `make_submission.py` gives `VALID_WITH_REVIEW` (only the two dataset licence
      strings are flagged; they are USGS public domain)
- [x] `participant_module` implements criterion status + evidence (parts B and C)

## 12. Shipped module (v0.1.0)

`my_submission/` is packed as `train/out/CICA_submission.zip` (60.5 MB, package hash
`d030430f18f1582cd9aa31d2cb202af65786176d6a308bbc05e9d4aad7d75588`, zip hash
`72a90d9ca1081f3432aad048165ee515f9434b8716157fdd18f4a84eb18f24b2`).

- CPU rules-only by default (`settings.use_llm = false`): pypdfium2 page text,
  header/footer/caption/reference filtering, description-matched criteria rules,
  TRL prior 6 with the ladder filled to that level, page-and-span evidence.
- Optional CUDA path loads `smollm3-3b-nf4` + the QLoRA adapter and trusts its TRL.
  It is bundled and declared but off: 0.68 vs 0.70 (rules) on the held-out SDG 14
  split, 14.3 s and 6.75 GiB per case.
- Public-case results (rules): mean A 0.7525, 54 exact, 41 off-by-one, 4 off-by-two+,
  0 invalid, 7.26 met criteria, 11.4 spans, 1.9 s, 0.26 GiB.
- Module source: `participant_module/{adapter,extract,rules,llm}.py`; no packages
  copied in; runtime deps come from the common base.
- Adapter use and transfer: `train/USING_THE_ADAPTER.md` (standalone loading, prompt
  format, hashes, handoff bundle, limits).
- Any change to the model, data, results or package must follow `SKILLS.md` (documentation
  and hash sync) and `AGENTS.md` (project context).

## 11. Files in this folder

- `setup_wsl.sh` — one-time environment build
- `build_data.py` — xlsx + PDFs -> JSONL
- `train_qlora.py` — QLoRA trainer; `--dry-run` validates token budgets
- `evaluate.py` — part-A scoring over a split
- `infer_check.py` — single-case generation, timing, peak VRAM
- `requirements-train.txt` — training-only dependencies
- `README.md` — short command reference
