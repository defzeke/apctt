# Training workspace (never submitted)

QLoRA fine-tune of `smollm3-3b-nf4` for one narrow job: turn retrieved report
excerpts into `{"trl": 1-9, "rationale": "..."}`. Only the LoRA adapter is shipped;
the base weights stay in the organizer's model store.

Status on 2026-11-04: the shipped adapter in `my_submission/assets/smollm3-trl-lora` is now
the `train/pipeline.py` adapter (module prompt, extractive rationale targets, r=8, 3 epochs,
231 s on the RTX 4060 Laptop). On the 25-case SDG-14 holdout it scores **0.70 mean_A with
both `evaluate.py` and `module_eval.py --llm`** (13 exact, 9 off-by-one, 3 off-by-two, 0
invalid; it predicts 6 for every case), the same as the rules alone. The adapter it replaced
(2026-11-03) scored 0.72 with `evaluate.py` but 0.68 inside the module. The submission still
ships rules-only (`settings.use_llm = false`); the adapter is bundled but off. See
`HANDOFF.md` for the partner guide and the TRL-6 collapse limitation.

Status on 2026-11-04: a B/C criterion-evidence ranker (cross-encoder, silver labels) is
trained in `train/out/criteria-ranker`. It is not shipped; see the ranker section below.

All commands run from the repository root. Generated files (`out/`, `data/pdfs/`) are
git-ignored, so a fresh clone must rebuild them.

## Environment (WSL2 on the training laptop, already built)

```bash
wsl -d Ubuntu
~/.venvs/apctt-train/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# 2.15.0+cu132 NVIDIA GeForce RTX 4060 Laptop GPU True
```

Rebuild from scratch if needed:

```bash
# from Git Bash in the repository root
MSYS_NO_PATHCONV=1 wsl -d Ubuntu --cd "$(cygpath -w "$PWD")" -- bash train/setup_wsl.sh
```

Training runs inside WSL2 because the Windows desktop reserves VRAM. The base model
is cached at `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/`.

## Commands

```bash
# data (fetch the PDFs once per clone)
cd cuda && python tools/get_public_cases.py --all --download --dir ../train/data/pdfs && cd ..
~/.venvs/apctt-train/bin/python train/build_data.py --holdout-sdg 14

# check token budget before spending GPU time
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl --max-len 896 --dry-run

# train (writes adapter + training_manifest.json)
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora

# score on the held-out SDG
~/.venvs/apctt-train/bin/python train/evaluate.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/data/val.jsonl --out train/out/eval.json

# one-case smoke test, with timing and peak VRAM
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora --index 0
```

## End-to-end pipeline (recursive PDFs)

`train/pipeline.py` builds the dataset from every PDF under `train/data/pdfs/`
(recursively), matches each file to a public case by SHA-256 (falling back to
`USGS-SDG-NNN` in the file name), extracts pages with pypdfium2, and keeps the
most TRL-dense pages and sentence windows. It writes `train.jsonl`, `val.jsonl`
and `dataset_report.json`, then runs `train_qlora.py` with a token-budget
precheck before the real QLoRA run:

```bash
# dataset + token stats only
~/.venvs/apctt-train/bin/python train/pipeline.py --dry-run

# full run; the adapter goes to train/out/pipeline-adapter by default
~/.venvs/apctt-train/bin/python train/pipeline.py

# replace the shipped adapter (then repack and refresh the hashes, SKILLS.md)
~/.venvs/apctt-train/bin/python train/pipeline.py \
  --adapter my_submission/assets/smollm3-trl-lora
```

Pass `--base` when the weights are already on disk, so nothing is downloaded:

```bash
~/.venvs/apctt-train/bin/python train/pipeline.py --dry-run \
  --base /home/zeke/models/SmolLM3-3B
```

Targets default to `--target extractive`: the assistant rationale is the verbatim
report sentence that best supports the organizer rationale (content-word overlap,
numbers weighted ×4; table-of-contents, dot-leader and figure-caption text is
rejected). Training excerpts are anchored on that sentence, so the model learns
"find the supporting sentence and copy it" instead of inventing a paraphrase.
`--target organizer` restores the old behaviour (copy the organizer summary).

```bash
# extractive targets, separate output dir, comparison adapter
~/.venvs/apctt-train/bin/python train/pipeline.py \
  --base /home/zeke/models/SmolLM3-3B \
  --out train/out/extractive-data \
  --adapter train/out/pipeline-adapter-extractive \
  --r 16 --epochs 6 --lr 1e-4
```

`--train-select label` (default) anchors only the training split; validation
excerpts stay label-free so the metric measures retrieval. `--oracle-val` (default
on) also writes `val-oracle.jsonl`, which repeats the validation records with the
target forced into the excerpt, separating "can it quote" from "did the selector
find the page".

`--max-chars` (default 1800) sets the excerpt budget; 2400 overflows the 896-token
limit on the longest reports, and the token-budget precheck then refuses to train.

The prompt is byte-for-byte `participant_module/llm.py:build_prompt`, so a new
adapter sees the same text at training and inference time. Outputs default to
`train/out/pipeline-data/` and `train/out/pipeline-adapter/` (both git-ignored).

## Criterion-evidence ranker (parts B and C, development only)

A fine-tuned `ms-marco-minilm-l6-v2` cross-encoder scores (criterion description,
sentence) pairs. Labels are silver: rule matches plus the sentence that best matches the
organizer rationale. It is not wired into the module.

```bash
# base model, once (needs network; about 92 MB)
(cd cuda && PYTHONPATH=src ~/.venvs/apctt-train/bin/python tools/get_models.py \
  ms-marco-minilm-l6-v2 --store ~/models/store)

# --base defaults to that store copy (or $APCTT_MODEL_STORE/...); pass it only to override
~/.venvs/apctt-train/bin/python train/build_criteria_data.py        # about 2 min
~/.venvs/apctt-train/bin/python train/train_ranker.py --eval-only   # untrained baseline
~/.venvs/apctt-train/bin/python train/train_ranker.py               # about 1 min on the GPU
~/.venvs/apctt-train/bin/python train/train_ranker.py \
  --base train/out/criteria-ranker --device cpu --eval-only         # CPU speed
```

Result on 2026-11-04, after the evidence-filter fix (SDG 14 held out, 25 reports): pair AUC
0.59 untrained and 0.91 trained (mostly relearning the rules). The rationale page was in the
top 3 for 0.48 of the reports with the model, against 0.44 with the rules (12 vs 11
reports, so within noise). A 2026-11-03 run before the fix gave 0.40. CPU speed is
79 pairs/s, so only rule candidates can be reranked within 180 s, not every sentence.

Test one or more cases, ranker against the shipped module, per criterion:

```bash
~/.venvs/apctt-train/bin/python train/ranker_check.py --case USGS-SDG-076 --device cpu
~/.venvs/apctt-train/bin/python train/ranker_check.py \
  --case USGS-SDG-012,USGS-SDG-040,USGS-SDG-061          # the 3 diagnostic cases
```

On the 25 held-out SDG 14 cases (2026-11-04), the ranker put the rationale page first at
the label's criterion for 7/25 cases, against 1/25 for the module. The module's filled
ladder cites one page for C01-C06. Reranking the rule candidates took 4-6 s per case on CPU.
Output: `train/out/ranker-check-{sdg14,smoke}.txt`.

## Safe envelope on this laptop (6 GB VRAM, 16 GB RAM)

- Keep every training example at **<= 900 tokens**. Longer sequences crash WSL with
  `CUDA driver error: device not ready` (memory pressure, not a clean OOM).
- `--max-len 896`, `--r 8`, `--batch 1`, `--grad-accum 4`, `adamw_torch` (the shipped
  run). `--r 16 --grad-accum 8` crashed at the first optimizer step; see `HANDOFF.md` §5.
- `build_data.py --max-chars 2400 --max-pages 4` produces ~780-token examples with
  no truncation on all 74 training records.
- Verified: 10-sample run -> loss 2.735; inference on USGS-SDG-076 -> TRL match,
  33.4 s for 160 new tokens, 2.06 GiB peak.

## Files

- `setup_wsl.sh` — one-time WSL/venv/torch setup
- `build_data.py` — xlsx + PDFs -> `data/{train,val}.jsonl`
- `pipeline.py` — recursive PDFs -> JSONL -> QLoRA, with a dry-run precheck
- `train_qlora.py` — QLoRA; `--dry-run` checks lengths, `--optim` selects the optimizer
- `evaluate.py` — exact / off-by-one / invalid JSON over a split
- `infer_check.py` — single-case generation with timing and peak VRAM
- `module_eval.py` — the shipped module over the public cases (`--llm` adds the adapter)
- `pack_adapter.py` — deterministic adapter-only handoff zip
- `build_criteria_data.py` — silver (criterion, sentence) pairs for parts B/C
- `train_ranker.py` — cross-encoder fine-tune and B/C evaluation; `--self-test` checks the metrics
- `ranker_check.py` — one-case ranker vs module comparison, per criterion; its
  `load_ranker` / `rank_criteria` are also used by the local UI (`ui/server.py`)
- `requirements-train.txt` — training-only dependencies (never shipped)
- `USING_THE_ADAPTER.md` — how to run the adapter standalone and hand it to others
- `TEST_PROMPTS.md` / `test_prompts.jsonl` — six labelled test prompts and how to run them
