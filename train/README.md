# Training workspace (never submitted)

QLoRA fine-tune of `smollm3-3b-nf4` for one narrow job: turn retrieved report
excerpts into `{"trl": 1-9, "rationale": "..."}`. Only the LoRA adapter is shipped;
the base weights stay in the organizer's model store.

Status on 2026-11-01: adapter trained (3 epochs, 18 min) into
`my_submission/assets/smollm3-trl-lora`, scored **0.72 mean_A** on the 25-case SDG-14
holdout (14 exact, 8 off-by-one, 3 off-by-two, 0 invalid). See `HANDOFF.md` for the
partner guide and the TRL-6 collapse limitation.

## Environment (WSL2, already built)

```bash
wsl -d Ubuntu-22.04
~/.venvs/apctt-train/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# 2.11.0+cu128 NVIDIA GeForce RTX 3050 True
```

Rebuild from scratch if needed:

```bash
wsl -d Ubuntu-22.04 -- bash "/mnt/c/Github Projects/apctt/train/setup_wsl.sh"
```

Training runs inside WSL2 because the Windows desktop reserves VRAM. The base model
is cached at `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/`.

## Commands

```bash
cd "/mnt/c/Github Projects/apctt"

# data (already done)
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

## Safe envelope on this laptop (6 GB VRAM, 16 GB RAM)

- Keep every training example at **<= 900 tokens**. Longer sequences crash WSL with
  `CUDA driver error: device not ready` (memory pressure, not a clean OOM).
- `--max-len 896`, `--r 8..16`, `--batch 1`, `--grad-accum 8`, `adamw_torch`.
- `build_data.py --max-chars 2400 --max-pages 4` produces ~780-token examples with
  no truncation on all 74 training records.
- Verified: 10-sample run -> loss 2.735; inference on USGS-SDG-076 -> TRL match,
  33.4 s for 160 new tokens, 2.06 GiB peak.

## Files

- `setup_wsl.sh` — one-time WSL/venv/torch setup
- `build_data.py` — xlsx + PDFs -> `data/{train,val}.jsonl`
- `train_qlora.py` — QLoRA; `--dry-run` checks lengths, `--optim` selects the optimizer
- `evaluate.py` — exact / off-by-one / invalid JSON over a split
- `infer_check.py` — single-case generation with timing and peak VRAM
- `requirements-train.txt` — training-only dependencies (never shipped)
- `USING_THE_ADAPTER.md` — how to run the adapter standalone and hand it to others
