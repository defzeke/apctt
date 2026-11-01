# Training workspace (never submitted)

Fine-tunes `smollm3-3b-nf4` (catalogue id, Apache-2.0) with QLoRA and produces a LoRA
adapter for `my_submission/assets/`. Only the adapter and a manifest are shipped; the
base weights stay in the organizer's model store.

## 0. Where it hooks into the competition

- Base model: `smollm3-3b-nf4` (3.08B, nf4, 3.40 GiB estimated inference peak).
- Team weights: `assets/smollm3-trl-lora/` declared with `"team_trained": true` and
  `"base_model": "smollm3-3b-nf4"` in `submission.json`.
- The adapter is trained for one narrow job: emit `{"trl": 1-9, "rationale": "..."}`
  from retrieved report pages. Criterion statuses and page evidence should stay a
  retrieval/rules problem, because the sheet has no per-criterion labels.

## 1. Environment (inside WSL2, not native Windows)

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Github\ Projects/apctt
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install 3.12
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-train/bin/python -r train/requirements-train.txt
.venv-train/bin/python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Expect `NVIDIA GeForce RTX 3050 True`.

## 2. Data

```bash
# PDFs (about 1.1 GB), from the kit directory
cd cuda
python tools/get_public_cases.py --all --download --dir ../train/data/pdfs
cd ..

# JSONL records; SDG 14 is held out for validation
.venv-train/bin/python train/build_data.py --holdout-sdg 14
wc -l train/data/train.jsonl train/data/val.jsonl
```

## 3. Train

```bash
.venv-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora
```

Default: `max_len=1536`, `r=16`, 3 epochs, batch 1, grad-accum 8, 4-bit base,
`paged_adamw_8bit`. OOM ladder: `--max-len 1024`, then `--r 8`, then `--max-len 768`.

Always train through WSL2: the Windows desktop reserves roughly 1.5 GB of the 6 GiB.

## 4. Smoke test

```bash
APCTT_MODEL_STORE="/mnt/c/Github Projects/apctt/models" \
.venv-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora --index 0
```

Check `parsed_ok`, `trl_match`, `seconds` and `peak_vram_gib` before packaging. On this
card expect roughly 3.4-4.0 GiB and a few seconds per short generation.

## 5. Package

Copy `cuda/licenses/Apache-2.0.txt` to `assets/smollm3-trl-lora/LICENSE`, add a `NOTICE`
crediting Hugging Face SmolLM3, then declare the adapter in `submission.json`:

```json
{"name": "smollm3-trl-lora", "path": "assets/smollm3-trl-lora",
 "team_trained": true, "base_model": "smollm3-3b-nf4",
 "license": "MIT", "description": "QLoRA r=16 adapter for JSON TRL synthesis"}
```

and the base model itself:

```json
{"catalog_id": "smollm3-3b-nf4"}
```

Then `python cuda/tools/make_submission.py my_submission`.

## Files

- `build_data.py` — xlsx + PDFs -> `data/{train,val}.jsonl`
- `train_qlora.py` — QLoRA, writes adapter + `training_manifest.json`
- `infer_check.py` — offline load + generate, prints TRL match and VRAM
- `requirements-train.txt` — training-only dependencies
