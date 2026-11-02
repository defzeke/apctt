# Using and handing over the fine-tuned model

Current state, 2026-11-03. This guide covers the QLoRA adapter in
`my_submission/assets/smollm3-trl-lora` and how to pass it to another person or team.
All commands run from the repository root.

## 1. What the artifact is

The team trained a **LoRA adapter**, not a full model. It contains about 58 MB of adapter
weights and must be loaded on top of the catalogue base model. The base weights (about
6.2 GB in bf16) are never part of the adapter and are not shipped in the submission.

| item | value |
|---|---|
| adapter folder | `my_submission/assets/smollm3-trl-lora/` |
| base model | `smollm3-3b-nf4` (`HuggingFaceTB/SmolLM3-3B`) |
| base revision | `a07cc9a04f16550a088caea529712d1d335b0ac1` |
| adapter licence | MIT (text: `participant_module/LICENSE`; `LICENSE-MIT` in the handoff bundle) |
| base licence | Apache-2.0 (`LICENSE` and `NOTICE` in the adapter folder are the base's; keep them) |
| training | QLoRA r=8, 3 epochs, 74 USGS reports, max length 896, seed 42 |
| output | one JSON object: `{"trl": 1-9, "rationale": "..."}` |

Hashes (LF files, as committed; a CRLF checkout changes the two JSON hashes):

```
821a287c639f13d075c70724d0bda33d29cb5e5bba2b0e936016bfd83b8feefd  adapter_config.json
17b781a9ae7c4c8f05f6472399d9b4aca280e61ec62a6335f6bedcbd011cdee2  adapter_model.safetensors
e110ad63e251400e2d913eb9e758fc4652f4af4f41ebe9d6b4d1332bcdc71425  training_manifest.json
```

What it does well: broad TRL sanity checks on USGS-style reports, with the strongest
signal at TRL 6. What it does not do: per-criterion status or evidence pages. The shipped
assessment module uses deterministic rules for that and treats the adapter as optional.

## 2. Current measured state

| test | result |
|---|---|
| held-out SDG 14 split (25 cases, rules only) | mean part-A 0.70 |
| held-out SDG 14 split (25 cases, adapter) | mean part-A 0.68 |
| 99 public cases (rules only) | mean part-A 0.7525, 0 invalid |
| always predict 6 | 0.753 on 99 public, 0.70 on SDG 14 |
| adapter cold load + 160 tokens (RTX 3050 6 GB) | about 33 s, 2.06 GiB peak VRAM |
| module with adapter, integrated | about 14 s per case after load, 6.75 GiB RSS |

The adapter does not beat "always predict 6" on the held-out public split. That is why the
submission ships with `settings.use_llm = false`. The adapter is still bundled so the CUDA
path can be reproduced and improved later.

## 3. Requirements

- Linux (WSL2 is fine) with Python 3.12 and the kit's common base environment.
- CUDA GPU with at least 6 GB VRAM for the 4-bit loader. The 4-bit config does not run on
  CPU.
- The base model available locally or in `APCTT_MODEL_STORE`:
  `$APCTT_MODEL_STORE/HuggingFaceTB/SmolLM3-3B/a07cc9a04f16550a088caea529712d1d335b0ac1`.
- `torch==2.11.0+cu128`, `transformers==5.19.0`, `peft==0.21.2`, `bitsandbytes==0.50.2`
  from the base environment. The training venv resolved transformers 5.22.1; the event runs
  5.19.0, so test there before enabling the adapter.

On this laptop the working venv is `~/.venvs/apctt-train` in WSL2 Ubuntu 22.04. Run GPU
work inside WSL: the Windows desktop reserves VRAM.

## 4. Use it inside the submission module

Copy the adapter folder into the module's `asset_root` (the submission already has it) and
pass:

```json
{
  "compute": {"device": "cuda:0"},
  "settings": {"use_llm": true, "max_pages": 300}
}
```

`participant_module/llm.py` then loads the base from `APCTT_MODEL_STORE` (or the Hugging
Face cache), applies the adapter, and uses the adapter TRL. Any load or parse failure is
ignored and the rules result stands, so this path degrades safely.

To try it locally:

```bash
~/.venvs/apctt-train/bin/python train/module_eval.py --llm \
  --only USGS-SDG-061,USGS-SDG-096
```

## 5. Use it standalone

```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Drop revision= below when BASE is a local folder or a model-store path.
BASE = "HuggingFaceTB/SmolLM3-3B"
ADAPTER = "my_submission/assets/smollm3-trl-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE, revision="a07cc9a04f16550a088caea529712d1d335b0ac1")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a technology-readiness assessor. Read the report excerpts and reply "
            "with one JSON object only: {\"trl\": <integer 1-9>, \"rationale\": \"<one sentence>\"}."
        ),
    },
    {
        "role": "user",
        "content": (
            "Assessment target: coupled hydrologic model (type: technology)\n"
            "Intended application: flood forecasting for water-resource management\n"
            "Report: example title (2019)\n\n"
            "Report excerpts:\n<2,000 to 2,400 characters of report text>\n\n"
            "Which TRL does the evidence support?"
        ),
    },
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=96, do_sample=False, temperature=None, top_p=None
    )
text = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()
print(text)
```

Rules that matter:

- `enable_thinking=False` is required by SmolLM3's chat template.
- Use `dtype=`, not `torch_dtype=` (transformers 5.x).
- Keep the prompt under 896 tokens; the adapter was trained on about 780-token prompts
  built from 4 pages at 2,400 characters per page.
- Decode with `do_sample=False` so results are repeatable.
- The reply is free text; extract the JSON object with a regex before parsing.

The prompt builder used in training is `train/build_data.py:build_user_text` (with
`pick_pages` for the excerpts); the training data format is one record per case in
`train/data/{train,val}.jsonl` with `messages` (system, user, assistant).

The module's builder, `participant_module/llm.py:build_prompt`, is **not** identical: it
omits the `(<year>)` after the report title and feeds rule-selected excerpts. Expect the
module path to score lower than `evaluate.py` (0.68 vs 0.72 on SDG 14) until the two are
aligned.

## 6. Smoke test after any transfer

```bash
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --base "$APCTT_MODEL_STORE/HuggingFaceTB/SmolLM3-3B/a07cc9a04f16550a088caea529712d1d335b0ac1" \
  --data train/data/val.jsonl --index 0
```

It prints the generated text, parsed JSON, whether the TRL matches, seconds and peak VRAM.
A healthy run parses JSON. On the RTX 3050 a single cold run takes about 20 to 35 seconds;
after load, `evaluate.py` averages about 14 seconds per case.

Ready-made prompts with known labels and recorded outputs: `train/TEST_PROMPTS.md` and
`train/test_prompts.jsonl` (six cases; `infer_check.py --data train/test_prompts.jsonl
--index N`, or score all six with `evaluate.py`).

Score a whole split:

```bash
~/.venvs/apctt-train/bin/python train/evaluate.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/data/val.jsonl --out train/out/eval.json
```

## 7. Handing it to someone else

Two different handovers:

- **The model alone** (for reuse, further training or review): the bundle below.
- **The runnable submission** (what the organizer runs): `train/out/CICA_submission.zip`,
  60.5 MB, zip SHA-256 `6d1e8831d33ba2c324137724412af504a8fbae4e99c624403b061e4a45d35cb4`,
  package SHA-256 `fd4bd882315c4b3647d150f80f8f98be8eb3fd054c723445bd8d42bd0f63fe9c`.
  It contains `participant_module/`, `assets/`, `submission.json`, `declaration.json` and
  `README.md`; run `cuda/tools/make_submission.py --check` on it after any change.

Send these files, and nothing else from the training workspace:

```
smollm3-trl-lora/
  adapter_config.json
  adapter_model.safetensors
  LICENSE            # Apache-2.0, the SmolLM3-3B base licence
  LICENSE-MIT        # MIT, the adapter's licence (copy of participant_module/LICENSE)
  NOTICE             # SmolLM3-3B attribution
  README.md          # model card
  training_manifest.json
train/infer_check.py
train/evaluate.py
train/data/val.jsonl  # optional: a small labelled sample so the smoke test runs
```

Do not send the base weights, the Hugging Face cache, the WSL venv, or the PDF corpus.

Build it with the deterministic packer (any Python 3, standard library only). It uses fixed
timestamps, so the same files give the same hash on any machine:

```bash
python train/pack_adapter.py
```

Current bundle:

```
train/out/CICA_smollm3-trl-lora.zip   55,940,109 bytes
sha256 e3587449e80dfdbed36728fd43190583d506861bb17dcb854d1c8ef2d2f0e612
```

The packer also prints every file hash. Give the receiver the zip hash and the three adapter
file hashes from §1 so they can verify the transfer.

The receiver needs the base model at the pinned revision. Either let `from_pretrained`
download it once with network access, or place it in `APCTT_MODEL_STORE`; then run the
smoke test in §6. Without a CUDA GPU the 4-bit loader is unsupported; they can merge the
adapter into a bf16 model for CPU inference, but expect minutes per answer, and the
rules-only module is a better CPU choice.

Licence obligations when passing it on:

- keep the adapter under MIT (`LICENSE-MIT`), and keep the base's `LICENSE` and `NOTICE`
  next to the weights;
- keep the SmolLM3-3B attribution and note that the base is Apache-2.0;
- do not relicense the base or claim it as team work.

If you publish it, use any channel that preserves the folder layout and publishes the
hashes. A private archive plus a checksum is enough; a model-hub repository works too, as
long as the base model is declared as the adapter's base, not uploaded.

## 8. Rebuilding the adapter from scratch

```bash
~/.venvs/apctt-train/bin/python train/build_data.py --holdout-sdg 14
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl --max-len 896 --dry-run
~/.venvs/apctt-train/bin/python train/train_qlora.py \
  --data train/data/train.jsonl \
  --out my_submission/assets/smollm3-trl-lora
```

`training_manifest.json` records the dataset hash, configuration, hardware and runtime.
The 74-record run took 1150.6 s (about 19 minutes) on the RTX 3050.

## 9. Known limits

- **TRL 6 collapse.** 54 of 99 public labels are 6 and the adapter learned that prior. It
  rarely says 5 or 7 without strong wording.
- **Weak labels.** Targets are the organizer's estimated TRL, not expert gold, and the
  public labels are not the hidden-case labels.
- **No per-criterion output.** The adapter predicts TRL and one rationale only.
- **Short context.** Prompts were built from 4 pages at 2,400 characters; longer inputs are
  truncated by the module before generation.
- **Prompt skew in the module.** `llm.py:build_prompt` differs from the training prompt
  (no report year, different excerpt selection).
- **CUDA only.** The 4-bit loader needs a GPU; CPU inference requires a merged bf16 model.
- **Not enabled in the shipped submission.** `settings.use_llm` is `false`; the measured
  adapter did not beat the rules-only prior on the held-out split.
