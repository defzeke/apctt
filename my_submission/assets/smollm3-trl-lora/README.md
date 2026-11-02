---
base_model: HuggingFaceTB/SmolLM3-3B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM3-3B
- lora
- qlora
license: mit
---

# smollm3-trl-lora

QLoRA adapter for `smollm3-3b-nf4` (HuggingFaceTB/SmolLM3-3B, revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`, Apache-2.0). The adapter weights are MIT.

## What it does

Given a USGS-style report excerpt and an assessment target, it returns one JSON object:
`{"trl": <integer 1-9>, "rationale": "<one sentence>"}`.

It predicts the technology readiness level only. It does not produce per-criterion status
or page evidence; use `participant_module` for that.

## Training

74 public APCTT cases, 3 epochs, max length 896, r=8, alpha=32, dropout 0.05, batch 1,
gradient accumulation 4, learning rate 2e-4, seed 42, bf16. Labels are the organizer's
estimated TRL from the public USGS sheet, not expert gold. Full configuration and runtime
are in `training_manifest.json`.

## Use

Load the base with bitsandbytes NF4, apply this adapter with PEFT, and use the SmolLM3
chat template with `enable_thinking=False`. See `train/USING_THE_ADAPTER.md` in the
project repository for a runnable example, the prompt format and the measured results.

## Licence files

`LICENSE` in this folder is the Apache-2.0 text of the SmolLM3-3B base and `NOTICE` carries
its attribution. The adapter weights themselves are MIT; the MIT text is
`participant_module/LICENSE` in the submission.

## Limits

The adapter learned the public label prior (most cases are TRL 6) and rarely predicts 5 or
7 without strong wording. It is disabled by default in the CICA submission because it did
not beat the rules-only prior on the held-out SDG 14 split.
