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

Version of 2026-11-04 (`train/pipeline.py`). 74 public APCTT cases (SDG 14 held out),
3 epochs, max length 896, r=8, alpha=32, dropout 0.05, batch 1, gradient accumulation 4,
learning rate 2e-4, seed 42, bf16, RTX 4060 Laptop, 231 s. The prompt is the module's own
(`participant_module/llm.py:build_prompt`). The rationale target is extractive: the report
sentence that best matches the organizer's reason, so the model quotes the excerpt instead
of paraphrasing. TRL labels are the organizer's estimates from the public USGS sheet, not
expert gold. Full configuration and runtime are in `training_manifest.json`.

## Use

Load the base with bitsandbytes NF4, apply this adapter with PEFT, and use the SmolLM3
chat template with `enable_thinking=False`. See `train/USING_THE_ADAPTER.md` in the
project repository for a runnable example, the prompt format and the measured results.

## Licence files

`LICENSE` in this folder is the Apache-2.0 text of the SmolLM3-3B base and `NOTICE` carries
its attribution. The adapter weights themselves are MIT; the MIT text is
`participant_module/LICENSE` in the submission.

## Limits

The adapter learned the public label prior (most cases are TRL 6): inside the module it
predicted 6 for all 25 held-out SDG 14 cases (mean part-A 0.70, the same as the rules; the
previous adapter scored 0.68). It is disabled by default in the CICA submission because it
does not beat the rules-only prior.
