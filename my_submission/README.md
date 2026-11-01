# CICA USGS TRL assessor

Stage 1 APCTT submission. The module reads a USGS report (PDF) and produces a TRL estimate,
a status for each of the 12 official checklist criteria, and page evidence with verbatim spans.

## What it does

`participant_module.adapter:AssessmentModule` implements the worker protocol (`load`, `assess`,
`reset_case`, `close`). For one case it:

1. extracts page text with pypdfium2 and drops running headers, page numbers, figure captions,
   reference lists, disclaimers and other boilerplate;
2. scores each sentence against the checklist criteria and keeps the best page evidence;
3. maps rules to criteria by their official description, so a changed criterion ID does not
   move a rule to another criterion;
4. fills the TRL ladder up to the predicted level, cites the closest evidence for each filled
   criterion, and abstains only when no readable text exists;
5. optionally refines the TRL with a bundled SmolLM3-3B QLoRA adapter (CUDA, off by default).

The default configuration is CPU-only and uses no model weights.

## Results on the 99 public cases

Rules-only, `train/module_eval.py`:

| metric | value |
|---|---|
| mean part-A score | 0.7525 |
| exact TRL | 54 / 99 |
| off by one | 41 / 99 |
| off by two or more | 4 / 99 |
| invalid results | 0 |
| mean wall time | 1.9 s per case |
| peak RSS | 0.26 GiB |

The public labels are overwhelmingly 6 (54 / 99), so the module predicts 6 unless the checklist
evidence is strong. On the held-out SDG 14 split the rules-only score is 0.70 and the adapter
scores 0.68, which is why the adapter is disabled by default. Both are close to the
always-predict-6 baseline (0.70 on that split); the organizer's public TRLs are estimates, not
expert labels.

The adapter was trained separately (see `declaration.json`): 74 reports, 3 epochs, r=8,
max length 896, seed 42, on an RTX 3050 6 GB. It is bundled under `assets/` and declared as
team-trained so the CUDA path stays reproducible.

## Configuration

`submission.json` passes:

- `settings.use_llm` (default `false`): load the catalogue model and adapter on CUDA and use the
  adapter TRL when it differs from the rules prior;
- `settings.max_pages` (default `300`).

The 180 s per-case budget is not a constraint for the rules path; a case takes about 2 s,
including PDF parsing.

## Licence

The module, rules and adapter are MIT (`participant_module/LICENSE`). The adapter is derived from
SmolLM3-3B (Apache-2.0); the base model's `LICENSE` and `NOTICE` are kept with the weights.
Third-party components are listed in `participant_module/THIRD_PARTY_NOTICES.md`.
