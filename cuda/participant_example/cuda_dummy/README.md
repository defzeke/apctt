# CUDA dummy example (DEMO only)

kit 0.2.1

This example proves the CUDA path end to end: synthetic training on a GPU → checkpoint →
a fresh worker loads it on CUDA → the API returns a result that depends on the model output.
It is **not** a TRL model, a baseline or an expert judgment. `predicted_trl` is the toy class
argmax + 1; `example_only=true` marks it as a format example. It runs with the official checklist
of `config/cuda-example.json` (decision D74), whose criteria it reports with the DEMO phrase rules.

## What runs where

| Step | Process | Command |
|---|---|---|
| CUDA probe | worker interpreter | `python scripts/check_cuda.py --worker-python <worker-python> --device cuda:0 --output <run>/cuda.json` |
| Training | worker interpreter, outside the API | `<worker-python> -m participant_example.cuda_dummy.train --device cuda:0 --output-dir artifacts/cuda_dummy` |
| Inference | fresh worker per case | `python -m apctt_server --profile dev --config config/cuda-example.json` |

The API process never imports torch and has no training endpoint.

## Rules the code enforces

- Only `cuda:N` devices. `--device cpu` or `auto` exits 2; missing CUDA exits 3. There is no CPU
  fallback, and no files are written unless training succeeded on CUDA.
- Training performs real forward/loss/backward/optimizer steps (at most 128), checks finite loss
  and gradients and a parameter change, and records seed, losses, device, torch/CUDA build and the
  checkpoint SHA-256 in `training_manifest.json`.
- The checkpoint is a CPU `state_dict` (`model.pt`) plus `model_config.json`. The adapter loads it
  with `map_location="cpu", weights_only=True`, checks the manifest hash and tensor shapes, then
  moves the model to the configured CUDA device and runs `eval()` + `inference_mode()`. It checks
  that model, input and output tensors are on CUDA.
- `weights_only=True` is not a full sandbox for untrusted weights; the worker process boundary is.
- The dummy refuses a non-example case schema. Until kit 0.5.0 it also refused every checklist but
  the DEMO one. Its GPU path was run with kit 0.3.0 (DEMO checklist) and has not been run with the
  official checklist (NOT RUN: no GPU host was used for kit 0.5.0; the CPU structure tests in
  tests/cuda_unit cover the change).

## Worker environment

Create a separate worker environment with a CUDA build of PyTorch that matches the host driver.
`config/cuda-example.json` expects it at `.venv-cuda/bin/python`; change the path in a local copy
(`config/*.local.json` is ignored by Git). Record the exact versions after the real GPU run
(`requirements-cuda-example.lock`, CUDA-02). Do not guess version numbers.

## Status

CPU structure tests (tests/cuda_unit) pass on a host without a GPU; a CPU run of the training code
is only a code smoke test. Real CUDA training, checkpoint, API round trip and recovery tests
(tests/cuda, CUDA-02) were run with kit 0.3.0 on an RTX 3070 Laptop GPU (driver 595.91.07,
torch 2.11.0+cu128, 2026-09-30, PDF input; the worker packages are in
`requirements-cuda-example.lock`). They were not re-run for 0.5.0 and are **NOT RUN** on the final
event machines or with real participant models. No trained weights are
shipped in this repository; train your own with the command above.
