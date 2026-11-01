# CUDA Quickstart (CUDA dummy example)

kit 0.5.0 | status: GPU/CUDA tests run with kit 0.3.0 on an RTX 3070 Laptop GPU (2026-09-30, PDF input); not re-run for 0.5.0; NOT RUN on the final event machines

The CUDA dummy shows the full GPU path with a tiny synthetic model (16 → 32 → 9 MLP). It is a
format example (`example_only=true`; it runs with the official checklist of
`config/cuda-example.json`, decision D74), not a TRL model. Details:
[participant_example/cuda_dummy/README.md](../../participant_example/cuda_dummy/README.md).

## A GPU is optional; your module uses the common base

A GPU is optional: a CPU-only module (`compute.device` `cpu`) is fully allowed and scored by the
same rules and the same 3-minute limit ([PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §7.2).
The GPU of the event machine, and the reference for every VRAM figure of the model catalogue, is an
NVIDIA RTX 3070 Laptop GPU with 8 GB VRAM (8 GiB; decisions D40 and D67). The event machines are 5
Asus TUF Dash F15 FX516PR-HN002 laptops (decision D76): Intel Core i7-11370H (4 cores, 8 threads),
16 GB of RAM, a 512 GB NVMe SSD and that GPU, on Ubuntu 24.04 (decision D80); at least 12 GB of
RAM is available to your worker ([PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §7.2).

This page is about the kit's CUDA dummy example only, which has its own small worker environment
`.venv-cuda` made from `requirements-cuda-example.lock` (torch 2.11.0+cu128, the same torch build
as the common base environment). Your own module does not use `.venv-cuda`: it runs in the
**common base environment**, the same for every team, which `tools/make_env.py` builds (README step
5). On Linux x86_64 with an NVIDIA GPU it builds the CUDA flavour of the base (`--flavor cuda`),
exactly the build of the event machines. To use the GPU with your module:

1. Probe the base's Python as in step 2 below:
   `.venv/bin/python scripts/check_cuda.py --worker-python "$PWD/../apctt-env/base/bin/python" --device cuda:0 --output /tmp/cuda-check.json`.
2. Set `compute` to `"device": "cuda:0", "require_cuda": true, "allow_cpu_fallback": false` in
   your local configuration, and `"compute": {"device": "cuda:0"}` in `submission.json`.
3. Stay within the 8 GB: one model of the catalogue at its documented context or batch is
   estimated to fit the budget of at most 7.0 GiB peak (estimates until the organizer measures
   them with `tools/hq/check_models_gpu.py`; [MODEL_CATALOG.md](MODEL_CATALOG.md)); several models
   at once, longer inputs or larger batches are your responsibility (`OUT_OF_MEMORY` fails the
   case).

## Order of steps

1. **Worker environment.** Create a separate environment with a CUDA build of PyTorch that matches
   your driver (Linux or WSL2). The API server environment stays free of torch.
   `config/cuda-example.json` expects `.venv-cuda/bin/python`; use a local copy
   (`config/cuda-example.local.json`) if your path differs. On Linux x86_64 the verified set is:
   ```bash
   uv venv --python 3.12 .venv-cuda
   uv pip install --python .venv-cuda/bin/python -r requirements-cuda-example.lock
   ```
   It pins torch 2.11.0+cu128 from the PyTorch cu128 index. A different driver or GPU may need a
   different CUDA build; the probe below tells you whether the one you installed works.
2. **Probe** with the exact interpreter the worker will use:
   ```bash
   .venv/bin/python scripts/check_cuda.py --worker-python "$PWD/.venv-cuda/bin/python" \
       --device cuda:0 --output /tmp/cuda-check.json
   ```
   Exit 0 means a 64×64 matmul ran and synchronized on the device. Exit 3 means CUDA is not usable
   (no torch, CPU-only build, no device, kernel failure). There is no CPU fallback.
3. **Train** (outside the server; at most 128 optimizer steps on synthetic data):
   ```bash
   .venv-cuda/bin/python -m participant_example.cuda_dummy.train \
       --device cuda:0 --output-dir artifacts/cuda_dummy
   ```
   This writes `model.pt` (CPU state_dict), `model_config.json` and `training_manifest.json`
   (seed, losses, device, torch/CUDA build, `checkpoint_sha256`). `--device cpu` is refused.
4. **Serve**: `.venv/bin/python -m apctt_server --profile dev --config config/cuda-example.json`.
   `/health/ready` is `503 CUDA_UNAVAILABLE` until the probe passes; the server never accepts a
   CUDA-configured job on CPU.
5. **POST/GET** `docs/EXAMPLE_REQUEST.json` as in [QUICKSTART.md](QUICKSTART.md). A fresh worker
   loads the checkpoint with `weights_only=True`, moves it to `cuda:0` and returns
   `predicted_trl = argmax + 1` with `confidence`. The job diagnostics show the worker-observed
   torch version and peak allocator memory.
6. **Automated GPU tests** (AC-29..AC-34) are run by the organizer.
   They were run with kit 0.3.0 on an NVIDIA RTX 3070 Laptop GPU (8 GiB; Ubuntu 24.04.5, driver
   595.91.07, torch 2.11.0+cu128) on 2026-09-30: CUDA training, checkpoint, CUDA inference through the
   server, timeout and crash recovery (AC-29..AC-34) with PDF input. Since 0.3.0 only the dummy's
   checklist check and `config/cuda-example.json` (official checklist, input limits) have changed; the
   runtime, `scripts/check_cuda.py`, the dummy's model, training and adapter code and the worker
   packages are unchanged. The GPU tests were not re-run for 0.5.0 and have not been run on the final
   event machines (Asus TUF Dash F15 FX516PR-HN002, decision D76): NOT RUN.
   The worker packages are in `requirements-cuda-example.lock`.

## Public models and the reference GPU

The organizer's public model catalogue is chosen for the reference GPU of the event machine, an
NVIDIA RTX 3070 Laptop GPU with 8 GB VRAM (decisions D40, D64 and D67): the list, sizes, VRAM
estimates and CPU options are in [MODEL_CATALOG.md](MODEL_CATALOG.md), and
`python tools/get_models.py ID --store DIR` downloads a model for development. The VRAM figures
are estimates until the organizer runs `tools/hq/check_models_gpu.py` on that GPU. The 4-bit
(bitsandbytes nf4) LLMs of the catalogue are marked `measured: false`: before the kit is published
the organizer measures them on the RTX 3070 Laptop GPU and removes any entry that does not load
within the budget (peak at most 7.0 GiB, load well inside the 3-minute limit of a case). They need
a CUDA GPU, and the 7-8B ones read 14.50-16.35 GB of weights from disk on every case (the event
machine's 16 GB of RAM cannot keep them in the file cache): time the load first. Load them with the code `python tools/get_models.py --snippet ID` prints.

## Versions and provenance

Do not guess versions. After a real GPU run, freeze the worker environment into
`requirements-cuda-example.lock` together with the driver, GPU and torch/CUDA build that were
observed. The checkpoint is identified by `checkpoint_sha256` in `training_manifest.json`; weights
are not committed to Git and are released only as a separately hash-verified asset.

## Licences

Installing the lock above downloads PyTorch, triton and NVIDIA's CUDA libraries from the official
PyTorch index. By installing them you accept their licences yourself, including NVIDIA's terms (the
CUDA Toolkit End User License Agreement and the other NVIDIA licence agreements shipped in the
NVIDIA CUDA library wheels and in `cuda-bindings`); `tools/make_env.py` installs the same packages
for the CUDA base, from the same index. Do not ship PyTorch, triton, `cuda-*` or the NVIDIA CUDA
libraries in your submission or in a web system; your submission contains no packages at all. Leave torch's other dependencies of
the common base (filelock, fsspec, jinja2, markupsafe, mpmath, networkx, setuptools, sympy,
typing-extensions) out of your submission bundle too; your web system, which runs in the common
base, may use these base packages. Details: [LICENSING.md](LICENSING.md) §6.

## What a pass means

A passing probe and dummy run show that the CUDA path works on that host. They do not show that
your real model fits in memory, runs within the event time limit, or that the evaluation host is
isolated. `empty_cache()` is housekeeping, not proof that memory was released.
