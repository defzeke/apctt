#!/usr/bin/env bash
# One-time training environment setup, run inside WSL2 Ubuntu.
#   from Git Bash in the repository root:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "$(cygpath -w "$PWD")" -- bash train/setup_wsl.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/apctt-train"
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
mkdir -p "$HOME/.venvs"
uv venv --python 3.12 "$VENV"

uv pip install --python "$VENV/bin/python" torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" -r "$REPO/train/requirements-train.txt"

"$VENV/bin/python" - <<'PY'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__, "peft", peft.__version__,
      "bitsandbytes", bitsandbytes.__version__)
PY
