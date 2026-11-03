# 1. verify and unpack
sha256sum CICA_smollm3-trl-lora.zip
unzip CICA_smollm3-trl-lora.zip -d ~/apctt-model
cd ~/apctt-model

# 2. environment
sudo apt update && sudo apt install -y python3.12-venv
python3.12 -m venv ~/.venvs/apctt-train
~/.venvs/apctt-train/bin/pip install --upgrade pip
~/.venvs/apctt-train/bin/pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
~/.venvs/apctt-train/bin/pip install "transformers>=5.19" "peft>=0.21" \
  "accelerate>=1.15" "bitsandbytes==0.50.2" pypdfium2 huggingface_hub

# 3. download the base model once (needs network; ~6 GB)
~/.venvs/apctt-train/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('HuggingFaceTB/SmolLM3-3B', \
revision='a07cc9a04f16550a088caea529712d1d335b0ac1', \
local_dir='$HOME/models/SmolLM3-3B')"

# 4. smoke test
~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --base "$HOME/models/SmolLM3-3B" \
  --data train/data/val.jsonl --index 0