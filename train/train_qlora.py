"""QLoRA fine-tune of a catalogue LLM (default: smollm3-3b-nf4) for JSON TRL output.

    python train_qlora.py --data data/train.jsonl --out ../my_submission/assets/smollm3-trl-lora

The base is loaded from the builder's bf16 safetensors and quantised with bitsandbytes
nf4, exactly as the event-time loader does it. Only the LoRA adapter is written; the
base weights are never copied into the submission. Run inside WSL2 to get the whole
6 GiB of the GPU (the Windows desktop reserves some).

OOM ladder: --max-len 896, then 768, then --r 8, then --grad-accum 4, then
`wsl --shutdown` on the training laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

BASE_ID = "smollm3-3b-nf4"
BASE_REPO = "HuggingFaceTB/SmolLM3-3B"
BASE_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base(repo: str, bf16: bool):
    model_kwargs = dict(
        local_files_only=Path(repo).is_dir(),
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        ),
    )
    dtype = torch.bfloat16 if bf16 else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **model_kwargs)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **model_kwargs)
    return model


def render(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:  # template without enable_thinking
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def make_encoder(tokenizer, max_len: int):
    def encode(record):
        messages = record["messages"]
        prompt_ids = tokenizer(
            render(tokenizer, messages[:-1], True), add_special_tokens=False
        )["input_ids"]
        full_ids = tokenizer(
            render(tokenizer, messages, False), add_special_tokens=False
        )["input_ids"][:max_len]
        labels = list(full_ids)
        mask = min(len(prompt_ids), len(full_ids))
        labels[:mask] = [-100] * mask
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}
    return encode


def make_collator(pad_token_id: int):
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, attention, labels = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE_REPO,
                        help="HF repo id or a local folder (e.g. the model store path)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-len", type=int, default=896)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--optim", default="adamw_torch",
                        help="paged_adamw_8bit, adamw_torch, adamw_8bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="tokenize only, print length and target-token stats, do not load the model")
    args = parser.parse_args(argv)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=Path(args.base).is_dir())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.map(make_encoder(tokenizer, args.max_len),
                          remove_columns=dataset.column_names)

    lengths = [len(item["input_ids"]) for item in dataset]
    targets = [sum(1 for token in item["labels"] if token != -100) for item in dataset]
    stats = {
        "records": len(dataset),
        "max_len": args.max_len,
        "tokens_max": max(lengths) if lengths else 0,
        "tokens_mean": round(sum(lengths) / max(1, len(lengths)), 1),
        "target_tokens_min": min(targets) if targets else 0,
        "target_tokens_mean": round(sum(targets) / max(1, len(targets)), 1),
        "records_truncated_to_no_target": sum(1 for value in targets if value == 0),
    }
    print(json.dumps(stats))
    if args.dry_run:
        return 0
    if args.out is None:
        parser.error("--out is required unless --dry-run is given")
    if stats["records_truncated_to_no_target"]:
        print(json.dumps({"status": "REFUSED", "reason": "TARGET_TRUNCATED",
                          "hint": "raise --max-len or lower build_data.py --max-chars"}))
        return 4

    # The CUDA check comes after the dry run: tokenization needs no GPU.
    if not torch.cuda.is_available():
        print(json.dumps({"status": "REFUSED", "reason": "CUDA_UNAVAILABLE"}))
        return 3
    bf16 = torch.cuda.is_bf16_supported()
    print(json.dumps({
        "status": "STARTING", "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "bf16": bf16, "base": args.base, "max_len": args.max_len,
    }))

    model = load_base(args.base, bf16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"trainable_params": trainable,
                      "trainable_pct": round(100 * trainable / 1e9, 4)}))

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model, train_dataset=dataset,
        data_collator=make_collator(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=args.optim,
            bf16=bf16,
            fp16=not bf16,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
        ),
    )
    trainer.train()
    torch.cuda.synchronize()

    model.save_pretrained(args.out)
    manifest = {
        "manifest_version": "train-manifest-v1",
        "base_model": BASE_ID,
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "dataset_file": Path(args.data).name,
        "dataset_sha256": file_sha256(Path(args.data)),
        "records": len(dataset),
        "max_len": args.max_len,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": args.r,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.05,
        "target_modules": TARGET_MODULES,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "optimizer": args.optim,
        "seed": args.seed,
        "compute_dtype": "bfloat16" if bf16 else "float16",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "seconds": round(time.time() - started, 1),
        "labels": "organizer-estimated TRL from USGS sheet; not expert gold",
    }
    (args.out / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "TRAINED", "adapter": str(args.out),
                      "seconds": manifest["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
