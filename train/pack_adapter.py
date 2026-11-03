"""Build the adapter-only handoff bundle, train/out/CICA_smollm3-trl-lora.zip.

Deterministic: fixed timestamps, sorted entries and fixed permissions, so the same files
give the same zip hash on any machine. Run from the repository root with any Python 3.

The adapter folder's LICENSE is the Apache-2.0 text of the SmolLM3-3B base; the adapter's
own MIT text is added as smollm3-trl-lora/LICENSE-MIT.
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

STAMP = (1980, 1, 1, 0, 0, 0)
ADAPTER = Path("my_submission/assets/smollm3-trl-lora")
EXTRA = {
    "smollm3-trl-lora/LICENSE-MIT": Path("my_submission/participant_module/LICENSE"),
    "train/infer_check.py": Path("train/infer_check.py"),
    "train/evaluate.py": Path("train/evaluate.py"),
    # Validation in the prompt format the shipped adapter was trained on (train/pipeline.py),
    # kept at the old in-zip path so the handoff commands do not change.
    "train/data/val.jsonl": Path("train/out/pipeline-data/val.jsonl"),
}


def entries() -> dict[str, Path]:
    found = {f"smollm3-trl-lora/{p.relative_to(ADAPTER).as_posix()}": p
             for p in ADAPTER.rglob("*") if p.is_file()}
    found.update(EXTRA)
    missing = [str(p) for p in found.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("train/out/CICA_smollm3-trl-lora.zip"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w") as bundle:
        for name, path in entries().items():
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    data = args.out.read_bytes()
    print(f"{args.out}  {len(data):,} bytes")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    for name, path in entries().items():
        print(f"  {hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
