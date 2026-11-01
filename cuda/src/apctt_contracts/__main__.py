"""Regenerate frozen machine-readable contracts: python -m apctt_contracts export <dir>."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .ipc import worker_protocol_schema


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "export":
        print("usage: python -m apctt_contracts export <contracts-dir>", file=sys.stderr)
        return 2
    target = Path(argv[1]) / "worker-protocol.json"
    text = json.dumps(worker_protocol_schema(), ensure_ascii=False, indent=2, sort_keys=True)
    target.write_text(text + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
