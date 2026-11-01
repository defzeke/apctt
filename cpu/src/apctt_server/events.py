"""SPEC-13 operational event log: JSON lines with IDs, phases, codes and timings; rotated."""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any

from apctt_diagnostics.responses import json_safe


def utc_now() -> str:
    now = dt.datetime.now(dt.UTC)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventLog:
    """Only fixed event names, IDs, codes and numbers. No request text or module output."""

    def __init__(self, path: Path, *, max_bytes: int = 10 * 1024 * 1024, backups: int = 3) -> None:
        self._logger = logging.getLogger(f"apctt.events.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._handler = handler
        self._logger.addHandler(handler)
        path.chmod(0o600)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": utc_now(), "event": event, **json_safe(fields)}
        self._logger.info(json.dumps(record, ensure_ascii=False, allow_nan=False))

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()
