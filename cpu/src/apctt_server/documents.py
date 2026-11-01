"""PDF document store and bounded page counting (0.3.0).

Admission decodes each PDF (apctt_contracts.cases), then this module counts its pages in a
separate short-lived process with a wall-clock limit, so pypdf never runs inside the API process
and a stuck parse cannot hold a request. Inputs are prepared by the organizer (D43), so the counter
has no CPU or memory limits; they also cannot be set portably (macOS rejects RLIMIT_AS).
Readable files are kept once under `server_data/documents/<sha256>.pdf` (content-addressed;
the same file sent again is stored once) and copied read-only into each run's scratch directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from apctt_contracts.pdf import document_filename

COUNT_TIMEOUT_S = 60.0


class CounterFailed(RuntimeError):
    """The page counter itself did not work (not started, crashed, timed out, bad output).

    This is a server problem, not a property of the submitted PDFs.
    """


class DocumentStore:
    def __init__(self, data_dir: Path) -> None:
        self.python = sys.executable
        self.root = data_dir / "documents"
        self.incoming = data_dir / "incoming"
        for directory in (self.root, self.incoming):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def path(self, sha256: str) -> Path:
        return self.root / f"{sha256}.pdf"

    def inspect_and_store(
        self, contents: Sequence[bytes], digests: Sequence[str]
    ) -> list[int | None]:
        """Page count per document (None = unreadable). Keeps the files only if all are readable.

        Raises CounterFailed when the counter itself fails.
        """
        stage = Path(tempfile.mkdtemp(dir=self.incoming))
        try:
            files = []
            for index, data in enumerate(contents):
                path = stage / document_filename(index)
                with open(path, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                files.append(path)
            counts = self._count(files)
            if all(count is not None for count in counts):
                for path, digest in zip(files, digests, strict=True):
                    target = self.path(digest)
                    if not target.exists():
                        os.chmod(path, 0o400)
                        os.replace(path, target)
            return counts
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _count(self, files: Sequence[Path]) -> list[int | None]:
        try:
            proc = subprocess.run(
                [self.python, "-I", "-m", "apctt_contracts.pdf", *map(str, files)],
                capture_output=True,
                timeout=COUNT_TIMEOUT_S,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CounterFailed(type(exc).__name__) from None
        if proc.returncode != 0:
            raise CounterFailed("exit status")
        try:
            value = json.loads(proc.stdout)
        except ValueError:
            raise CounterFailed("output") from None
        if not isinstance(value, list) or len(value) != len(files):
            raise CounterFailed("output")
        return [
            v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else None for v in value
        ]
