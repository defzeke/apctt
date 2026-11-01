"""SPEC-10 canonical JSON identity and package manifests.

Canonicalization `python-json-sortkeys-v1`: strict parse, then json.dumps with sorted keys, compact
separators, ensure_ascii=False and allow_nan=False, UTF-8 encoded. Not RFC 8785.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import CANONICALIZATION

EXCLUDED_DIRS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
EXCLUDED_FILES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


def canonical_bytes(value: object) -> bytes:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_hex(canonical_bytes(value))


def input_sha256(case: object) -> str:
    return canonical_sha256(case)


def effective_sha256(
    *,
    input_sha256: str,
    config_sha256: str,
    verified_package_sha256: str | None,
    deployment_scope: str,
) -> str:
    return canonical_sha256(
        {
            "canonicalization": CANONICALIZATION,
            "config_sha256": config_sha256,
            "deployment_scope": deployment_scope,
            "input_sha256": input_sha256,
            "verified_package_sha256": verified_package_sha256,
        }
    )


class PackageDigestError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PackageEntry:
    path: str
    size: int
    sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_manifest(roots: Sequence[tuple[str, Path]]) -> list[PackageEntry]:
    """List regular files under labelled roots. Symlinks and special files are rejected."""
    entries: list[PackageEntry] = []
    for label, root in roots:
        if root.is_symlink():
            raise PackageDigestError("symlink_rejected")
        if not root.is_dir():
            raise PackageDigestError("not_a_directory")
        for current, dirs, files in os.walk(root, followlinks=False):
            base = Path(current)
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
            if any((base / d).is_symlink() for d in dirs):
                raise PackageDigestError("symlink_rejected")
            for name in sorted(files):
                if name in EXCLUDED_FILES or name.endswith(EXCLUDED_SUFFIXES):
                    continue
                path = base / name
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    raise PackageDigestError("symlink_rejected")
                if not stat.S_ISREG(info.st_mode):
                    raise PackageDigestError("special_file")
                relative = f"{label}/{path.relative_to(root).as_posix()}"
                entries.append(PackageEntry(relative, info.st_size, _file_sha256(path)))
    return sorted(entries, key=lambda entry: entry.path)


def package_digest(entries: Sequence[PackageEntry]) -> str:
    return canonical_sha256(
        {
            "canonicalization": CANONICALIZATION,
            "files": [{"path": e.path, "sha256": e.sha256, "size": e.size} for e in entries],
        }
    )
