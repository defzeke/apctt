"""SPEC-10 canonical hashes, strict JSON and raw identity: AC-20."""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest

from apctt_contracts.hashing import (
    PackageDigestError,
    canonical_bytes,
    canonical_sha256,
    effective_sha256,
    input_sha256,
    package_digest,
    package_manifest,
    sha256_hex,
)
from apctt_contracts.strict_json import StrictJSONError, loads_strict

from .fixtures import case, load_json, result


def test_ac20_hash_and_raw_identity(tmp_path: Path) -> None:
    # The committed example hash is reproduced from the synthetic case.
    request = load_json("docs/EXAMPLE_REQUEST.json")
    response = load_json("docs/EXAMPLE_RESPONSE.json")
    assert input_sha256(request["case"]) == response["meta"]["input_sha256"]

    # Canonical identity ignores key order and whitespace only.
    value = case()
    reordered = json.loads(json.dumps(value, sort_keys=False)[::1])
    spaced = json.dumps(dict(reversed(list(value.items()))), indent=4).encode()
    assert input_sha256(loads_strict(spaced)) == input_sha256(value) == input_sha256(reordered)
    swapped = case()
    swapped["documents"].reverse()
    assert input_sha256(swapped) != input_sha256(value)
    nfd = case()
    nfd["title"] = unicodedata.normalize("NFD", "Café")
    nfc = case()
    nfc["title"] = unicodedata.normalize("NFC", "Café")
    assert input_sha256(nfd) != input_sha256(nfc)
    assert canonical_bytes({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode()

    # Effective identity changes with config, package digest and deployment scope.
    base = {
        "input_sha256": input_sha256(value),
        "config_sha256": canonical_sha256({"seed": 42}),
        "verified_package_sha256": "a" * 64,
        "deployment_scope": "dev:demo",
    }
    reference = effective_sha256(**base)
    assert effective_sha256(**base) == reference
    for key, other in (
        ("config_sha256", canonical_sha256({"seed": 43})),
        ("verified_package_sha256", None),
        ("deployment_scope", "evaluation:demo"),
        ("input_sha256", input_sha256(swapped)),
    ):
        assert effective_sha256(**dict(base, **{key: other})) != reference

    # Raw bytes identify formatting; the parsed result keeps its meaning.
    compact = json.dumps(result(), separators=(",", ":")).encode()
    pretty = json.dumps(result(), indent=2).encode()
    assert sha256_hex(compact) != sha256_hex(pretty)
    assert canonical_sha256(loads_strict(compact)) == canonical_sha256(loads_strict(pretty))
    assert loads_strict(pretty) == result()

    # Strict parsing rejects ambiguous or non-JSON input with fixed reasons.
    for raw, reason in (
        (b'{"a":1,"a":2}', "duplicate_key"),
        (b'{"a":NaN}', "non_finite_number"),
        (b'{"a":-Infinity}', "non_finite_number"),
        (b'{"a":1e400}', "non_finite_number"),
        (b'{"a":"\xff"}', "invalid_utf8"),
        (b'{"a":"\\ud800"}', "lone_surrogate"),
        (b"\xef\xbb\xbf{}", "byte_order_mark"),
        (b'{"a":', "syntax"),
        (b"[" * 100_000 + b"]" * 100_000, "too_deep"),
        (b'{"a":' + b"9" * 5000 + b"}", "number_out_of_range"),
    ):
        with pytest.raises(StrictJSONError) as caught:
            loads_strict(raw)
        assert caught.value.reason == reason, raw[:20]
    assert loads_strict(b'{"a":[1,2.5,"x",null,true]}') == {"a": [1, 2.5, "x", None, True]}

    # Package digest: deterministic, content-sensitive, caches excluded, symlinks rejected.
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n")
    (pkg / "sub" / "b.bin").write_bytes(b"\x00\x01")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"cache")
    entries = package_manifest([("package", pkg)])
    assert [e.path for e in entries] == ["package/a.py", "package/sub/b.bin"]
    first = package_digest(entries)
    assert package_digest(package_manifest([("package", pkg)])) == first
    (pkg / "a.py").write_text("A = 2\n")
    assert package_digest(package_manifest([("package", pkg)])) != first
    os.symlink(pkg / "a.py", pkg / "link.py")
    with pytest.raises(PackageDigestError) as caught_link:
        package_manifest([("package", pkg)])
    assert caught_link.value.reason == "symlink_rejected"
