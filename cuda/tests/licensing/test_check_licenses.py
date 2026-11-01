"""Licence checker (tools/check_licenses.py) against contracts/license-policy.json.

Every input is synthetic and built in a temporary directory: tiny wheels with only a METADATA file,
installed packages, package-lock.json files, submission.json and declaration.json. The licence
strings are the ones real packages declare. Nothing here needs network access. Exit codes: 0 all
allowed, 1 any denied, 2 only review items, 3 usage or input error.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import re
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "check_licenses.py"
POLICY = ROOT / "contracts" / "license-policy.json"
GUIDE = ROOT / "docs" / "participant" / "LICENSING.md"

CATALOG = ROOT / "contracts" / "model-catalog.json"
QWEN_15B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_15B_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN_15B_ID = "qwen2.5-1.5b-instruct"
# The kit's licence texts: D72 checks that participant_module/LICENSE holds the MIT text.
MIT_TEXT = (
    (ROOT / "licenses/MIT.txt").read_text().replace("<year> <copyright holders>", "2026 Team")
)
APACHE_TEXT = (ROOT / "licenses/Apache-2.0.txt").read_text()
SHORT_MIT = "MIT License\n\nCopyright (c) 2026 Example Team\n\nPermission is hereby granted...\n"
NON_COMMERCIAL_TEXT = "Bundled data: for non-commercial research purposes only.\n"
COMMONS_CLAUSE_TEXT = '"Commons Clause" License Condition v1.0\nno right to Sell the Software.\n'
AGPL_PLUS = "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
BSD = "License :: OSI Approved :: BSD License"
APACHE = "License :: OSI Approved :: Apache Software License"
MIT = "License :: OSI Approved :: MIT License"
GPL3 = "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
PROPRIETARY = "License :: Other/Proprietary License"
SEE_LICENSE = "SEE LICENSE IN LICENSE.md"
# primevue 5.0.2's PrimeUI-licensed dependencies (npm, 2026-10-10), and primeicons 8.
PRIMEUI_PACKAGES = (
    ("@primevue/core", "5.0.2"),
    ("@primevue/icons", "5.0.2"),
    ("@primeuix/styles", "3.0.1"),
    ("@primeuix/styled", "1.0.0"),
    ("@primeuix/utils", "0.8.2"),
    ("@primeuix/motion", "1.0.0"),
    ("@primeicons/core", "8.0.2"),
    ("@primeicons/vue", "8.0.2"),
    ("@primeui/license-manager", "1.1.0"),
    ("primeicons", "8.0.2"),
)
# primevue 4.5.5's MIT dependencies (@primeuix/styles ^2.0.3, styled ^0.7.4, utils ^0.6.2).
PRIMEVUE_4_PACKAGES = (
    ("@primevue/core", "4.5.5"),
    ("@primevue/icons", "4.5.5"),
    ("@primeuix/styles", "2.0.3"),
    ("@primeuix/styled", "0.7.4"),
    ("@primeuix/utils", "0.6.2"),
    ("primeicons", "7.0.0"),
)
# The package.json files of pdfjs-dist 6.4.299, apexcharts 5.0.0 and handsontable 6.2.2 exactly as
# their npm tarballs ship them.
MANIFESTS: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parent / "npm_manifests.json").read_text(encoding="utf-8")
)["manifests"]
# Every copyleft or use-restricted "License ::" classifier in trove-classifiers 2026.9.21.13, with
# the status its mapped identifier gets.
OFFICIAL_CLASSIFIERS = {
    "License :: OSI Approved :: GNU Affero General Public License v3": "denied",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)": "denied",
    "License :: OSI Approved :: GNU General Public License (GPL)": "denied",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)": "denied",
    "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)": "denied",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "denied",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)": "denied",
    "License :: OSI Approved :: European Union Public Licence 1.0 (EUPL 1.0)": "denied",
    "License :: OSI Approved :: European Union Public Licence 1.1 (EUPL 1.1)": "denied",
    "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)": "denied",
    "License :: OSI Approved :: Open Software License 3.0 (OSL-3.0)": "denied",
    "License :: Aladdin Free Public License (AFPL)": "denied",
    "License :: Free For Educational Use": "denied",
    "License :: Free For Home Use": "denied",
    "License :: Free for non-commercial use": "denied",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)": "review",
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": "review",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "review",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)": "review",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)": "review",
    "License :: OSI Approved :: Eclipse Public License 1.0 (EPL-1.0)": "review",
    "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)": "review",
    "License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)": (
        "review"
    ),
    "License :: OSI Approved :: Mozilla Public License 1.0 (MPL)": "review",
    "License :: OSI Approved :: Mozilla Public License 1.1 (MPL 1.1)": "review",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "allowed",
    "License :: OSI Approved :: Artistic License": "review",
    "License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)": (
        "review"
    ),
    "License :: CeCILL-B Free Software License Agreement (CECILL-B)": "review",
    "License :: CeCILL-C Free Software License Agreement (CECILL-C)": "review",
    "License :: OSI Approved :: Common Public License": "review",
    "License :: OSI Approved :: IBM Public License": "review",
    "License :: OSI Approved :: GNU Free Documentation License (FDL)": "review",
    "License :: OSI Approved :: Nethack General Public License": "review",
    "License :: OSI Approved :: Qt Public License (QPL)": "review",
    "License :: OSI Approved :: Sleepycat License": "review",
    "License :: OSI Approved :: Sun Public License": "review",
    "License :: OSI Approved :: Apple Public Source License": "review",
    "License :: Netscape Public License (NPL)": "review",
    "License :: OSI Approved :: Ricoh Source Code Public License": "review",
    "License :: OSI Approved :: Motosoto License": "review",
    "License :: OSI Approved :: Nokia Open Source License": "review",
    "License :: Nokia Open Source License (NOKOS)": "review",
}


# ---------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_licenses", TOOL)
    assert spec is not None and spec.loader is not None, TOOL
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_licenses"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def policy(tool: ModuleType) -> Any:
    return tool.load_policy(POLICY)


def metadata(
    name: str,
    version: str,
    expression: str | None,
    license: str | None,
    classifiers: tuple[str, ...],
) -> str:
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if expression is not None:
        lines.append(f"License-Expression: {expression}")
    if license is not None:
        lines.append("License: " + license.replace("\n", "\n        "))
    lines += [f"Classifier: {c}" for c in classifiers]
    return "\n".join(lines) + "\n\nDescription.\n"


def make_wheel(
    directory: Path,
    name: str,
    version: str,
    *,
    expression: str | None = None,
    license: str | None = None,
    classifiers: tuple[str, ...] = (),
    files: dict[str, str] | None = None,
) -> Path:
    """A minimal wheel: METADATA, WHEEL and any extra members ({dist} expands to name-version)."""
    dist = f"{re.sub(r'[-.]+', '_', name)}-{version}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dist}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        text = metadata(name, version, expression, license, classifiers)
        archive.writestr(f"{dist}.dist-info/METADATA", text)
        archive.writestr(f"{dist}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        for member, body in (files or {}).items():
            archive.writestr(member.format(dist=dist), body)
    return path


def make_installed(
    directory: Path,
    name: str,
    version: str,
    *,
    expression: str | None = None,
    license: str | None = None,
    classifiers: tuple[str, ...] = (),
    record: tuple[str, ...] = (),
    editable: bool = False,
) -> Path:
    """An installed package: <dist>.dist-info with METADATA and RECORD."""
    folder = directory / f"{re.sub(r'[-.]+', '_', name)}-{version}.dist-info"
    folder.mkdir(parents=True)
    (folder / "METADATA").write_text(metadata(name, version, expression, license, classifiers))
    (folder / "RECORD").write_text("".join(f"{member},,\n" for member in record))
    if editable:
        direct = {"url": "file:///work/team", "dir_info": {"editable": True}}
        (folder / "direct_url.json").write_text(json.dumps(direct))
    return folder


def make_bundle(
    root: Path,
    *,
    team_license: str | None = "MIT",
    license_text: str | None = MIT_TEXT,
    models: list[dict[str, Any]] | None = None,
    submission: dict[str, Any] | None = None,
    lock: str = "",
    declaration: dict[str, Any] | None = None,
    with_declaration: bool = True,
) -> Path:
    """A Stage 1 bundle in the legacy D51 layout (participant_module/, assets/, wheels/,
    requirements.lock, submission.json, declaration.json; no data sources unless given), so that
    the checker is also exercised on folders that D59/D64 no longer ship: tools/make_submission.py
    refuses wheels/ and leaves requirements.lock out of the minimal bundle."""
    module = root / "participant_module"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    (module / "adapter.py").write_text("class AssessmentModule:\n    pass\n")
    if license_text is not None:
        (module / "LICENSE").write_text(license_text)
    (root / "assets").mkdir()
    (root / "wheels").mkdir()
    if submission is None:
        submission = {"models": models or []}
        if team_license is not None:
            submission["license"] = team_license
    (root / "submission.json").write_text(json.dumps(submission, indent=2))
    (root / "requirements.lock").write_text(lock)
    if with_declaration:
        content = declaration if declaration is not None else {"data_sources": []}
        (root / "declaration.json").write_text(json.dumps(content, indent=2))
    return root


def make_web(root: Path, *, license_text: str | None = MIT_TEXT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if license_text is not None:
        (root / "LICENSE").write_text(license_text)
    (root / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n")
    return root


def write_lock(folder: Path, packages: dict[str, dict[str, Any]], version: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lock = {"name": "w", "lockfileVersion": version, "packages": packages}
    (folder / "package-lock.json").write_text(json.dumps(lock))


def run(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], *args: str, policy: Path = POLICY
) -> tuple[int, dict[str, Any]]:
    code = tool.main([*args, "--policy", str(policy), "--json"])
    return code, json.loads(capsys.readouterr().out)


def item(report: dict[str, Any], name: str, kind: str | None = None) -> dict[str, Any]:
    found = [
        i for i in report["items"] if i["name"] == name and (kind is None or i["kind"] == kind)
    ]
    assert len(found) == 1, (name, kind, [(i["kind"], i["name"]) for i in report["items"]])
    return found[0]


def statuses(report: dict[str, Any]) -> dict[str, str]:
    return {f"{i['kind']}:{i['name']}": i["status"] for i in report["items"]}


def policy_with(tmp_path: Path, packages: list[dict[str, Any]]) -> Path:
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    raw["packages"] += packages
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    return path


# ------------------------------------------------------------------------ policy file


def test_policy_states_the_confirmed_rules(policy: Any) -> None:
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    assert raw["policy_version"] == "apctt-license-policy-v2"
    assert raw["version"] == "0.4.0"
    assert "2026-10-10" in raw["status"] and "confirmed" in raw["status"]
    assert "D72" in raw["status"] and "2026-10-13" in raw["status"]
    assert raw["updated"] == "2026-10-13"
    # D63: no exceptions and no advance rulings; review items are decided after the submission.
    for text in (json.dumps(raw["statuses"]), json.dumps(raw["models"]), raw["datasets"]["note"]):
        assert "written exception" not in text and "unless the organizer" not in text, text
    assert "after the submission" in raw["statuses"]["review"]
    assert "legal advice" not in raw["notice"].lower() and "SPEC-14" in raw["notice"]  # D79
    assert "HQ decides flagged items after submission" in raw["notice"]
    assert "person" not in raw["statuses"]["review"] and "HQ decides" in raw["statuses"]["review"]
    # D72 (2026-10-13): the team's own work is MIT only; third parties keep their licences.
    assert raw["team_license"]["allowed"] == ["MIT"]
    team = raw["team_license"]
    for key in ("bundle", "web", "rule"):
        assert "Apache-2.0 or MIT" not in team[key], key
    assert "D72" in team["rule"] and "Apache-2.0 is no longer accepted" in team["rule"]
    assert "own licences" in team["rule"] and "NOTICE" in team["rule"]
    assert "team_license" not in json.dumps(raw["settled"])
    assert raw["models"]["allowed_licenses"] == ["Apache-2.0", "MIT"]  # catalogue models (D64)
    lists = raw["licenses"]
    allow = set(lists["allow"]) | set(lists["allow_with_conditions"])
    assert not allow & set(lists["review"]) and not allow & set(lists["deny"])
    assert not set(lists["review"]) & set(lists["deny"])
    for pattern in ("AGPL-*", "GPL-*", "SSPL-*", "EUPL-*", "OSL-*", "RPL-*", "CC-BY-NC*"):
        assert pattern in lists["deny_patterns"], pattern
    for pattern in ("LGPL-*", "EPL-*", "CDDL-*", "CC-BY-ND-*"):
        assert pattern in lists["review_patterns"], pattern
    assert "MPL-2.0" in lists["allow_with_conditions"]
    exceptions = lists["exceptions"]
    assert exceptions["LLVM-exception"]["status"] == "keep"
    assert exceptions["Classpath-exception-2.0"]["status"] == "review"
    assert exceptions["Font-exception-2.0"]["status"] == "allowed"
    assert "pdfjs-dist" in exceptions["Font-exception-2.0"]["note"]
    # Repair round 1: an allowed or review exception names the GPL bases it lifts; the two GPL
    # carve-outs were a builder addition, adopted as rules by decision D81.
    for name, exception in exceptions.items():
        if exception["status"] in ("allowed", "review"):
            assert exception["bases"] and all(b.startswith("GPL-") for b in exception["bases"])
    settled = " ".join(raw["settled"])
    assert "GCC-exception-3.1" in settled and "Font-exception-2.0" in settled and "D81" in settled
    for name in ("GCC-exception-3.1", "Font-exception-2.0", "Classpath-exception-2.0"):
        assert exceptions[name]["decision"] == "D81", name
    # Repair round 2: Classpath-exception-2.0 (review on GPL-2.0) is a builder addition too, and
    # Elastic-2.0 (review; it bars offering the software as a hosted service) stays review, decided
    # by HQ after the submission (D81).
    assert "Classpath-exception-2.0" in settled and "Elastic-2.0" in settled
    assert "settled" in raw and "builder_addition" not in json.dumps(exceptions)
    # Repair round 2: Commons Clause is refused as a plain identifier too.
    assert "Commons-Clause" in lists["deny"]
    rules = {name: rule for rule in raw["packages"] for name in rule["names"]}
    assert rules["mineru"]["status"] == "denied"
    assert "min_version" not in rules["handsontable"]  # confirmed: "Handsontable", no version
    # Every table target is a valid expression of identifiers the policy classifies.
    for target in list(raw["license_strings"].values()) + list(raw["classifiers"].values()):
        assert policy.verdict(target)[0] in ("allowed", "review", "denied"), target
        assert all(policy.known(i) for i in policy.ids(target)), target
    for key in raw["license_strings"]:
        assert key == " ".join(key.lower().split()), key
    base = raw["organizer_base"]
    assert {"torch", "triton", "nvidia-*-cu1*", "cuda-*"} <= set(base["proprietary_patterns"])
    # D64: the model catalogue is the only list of public models (no allowlist that can drift).
    assert "allowlist" not in raw["models"]
    assert raw["models"]["catalog"] == "contracts/model-catalog.json"
    assert "D64" in raw["models"]["rule"] and "refused" in raw["models"]["rule"]
    for entry in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]:
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry
        assert entry["license"] in raw["models"]["allowed_licenses"], entry
        denied = [d["pattern"] for d in raw["models"]["deny"]]
        assert not any(fnmatch.fnmatchcase(entry["repo"], d) for d in denied), entry["repo"]
    names = {n for rule in raw["packages"] for n in rule["names"]}
    for name in ("pymupdf", "surya-ocr", "marker-pdf", "mineru", "@ckeditor/", "@mui/x-*-pro"):
        assert name in names, name


def test_licence_lists_of_the_participant_guide_match_the_policy() -> None:
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    guide = GUIDE.read_text(encoding="utf-8")
    lists = guide.split("### 3.1 Licence lists", 1)[1].split("\n### ", 1)[0]
    allowed = next(line for line in lists.splitlines() if line.startswith("| Allowed |"))
    for licence in raw["licenses"]["allow"]:
        if not licence.startswith("LicenseRef-"):
            assert licence in allowed, licence
    for word in ("AGPL", "GPL", "SSPL", "EUPL", "OSL", "RPL", "LGPL", "EPL", "CDDL", "MPL-2.0"):
        assert word in lists, word
    assert f"licence policy {raw['version']}" in guide.splitlines()[2]


# ------------------------------------------------------- declared strings -> SPDX -> status


@pytest.mark.parametrize(
    ("raw", "expression", "status"),
    [
        # SPDX expressions as real packages declare them.
        ("MIT", "MIT", "allowed"),
        ("apache-2.0", "Apache-2.0", "allowed"),  # Hugging Face spelling
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", None, "allowed"),  # numpy
        ("Apache-2.0 OR BSD-2-Clause", None, "allowed"),  # packaging
        ("Apache-2.0 AND CNRI-Python", None, "allowed"),  # regex
        ("(MPL-2.0 OR Apache-2.0)", "MPL-2.0 OR Apache-2.0", "allowed"),  # dompurify
        ("(MIT AND BSD-3-Clause)", "MIT AND BSD-3-Clause", "allowed"),  # sha.js
        ("(CC-BY-4.0 AND OFL-1.1 AND MIT)", None, "allowed"),  # @fortawesome/fontawesome-free
        ("LicenseRef-NVIDIA-Proprietary", None, "review"),  # nvidia-nccl-cu12 2.28.9
        ("LicenseRef-MinerU-Open-Source-License", None, "review"),  # mineru 4.0.11
        # Strong copyleft is denied, weak copyleft is review, MPL-2.0 is allowed (file level).
        ("AGPL-3.0-only", None, "denied"),
        ("AGPL-3.0", None, "denied"),
        ("GPL-2.0-or-later", None, "denied"),
        ("GPL-2.0+", None, "denied"),
        ("GPL-3.0-only", None, "denied"),
        ("SSPL-1.0", None, "denied"),
        ("EUPL-1.2", None, "denied"),
        ("OSL-3.0", None, "denied"),
        ("RPL-1.5", None, "denied"),
        ("LGPL-2.1-or-later", None, "review"),
        ("LGPL-2.1+", None, "review"),
        ("EPL-2.0", None, "review"),
        ("CDDL-1.0", None, "review"),
        ("MPL-2.0", None, "allowed"),
        # Use-restricted licences.
        ("CC-BY-NC-4.0", None, "denied"),
        ("cc-by-nc-sa-4.0", None, "denied"),
        ("CC-BY-ND-4.0", None, "review"),
        ("CC-BY-SA-4.0", None, "review"),
        ("BUSL-1.1", None, "denied"),
        ("PolyForm-Noncommercial-1.0.0", None, "denied"),
        ("Elastic-2.0", None, "review"),
        # OR: the most permissive allowed choice; a paid alternative is no choice.
        ("MIT OR GPL-3.0-only", None, "allowed"),
        ("GPL-3.0-only OR LicenseRef-Commercial", None, "denied"),
        ("AGPL-3.0-only OR LicenseRef-Proprietary", None, "denied"),
        # Repair round 1 (d): a custom commercial or proprietary LicenseRef is no free choice.
        ("GPL-3.0-only OR LicenseRef-Artifex-Commercial", None, "denied"),
        ("AGPL-3.0-or-later OR LicenseRef-Acme-Proprietary-1.0", None, "denied"),
        ("MIT OR LicenseRef-Acme-Commercial", None, "allowed"),
        ("LGPL-3.0-only OR GPL-3.0-only", None, "review"),
        ("MIT AND GPL-3.0-only", None, "denied"),
        # WITH: only listed exceptions change the result.
        ("Apache-2.0 WITH LLVM-exception", None, "allowed"),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", None, "allowed"),
        ("GPL-2.0-only WITH Font-exception-2.0", None, "allowed"),  # pdfjs-dist Liberation fonts
        ("GPL-2.0-only WITH Classpath-exception-2.0", None, "review"),
        ("Apache-2.0 WITH Commons-Clause", None, "denied"),
        ("MIT WITH Some-Unknown-exception", None, "review"),
        ("GPL-2.0-only WITH Linux-syscall-note", None, "denied"),
        # Repair round 1: an exception lifts only the GPL base it belongs to; a denied base stays
        # denied, and on any other base the exception is at least review.
        ("AGPL-3.0-only WITH GCC-exception-3.1", None, "denied"),
        ("SSPL-1.0 WITH Font-exception-2.0", None, "denied"),
        ("CC-BY-NC-4.0 WITH Font-exception-2.0", None, "denied"),
        ("BUSL-1.1 WITH GCC-exception-3.1", None, "denied"),
        ("LicenseRef-Commercial WITH GCC-exception-3.1", None, "denied"),
        ("AGPL-3.0-only WITH Classpath-exception-2.0", None, "denied"),
        ("Apache-2.0 WITH GCC-exception-3.1", None, "review"),
        ("MIT WITH Font-exception-2.0", None, "review"),
        # Repair round 2: Commons Clause written as a plain identifier is refused wherever it
        # appears, also as an OR "alternative" (it is a restriction, not a licence).
        ("Commons-Clause", None, "denied"),
        ("Apache-2.0 AND Commons-Clause", None, "denied"),
        ("MIT OR Commons-Clause", None, "denied"),
        ("MIT OR LicenseRef-Commons-Clause", None, "denied"),
        # Repair round 2: the deprecated GPL-*-with-* identifiers are read as their WITH form, so
        # they get the same result as the modern spelling (no review-list shortcut).
        (
            "GPL-2.0-with-classpath-exception",
            "GPL-2.0-only WITH Classpath-exception-2.0",
            "review",
        ),
        ("GPL-2.0-or-later WITH Classpath-exception-2.0", None, "review"),
        ("GPL-3.0-with-GCC-exception", "GPL-3.0-only WITH GCC-exception-3.1", "allowed"),
        ("GPL-2.0-with-GCC-exception", "GPL-2.0-only WITH GCC-exception-2.0", "denied"),
        ("GPL-2.0-with-font-exception", "GPL-2.0-only WITH Font-exception-2.0", "allowed"),
        ("GPL-2.0-with-autoconf-exception", "GPL-2.0-only WITH Autoconf-exception-2.0", "denied"),
        ("GPL-2.0-with-bison-exception", "GPL-2.0-or-later WITH Bison-exception-2.2", "denied"),
        ("MIT AND GPL-2.0-with-GCC-exception", None, "denied"),
        # Free-text License fields of real packages, through the exact table.
        ("MIT License", "MIT", "allowed"),
        ("The MIT License (MIT)", "MIT", "allowed"),
        ("Apache 2.0 License", "Apache-2.0", "allowed"),  # transformers 5.19.0
        ("Apache License, Version 2.0", "Apache-2.0", "allowed"),
        ("Apache Software License", "LicenseRef-Apache-unspecified", "allowed"),
        ("BSD License", "LicenseRef-BSD-unspecified", "allowed"),
        ("3-Clause BSD License", "BSD-3-Clause", "allowed"),
        ("  bsd   3-clause   LICENSE ", "BSD-3-Clause", "allowed"),
        ("GPLv3", "GPL-3.0-only", "denied"),
        ("GPL v2 or later", "GPL-2.0-or-later", "denied"),
        ("GPL-2.0 or later", "GPL-2.0-or-later", "denied"),  # never read as an OR
        ("LGPLv2.1+", "LGPL-2.1-or-later", "review"),
        ("AGPLv3", "AGPL-3.0-only", "denied"),
        (
            "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License",  # pymupdf 1.28.2
            "AGPL-3.0-only OR LicenseRef-Commercial",
            "denied",
        ),
        ("NVIDIA Proprietary Software", "LicenseRef-NVIDIA-Proprietary", "review"),
        ("Commercial", "LicenseRef-Commercial", "denied"),
        ("Public Domain", "LicenseRef-Public-Domain", "review"),
        ("UNLICENSED", "LicenseRef-Proprietary", "review"),  # npm: no permission granted
        ("CC BY 4.0", "CC-BY-4.0", "allowed"),
        ("CC BY-NC 4.0", "CC-BY-NC-4.0", "denied"),
        ("llama3.2", "LicenseRef-Llama-3.2-Community", "denied"),
        ("gemma", "LicenseRef-Gemma-Terms", "denied"),
        ("other", "LicenseRef-Other", "review"),
        # Identifiers the policy does not list are review.
        ("Apache-1.1", "Apache-1.1", "review"),
        ("Custom", "Custom", "review"),
    ],
)
def test_declared_strings(policy: Any, raw: str, expression: str | None, status: str) -> None:
    normalised = policy.normalise(raw)
    assert normalised is not None, raw
    if expression is not None:
        assert normalised == expression
    assert policy.verdict(normalised)[0] == status, (raw, normalised)


@pytest.mark.parametrize(
    ("raw", "expression"),
    [
        ("Apache2.0", "Apache-2.0"),  # rank-bm25 0.2.2
        ("Apache2", "Apache-2.0"),
        ("Apache 2.0", "Apache-2.0"),
        ("Apache-2.0", "Apache-2.0"),
        ("Apache License 2.0", "Apache-2.0"),
        ("Apache License, Version 2.0", "Apache-2.0"),
        ("Apache Licence, Version 2.0", "Apache-2.0"),
        ("Apache License Version 2", "Apache-2.0"),
        ("The Apache License, Version 2.0", "Apache-2.0"),
        ("The Apache Software License, Version 2.0", "Apache-2.0"),
        ("MIT License", "MIT"),
        ("MIT Licence", "MIT"),
        ("The MIT Licence", "MIT"),
        ("MIT-License", "MIT"),
    ],
)
def test_unambiguous_apache_and_mit_spellings(policy: Any, raw: str, expression: str) -> None:
    assert policy.normalise(raw) == expression, raw
    assert policy.verdict(expression)[0] == "allowed"  # both stay allowed for third parties
    assert policy.team_cleared(expression) == (expression == "MIT")  # the team: MIT only (D72)


@pytest.mark.parametrize(
    "raw",
    [
        "BSD-3-Clause, Apache-2.0, dependency licenses",  # pypdfium2
        "SEE LICENSE IN LICENSE.md",  # @ckeditor/*, later tinymce
        "Dual License",  # python-dateutil's License field
        "MIT with Commons Clause",
        "Apache 2.0 for research purposes",
        "MIT License (no commercial use without permission)",
        "Copyright (c) 2005-2024, NumPy Developers.\nAll rights reserved.\n\nRedistribution ...",
    ],
)
def test_other_text_is_an_unrecognised_licence_string(policy: Any, raw: str) -> None:
    assert policy.normalise(raw) is None


def test_deprecated_identifiers_are_read_in_their_modern_form(policy: Any) -> None:
    # Policy.verdict on the raw identifier (not normalised first) gives the modern form's result.
    for old, new in (
        ("GPL-2.0-with-classpath-exception", "GPL-2.0-only WITH Classpath-exception-2.0"),
        ("GPL-2.0-with-GCC-exception", "GPL-2.0-only WITH GCC-exception-2.0"),
        ("GPL-3.0-with-GCC-exception", "GPL-3.0-only WITH GCC-exception-3.1"),
        ("GPL-2.0-with-font-exception", "GPL-2.0-only WITH Font-exception-2.0"),
    ):
        assert policy.verdict(old) == policy.verdict(new), old
        assert policy.known(old), old


def test_every_official_copyleft_classifier_is_mapped(policy: Any) -> None:
    # Repair round 2: EUPL 1.0/1.1 and OSL-3.0 classifiers were unmapped, so a wheel declaring
    # only one of them was review ("unrecognised licence classifier") instead of denied.
    for classifier, status in OFFICIAL_CLASSIFIERS.items():
        assert classifier in policy.classifiers, classifier
        assert policy.verdict(policy.classifiers[classifier])[0] == status, classifier


def test_missing_licences(policy: Any) -> None:
    for raw in ("", "UNKNOWN", "none", "  "):
        assert policy.normalise(raw) == ""
    assert policy.verdict("")[0] == "review"


def test_conditions_are_reported(policy: Any) -> None:
    status, conditions = policy.verdict("(CC-BY-4.0 AND OFL-1.1 AND MIT)")
    assert status == "allowed" and len(conditions) == 2
    status, conditions = policy.verdict("GPL-2.0-only WITH Font-exception-2.0")
    assert status == "allowed" and "unmodified" in conditions[0]


# ---------------------------------------------------------------- Python package metadata


def test_python_metadata_order_and_real_strings(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    wheels = bundle / "wheels"
    # License-Expression first, whatever the classifiers and License field say.
    make_wheel(
        wheels, "numpy", "2.5.3", expression="BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"
    )
    make_wheel(wheels, "exprlib", "1.0", expression="MIT", license="GPL", classifiers=(GPL3,))
    # Then classifiers: python-dateutil ("Dual License" + BSD and Apache classifiers).
    make_wheel(
        wheels, "python-dateutil", "2.9.0", license="Dual License", classifiers=(BSD, APACHE)
    )
    make_wheel(wheels, "fieldlib", "1.0", license="GPL", classifiers=(MIT,))
    # Then the License field: transformers declares only "Apache 2.0 License".
    make_wheel(wheels, "transformers", "5.19.0", license="Apache 2.0 License")
    make_wheel(
        wheels, "pypdfium2", "5.14.0", license="BSD-3-Clause, Apache-2.0, dependency licenses"
    )
    make_wheel(wheels, "mysterylib", "0.1")
    make_wheel(wheels, "oddclassifier", "1.0", classifiers=("License :: Some Unknown Licence",))
    code, report = run(tool, capsys, "--bundle", str(bundle))
    numpy = item(report, "numpy")
    assert numpy["status"] == "allowed" and numpy["origin"] == "License-Expression"
    assert item(report, "exprlib")["license"] == "MIT"
    dateutil = item(report, "python-dateutil")
    assert dateutil["status"] == "allowed" and dateutil["origin"] == "Classifier"
    assert " OR " in dateutil["license"]
    assert item(report, "fieldlib")["status"] == "allowed"  # the classifier wins over the field
    transformers = item(report, "transformers")
    assert transformers["license"] == "Apache-2.0" and transformers["origin"] == "License"
    pdfium = item(report, "pypdfium2")
    assert pdfium["status"] == "review" and "unrecognised licence string" in pdfium["reasons"][0]
    mystery = item(report, "mysterylib")
    assert mystery["status"] == "review" and "no licence declared" in mystery["reasons"][0]
    odd = item(report, "oddclassifier")
    assert odd["status"] == "review" and "unrecognised licence classifier" in odd["reasons"][0]
    assert code == 2


def test_several_classifiers(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    make_wheel(bundle / "wheels", "permissive", "1.0", classifiers=(MIT, APACHE))
    make_wheel(bundle / "wheels", "mixed", "1.0", classifiers=(MIT, GPL3))
    make_wheel(bundle / "wheels", "gplorpaid", "1.0", classifiers=(AGPL_PLUS, PROPRIETARY))
    eupl = "License :: OSI Approved :: European Union Public Licence 1.1 (EUPL 1.1)"
    make_wheel(bundle / "wheels", "eupllib", "1.0", classifiers=(eupl,))
    osl = "License :: OSI Approved :: Open Software License 3.0 (OSL-3.0)"
    make_wheel(bundle / "wheels", "osllib", "1.0", classifiers=(osl,))
    code, report = run(tool, capsys, "--bundle", str(bundle))
    assert item(report, "permissive")["status"] == "allowed"
    mixed = item(report, "mixed")
    assert mixed["status"] == "review"
    assert any("different statuses" in reason for reason in mixed["reasons"]), mixed
    assert item(report, "gplorpaid")["status"] == "denied"  # copyleft or a paid licence
    for name, licence in (("eupllib", "EUPL-1.1"), ("osllib", "OSL-3.0")):
        found = item(report, name)
        assert found["status"] == "denied" and found["license"] == licence, found
        assert not any("unrecognised" in reason for reason in found["reasons"]), found
    assert code == 1


def test_licence_file_bodies_are_not_read(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Review finding (a): words such as "non-commercial" in bundled licence files (OpenCV,
    # pyarrow, rapidocr) made permissive packages review. Only the metadata counts now.
    bundle = make_bundle(tmp_path / "submission")
    licences = {
        "{dist}.dist-info/licenses/LICENSE": APACHE_TEXT,
        "{dist}.dist-info/licenses/LICENSE-3RD-PARTY.txt": NON_COMMERCIAL_TEXT,
    }
    make_wheel(
        bundle / "wheels", "opencv-python", "4.13.0.92", license="Apache 2.0", files=licences
    )
    make_wheel(bundle / "wheels", "pyarrow", "23.0.0", expression="Apache-2.0", files=licences)
    make_wheel(bundle / "wheels", "rapidocr", "3.4.0", license="Apache-2.0", files=licences)
    make_wheel(
        bundle / "wheels",
        "clauselib",
        "1.0",
        expression="MIT",
        files={"{dist}.dist-info/licenses/LICENSE": MIT_TEXT + COMMONS_CLAUSE_TEXT},
    )
    make_wheel(
        bundle / "wheels",
        "fileonly",
        "1.0",
        files={"{dist}.dist-info/licenses/LICENSE": MIT_TEXT},
    )
    code, report = run(tool, capsys, "--bundle", str(bundle))
    for name in ("opencv-python", "pyarrow", "rapidocr", "clauselib"):
        assert item(report, name)["status"] == "allowed", item(report, name)
    assert item(report, "fileonly")["status"] == "review"  # no declared licence
    assert code == 2


# ---------------------------------------------------------------------------- name rules


def test_python_name_rules_override_metadata(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    wheels = bundle / "wheels"
    make_wheel(wheels, "surya-ocr", "0.22.1", license="Apache-2.0")
    make_wheel(wheels, "marker-pdf", "2.0.0", license="Apache-2.0")
    make_wheel(
        wheels,
        "PyMuPDF",
        "1.28.2",
        license="Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License",
    )
    make_wheel(wheels, "mineru", "4.0.11", expression="LicenseRef-MinerU-Open-Source-License")
    make_wheel(
        wheels,
        "Unidecode",
        "1.4.0",
        license="GPL",
        classifiers=("License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",),
    )
    code, report = run(tool, capsys, "--bundle", str(bundle))
    surya = item(report, "surya-ocr")
    assert surya["status"] == "denied" and surya["license"] == "Apache-2.0"
    assert any("OpenRAIL" in reason for reason in surya["reasons"]), surya
    assert item(report, "marker-pdf")["status"] == "denied"
    pymupdf = item(report, "pymupdf")
    assert pymupdf["status"] == "denied"
    assert any("pypdf" in reason for reason in pymupdf["reasons"]), pymupdf
    # Repair round 1: the MinerU licence is revenue-capped (D53 rule 3), so it is denied.
    mineru = " ".join(item(report, "mineru")["reasons"])
    assert item(report, "mineru")["status"] == "denied", mineru
    assert "USD 20 million" in mineru and "100 million" in mineru, mineru
    unidecode = item(report, "unidecode")
    assert unidecode["status"] == "denied"
    rule = next(r for r in unidecode["reasons"] if r.startswith("policy rule for unidecode"))
    assert "GPL-2.0-or-later" in rule and "PyQt6" not in rule and "kiwipiepy" not in rule, rule
    assert code == 1


def test_npm_name_scope_and_version_rules(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    write_lock(
        web,
        {
            "": {"name": "w", "license": "MIT"},
            "node_modules/apexcharts": {"version": "5.0.0", "license": "MIT"},
            "node_modules/old/node_modules/apexcharts": {"version": "4.7.0", "license": "MIT"},
            "node_modules/@ckeditor/ckeditor5-core": {
                "version": "49.0.0",
                "license": "SEE LICENSE IN LICENSE.md",
            },
            "node_modules/ckeditor5": {"version": "47.0.0", "license": "SEE LICENSE IN LICENSE.md"},
            "node_modules/@mui/x-data-grid-pro": {
                "version": "8.0.0",
                "license": "SEE LICENSE IN LICENSE",
            },
            "node_modules/@mui/x-data-grid": {"version": "8.0.0", "license": "MIT"},
            "node_modules/tinymce": {"version": "6.8.5", "license": "MIT"},
            "node_modules/editor/node_modules/tinymce": {
                "version": "8.9.3",
                "license": "SEE LICENSE IN license.md",
            },
            "node_modules/primevue": {"version": "5.0.2", "license": "SEE LICENSE IN LICENSE.md"},
            "node_modules/mapbox-gl": {
                "version": "1.13.3",
                "license": "SEE LICENSE IN LICENSE.txt",
            },
            "node_modules/highcharts": {
                "version": "11.4.0",
                "license": "https://www.highcharts.com/license",
            },
            "node_modules/pdfjs-dist": {"version": "6.4.299", "license": "Apache-2.0"},
            # Repair round 1: the PrimeUI-licensed packages around primevue 5 (versions and
            # licences as published on npm), and the MIT versions primevue 4.5.5 installs.
            **{
                f"node_modules/{name}": {"version": version, "license": SEE_LICENSE}
                for name, version in PRIMEUI_PACKAGES
            },
            **{
                f"node_modules/primevue4/node_modules/{name}": {
                    "version": version,
                    "license": "MIT",
                }
                for name, version in PRIMEVUE_4_PACKAGES
            },
            "node_modules/handsontable": {"version": "6.2.2", "license": "MIT"},
        },
    )
    code, report = run(tool, capsys, "--web", str(web))
    found = {(i["name"], i["version"]): i for i in report["items"] if i["kind"] == "npm"}
    for name, version in PRIMEUI_PACKAGES:
        assert found[(name, version)]["status"] == "denied", found[(name, version)]
        assert "PrimeUI" in " ".join(found[(name, version)]["reasons"])
    for name, version in PRIMEVUE_4_PACKAGES:
        assert found[(name, version)]["status"] == "allowed", found[(name, version)]
    # The confirmed rule names Handsontable without a version (6.2.2 is MIT: an exception).
    assert found[("handsontable", "6.2.2")]["status"] == "denied"
    expected = {
        ("apexcharts", "5.0.0"): "denied",
        ("apexcharts", "4.7.0"): "allowed",
        ("@ckeditor/ckeditor5-core", "49.0.0"): "denied",  # review finding (c): scope rule
        ("ckeditor5", "47.0.0"): "denied",
        ("@mui/x-data-grid-pro", "8.0.0"): "denied",
        ("@mui/x-data-grid", "8.0.0"): "allowed",
        ("tinymce", "6.8.5"): "allowed",
        ("tinymce", "8.9.3"): "denied",
        ("primevue", "5.0.2"): "denied",
        ("mapbox-gl", "1.13.3"): "review",  # BSD-3-Clause file, unrecognised metadata
        ("highcharts", "11.4.0"): "denied",
        ("pdfjs-dist", "6.4.299"): "allowed",
    }
    assert {key: found[key]["status"] for key in expected} == expected
    pdfjs = found[("pdfjs-dist", "6.4.299")]
    assert any("LICENSE_LIBERATION" in condition for condition in pdfjs["conditions"]), pdfjs
    assert code == 1


def test_published_allow_rules(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    policy = policy_with(
        tmp_path,
        [
            {
                "ecosystem": "pypi",
                "names": ["pypdfium2", "pymupdf"],
                "status": "allowed",
                "reason": "example",
            },
            {
                "ecosystem": "pypi",
                "names": ["gpulib"],
                "status": "allowed",
                "reason": "licence only",
            },
            {
                "ecosystem": "pypi",
                "names": ["gpulib2"],
                "status": "allowed",
                "reason": "including its NVIDIA binaries",
                "embedded_binaries_ok": True,
            },
        ],
    )
    bundle = make_bundle(tmp_path / "submission")
    make_wheel(
        bundle / "wheels",
        "pypdfium2",
        "5.14.0",
        license="BSD-3-Clause, Apache-2.0, dependency licenses",
    )
    make_wheel(bundle / "wheels", "pymupdf", "1.28.2", license="AGPL-3.0")
    for name in ("gpulib", "gpulib2"):
        make_wheel(
            bundle / "wheels",
            name,
            "1.0",
            license="MIT",
            files={f"{name}.libs/libcudart.so.12": "binary"},
        )
    code, report = run(tool, capsys, "--bundle", str(bundle), policy=policy)
    pdfium = item(report, "pypdfium2")
    assert pdfium["status"] == "allowed"
    assert any("policy rule" in reason for reason in pdfium["reasons"]), pdfium
    assert not any("exception" in reason for reason in pdfium["reasons"]), pdfium
    assert item(report, "pymupdf")["status"] == "denied"  # a deny rule still wins
    gpulib = item(report, "gpulib")
    assert gpulib["status"] == "review"
    assert any("libcudart" in reason for reason in gpulib["reasons"]), gpulib
    assert item(report, "gpulib2")["status"] == "allowed"
    assert code == 1


# ----------------------------------------------------------- organizer base (D52)


def test_common_cuda_base_in_a_team_bundle_is_denied(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path / "submission",
        lock="triton==3.6.0 \\\n    --hash=sha256:00\nTyping_Extensions==4.16.0\n",
    )
    wheels = bundle / "wheels"
    make_wheel(wheels, "torch", "2.11.0+cu128", license="BSD-3-Clause")
    make_wheel(
        wheels, "nvidia-nvtx-cu12", "12.8.90", license="Apache 2.0", classifiers=(PROPRIETARY,)
    )
    make_wheel(
        wheels, "nvidia-cublas", "13.8.1.7", license="NVIDIA Proprietary Software"
    )  # CUDA 13
    make_wheel(wheels, "nvidia-newlib", "1.0", expression="LicenseRef-NVIDIA-Proprietary")
    make_wheel(wheels, "jinja2", "3.1.6", classifiers=(BSD,))
    make_wheel(wheels, "nvidia-ml-py", "13.580.82", license="BSD", classifiers=(BSD,))
    make_wheel(
        wheels,
        "visionlib",
        "0.1",
        license="BSD",
        files={"visionlib.libs/libcudart.41118559.so.12": "binary"},
    )
    code, report = run(tool, capsys, "--bundle", str(bundle))
    for name in (
        "torch",
        "nvidia-nvtx-cu12",
        "nvidia-cublas",
        "nvidia-newlib",
        "jinja2",
        "triton",
        "typing-extensions",
    ):
        entry = item(report, name)
        assert entry["status"] == "denied", entry
        assert any("D52" in reason for reason in entry["reasons"]), entry
    assert item(report, "nvidia-ml-py")["status"] == "allowed"
    vision = item(report, "visionlib")
    assert vision["status"] == "review" and "libcudart" in " ".join(vision["reasons"])
    assert code == 1


def test_common_cuda_base_in_a_web_system(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "requirements.txt").write_text("flask==3.1.3\njinja2==3.1.6\ntorch==2.11.0+cu128\n")
    make_wheel(web / "wheels", "flask", "3.1.3", expression="BSD-3-Clause")
    make_wheel(web / "wheels", "jinja2", "3.1.6", classifiers=(BSD,))
    code, report = run(tool, capsys, "--web", str(web))
    assert item(report, "jinja2")["status"] == "allowed"  # open-source base pins are fine here
    torch = item(report, "torch")
    assert torch["status"] == "denied" and torch["kind"] == "lock"
    assert code == 1


# ---------------------------------------------------------------- team licence (rule 1)


@pytest.mark.parametrize(
    ("declared", "text", "status"),
    [
        ("MIT", MIT_TEXT, "allowed"),
        ("Apache-2.0", APACHE_TEXT, "denied"),  # D72: MIT only for the team's own work
        ("mit", MIT_TEXT, "allowed"),
        ("MIT", APACHE_TEXT, "denied"),  # D72: the LICENSE must hold the MIT text
        ("MIT", SHORT_MIT, "denied"),  # the permission and warranty paragraphs are missing
        ("MIT", MIT_TEXT.upper().replace("\n", "\n  "), "allowed"),  # case, line breaks, spaces
        ("MIT", MIT_TEXT.replace('"', "\u201c"), "allowed"),  # typographic quotes
        ("BSD-3-Clause", MIT_TEXT, "denied"),
        ("GPL-3.0-only", MIT_TEXT, "denied"),
        ("LicenseRef-Proprietary", MIT_TEXT, "denied"),
        ("MIT with restrictions", MIT_TEXT, "denied"),
        # Repair round 1: an exception added to Apache-2.0 or MIT is another licence.
        ("MIT WITH Commons-Clause", MIT_TEXT, "denied"),
        ("Apache-2.0 WITH Commons-Clause", APACHE_TEXT, "denied"),
        ("Apache-2.0 WITH LLVM-exception", APACHE_TEXT, "denied"),
        ("MIT WITH Font-exception-2.0", MIT_TEXT, "denied"),
        ("MIT OR Apache-2.0", MIT_TEXT, "denied"),  # D72: declare MIT alone
        ("Apache-2.0 AND MIT", MIT_TEXT, "denied"),
        (None, MIT_TEXT, "denied"),
        ("MIT", None, "denied"),
        ("MIT", "  \n", "denied"),
    ],
    ids=lambda value: " ".join(value.split())[:24] if isinstance(value, str) else None,
)
def test_team_licence_in_a_bundle(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    declared: str | None,
    text: str | None,
    status: str,
) -> None:
    bundle = make_bundle(tmp_path / "submission", team_license=declared, license_text=text)
    code, report = run(tool, capsys, "--bundle", str(bundle))
    team = item(report, "participant_module", "team")
    assert team["status"] == status, team
    assert code == {"allowed": 0, "denied": 1}[status]


def test_apache_is_refused_for_the_team_licence_with_a_clear_message(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D72 (2026-10-13): the team's own work is MIT only. A team that still declares Apache-2.0
    (allowed until 0.4.0) is told what changed; third-party components keep their licences."""
    apache = {"team_license": "Apache-2.0", "license_text": APACHE_TEXT}
    bundle = make_bundle(tmp_path / "submission", **apache)
    code, report = run(tool, capsys, "--bundle", str(bundle))
    team = item(report, "participant_module", "team")
    reasons = " ".join(team["reasons"])
    assert team["status"] == "denied" and code == 1
    for phrase in ("must be MIT", "D72", "Apache-2.0 is no longer accepted for the team's own work",
                   "participant_module/LICENSE"):  # fmt: skip
        assert phrase in reasons, (phrase, reasons)
    web = make_web(tmp_path / "web", license_text=APACHE_TEXT)
    (web / "package.json").write_text(json.dumps({"name": "ui", "license": "Apache-2.0"}))
    code, report = run(tool, capsys, "--web", str(web))
    team = item(report, "web system", "team")
    assert team["status"] == "denied" and code == 1
    assert "Apache-2.0 is no longer accepted" in " ".join(team["reasons"]), team
    missing = make_web(tmp_path / "no-licence", license_text=None)
    code, report = run(tool, capsys, "--web", str(missing))
    assert "add the MIT text" in " ".join(item(report, "web system", "team")["reasons"])


def test_weights_derived_from_a_catalogue_model_keep_its_licence(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D72: weights fine-tuned from a catalogue model are the team's own work under MIT and stay
    subject to that model's licence: the item says to keep its LICENSE and NOTICE."""
    bundle = make_bundle(
        tmp_path / "submission",
        models=[{"name": "ours", "team_trained": True, "base_model": QWEN_15B_ID,
                 "path": "assets"}],
    )  # fmt: skip
    code, report = run(tool, capsys, "--bundle", str(bundle))
    found = item(report, "ours", "model")
    conditions = " ".join(found["conditions"])
    assert found["status"] == "allowed" and code == 0, found
    assert QWEN_15B_ID in conditions and "Apache-2.0" in conditions, conditions
    assert "keep its LICENSE and NOTICE" in conditions, conditions


def test_bundle_without_a_readable_submission_json_is_denied(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    for broken in (None, "{not json", "[]"):
        if broken is None:
            (bundle / "submission.json").unlink()
        else:
            (bundle / "submission.json").write_text(broken)
        code, report = run(tool, capsys, "--bundle", str(bundle))
        assert code == 1 and item(report, "participant_module", "team")["status"] == "denied"


def test_team_licence_of_a_web_system(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    good = make_web(tmp_path / "good")
    (good / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    code, report = run(tool, capsys, "--web", str(good))
    team = item(report, "web system", "team")
    assert team["status"] == "allowed" and team["license"] == "MIT", team
    assert code == 0, statuses(report)

    missing = make_web(tmp_path / "missing", license_text=None)
    code, report = run(tool, capsys, "--web", str(missing))
    assert item(report, "web system", "team")["status"] == "denied" and code == 1

    other = make_web(tmp_path / "other")
    write_lock(other / "frontend", {"": {"name": "ui", "license": "GPL-3.0-only"}})
    (other / "frontend/package.json").write_text(
        json.dumps({"name": "ui", "license": "GPL-3.0-only"})
    )
    code, report = run(tool, capsys, "--web", str(other))
    team = item(report, "web system", "team")
    assert team["status"] == "denied" and "frontend/package.json" in " ".join(team["reasons"])
    assert code == 1

    # Repair round 1: MIT with an exception is not MIT (and Apache-2.0 is refused, D72).
    for index, declared in enumerate(("MIT WITH Commons-Clause", "Apache-2.0 WITH LLVM-exception")):
        excepted = make_web(tmp_path / f"exception-{index}")
        (excepted / "package.json").write_text(json.dumps({"name": "ui", "license": declared}))
        code, report = run(tool, capsys, "--web", str(excepted))
        team = item(report, "web system", "team")
        assert team["status"] == "denied" and declared in " ".join(team["reasons"]), team
        assert code == 1

    # Repair round 2: the legacy "licenses" list of the team's own package.json counts as well.
    for index, (types, expected) in enumerate(((["GPL-3.0"], "denied"), (["MIT"], "allowed"))):
        legacy = make_web(tmp_path / f"legacy-{index}")
        package = {"name": "app", "licenses": [{"type": t} for t in types]}
        (legacy / "package.json").write_text(json.dumps(package))
        write_lock(legacy, {"": {"name": "app"}})
        code, report = run(tool, capsys, "--web", str(legacy))
        team = item(report, "web system", "team")
        assert team["status"] == expected, team
        assert code == {"denied": 1, "allowed": 0}[expected], statuses(report)
        if expected == "denied":
            assert "GPL-3.0" in " ".join(team["reasons"]), team


# ---------------------------------------------------------------- models and data


@pytest.mark.parametrize(
    ("entry", "status"),
    [
        # D64: catalogue models by id; the catalogue holds repository, revision and licence.
        ({"catalog_id": QWEN_15B_ID}, "allowed"),
        ({"catalog_id": QWEN_15B_ID, "license": "Apache-2.0"}, "allowed"),
        ({"catalog_id": QWEN_15B_ID, "license": "MIT"}, "review"),  # contradicts the catalogue
        ({"catalog_id": "no-such-model"}, "denied"),
        ({"catalog_id": QWEN_15B_ID, "path": "assets/qwen"}, "denied"),  # public weights shipped
        # The same model named by repository and its catalogue revision is that catalogue model.
        ({"repo": QWEN_15B, "revision": QWEN_15B_REVISION, "license": "Apache-2.0"}, "allowed"),
        (
            {"repo": QWEN_15B, "revision": QWEN_15B_REVISION.upper(), "license": "apache-2.0"},
            "allowed",
        ),
        ({"repo": QWEN_15B, "revision": "0" * 40, "license": "Apache-2.0"}, "denied"),
        ({"repo": QWEN_15B, "license": "Apache-2.0"}, "denied"),
        # Any other public model is refused, whatever its licence (D64).
        (
            {
                "repo": "meta-llama/Llama-3.2-1B-Instruct",
                "revision": "9" * 40,
                "license": "llama3.2",
            },
            "denied",
        ),
        ({"repo": "Qwen/Qwen2.5-3B-Instruct", "revision": "a" * 40, "license": "other"}, "denied"),
        ({"repo": "google/gemma-2-2b-it", "revision": "1" * 40, "license": "gemma"}, "denied"),
        ({"repo": "datalab-to/surya-ocr-2", "revision": "1" * 40, "license": "other"}, "denied"),
        ({"repo": "example-org/permissive", "revision": "2" * 40, "license": "mit"}, "denied"),
        ({"repo": "example-org/bsd", "revision": "2" * 40, "license": "bsd-3-clause"}, "denied"),
        ({"repo": "example-org/nc", "revision": "3" * 40, "license": "cc-by-nc-4.0"}, "denied"),
        ({"repo": "example-org/rail", "revision": "4" * 40, "license": "openrail"}, "denied"),
        ({"repo": "example-org/unlabelled", "revision": "6" * 40}, "denied"),
        ({"repo": "example-org/branch", "revision": "main", "license": "MIT"}, "denied"),
        ({"license": "MIT"}, "denied"),
        # The deny list also applies to a model named by "name" or by a Hugging Face URL.
        (
            {
                "name": "my-llm",
                "url": "https://huggingface.co/google/gemma-2-2b-it/resolve/main/model.safetensors",
                "sha256": "b" * 64,
                "license": "apache-2.0",
            },
            "denied",
        ),
        (
            {
                "name": "detector",
                "url": "https://example.org/d.onnx",
                "sha256": "a" * 64,
                "license": "MIT",
            },
            "denied",
        ),
        # Repair round 1: every Gemma-licensed repository, whoever publishes it.
        (
            {"repo": "google/embeddinggemma-300m", "revision": "1" * 40, "license": "apache-2.0"},
            "denied",
        ),
        (
            {"repo": "unsloth/gemma-3-1b-it", "revision": "1" * 40, "license": "apache-2.0"},
            "denied",
        ),
        # Repair round 2: copies and derivatives of Llama published outside meta-llama/.
        ({"repo": "unsloth/Llama-3.2-1B-Instruct", "revision": "1" * 40}, "denied"),
        ({"repo": "unsloth/Meta-Llama-3.1-8B-Instruct", "revision": "1" * 40}, "denied"),
        ({"repo": "NousResearch/Hermes-3-Llama-3.1-8B", "revision": "1" * 40}, "denied"),
        ({"repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "revision": "1" * 40}, "denied"),
        (
            {
                "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "revision": "1" * 40,
                "license": "apache-2.0",
            },
            "denied",
        ),
        # Repair round 1: "team_trained" does not hide a third-party model or another licence.
        (
            {
                "repo": "meta-llama/Llama-3.1-8B-Instruct",
                "license": "llama3.1",
                "team_trained": True,
                "path": "assets/m",
            },
            "denied",
        ),
        (
            {
                "name": "my-llama-finetune",
                "url": "https://huggingface.co/meta-llama/Llama-3.2-1B",
                "team_trained": True,
                "path": "assets/m",
            },
            "denied",
        ),
        ({"name": "ours", "team_trained": True, "license": "cc-by-nc-4.0", "path": "a"}, "denied"),
        ({"name": "ours", "team_trained": True, "license": "BSD-3-Clause", "path": "a"}, "denied"),
        # D72: the team's own weights are MIT only, like its code.
        ({"name": "ours", "team_trained": True, "license": "Apache-2.0", "path": "a"}, "denied"),
        ({"name": "ours", "team_trained": True, "license": "MIT", "path": "assets"}, "allowed"),
        # D64: a fine-tune of a model outside the catalogue is refused, of a catalogue model fine.
        ({"name": "ours", "team_trained": True, "repo": "example-org/base", "path": "a"}, "denied"),
        (
            {
                "name": "ours",
                "team_trained": True,
                "url": "https://example.org/w.onnx",
                "path": "a",
            },
            "denied",
        ),
        ({"name": "ours", "team_trained": True, "base_model": "llama-3", "path": "a"}, "denied"),
        (
            {"name": "ours", "team_trained": True, "base_model": QWEN_15B_ID, "path": "assets"},
            "allowed",
        ),
        ({"name": "ours", "team_trained": True, "repo": QWEN_15B, "path": "assets"}, "allowed"),
    ],
)
def test_model_rule(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    entry: dict[str, Any],
    status: str,
) -> None:
    bundle = make_bundle(tmp_path / "submission", models=[entry])
    code, report = run(tool, capsys, "--bundle", str(bundle))
    models = [i for i in report["items"] if i["kind"] == "model"]
    assert len(models) == 1 and models[0]["status"] == status, models
    assert code == {"allowed": 0, "review": 2, "denied": 1}[status]


def test_model_copies_team_weights_and_undeclared_files(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path / "submission",
        models=[
            {"catalog_id": QWEN_15B_ID},
            {"catalog_id": QWEN_15B_ID, "path": "assets/with-licence"},
            {"name": "our-classifier", "team_trained": True, "path": "assets/ours/model.pt"},
            {"name": "our-other", "team_trained": True},
            {
                "name": "our-declared",
                "team_trained": True,
                "license": "mit",
                "path": "assets/ours/model.pt",
            },
            {
                "repo": "example-org/base-model",
                "team_trained": True,
                "path": "assets/ours/model.pt",
            },
        ],
    )
    (bundle / "assets/with-licence").mkdir()
    (bundle / "assets/with-licence/LICENSE").write_text(APACHE_TEXT)
    (bundle / "assets/with-licence/model.safetensors").write_bytes(b"weights")
    (bundle / "assets/ours").mkdir()
    (bundle / "assets/ours/model.pt").write_bytes(b"weights")
    (bundle / "assets/stray").mkdir()
    (bundle / "assets/stray/model.safetensors").write_bytes(b"weights")
    (bundle / "assets/bin").mkdir()
    (bundle / "assets/bin/libocr.so.1").write_bytes(b"\x7fELF")
    (bundle / "assets/notes.txt").write_text("plain text is fine")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    ours = item(report, "our-classifier", "model")
    assert ours["status"] == "allowed" and ours["license"] == "MIT", ours  # the team's licence
    assert item(report, "our-other", "model")["status"] == "review"  # needs a path
    declared = item(report, "our-declared", "model")
    assert declared["status"] == "allowed" and declared["license"] == "MIT", declared
    finetune = item(report, "example-org/base-model", "model")
    assert finetune["status"] == "denied", finetune
    assert "outside the model catalogue" in " ".join(finetune["reasons"]), finetune
    qwen = [i for i in report["items"] if i["kind"] == "model" and i["name"] == QWEN_15B_ID]
    assert [q["status"] for q in qwen] == ["allowed", "denied"], qwen
    assert qwen[0]["license"] == "Apache-2.0" and qwen[0]["origin"] == "model catalogue"
    assert qwen[0]["source"] == f"{QWEN_15B}@{QWEN_15B_REVISION}"
    assert "not shipped" in " ".join(qwen[1]["reasons"])
    assets = {i["name"]: i for i in report["items"] if i["kind"] == "asset"}
    assert set(assets) == {"assets/stray", "assets/bin"}, sorted(assets)
    assert all(a["status"] == "review" for a in assets.values())
    assert code == 1


def test_team_trained_names_of_third_party_model_families_are_review(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Repair round 2: {"name": "my-gemma-ft", "team_trained": true} gave exit 0, because the model
    # deny patterns need an organisation ("*/*gemma*", "meta-llama/*") and a bare name has none.
    bundle = make_bundle(
        tmp_path / "submission",
        models=[
            {"name": "my-gemma-ft", "team_trained": True, "path": "assets/own"},
            {"name": "Llama3-tuned", "team_trained": True, "license": "MIT", "path": "assets/own"},
            {"name": "exaone-distilled", "team_trained": True, "path": "assets/own"},
            {"name": "our-classifier", "team_trained": True, "path": "assets/own"},
            # D64: the declared catalogue base explains a matching family name.
            {
                "name": "my-qwen-ft",
                "team_trained": True,
                "base_model": QWEN_15B_ID,
                "path": "assets/own",
            },
            {
                "name": "my-gemma-lora",
                "team_trained": True,
                "base_model": QWEN_15B_ID,
                "path": "assets/own",
            },
        ],
    )
    (bundle / "assets/own").mkdir()
    (bundle / "assets/own/model.pt").write_bytes(b"weights")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    for name, family in (
        ("my-gemma-ft", "Gemma"),
        ("Llama3-tuned", "Llama"),
        ("exaone-distilled", "EXAONE"),
    ):
        found = item(report, name, "model")
        reasons = " ".join(found["reasons"])
        assert found["status"] == "review" and family in reasons, found
        assert "not team-trained" in reasons, found
    assert item(report, "our-classifier", "model")["status"] == "allowed"
    assert item(report, "my-qwen-ft", "model")["status"] == "allowed"
    gemma = item(report, "my-gemma-lora", "model")
    assert gemma["status"] == "review" and "Gemma" in " ".join(gemma["reasons"]), gemma
    assert code == 2


def test_report_text_of_a_denied_team_trained_entry(tmp_path: Path) -> None:
    # Repair round 2: a team-trained entry denied for naming meta-llama/ showed the team licence in
    # the licence column and "policy).; names a third-party model" in its reasons.
    bundle = make_bundle(
        tmp_path / "submission",
        team_license="Apache-2.0",
        license_text=APACHE_TEXT,
        models=[
            {
                "name": "ft",
                "team_trained": True,
                "repo": "meta-llama/Llama-3.2-1B",
                "path": "assets/own",
            }
        ],
    )
    (bundle / "assets/own").mkdir()
    (bundle / "assets/own/model.pt").write_bytes(b"weights")
    notices = tmp_path / "notices.md"
    result = cli("--bundle", str(bundle), "--notices", str(notices))
    assert result.returncode == 1, result.stdout
    row = next(line for line in result.stdout.splitlines() if "meta-llama/Llama-3.2-1B" in line)
    assert row.startswith("denied") and "Apache-2.0" not in row and ".;" not in row, row
    report = json.loads(cli("--bundle", str(bundle), "--json").stdout)
    model = item(report, "meta-llama/Llama-3.2-1B", "model")
    assert model["license"] == "", model
    not_cleared = notices.read_text(encoding="utf-8").split("## Not cleared", 1)[1]
    line = next(x for x in not_cleared.splitlines() if "meta-llama/Llama-3.2-1B" in x)
    assert "Apache-2.0" not in line and ".;" not in line, line


def test_dataset_licences(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path / "submission",
        declaration={
            "data_sources": [
                {"source": "Open corpus", "license": "CC-BY-4.0"},
                {"source": "Short-name corpus", "license": "CC BY 4.0"},
                {"source": "NC corpus", "license": "CC BY-NC 4.0"},
                {"source": "SA corpus", "license": "CC BY-SA 4.0"},
                {"source": "Free-text corpus", "license": "CC BY 4.0 for research"},
                {"source": "Undeclared corpus"},
            ]
        },
    )
    code, report = run(tool, capsys, "--bundle", str(bundle))
    found = {i["name"]: i["status"] for i in report["items"] if i["kind"] == "dataset"}
    assert found == {
        "Open corpus": "allowed",
        "Short-name corpus": "allowed",
        "NC corpus": "denied",
        "SA corpus": "review",
        "Free-text corpus": "review",
        "Undeclared corpus": "review",
    }
    assert code == 1
    (bundle / "declaration.json").unlink()
    code, report = run(tool, capsys, "--bundle", str(bundle))
    assert item(report, "declaration.json", "dataset")["status"] == "review" and code == 2


# ------------------------------------------------------------- bundle layout and vendoring


def test_virtual_environment_outside_the_package_is_a_denied_layout_item(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Review finding (d): README layout my_submission/.venv-worker/ is not part of the bundle.
    bundle = make_bundle(tmp_path / "submission")
    venv = bundle / ".venv-worker"
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    make_installed(site, "pylint", "4.0.0", license="GPL-2.0-or-later")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    layout = item(report, ".venv-worker", "layout")
    assert layout["status"] == "denied" and "D51" in " ".join(layout["reasons"]), layout
    assert not [i for i in report["items"] if i["name"] == "pylint"]  # not read
    assert code == 1


def test_virtual_environment_inside_wheels_is_a_denied_layout_item(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Repair round 1: wheels/ holds wheel files; an environment there is not delivered either.
    bundle = make_bundle(tmp_path / "submission")
    make_wheel(bundle / "wheels", "examplelib", "1.0", expression="MIT")
    venv = bundle / "wheels/.venv"
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    make_installed(site, "six", "1.17.0", license="MIT")
    loose = bundle / "wheels/site-packages"
    loose.mkdir()
    make_installed(loose, "pylint", "4.0.0", license="GPL-2.0-or-later")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    for name in ("wheels/.venv", "wheels/site-packages"):
        layout = item(report, name, "layout")
        assert layout["status"] == "denied" and "D51" in " ".join(layout["reasons"]), layout
    assert not [i for i in report["items"] if i["name"] in ("six", "pylint")]  # not read
    assert item(report, "examplelib")["status"] == "allowed"
    assert not [n for n in report["notes"] if "installed Python environment" in n]
    assert code == 1


def test_third_party_code_inside_participant_module_is_reviewed_by_its_metadata(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    vendor = bundle / "participant_module/_vendor"
    make_installed(vendor, "six", "1.17.0", license="MIT", record=("six.py",))
    make_installed(vendor, "borb", "3.0.9", classifiers=(AGPL_PLUS,))
    # Review finding (e): an unpacked sdist (PKG-INFO, setup.py, egg-info) is not team code.
    sdist = vendor / "tablib-3.8.0"
    (sdist / "tablib.egg-info").mkdir(parents=True)
    pkg_info = metadata("tablib", "3.8.0", None, "MIT", ())
    (sdist / "PKG-INFO").write_text(pkg_info)
    (sdist / "tablib.egg-info/PKG-INFO").write_text(pkg_info)
    (sdist / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    agpl = vendor / "agpltool-1.0"
    agpl.mkdir()
    (agpl / "PKG-INFO").write_text(metadata("agpltool", "1.0", "AGPL-3.0-only", None, ()))
    (agpl / "pyproject.toml").write_text("[project]\nname = 'agpltool'\n")
    shipped = bundle / "participant_module/.venv"
    (shipped / "lib/python3.12/site-packages").mkdir(parents=True)
    (shipped / "pyvenv.cfg").write_text("home = /usr/bin\n")
    make_installed(shipped / "lib/python3.12/site-packages", "pylint", "4.0.0", license="GPL-2.0")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    six = item(report, "six")
    assert six["status"] == "review" and six["license"] == "MIT"
    assert any("inside participant_module/" in reason for reason in six["reasons"]), six
    tablib = item(report, "tablib")
    assert tablib["status"] == "review" and tablib["license"] == "MIT", tablib
    assert item(report, "borb")["status"] == "denied"
    assert item(report, "agpltool")["status"] == "denied"
    assert item(report, "pylint")["status"] == "denied"
    assert any(".venv" in note for note in report["notes"]), report["notes"]
    assert code == 1


def test_wheels_anywhere_and_linked_wheels(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission", lock="examplelib==1.0\nmissinglib==2.0\n")
    make_wheel(bundle / "wheels", "examplelib", "1.0", expression="MIT")
    make_wheel(bundle / "wheels/extra", "PyMuPDF", "1.28.2", license="AGPL-3.0")
    make_wheel(bundle / "assets", "borb", "3.0.9", license="AGPL-3.0-or-later")
    outside = make_wheel(tmp_path / "outside", "konlpy", "0.6.0", license="GPL-3.0-only")
    (bundle / "wheels" / outside.name).symlink_to(outside)
    code, report = run(tool, capsys, "--bundle", str(bundle))
    assert item(report, "examplelib")["status"] == "allowed"
    for name in ("pymupdf", "borb"):
        assert item(report, name)["status"] == "denied", item(report, name)
    linked = item(report, "konlpy")
    assert linked["status"] == "review" and "symbolic link" in linked["reasons"][0], linked
    missing = item(report, "missinglib", "lock")
    assert missing["status"] == "review" and "no wheel" in missing["reasons"][0], missing
    assert code == 1


# ---------------------------------------------------------------- Stage 2 web: npm


def test_web_package_lock(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "node_modules/x").mkdir(parents=True)
    (web / "node_modules/x/package.json").write_text(json.dumps({"license": "BSD-3-Clause"}))
    write_lock(
        web,
        {
            "": {"name": "team-web", "version": "1.0.0", "license": "MIT"},
            "node_modules/react": {"version": "19.3.0", "license": "MIT"},
            "node_modules/react/node_modules/nested": {"version": "1.2.3", "license": "ISC"},
            "node_modules/@fortawesome/fontawesome-free": {
                "version": "7.3.1",
                "license": "(CC-BY-4.0 AND OFL-1.1 AND MIT)",
            },
            "node_modules/agpl-widget": {"version": "1.0.0", "license": "AGPL-3.0-only"},
            "node_modules/legacy": {
                "version": "1.0.0",
                "licenses": [{"type": "MIT"}, {"type": "Apache-2.0"}],
            },
            "node_modules/mystery": {"version": "0.1.0"},
            "node_modules/x": {"version": "1.0.0"},
            "node_modules/../../outside": {"version": "9.9.9"},
        },
    )
    code, report = run(tool, capsys, "--web", str(web))
    assert item(report, "react")["status"] == "allowed"
    assert item(report, "react")["source"] == "node_modules/react"
    assert item(report, "nested")["status"] == "allowed"
    awesome = item(report, "@fortawesome/fontawesome-free")
    assert awesome["status"] == "allowed" and awesome["conditions"], awesome
    assert item(report, "agpl-widget")["status"] == "denied"
    assert item(report, "legacy")["license"] == "MIT OR Apache-2.0"
    assert item(report, "mystery")["status"] == "review"
    x = item(report, "x")
    assert x["status"] == "allowed" and x["origin"] == "package.json"
    escaped = [i for i in report["items"] if i["version"] == "9.9.9"]
    assert len(escaped) == 1 and escaped[0]["status"] == "review"
    assert code == 1


DEV_LOCK = {
    "": {"name": "w", "license": "MIT", "devDependencies": {"gpl-build-tool": "2.0.0"}},
    "node_modules/react": {"version": "19.3.0", "license": "MIT"},
    "node_modules/gpl-build-tool": {"version": "2.0.0", "license": "GPL-3.0-or-later", "dev": True},
}


@pytest.mark.parametrize("layout", ["lock", "hidden-lock", "node_modules", "package.json"])
def test_dev_only_npm_packages_are_a_note_with_or_without_a_lock(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, layout: str
) -> None:
    # Review finding (g): the same project gives the same result however it is uploaded.
    web = make_web(tmp_path / "web")
    package = {
        "name": "w",
        "license": "MIT",
        "dependencies": {"react": "19.3.0"},
        "devDependencies": {"gpl-build-tool": "2.0.0"},
    }
    (web / "package.json").write_text(json.dumps(package))
    if layout == "lock":
        write_lock(web, DEV_LOCK)
    if layout == "hidden-lock":
        write_lock(web / "node_modules", DEV_LOCK)
        (web / "node_modules/package-lock.json").rename(web / "node_modules/.package-lock.json")
    if layout == "node_modules":
        for name, entry in DEV_LOCK.items():
            if name:
                folder = web / name
                folder.mkdir(parents=True)
                manifest = {"name": name.split("/")[-1], **entry}
                (folder / "package.json").write_text(json.dumps(manifest))
    code, report = run(tool, capsys, "--web", str(web))
    assert not [i for i in report["items"] if i["name"] == "gpl-build-tool"], report["items"]
    note = next(n for n in report["notes"] if "dev-only" in n)
    assert "gpl-build-tool" in note, note
    if layout != "package.json":  # without a lock or node_modules/ the licence is not known
        assert "GPL-3.0-or-later: denied" in note, note
    assert item(report, "react")["kind"] == "npm"
    # Repair round 2: a dev-only package that is not allowed was only a note (exit 0); a build
    # may bundle it (Vite, SvelteKit), so a person confirms it is not shipped.
    dev = item(report, "devDependencies", "npm")
    assert dev["status"] == "review" and "gpl-build-tool" in " ".join(dev["reasons"]), dev
    assert "bundle" in " ".join(dev["reasons"]), dev
    assert code == 2, statuses(report)


def test_allowed_dev_only_npm_packages_stay_a_note(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(
        json.dumps({"name": "w", "license": "MIT", "devDependencies": {"vite": "7.1.0"}})
    )
    write_lock(
        web,
        {
            "": {"name": "w", "license": "MIT", "devDependencies": {"vite": "7.1.0"}},
            "node_modules/vite": {"version": "7.1.0", "license": "MIT", "dev": True},
        },
    )
    code, report = run(tool, capsys, "--web", str(web))
    assert any("dev-only" in note for note in report["notes"]), report["notes"]
    assert not [i for i in report["items"] if i["kind"] == "npm"], report["items"]
    assert code == 0, statuses(report)


def test_npm_workspaces_are_read_once(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Review finding (h): a workspace package is listed in the root lock and has its own
    # package.json; it is the team's own code and its dependencies are counted once.
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(
        json.dumps({"name": "root", "license": "MIT", "workspaces": ["packages/*"]})
    )
    (web / "packages/ui").mkdir(parents=True)
    (web / "packages/ui/package.json").write_text(
        json.dumps({"name": "team-ui", "license": "MIT", "dependencies": {"react": "19.3.0"}})
    )
    write_lock(
        web,
        {
            "": {"name": "root", "license": "MIT", "workspaces": ["packages/*"]},
            "packages/ui": {"name": "team-ui", "version": "1.0.0", "license": "MIT"},
            "node_modules/team-ui": {"resolved": "packages/ui", "link": True},
            "node_modules/react": {"version": "19.3.0", "license": "MIT"},
        },
    )
    code, report = run(tool, capsys, "--web", str(web))
    names = [i["name"] for i in report["items"] if i["kind"] == "npm"]
    assert names == ["react"], report["items"]
    assert item(report, "web system", "team")["status"] == "allowed"
    assert code == 0, statuses(report)
    # A workspace under a licence other than MIT is the team's licence problem (D72).
    lock = json.loads((web / "package-lock.json").read_text())
    lock["packages"]["packages/ui"]["license"] = "BSD-3-Clause"
    (web / "package-lock.json").write_text(json.dumps(lock))
    code, report = run(tool, capsys, "--web", str(web))
    assert item(report, "web system", "team")["status"] == "denied" and code == 1


def test_vendored_package_folders_are_third_party_items(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    write_lock(web, {"": {"name": "ui", "license": "MIT"}})
    copied = web / "public/pdfjs"  # inside a locked project, not installed by its lock
    copied.mkdir(parents=True)
    # Repair round 2: the real pdfjs-dist 6.4.299 manifest, with its optionalDependencies (the
    # synthetic one without them hid that the folder was read as the team's sub-project).
    (copied / "package.json").write_text(json.dumps(MANIFESTS["pdfjs-dist"], indent=2))
    (copied / "pdf.min.mjs").write_text("/* pdf.js */\n")
    gpl = web / "static/vendor/gplwidget"  # under a vendor-like folder
    gpl.mkdir(parents=True)
    (gpl / "package.json").write_text(
        json.dumps({"name": "gplwidget", "version": "1.0.0", "license": "GPL-3.0-only"})
    )
    # A copied library that declares only peer dependencies is not the team's sub-project either.
    viewer = web / "public/viewer"
    viewer.mkdir(parents=True)
    (viewer / "package.json").write_text(
        json.dumps(
            {
                "name": "pdf-viewer-lite",
                "version": "2.0.0",
                "license": "MIT",
                "peerDependencies": {"react": ">=18"},
            }
        )
    )
    code, report = run(tool, capsys, "--web", str(web))
    pdfjs = item(report, "pdfjs-dist")
    assert pdfjs["status"] == "allowed" and pdfjs["source"] == "public/pdfjs/package.json"
    assert pdfjs["origin"] == "vendored package.json", pdfjs
    assert any("LICENSE_LIBERATION" in condition for condition in pdfjs["conditions"]), pdfjs
    canvas = item(report, "@napi-rs/canvas", "npm")
    assert canvas["status"] == "review" and canvas["source"] == "public/pdfjs/package.json"
    viewer_item = item(report, "pdf-viewer-lite", "npm")
    assert viewer_item["status"] == "allowed" and viewer_item["origin"] == "vendored package.json"
    assert item(report, "react", "npm")["status"] == "review"
    assert item(report, "gplwidget")["status"] == "denied"
    assert item(report, "web system", "team")["status"] == "allowed"  # not the team's licence
    assert not [i for i in report["items"] if i["kind"] == "asset"]  # covered by package.json
    assert code == 1


@pytest.mark.parametrize("root_lock", [True, False])
@pytest.mark.parametrize("parent", ["public", "static/js"])
def test_copied_packages_the_policy_names_are_third_party_items_anywhere(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    root_lock: bool,
    parent: str,
) -> None:
    # Repair round 2: the real apexcharts 5.0.0 and handsontable 6.2.2 manifests (MIT, with
    # dependencies) copied outside a vendor-like folder were read as the team's sub-projects, so
    # the denied packages were never listed (exit 2); pdfjs-dist lost its licence-file condition.
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    if root_lock:
        write_lock(web, {"": {"name": "ui", "license": "MIT"}})
    for name, manifest in MANIFESTS.items():
        (web / parent / name).mkdir(parents=True)
        (web / parent / name / "package.json").write_text(json.dumps(manifest, indent=2))
    notices = tmp_path / "notices.md"
    code, report = run(tool, capsys, "--web", str(web), "--notices", str(notices))
    for name, version, status in (
        ("apexcharts", "5.0.0", "denied"),
        ("handsontable", "6.2.2", "denied"),
        ("pdfjs-dist", "6.4.299", "allowed"),
    ):
        found = item(report, name, "npm")
        assert (found["version"], found["status"]) == (version, status), found
        assert found["source"] == f"{parent}/{name}/package.json", found
        assert found["origin"] == "vendored package.json", found
    pdfjs = item(report, "pdfjs-dist", "npm")
    assert any("LICENSE_LIBERATION" in condition for condition in pdfjs["conditions"]), pdfjs
    for dependency in ("@svgdotjs/svg.js", "@yr/monotone-cubic-spline", "moment", "pikaday"):
        found = item(report, dependency, "npm")
        assert found["status"] == "review" and "copied package" in found["reasons"][0], found
    assert not [i for i in report["items"] if i["name"] in ("webpack", "jest")]  # dev tooling
    assert item(report, "web system", "team")["status"] == "allowed", statuses(report)
    assert "| pdfjs-dist | 6.4.299 | Apache-2.0 |" in notices.read_text(encoding="utf-8")
    assert code == 1


def test_name_rules_apply_to_the_own_package_json_of_every_project_folder(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A copied package that brought its own node_modules/ is read as a project; its own name and
    # version still go through the name rules.
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    write_lock(web, {"": {"name": "ui", "license": "MIT"}})
    copied = web / "public/grid"
    (copied / "node_modules/moment").mkdir(parents=True)
    (copied / "package.json").write_text(json.dumps(MANIFESTS["handsontable"], indent=2))
    (copied / "node_modules/moment/package.json").write_text(
        json.dumps({"name": "moment", "version": "2.20.1", "license": "MIT"})
    )
    code, report = run(tool, capsys, "--web", str(web))
    grid = item(report, "handsontable", "npm")
    assert grid["status"] == "denied" and grid["source"] == "public/grid/package.json", grid
    assert "copied" in " ".join(grid["reasons"]), grid
    assert item(report, "moment", "npm")["status"] == "allowed"
    assert code == 1


def test_sub_project_licence_reason_names_the_vendor_folder_alternative(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A copied ISC package with dependencies outside vendor/ is read as the team's sub-project;
    # the refusal says how to mark it as a copied package instead.
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    write_lock(web, {"": {"name": "ui", "license": "MIT"}})
    (web / "public/isclib").mkdir(parents=True)
    (web / "public/isclib/package.json").write_text(
        json.dumps(
            {"name": "isclib", "version": "1.0.0", "license": "ISC", "dependencies": {"a": "1"}}
        )
    )
    code, report = run(tool, capsys, "--web", str(web))
    team = item(report, "web system", "team")
    reasons = " ".join(team["reasons"])
    assert team["status"] == "denied" and "public/isclib/package.json" in reasons, team
    assert "vendor/" in reasons, team
    assert code == 1


@pytest.mark.parametrize("frontend_lock", ["yarn.lock", "pnpm-lock.yaml", "private", "none"])
def test_sub_project_without_an_npm_lock_is_the_teams_project(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, frontend_lock: str
) -> None:
    # Repair round 1: a root package-lock.json used only for tooling made frontend/ (no npm lock
    # of its own) look like a copied third-party package: its licence was not checked as the
    # team's and its dependencies (Highcharts, ApexCharts 5) were never listed.
    web = make_web(tmp_path / "web")
    root = {"name": "tooling", "private": True, "license": "MIT", "devDependencies": {"husky": "9"}}
    (web / "package.json").write_text(json.dumps(root))
    write_lock(
        web,
        {
            "": {"name": "tooling", "license": "MIT", "devDependencies": {"husky": "9"}},
            "node_modules/husky": {"version": "9.1.7", "license": "MIT", "dev": True},
        },
    )
    frontend = web / "frontend"
    frontend.mkdir()
    package: dict[str, Any] = {
        "name": "ui",
        "version": "0.1.0",
        "license": "BSD-3-Clause",
        "dependencies": {
            "highcharts": "^11.4.0",
            "apexcharts": "^5.3.0",
            "ckeditor5": "^47.0.0",
            "react": "19.3.0",
        },
    }
    if frontend_lock == "private":
        package["private"] = True
    elif frontend_lock != "none":
        (frontend / frontend_lock).write_text("# lockfile\n")
    (frontend / "package.json").write_text(json.dumps(package))
    code, report = run(tool, capsys, "--web", str(web))
    for name in ("highcharts", "apexcharts", "ckeditor5"):
        found = item(report, name, "npm")
        assert found["status"] == "denied" and found["source"] == "frontend/package.json", found
    assert item(report, "react", "npm")["status"] == "review"  # licence not read without a lock
    team = item(report, "web system", "team")
    reasons = " ".join(team["reasons"])
    assert team["status"] == "denied" and "frontend/package.json" in reasons, team
    assert "BSD-3-Clause" in reasons
    assert not [i for i in report["items"] if i["name"] == "ui"]  # not a third-party item
    if frontend_lock in ("yarn.lock", "pnpm-lock.yaml"):
        unread = item(report, f"frontend/{frontend_lock}", "npm")
        assert unread["status"] == "review" and "not read" in unread["reasons"][0], unread
    assert code == 1


def test_dependencies_of_a_vendored_package_are_listed(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Repair round 1: plugins/my-plugin (a vendor-like folder, no lock) depended on ckeditor5 and
    # the check exited 0, because only the folder's own "license" was read.
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "w", "license": "MIT"}))
    plugin = web / "plugins/my-plugin"
    plugin.mkdir(parents=True)
    manifest = {
        "name": "my-plugin",
        "version": "1.0.0",
        "license": "MIT",
        "dependencies": {"ckeditor5": "^47.0.0", "chart.js": "4.5.0"},
        "peerDependencies": {"highcharts": ">=11"},
        "devDependencies": {"gpl-build-tool": "2.0.0"},
    }
    (plugin / "package.json").write_text(json.dumps(manifest))
    code, report = run(tool, capsys, "--web", str(web))
    vendored = item(report, "my-plugin", "npm")
    assert vendored["status"] == "allowed" and vendored["origin"] == "vendored package.json"
    for name in ("ckeditor5", "highcharts"):
        found = item(report, name, "npm")
        assert found["status"] == "denied", found
        assert found["source"] == "plugins/my-plugin/package.json", found
    chart = item(report, "chart.js", "npm")
    assert chart["status"] == "review" and chart["version"] == "4.5.0", chart
    assert not [i for i in report["items"] if i["name"] == "gpl-build-tool"]  # not shipped
    assert code == 1

    # A dependency that the enclosing project's lock already lists is not repeated.
    write_lock(
        web,
        {
            "": {"name": "w", "license": "MIT"},
            "node_modules/chart.js": {"version": "4.5.0", "license": "MIT"},
        },
    )
    code, report = run(tool, capsys, "--web", str(web))
    chart = item(report, "chart.js", "npm")
    assert chart["status"] == "allowed" and chart["source"] == "node_modules/chart.js", chart
    assert item(report, "ckeditor5", "npm")["status"] == "denied" and code == 1


def test_web_without_a_lock(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(
        json.dumps(
            {
                "name": "w",
                "license": "MIT",
                "dependencies": {"highcharts": "^11.4.0", "chart.js": "4.5.0"},
            }
        )
    )
    (web / "yarn.lock").write_text("# yarn lockfile v1\n")
    code, report = run(tool, capsys, "--web", str(web))
    assert item(report, "highcharts")["status"] == "denied"
    chart = item(report, "chart.js")
    assert chart["status"] == "review" and chart["version"] == "4.5.0"
    assert item(report, "yarn.lock")["status"] == "review"
    assert code == 1


def test_symbolic_links_in_node_modules(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "package.json").write_text(json.dumps({"name": "w", "license": "MIT"}))
    (tmp_path / "store/highcharts").mkdir(parents=True)
    (web / "node_modules").mkdir()
    (web / "node_modules/highcharts").symlink_to(tmp_path / "store/highcharts")
    code, report = run(tool, capsys, "--web", str(web))
    links = item(report, "node_modules/", "npm")
    assert links["status"] == "review" and "symbolic link" in links["reasons"][0], links
    assert code == 2


# --------------------------------------------------------- Stage 2 web: Python and files


def test_web_python_requirements_and_environments(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "backend").mkdir()
    (web / "backend/requirements.txt").write_text(
        "# web backend\n"
        "fastapi==0.135.0 \\\n    --hash=sha256:00\n"
        "uvicorn[standard]==0.30.1 ; python_version >= '3.10'\n"
        "httpx>=0.27\n"
        "pymupdf\n"
        "-r extra.txt\n"
        "-e git+https://example.org/team/helper.git#egg=helper\n"
        "pdfkit @ https://example.org/pdfkit-1.0-py3-none-any.whl\n"
        "-e .\n"
        "--index-url https://example.org/simple\n"
    )
    (web / "backend/extra.txt").write_text("unidecode==1.3.8\n")
    make_wheel(web / "backend/wheels", "fastapi", "0.135.0", expression="MIT")
    (web / "svc").mkdir()
    (web / "svc/uv.lock").write_text('version = 1\n[[package]]\nname = "magic-pdf"\n')
    (web / "svc/pyproject.toml").write_text('[project]\nname = "svc"\n')
    venv = web / ".venv"
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    make_installed(site, "borb", "3.0.9", classifiers=(AGPL_PLUS,))
    make_installed(site, "team-web", "0.1.0", editable=True)  # the team's own project
    egg = web / "backend/team_web.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text(metadata("team-web", "0.1.0", None, None, ()))
    (web / "backend/pyproject.toml").write_text('[project]\nname = "team-web"\n')
    code, report = run(tool, capsys, "--web", str(web))
    assert item(report, "fastapi")["status"] == "allowed"
    assert item(report, "uvicorn")["version"] == "0.30.1"
    httpx = item(report, "httpx")
    assert httpx["status"] == "review" and any("not pinned" in r for r in httpx["reasons"])
    for name in ("pymupdf", "unidecode", "borb"):
        assert item(report, name)["status"] == "denied", item(report, name)
    assert item(report, "helper")["status"] == "review"
    assert "direct URL" in item(report, "pdfkit")["reasons"][0]
    for name in ("svc/uv.lock", "svc/pyproject.toml"):
        unread = item(report, name, "lock")
        assert unread["status"] == "review" and "not read" in unread["reasons"][0], unread
    assert not [i for i in report["items"] if i["name"] in ("team-web", "backend/pyproject.toml")]
    assert any(".venv" in note for note in report["notes"]), report["notes"]
    assert code == 1


def test_installed_environments_are_left_out_and_web_packages_come_from_the_base(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D59/D64: nothing ships packages, so an installed environment is to be removed (not
    replaced by wheels and a lock file, the D51 advice), and a web system uses only the packages
    of the common base: an allowed licence outside the base is review."""
    web = make_web(tmp_path / "web")
    site = web / ".venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (web / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    make_installed(site, "rank_bm25", "0.2.2", license="Apache-2.0")
    make_installed(site, "jinja2", "3.1.6", license="BSD-3-Clause")
    code, report = run(tool, capsys, "--web", str(web))
    bm25 = item(report, "rank-bm25")
    assert bm25["status"] == "review", bm25
    assert any("not in the common base" in r for r in bm25["reasons"]), bm25
    assert item(report, "jinja2")["status"] == "allowed"
    bundle = make_bundle(tmp_path / "submission")
    venv = bundle / "participant_module/.venv"
    (venv / "lib/python3.12/site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    _, bundled = run(tool, capsys, "--bundle", str(bundle))
    for notes, where in ((report["notes"], "the web system"), (bundled["notes"], "the upload")):
        text = " ".join(notes)
        assert f"remove it from {where}" in text and "ships no packages (D64)" in text, notes
        assert "Ship wheels" not in text and "lock file" not in text
    assert code == 2


def test_a_catalogue_id_named_by_a_team_trained_entry_is_its_base(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A team-trained entry that names a catalogue model by catalog_id (qwen3-0.6b) is not
    "from a public model outside the catalogue"; the checker asks for base_model instead."""
    bundle = make_bundle(
        tmp_path / "submission",
        models=[{"catalog_id": "qwen3-0.6b", "name": "our-tuned", "team_trained": True,
                 "path": "assets/own"}],
    )  # fmt: skip
    (bundle / "assets/own").mkdir()
    (bundle / "assets/own/adapter.safetensors").write_bytes(b"weights")
    code, report = run(tool, capsys, "--bundle", str(bundle))
    found = item(report, "qwen3-0.6b", "model")
    text = " ".join(found["reasons"] + found["conditions"])
    assert "outside the catalogue" not in text, found
    assert '"base_model": "qwen3-0.6b"' in text, found
    assert found["status"] == "allowed", found


def test_plain_front_end_files(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    web = make_web(tmp_path / "web")
    (web / "static/js").mkdir(parents=True)
    (web / "static/js/app.js").write_text("console.log('team code');\n")  # team's own source
    (web / "static/js/jquery.min.js").write_text("/*! jQuery */\n")
    (web / "static/css").mkdir(parents=True)
    (web / "static/css/site.css").write_text("body {}\n")
    (web / "static/lib/chart").mkdir(parents=True)
    (web / "static/lib/chart/chart.umd.js").write_text("/*! Chart.js */\n")
    (web / "static/lib/chart/LICENSE.md").write_text(MIT_TEXT)
    (web / "dist/assets").mkdir(parents=True)
    (web / "dist/assets/index-4f2a9c.js").write_text("/* build output, no package.json */\n")
    frontend = web / "frontend"
    (frontend / "src/lib").mkdir(parents=True)
    (frontend / "src/lib/util.js").write_text("export const x = 1;\n")  # team source, npm project
    (frontend / "public").mkdir()
    (frontend / "public/plotly.min.js").write_text("/* copied */\n")
    (frontend / "dist").mkdir()
    (frontend / "dist/index.min.js").write_text("/* build output of the npm project */\n")
    (frontend / "package.json").write_text(json.dumps({"name": "ui", "license": "MIT"}))
    write_lock(frontend, {"": {"name": "ui", "license": "MIT"}})
    code, report = run(tool, capsys, "--web", str(web))
    assets = {i["name"]: i for i in report["items"] if i["kind"] == "asset"}
    assert set(assets) == {"static/js", "static/lib/chart", "dist/assets", "frontend/public"}, (
        sorted(assets)
    )
    assert "jquery.min.js" in assets["static/js"]["reasons"][0]
    assert "app.js" not in assets["static/js"]["reasons"][0]
    assert all(a["status"] == "review" for a in assets.values())
    assert code == 2


# ---------------------------------------------------------------------- CLI behaviour


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_exit_codes_table_and_notices(tmp_path: Path) -> None:
    clean = make_bundle(tmp_path / "clean")
    make_wheel(clean / "wheels", "examplelib", "1.0", expression="MIT")
    review = make_bundle(tmp_path / "review")
    make_wheel(review / "wheels", "mysterylib", "0.1")
    denied = make_bundle(tmp_path / "denied")
    make_wheel(denied / "wheels", "copyleftpdf", "1.0", expression="GPL-3.0-only")
    make_wheel(denied / "wheels", "okpdf", "1.0", expression="BSD-3-Clause")

    ok = cli("--bundle", str(clean))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "legal advice" not in ok.stdout  # D79: no disclaimer wording
    assert "HQ decides flagged items after submission" in ok.stdout
    assert re.search(r"^status\s+kind\s+component\s+version\s+licence", ok.stdout, re.M)
    assert re.search(r"^allowed\s+wheel\s+examplelib\s+1\.0\s+MIT", ok.stdout, re.M)
    assert "summary: 2 allowed, 0 review, 0 denied -> exit 0" in ok.stdout

    assert cli("--bundle", str(review)).returncode == 2
    notices = tmp_path / "out" / "THIRD_PARTY_NOTICES.md"
    bad = cli("--bundle", str(denied), "--notices", str(notices))
    assert bad.returncode == 1 and "copyleftpdf" in bad.stdout
    draft = notices.read_text(encoding="utf-8")
    tables, not_cleared = draft.split("## Not cleared")
    assert "| okpdf | 1.0 | BSD-3-Clause |" in tables
    assert "copyleftpdf" not in tables and "| copyleftpdf | 1.0 | GPL-3.0-only |" in not_cleared
    assert "participant_module" not in draft  # the team's own code is not a third-party notice

    both = cli("--bundle", str(clean), "--web", str(make_web(tmp_path / "web")), "--json")
    assert both.returncode == 0, both.stdout
    report = json.loads(both.stdout)
    assert {i["scope"] for i in report["items"]} == {"bundle", "web"}
    assert report["policy"]["version"] == json.loads(POLICY.read_text())["version"]
    assert report["policy"]["version"] == "0.4.0" and report["exit_code"] == 0
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert report["catalog"]["version"] == catalog["catalog_version"]
    assert len(report["catalog"]["sha256"]) == 64

    assert cli().returncode == 3  # no input
    assert cli("--bundle", str(tmp_path / "missing")).returncode == 3
    assert cli("--bundle", str(clean), "--unknown-option").returncode == 3
    broken = tmp_path / "broken-policy.json"
    broken.write_text("{}")
    assert cli("--bundle", str(clean), "--policy", str(broken)).returncode == 3


def test_checker_never_opens_a_network_connection(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert not imported & {"socket", "ssl", "urllib", "http", "requests", "httpx", "ftplib"}

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    bundle = make_bundle(tmp_path / "submission")
    make_wheel(bundle / "wheels", "examplelib", "1.0", expression="MIT")
    code, _ = run(tool, capsys, "--bundle", str(bundle))
    assert code == 0


def test_untrusted_text_is_sanitised(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path / "submission")
    make_wheel(bundle / "wheels", "pipe|name", "1.0", license="MIT \x1b[31mred\x07")
    notices = tmp_path / "notices.md"
    code = tool.main(["--bundle", str(bundle), "--policy", str(POLICY), "--notices", str(notices)])
    table = capsys.readouterr().out
    assert "\x1b" not in table and "\x07" not in table
    draft = notices.read_text(encoding="utf-8")
    assert "\x1b" not in draft and "pipe\\|name" in draft
    assert code == 2


def test_licensing_guide_example_is_accepted(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    text = GUIDE.read_text(encoding="utf-8")
    match = re.search(r"<!-- example: submission-licence -->\n```json\n(.*?)```", text, re.S)
    assert match, "LICENSING.md lost its submission.json example"
    submission = json.loads(match.group(1))
    bundle = make_bundle(tmp_path / "submission", submission=submission)
    for model in submission["models"]:
        if "path" in model:
            (bundle / model["path"]).mkdir(parents=True)
            (bundle / model["path"] / "LICENSE").write_text(APACHE_TEXT)
    code, report = run(tool, capsys, "--bundle", str(bundle))
    assert code == 0, statuses(report)
    assert "python tools/check_licenses.py --bundle" in text
    for option in ("--web", "--json", "--notices"):
        assert option in text


def test_the_licence_file_must_hold_the_mit_text_d72(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D72: with "license": "MIT", participant_module/LICENSE (and the LICENSE at the root of the
    web code) must contain the MIT text: its permission, condition and warranty paragraphs, matched
    after normalising case, quotes, punctuation and white space. The Apache-2.0 text is refused
    with a message that says so; the MIT text followed by the Apache-2.0 text is a review item."""
    bundle = make_bundle(tmp_path / "apache", license_text=APACHE_TEXT)
    code, report = run(tool, capsys, "--bundle", str(bundle))
    team = item(report, "participant_module", "team")
    reasons = " ".join(team["reasons"])
    assert team["status"] == "denied" and code == 1
    for phrase in ("participant_module/LICENSE holds the Apache License 2.0 text", "D72",
                   "licenses/MIT.txt"):  # fmt: skip
        assert phrase in reasons, (phrase, reasons)
    other = make_bundle(tmp_path / "other", license_text="All rights reserved by Team 01.\n")
    code, report = run(tool, capsys, "--bundle", str(other))
    reasons = " ".join(item(report, "participant_module", "team")["reasons"])
    assert code == 1 and "does not contain the MIT licence text" in reasons, reasons
    both = make_bundle(tmp_path / "both", license_text=MIT_TEXT + "\n" + APACHE_TEXT)
    code, report = run(tool, capsys, "--bundle", str(both))
    team = item(report, "participant_module", "team")
    assert team["status"] == "review" and code == 2, team
    assert "also holds the Apache License 2.0 text" in " ".join(team["reasons"])
    web = make_web(tmp_path / "web", license_text=APACHE_TEXT)
    code, report = run(tool, capsys, "--web", str(web))
    team = item(report, "web system", "team")
    assert team["status"] == "denied" and code == 1
    assert "web code holds the Apache License 2.0 text" in " ".join(team["reasons"]), team
    good = make_web(tmp_path / "good-web")
    code, report = run(tool, capsys, "--web", str(good))
    assert item(report, "web system", "team")["status"] == "allowed", report


def test_a_utf16_licence_file_is_read_and_other_encodings_are_named(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D72 text check: an editor may save the MIT text as UTF-16 with a byte order mark (Windows
    Notepad "Unicode"); it is read as such and passes. A file that still does not hold the MIT text
    is refused with a message that names the expected encoding (UTF-8 text)."""
    for name, data in (
        ("utf16-le", MIT_TEXT.replace("\n", "\r\n").encode("utf-16")),
        ("utf16-be", "﻿".encode("utf-16-be") + MIT_TEXT.encode("utf-16-be")),
        ("utf8-bom", "﻿".encode() + MIT_TEXT.replace("\n", "\r\n").encode()),
    ):
        bundle = make_bundle(tmp_path / name)
        (bundle / "participant_module" / "LICENSE").write_bytes(data)
        code, report = run(tool, capsys, "--bundle", str(bundle))
        assert item(report, "participant_module", "team")["status"] == "allowed", (name, report)
        assert code == 0, name
        web = make_web(tmp_path / f"web-{name}")
        (web / "LICENSE").write_bytes(data)
        code, report = run(tool, capsys, "--web", str(web))
        assert item(report, "web system", "team")["status"] == "allowed", (name, report)
    other = make_bundle(tmp_path / "latin-1")
    (other / "participant_module" / "LICENSE").write_bytes("Licence \xe9crite.\n".encode("latin-1"))
    code, report = run(tool, capsys, "--bundle", str(other))
    reasons = " ".join(item(report, "participant_module", "team")["reasons"])
    assert code == 1 and "does not contain the MIT licence text" in reasons, reasons
    assert "saved as UTF-8 text" in reasons, reasons
