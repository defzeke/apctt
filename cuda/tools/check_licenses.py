"""Flag the licences in a Stage 1 bundle or a Stage 2 web system against the licence policy.

    python tools/check_licenses.py --bundle SUBMISSION_DIR [--web WEB_DIR]
                                   [--policy contracts/license-policy.json]
                                   [--json] [--notices THIRD_PARTY_NOTICES.md]

Licences come only from declared metadata: Python License-Expression, else "License ::"
classifiers, else the License field; npm "license"; submission.json models; declaration.json
data sources. A declared string is read through the policy's exact table or as an SPDX
expression; anything else is review. Licence texts and file contents are not read, with one
exception: the team's own licence file (participant_module/LICENSE of a bundle, the LICENSE at the
root of a web system) must hold the MIT text (decision D72): its permission, condition and warranty
paragraphs, compared after case, quotes, punctuation and white space are normalised; the Apache
License 2.0 text there is denied with a message that says so. Name rules
in the policy flag known problem packages whatever their metadata says. Public models: only the
model catalogue (contracts/model-catalog.json, decision D64) and team-trained weights; any other
public model is denied. HQ decides review items after the submission (D63).

Exit codes: 0 everything allowed, 1 something denied, 2 only review items, 3 usage or input
error. Standard library only, Python 3.10+, no network. It flags components from their declared
metadata; HQ decides flagged items after submission.
"""

from __future__ import annotations

import argparse
import codecs
import email.parser
import fnmatch
import hashlib
import json
import os
import re
import sys
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import model_catalog  # noqa: E402 - the kit's catalogue loader (D64)

DEFAULT_POLICY = TOOLS.parent / "contracts" / "license-policy.json"
DEFAULT_CATALOG = model_catalog.CATALOG
REPORT_VERSION = "apctt-license-report-v2"
NOTICE = "Flags components from their declared metadata; HQ decides flagged items after submission."
ALLOWED, REVIEW, DENIED = "allowed", "review", "denied"
RANK = {ALLOWED: 0, REVIEW: 1, DENIED: 2}
KINDS = ("team", "layout", "wheel", "lock", "npm", "model", "dataset", "asset")
MISSING = frozenset({"", "unknown", "none", "noassertion", "n/a", "undefined"})
CHECKER = "checker"  # origin of items about the upload itself rather than a component
MAX_TEXT = 4 * 1024 * 1024
MAX_JSON = 64 * 1024 * 1024
MAX_FOLDERS = 200
NPM_LOCKS = ("package-lock.json", "npm-shrinkwrap.json", "node_modules/.package-lock.json")
NPM_FILES = frozenset({"package.json", "package-lock.json", "npm-shrinkwrap.json"})
OTHER_JS_LOCKS = ("bun.lock", "bun.lockb", "pnpm-lock.yaml", "yarn.lock")
LIBRARY = re.compile(r"\.(?:so(?:\.\d+)*|dll|dylib|exe)$", re.I)
HF_URL = re.compile(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([^/\s?#]+/[^/\s?#]+)", re.I)
GENERIC_CLASSIFIERS = frozenset({"License :: OSI Approved", "License :: DFSG approved"})
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;]*)$")
RUNTIME_SECTIONS = ("dependencies", "optionalDependencies", "peerDependencies")
SECTIONS = (*RUNTIME_SECTIONS, "devDependencies")


class InputError(Exception):
    """A problem with the command line, the policy or an input directory (exit code 3)."""


def clean(value: object, limit: int = 300) -> str:
    """Untrusted text for terminal, JSON and Markdown: printable characters only, bounded."""
    text = value if isinstance(value, str) else "" if value is None else str(value)
    text = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def worst(*statuses: str) -> str:
    return max(statuses, key=RANK.__getitem__)


def canonical_pypi(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def matches(name: str, patterns: Iterable[str]) -> bool:
    """Exact names, glob patterns, and npm scope prefixes ending in "/" ("@ckeditor/")."""
    name = name.lower()
    return any(
        name.startswith(p.lower()) if p.endswith("/") else fnmatch.fnmatchcase(name, p.lower())
        for p in patterns
    )


def at_least(version: str, minimum: str) -> bool:
    """version >= minimum by release numbers; an unknown version counts as matching."""

    def release(text: str) -> list[int]:
        found = re.match(r"v?(\d+(?:\.\d+)*)", text.strip())
        return [int(part) for part in found.group(1).split(".")] if found else []

    have, need = release(version), release(minimum)
    width = max(len(have), len(need))
    return not have or (have + [0] * width)[:width] >= (need + [0] * width)[:width]


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def inside(root: Path, relative: str) -> Path | None:
    """root/relative when it stays inside root (symbolic links resolved), else None."""
    if not relative or relative.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", relative):
        return None
    real_root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(real_root, relative))
    if candidate != real_root and not candidate.startswith(real_root + os.sep):
        return None
    return Path(candidate)


def read_text(path: Path, limit: int = MAX_TEXT) -> str | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_json(path: Path, limit: int = MAX_JSON) -> tuple[Any, str | None]:
    text = read_text(path, limit)
    if text is None:
        return None, "is missing" if not path.is_file() else "is not readable or too large"
    try:
        return json.loads(text), None
    except ValueError as exc:
        return None, f"is not valid JSON ({clean(exc, 120)})"


def mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def present(path: Path) -> bool:
    return bool((read_text(path) or "").strip())


def walk(root: Path, skip: Callable[[str], bool]) -> Iterable[tuple[Path, list[str], list[str]]]:
    """os.walk in sorted order, never into symbolic links or folders skip() names."""
    for directory, subdirs, files in os.walk(root):
        here = Path(directory)
        subdirs[:] = sorted(d for d in subdirs if not skip(d) and not (here / d).is_symlink())
        yield here, subdirs, sorted(files)


# ------------------------------------------------------------------- SPDX expressions

Node = tuple[Any, ...]  # ("id", x) | ("with", id, exception) | ("and" | "or", [nodes])
TOKEN = re.compile(r"\(|\)|[A-Za-z0-9][A-Za-z0-9.+:_-]*")
OPERATORS = ("AND", "OR", "WITH")


def parse(text: str) -> Node | None:
    """An SPDX expression; operators only in upper case ("GPL-2.0 or later" is not one)."""
    tokens = TOKEN.findall(text)
    if not tokens or "".join(tokens) != "".join(text.split()):
        return None
    position = 0

    def take(*refused: str) -> str:
        nonlocal position
        if position >= len(tokens) or tokens[position] in refused:
            raise ValueError(text)
        position += 1
        return str(tokens[position - 1])

    def chain(operator: str, lower: Callable[[], Node]) -> Node:
        parts = [lower()]
        while position < len(tokens) and tokens[position] == operator:
            take()
            parts.append(lower())
        return parts[0] if len(parts) == 1 else (operator.lower(), parts)

    def primary() -> Node:
        word = take(")", *OPERATORS)
        if word == "(":
            inner = chain("OR", conjunction)
            if take() != ")":
                raise ValueError(text)
            return inner
        if position < len(tokens) and tokens[position] == "WITH":
            take()
            return ("with", word, take("(", ")", *OPERATORS))
        return ("id", word)

    def conjunction() -> Node:
        return chain("AND", primary)

    try:
        tree = chain("OR", conjunction)
    except ValueError:
        return None
    return tree if position == len(tokens) else None


def render(node: Node, top: bool = True) -> str:
    if node[0] == "id":
        return str(node[1])
    if node[0] == "with":
        return f"{node[1]} WITH {node[2]}"
    text = f" {node[0].upper()} ".join(render(part, top=False) for part in node[1])
    return text if top else f"({text})"


def node_ids(node: Node) -> list[str]:
    if node[0] in ("id", "with"):
        return [str(node[1])]
    return [i for part in node[1] for i in node_ids(part)]


# ------------------------------------------------------------------------------ policy


@dataclass(frozen=True)
class Rule:
    ecosystem: str
    names: tuple[str, ...]
    status: str  # allowed (a published policy ruling), note, review or denied
    reason: str
    min_version: str = ""
    embedded_ok: bool = False

    @staticmethod
    def load(r: dict[str, Any]) -> Rule:
        more = f" Alternatives: {', '.join(r['alternatives'])}." if r.get("alternatives") else ""
        version, embedded = r.get("min_version", ""), r.get("embedded_binaries_ok") is True
        return Rule(
            r["ecosystem"], tuple(r["names"]), r["status"], r["reason"] + more, version, embedded
        )

    def applies(self, ecosystem: str, name: str, version: str) -> bool:
        key = canonical_pypi(name) if ecosystem == "pypi" else name
        return (
            self.ecosystem == ecosystem
            and matches(key, self.names)
            and (not self.min_version or at_least(version, self.min_version))
        )


class Policy:
    """contracts/license-policy.json with case-insensitive lookups. A missing key or a wrong
    type is an input error (load_policy)."""

    def __init__(
        self, raw: dict[str, Any], digest: str, catalog: model_catalog.Catalog | None = None
    ) -> None:
        self.sha256 = digest
        self.catalog = catalog
        self.policy_version, self.version = raw["policy_version"], raw["version"]
        team, lists, base, models, web = (
            raw[k] for k in ("team_license", "licenses", "organizer_base", "models", "web")
        )
        self.team_allowed = {i.lower() for i in team["allowed"]}
        self.web_licence_files: list[str] = team["web_license_files"]
        self.status: dict[str, str] = {}
        self.names: dict[str, str] = {}
        for key, status in (("allow", ALLOWED), ("review", REVIEW), ("deny", DENIED)):
            for name in lists[key]:
                self.status[name.lower()], self.names[name.lower()] = status, name
        conditions = lists["allow_with_conditions"]
        self.conditions = {k.lower(): f"{k}: {v}" for k, v in conditions.items()}
        self.names.update({k.lower(): k for k in conditions})
        self.deny_patterns = [p.lower() for p in lists["deny_patterns"]]
        self.review_patterns = [p.lower() for p in lists["review_patterns"]]
        self.paid = {i.lower() for i in lists["paid_alternatives"]}
        self.paid_patterns = [p.lower() for p in lists["paid_alternative_patterns"]]
        # Commons Clause is a restriction, not a licence: named anywhere, it denies the expression.
        self.restrictions = {i.lower() for i in lists["restrictions"]}
        # Deprecated identifiers (GPL-2.0-with-classpath-exception) count as their modern form.
        self.deprecated: dict[str, Node] = {}
        for old, new in lists["deprecated"].items():
            tree = parse(new)
            if tree is None:
                raise ValueError(f"licenses.deprecated.{old}: {new!r} is not an SPDX expression")
            self.deprecated[old.lower()] = tree
        # name -> (spelling, effect, note, bases): an allowed or review exception applies only to
        # the base licences it names (GCC-exception-3.1 lifts GPL, never AGPL or SSPL).
        self.exceptions: dict[str, tuple[str, str, str, list[str]]] = {}
        for name, entry in lists["exceptions"].items():
            effect, bases = entry["status"], [b.lower() for b in entry.get("bases", [])]
            if effect not in ("keep", ALLOWED, REVIEW, DENIED):
                raise ValueError(f"licenses.exceptions.{name}: keep, allowed, review or denied")
            if effect in (ALLOWED, REVIEW) and not bases:
                raise ValueError(f"licenses.exceptions.{name}: an {effect} exception needs bases")
            self.exceptions[name.lower()] = (name, effect, entry.get("note") or "", bases)
        self.strings: dict[str, str] = {
            " ".join(k.lower().split()): v for k, v in raw["license_strings"].items()
        }
        self.classifiers: dict[str, str] = raw["classifiers"]
        for target in [*self.strings.values(), *self.classifiers.values()]:
            if parse(target) is None:
                raise ValueError(f"{target!r} is not an SPDX expression")
        self.base_packages = {canonical_pypi(n) for n in base["packages"]}
        self.base_patterns: list[str] = base["proprietary_patterns"]
        self.embedded_files: list[str] = base["embedded_nvidia_files"]
        self.embedded_reason: str = base["embedded_reason"]
        self.rules = [Rule.load(r) for r in raw["packages"]]
        if {r.status for r in self.rules} - {ALLOWED, "note", REVIEW, DENIED}:
            raise ValueError("packages[].status must be allowed, note, review or denied")
        self.model_licences = {i.lower() for i in models["allowed_licenses"]}
        self.model_licence_files: list[str] = models["license_file_names"]
        self.weight_suffixes = tuple(s.lower() for s in models["weight_suffixes"])
        self.catalog_file: str = models["catalog"]
        self.model_deny = [(m["pattern"], m["reason"]) for m in models["deny"]]
        self.name_families = [(m["pattern"], m["family"]) for m in models["name_families"]]
        self.notices_names: list[str] = web["notices_file_names"]
        self.skip_dirs = set(web["skip_dirs"])
        self.build_dirs = {d.lower() for d in web["build_output_dirs"]}
        self.vendor_dirs = {d.lower() for d in web["vendor_dir_names"]}
        self.front_end = tuple(s.lower() for s in web["front_end_suffixes"])
        self.unread_files = set(web["unread_dependency_files"])

    def classify(self, licence: str) -> tuple[str, list[str]]:
        """One identifier: the explicit lists, then the deny patterns; anything else is review.
        '+', '-or-later' and '-only' are also tried without the suffix (GPL-2.0+ is GPL)."""
        lowered = licence.lower()
        for key in (lowered, lowered.removesuffix("+"), re.sub(r"-(or-later|only)$", "", lowered)):
            if key in self.conditions:
                return ALLOWED, [self.conditions[key]]
            if key in self.status:
                return self.status[key], []
        if any(fnmatch.fnmatchcase(lowered, p) for p in self.deny_patterns):
            return DENIED, []
        return REVIEW, []

    def known(self, licence: str) -> bool:
        lowered = licence.lower()
        patterns = self.deny_patterns + self.review_patterns
        known = lowered in self.status or lowered in self.conditions or lowered in self.deprecated
        return known or any(fnmatch.fnmatchcase(lowered, p) for p in patterns)

    def paid_alternative(self, licence: str) -> bool:
        lowered = licence.lower()
        patterns = self.paid_patterns
        return lowered in self.paid or any(fnmatch.fnmatchcase(lowered, p) for p in patterns)

    def named(self, ecosystem: str, name: str) -> bool:
        """A name rule of the policy names this package (in any version)."""
        key = canonical_pypi(name) if ecosystem == "pypi" else name
        return any(r.ecosystem == ecosystem and matches(key, r.names) for r in self.rules)

    def evaluate(self, node: Node, in_or: bool = False) -> tuple[str, list[str]]:
        if node[0] == "id" and str(node[1]).lower() in self.deprecated:
            return self.evaluate(self.deprecated[str(node[1]).lower()], in_or)
        if node[0] == "id":
            if in_or and self.paid_alternative(str(node[1])):
                return DENIED, []  # a paid or proprietary alternative is not a free choice
            return self.classify(str(node[1]))
        if node[0] == "with":
            base = str(node[1])
            status, conditions = self.classify(base)
            exception = self.exceptions.get(str(node[2]).lower())
            if exception is None:
                return worst(status, REVIEW), conditions
            name, effect, note, bases = exception
            notes = conditions + ([f"{name}: {note}"] if note else [])
            if effect == "keep":
                return status, notes
            if effect == DENIED:
                return DENIED, conditions
            if not any(fnmatch.fnmatchcase(base.lower(), b) for b in bases):
                return worst(status, REVIEW), conditions  # not a base this exception lifts
            return effect, notes
        results = [self.evaluate(part, node[0] == "or") for part in node[1]]
        if node[0] == "and":
            return worst(*(r[0] for r in results)), [c for r in results for c in r[1]]
        return min(results, key=lambda r: (RANK[r[0]], len(r[1])))

    def verdict(self, expression: str) -> tuple[str, list[str]]:
        tree = parse(expression) if expression else None
        if tree is None:
            return REVIEW, []
        if any(i.lower() in self.restrictions for i in node_ids(tree)):
            return DENIED, []  # MIT OR Commons-Clause: a restriction is not an alternative
        status, conditions = self.evaluate(tree)
        return status, sorted(set(conditions))

    def ids(self, expression: str) -> list[str]:
        tree = parse(expression) if expression else None
        return node_ids(tree) if tree is not None else []

    def team_cleared(self, expression: str) -> bool:
        """The team licence (D72, superseding D53 rule 1): only the identifiers of the policy's
        team_license.allowed (MIT), with no WITH exception ("MIT WITH Commons-Clause" is another
        licence) and no other licence joined by AND or OR."""
        tree = parse(expression) if expression else None

        def plain(node: Node) -> bool:
            if node[0] == "id":
                return str(node[1]).lower() in self.team_allowed
            return node[0] in ("and", "or") and all(plain(part) for part in node[1])

        return tree is not None and plain(tree)

    def normalise(self, raw: str) -> str | None:
        """A declared licence string as an SPDX expression: "" when none is declared, None when
        the text is neither in the license_strings table nor an SPDX expression."""
        text = " ".join(raw.split())
        if text.lower() in MISSING or text.lower() in self.strings:
            return self.strings.get(text.lower(), "")
        tree = parse(text)

        def spell(node: Node) -> Node:
            if node[0] in ("and", "or"):
                return (node[0], [spell(part) for part in node[1]])
            if node[0] == "id" and str(node[1]).lower() in self.deprecated:
                return spell(self.deprecated[str(node[1]).lower()])
            name = self.names.get(str(node[1]).lower(), node[1])
            if node[0] == "id":
                return ("id", name)
            return ("with", name, self.exceptions.get(str(node[2]).lower(), (node[2],))[0])

        return render(spell(tree)) if tree is not None else None


def load_policy(path: Path, catalog: Path | None = None) -> Policy:
    """The policy, with the model catalogue it names (default: the kit's catalogue)."""
    try:
        loaded = model_catalog.load(catalog or DEFAULT_CATALOG)
    except model_catalog.CatalogError as exc:
        raise InputError(f"model catalogue: {clean(exc, 300)}") from None
    try:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        return Policy(json.loads(data.decode("utf-8")), digest, loaded)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InputError(f"policy {path}: {type(exc).__name__} {clean(exc, 200)}") from None


# ------------------------------------------------------------------------------ report


@dataclass
class Item:
    scope: str
    kind: str
    name: str
    version: str = ""
    license: str = ""
    declared: str = ""
    origin: str = ""
    status: str = ALLOWED
    reasons: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    source: str = ""

    def flag(self, status: str, reason: str = "") -> None:
        self.status = worst(self.status, status)
        if reason and reason not in self.reasons:
            self.reasons.append(reason)


@dataclass
class Report:
    policy: Policy
    inputs: dict[str, str | None]
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {status: sum(i.status == status for i in self.items) for status in RANK}

    def exit_code(self) -> int:
        found = {item.status for item in self.items}
        return 1 if DENIED in found else 2 if REVIEW in found else 0

    def sorted_items(self) -> list[Item]:
        def order(i: Item) -> tuple[bool, int, str, str]:
            return i.scope != "bundle", KINDS.index(i.kind), i.name.lower(), i.version

        return sorted(self.items, key=order)

    def as_json(self) -> dict[str, Any]:
        policy = {k: getattr(self.policy, k) for k in ("policy_version", "version", "sha256")}
        catalog = self.policy.catalog
        return {
            "report_version": REPORT_VERSION,
            "tool": "tools/check_licenses.py",
            "notice": NOTICE,
            "policy": policy,
            "catalog": {
                "version": catalog.version,
                "status": catalog.status,
                "sha256": catalog.sha256,
            }
            if catalog
            else None,  # fmt: skip
            "inputs": self.inputs,
            "items": [asdict(item) for item in self.sorted_items()],
            "notes": self.notes,
            "summary": self.summary(),
            "exit_code": self.exit_code(),
        }


def judge(item: Item, expression: str | None, policy: Policy) -> None:
    """Classify a normalised licence (None: an unrecognised string, kept in item.declared)."""
    if expression is None:
        item.flag(REVIEW, f'unrecognised licence string "{clean(item.declared, 120)}"')
        return
    item.license = expression
    if not expression:
        item.flag(REVIEW, "no licence declared")
        return
    status, conditions = policy.verdict(expression)
    item.conditions += [c for c in conditions if c not in item.conditions]
    if status == DENIED:
        item.flag(DENIED, f"{expression} is denied by the policy")
    elif status == REVIEW:
        unknown = any(not policy.known(i) for i in policy.ids(expression))
        item.flag(
            REVIEW, f"{expression}: {'not in the policy lists' if unknown else 'needs review'}"
        )


def declare(item: Item, raw: str, origin: str, policy: Policy) -> None:
    item.declared, item.origin = clean(raw), origin
    judge(item, policy.normalise(raw), policy)


def apply_rules(item: Item, ecosystem: str, policy: Policy) -> bool:
    """Name rules. Returns True when a published allow ruling also covers NVIDIA binaries."""
    found = [r for r in policy.rules if r.applies(ecosystem, item.name, item.version)]
    for rule in found:
        label = f"policy rule for {item.name}" + (
            f" >= {rule.min_version}" if rule.min_version else ""
        )
        if rule.status in (REVIEW, DENIED):
            item.flag(rule.status, f"{label}: {rule.reason}")
        elif rule.status == "note":
            item.conditions.append(f"{label}: {rule.reason}")
    rulings = [r for r in found if r.status == ALLOWED]
    if rulings and not any(r.status in (REVIEW, DENIED) for r in found):
        item.status = ALLOWED
        item.reasons.append(f"published policy rule (allowed for every team): {rulings[0].reason}")
        return any(r.embedded_ok for r in rulings)
    return False


def checker_item(scope: str, kind: str, name: str, status: str, reason: str) -> Item:
    item = Item(scope, kind, clean(name, 200), origin=CHECKER, source=clean(name, 200))
    item.flag(status, reason)
    return item


# ----------------------------------------------------------------------- Python packages


def is_base(name: str, licence: str, policy: Policy, scope: str) -> bool:
    if matches(name, policy.base_patterns):
        return True
    if scope == "bundle" and canonical_pypi(name) in policy.base_packages:
        return True
    return name.startswith("nvidia-") and "licenseref-nvidia-proprietary" in licence.lower()


def base_reason(scope: str) -> str:
    if scope == "bundle":
        return (
            "part of the organizer's common CUDA base (D52): the organizer installs it from the "
            "official index; remove it from the bundle, which ships no packages (D64)"
        )
    return (
        "PyTorch, triton and the NVIDIA CUDA libraries are installed only by the organizer (D52) "
        "and never redistributed; a web system reaches the model through the kit server API"
    )


def python_licence(item: Item, headers: Any, policy: Policy) -> None:
    """License-Expression, else the License classifiers, else the License field."""
    expression = str(headers.get("License-Expression") or "").strip()
    if expression:
        declare(item, expression, "License-Expression", policy)
        return
    found = [
        " :: ".join(p.strip() for p in str(c).split("::"))
        for c in headers.get_all("Classifier") or []
    ]
    found = [c for c in found if c.startswith("License ::") and c not in GENERIC_CLASSIFIERS]
    mapped = list(dict.fromkeys(policy.classifiers[c] for c in found if c in policy.classifiers))
    unmapped = [c for c in found if c not in policy.classifiers]
    if unmapped:
        item.flag(REVIEW, f'unrecognised licence classifier "{clean(unmapped[0], 120)}"')
    if not mapped:
        value = headers.get("License")
        declare(item, str(value or ""), "License" if value else "none", policy)
        return
    item.declared = clean("; ".join(c.rsplit(" :: ", 1)[-1] for c in found))
    item.origin = "Classifier"
    judge(item, " OR ".join(f"({m})" if " " in m else m for m in mapped), policy)
    if len({policy.verdict(m)[0] for m in mapped}) > 1 and item.status != DENIED:
        item.flag(REVIEW, "classifiers with different statuses: alternatives or parts?")


def python_item(
    text: str, stem: str, members: list[str], source: str, policy: Policy, scope: str
) -> Item:
    headers = email.parser.HeaderParser().parsestr(text)
    parts = stem.split("-")
    name = canonical_pypi(clean(headers.get("Name") or parts[0], 120))
    version = clean(headers.get("Version") or (parts[1] if len(parts) > 1 else ""), 60)
    item = Item(scope, "wheel", name, version, source=source)
    python_licence(item, headers, policy)
    if is_base(item.name, item.license, policy, scope):
        item.status, item.reasons = DENIED, [base_reason(scope)]
        return item
    embedded_ok = apply_rules(item, "pypi", policy)
    names = {Path(m).name for m in members}
    binaries = sorted(n for n in names if matches(n, policy.embedded_files))
    if binaries and not embedded_ok:
        item.flag(REVIEW, f"{policy.embedded_reason} ({', '.join(binaries[:6])})")
    return item


def unread_item(scope: str, stem: str, source: str, reason: str) -> Item:
    parts = stem.split("-")
    item = Item(scope, "wheel", canonical_pypi(parts[0]), parts[1] if len(parts) > 1 else "")
    item.source = source
    item.flag(REVIEW, reason)
    return item


def wheel_item(path: Path, root: Path, policy: Policy, scope: str) -> Item:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            meta = next((m for m in members if re.fullmatch(r"[^/]+\.dist-info/METADATA", m)), "")
            if not meta or archive.getinfo(meta).file_size > MAX_TEXT:
                return unread_item(scope, path.name, rel(path, root), "no readable METADATA")
            text = archive.read(meta).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        reason = f"not a readable wheel ({clean(exc, 120)})"
        return unread_item(scope, path.name, rel(path, root), reason)
    return python_item(text, path.name, members, rel(path, root), policy, scope)


def installed_item(folder: Path, metadata: Path, root: Path, policy: Policy, scope: str) -> Item:
    """*.dist-info/METADATA, *.egg-info/PKG-INFO, an *.egg-info file, or an unpacked source
    package (PKG-INFO at its root); RECORD or installed-files.txt lists the files."""
    stem = folder.name.rsplit(".", 1)[0] if folder.name.endswith("-info") else folder.name
    text = read_text(metadata)
    if text is None:
        return unread_item(scope, stem, rel(folder, root), "installed package without metadata")
    record = read_text(folder / "RECORD") or read_text(folder / "installed-files.txt") or ""
    members = [line.split(",", 1)[0] for line in record.splitlines()]
    return python_item(text, stem, members, rel(folder, root), policy, scope)


def own_project(folder: Path) -> bool:
    """The team's own project installed in development mode: an editable *.dist-info, or an
    *.egg-info next to (or in src/ under) setup files, unless that folder is an unpacked source
    package (a PKG-INFO of its own: third-party code)."""
    if folder.name.endswith(".dist-info"):
        info = mapping(mapping(read_json(folder / "direct_url.json", MAX_TEXT)[0]).get("dir_info"))
        return info.get("editable") is True
    project = folder.parent.parent if folder.parent.name == "src" else folder.parent
    setup = any((project / n).is_file() for n in ("pyproject.toml", "setup.py", "setup.cfg"))
    return folder.name.endswith(".egg-info") and setup and not (project / "PKG-INFO").is_file()


@dataclass
class PythonFiles:
    wheels: list[Path] = field(default_factory=list)
    links: list[Path] = field(default_factory=list)
    metadata: list[tuple[Path, Path]] = field(default_factory=list)  # (package, metadata file)
    environments: list[Path] = field(default_factory=list)
    excluded: list[Path] = field(default_factory=list)  # environments that are not delivered


def find_python(root: Path, delivered: Callable[[Path], bool]) -> PythonFiles:
    """Wheels and installed packages anywhere except node_modules/ and .git; a virtual
    environment (pyvenv.cfg) or site-packages that delivered() rejects is not read."""
    found = PythonFiles()
    for here, subdirs, files in walk(root, lambda d: d in ("node_modules", ".git")):
        environment = here != root and ("pyvenv.cfg" in files or here.name == "site-packages")
        if environment and not any(here.is_relative_to(e) for e in found.environments):
            if not delivered(here):
                found.excluded.append(here)
                subdirs[:] = []
                continue
            found.environments.append(here)
        if "PKG-INFO" in files and here != root:
            found.metadata.append((here, here / "PKG-INFO"))
        for name in [d for d in subdirs if d.endswith((".dist-info", ".egg-info"))]:
            info = "METADATA" if name.endswith(".dist-info") else "PKG-INFO"
            found.metadata.append((here / name, here / name / info))
            subdirs.remove(name)
        for path in (here / f for f in files if f.endswith((".whl", ".egg-info"))):
            if path.name.endswith(".egg-info"):
                found.metadata.append((path, path))  # read_text() refuses symbolic links
            else:
                (found.links if path.is_symlink() else found.wheels).append(path)
    return found


@dataclass
class Requirement:
    name: str
    version: str
    source: str
    problem: str = ""


def direct_name(target: str) -> str:
    egg = re.search(r"[#&]egg=([A-Za-z0-9._-]+)", target)
    base = target.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return egg.group(1) if egg else re.split(r"-\d", base, maxsplit=1)[0]


def read_requirements(path: Path, root: Path, seen: set[str]) -> list[Requirement]:
    """requirements*.txt / *.lock: exact == pins; -r is followed inside root; -c, local -e and
    other pip options add no package."""
    real = os.path.realpath(path)
    text = read_text(path) if real not in seen else None
    seen.add(real)
    source, found = rel(path, root), []
    for line in re.sub(r"\\\r?\n", " ", text or "").splitlines():
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        option = re.match(r"(-r|--requirement|-e|--editable)[\s=]+(\S+)", line)
        if option and option.group(1) in ("-r", "--requirement"):
            target = inside(root, rel(path.parent / option.group(2), root))
            if target is not None and target.is_file():
                found += read_requirements(target, root, seen)
            else:
                found.append(Requirement(option.group(2), "", source, "-r file not read"))
        elif option and re.search(r"://|^git\+", option.group(2)):
            found.append(Requirement(direct_name(option.group(2)), "", source, "editable URL"))
        elif line and not line.startswith("-"):
            line = re.split(r"\s--", line, maxsplit=1)[0]
            pin = REQUIREMENT.match(line.split(";")[0].strip())
            if pin is not None and pin.group(2).strip().startswith("@"):
                found.append(Requirement(pin.group(1), "", source, "direct URL reference"))
                continue
            if pin is None or "://" in line or line.startswith((".", "/")):
                found.append(Requirement(direct_name(line), "", source, "direct URL or file"))
                continue
            exact = re.fullmatch(r"===?\s*([A-Za-z0-9.+!_-]+)", pin.group(2).strip())
            problem = "" if exact else f"not pinned ({clean(pin.group(2), 60) or 'no version'})"
            version = exact.group(1) if exact else ""
            found.append(Requirement(pin.group(1), version, source, problem))
    return found


def requirement_item(requirement: Requirement, policy: Policy, scope: str) -> Item:
    name = canonical_pypi(clean(requirement.name, 120))
    item = Item(scope, "lock", name, clean(requirement.version, 60), origin=requirement.source)
    item.source = requirement.source
    if is_base(name, "", policy, scope):
        item.flag(DENIED, base_reason(scope))
        return item
    if requirement.problem.startswith("not pinned"):
        item.flag(REVIEW, f"{requirement.problem}: pin an exact version with ==")
    elif requirement.problem:
        item.flag(REVIEW, f"{requirement.problem} in {requirement.source}")
    item.flag(REVIEW, f"listed in {requirement.source} but no wheel found: licence not read")
    apply_rules(item, "pypi", policy)
    return item


def scan_python(
    root: Path, scope: str, requirements: list[Requirement], found: PythonFiles, policy: Policy
) -> list[Item]:
    items: dict[tuple[str, str], Item] = {}
    for wheel in found.wheels:
        item = wheel_item(wheel, root, policy, scope)
        items.setdefault((item.name, item.version), item)
    for folder, metadata in found.metadata:
        if not own_project(folder):
            item = installed_item(folder, metadata, root, policy, scope)
            items.setdefault((item.name, item.version), item)
    result = list(items.values())
    for item in result:
        if scope == "bundle" and item.source.startswith("participant_module/"):
            item.flag(
                REVIEW,
                "third-party package inside participant_module/: the organizer reviews it after "
                "the submission; copy pure-Python code without package metadata, keep its licence "
                "files and list it in THIRD_PARTY_NOTICES.md (D66)",
            )
    for link in found.links:
        reason = "symbolic link to a wheel: not followed (every team runs in the common base)"
        result.append(unread_item(scope, link.name, rel(link, root), reason))
        result[-1].origin = CHECKER
    names = {item.name for item in result}
    for requirement in requirements:
        if canonical_pypi(requirement.name) not in names:
            names.add(canonical_pypi(requirement.name))
            result.append(requirement_item(requirement, policy, scope))
    return result


def environment_notes(scope: str, root: Path, found: PythonFiles, report: Report) -> None:
    where = "the upload" if scope == "bundle" else "the web system"
    for environment in found.environments:
        report.notes.append(
            f"{scope}: {rel(environment, root)} is an installed Python environment: remove it "
            f"from {where}; every team runs in the common base and ships no packages (D64). Its "
            "packages were read from their metadata to show their licences"
        )


def outside_base(items: Iterable[Item]) -> None:
    """Stage 2 web code runs in the common base (contracts/base-environment.json) and uses only
    its packages (D64)."""
    common = {canonical_pypi(name) for name in model_catalog.base_packages()}
    for item in items:
        if item.kind not in ("wheel", "lock") or item.status == DENIED:
            continue
        if common and canonical_pypi(item.name) not in common:
            item.flag(
                REVIEW,
                "not in the common base: the web system runs in the common base and uses only "
                "its packages, and no package is added for a team (D64)",
            )


# --------------------------------------------------------------- Stage 1 bundle (D51)


TEAM_LICENCE = "MIT"  # D72 (2026-10-13): the team's own work is MIT only
NO_APACHE = (
    "Apache-2.0 is no longer accepted for the team's own work (D72, 2026-10-13): declare MIT and "
    "put the MIT text in participant_module/LICENSE (Stage 2: LICENSE at the root of the web "
    "code); third-party components keep their own licences"
)


# The three paragraphs of the MIT text (licenses/MIT.txt) that a team's LICENSE must contain (D72).
MIT_PARAGRAPHS = (
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software"
    ' and associated documentation files (the "Software"), to deal in the Software without'
    " restriction, including without limitation the rights to use, copy, modify, merge, publish,"
    " distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom"
    " the Software is furnished to do so, subject to the following conditions:",
    "The above copyright notice and this permission notice shall be included in all copies or"
    " substantial portions of the Software.",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,'
    " INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR"
    " PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE"
    " FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR"
    " OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER"
    " DEALINGS IN THE SOFTWARE.",
)
APACHE_TITLE = "apache license version 2 0 january 2004"  # the Apache-2.0 heading, normalised


def plain_words(text: str) -> str:
    """Lower-case words and numbers only: quotes, punctuation and line breaks do not matter."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def licence_text(path: Path) -> str:
    """A team licence file as text: UTF-8 (a byte order mark is ignored), or UTF-16 when it starts
    with a UTF-16 byte order mark, as some editors save "Unicode" text."""
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TEXT:
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def team_text(team: Item, path: Path, where: str) -> None:
    """D72: the team's licence file must hold the MIT text (the three paragraphs of
    MIT_PARAGRAPHS). The Apache-2.0 text instead is denied with a message that says so; the MIT
    text together with the Apache-2.0 text is review (which licence is granted?)."""
    text = plain_words(licence_text(path))
    apache = APACHE_TITLE in text
    fix = "licenses/MIT.txt of the kit, with your team in the copyright line, saved as UTF-8 text"
    if all(plain_words(paragraph) in text for paragraph in MIT_PARAGRAPHS):
        if apache:
            team.flag(
                REVIEW,
                f"{where} also holds the Apache License 2.0 text: the team's own work is licensed "
                f"under MIT alone (D72); keep only the MIT text ({fix})",
            )
        return
    if apache:
        team.flag(
            DENIED,
            f"{where} holds the Apache License 2.0 text, not the MIT text: the team's own work is "
            f"MIT only (D72); replace it with the MIT text ({fix})",
        )
    else:
        team.flag(
            DENIED,
            f"{where} does not contain the MIT licence text (its permission, condition and "
            f"warranty paragraphs; the team's own work is MIT only, D72): use {fix}",
        )


def team_licence(team: Item, declared: object, where: str, policy: Policy, hint: str = "") -> None:
    """The team licence (D72, superseding D53 rule 1): the team's own work is MIT only; anything
    else is denied, and Apache-2.0, allowed until D72, gets a message that says what changed."""
    text = declared if isinstance(declared, str) else ""
    expression = policy.normalise(text) or ""
    if not team.license and not team.declared:
        team.declared, team.license = clean(text), expression
    if not text.strip():
        team.flag(DENIED, f"{where} is missing: declare {TEAM_LICENCE}{hint}")
    elif not policy.team_cleared(expression):
        reason = f'{where} is "{clean(text, 80)}": the team\'s own work must be {TEAM_LICENCE}'
        reason += f" alone, without a WITH clause, added terms or another licence (D72){hint}"
        if "apache-2.0" in (i.lower() for i in policy.ids(expression)):
            reason += f"; {NO_APACHE}"
        team.flag(DENIED, reason)


def model_files(item: Item, root: Path, relative: object, policy: Policy, own: bool) -> None:
    """The weights must be in the bundle; a third-party model copy needs its licence file next
    to them (presence only)."""
    target = inside(root, relative) if isinstance(relative, str) else None
    if target is None or not target.exists():
        item.flag(REVIEW, f'"path" {clean(relative, 120)} is outside the bundle or not found')
        return
    names = [p.name for p in (target if target.is_dir() else target.parent).iterdir()]
    if not own and not any(matches(n, policy.model_licence_files) for n in names):
        item.flag(REVIEW, f"no licence file next to the weights in {clean(relative, 120)}")


OUTSIDE = (
    "outside the model catalogue: only catalogue models (contracts/model-catalog.json, by "
    "catalog_id) and team-trained weights may be used; other public models are refused (D64)"
)
NOT_SHIPPED = (
    "public model weights are not shipped: the organizer provides catalogue models in "
    "APCTT_MODEL_STORE (D64); name the model by catalog_id without a path"
)


def catalogue_model(item: Item, model: model_catalog.Model, entry: dict[str, Any]) -> None:
    """A catalogue model: allowed with the catalogue's licence; never shipped in the bundle."""
    declared = item.license
    item.license, item.origin = model.license, "model catalogue"
    item.version = model.revision
    item.source = f"{model.repo}@{model.revision}"
    if declared and declared != model.license:
        item.flag(REVIEW, f"declared {declared}; the model catalogue says {model.license}")
    if entry.get("path"):
        item.flag(DENIED, NOT_SHIPPED)


def by_repository(item: Item, repo: str, entry: dict[str, Any], policy: Policy) -> None:
    """A model named by repository: that catalogue model at its catalogue revision, else denied."""
    model = policy.catalog.by_repo(repo) if policy.catalog else None
    revision = clean(entry.get("revision") or "", 80).lower()
    if model is None:
        item.flag(DENIED, OUTSIDE)
    elif revision != model.revision:
        shown = revision or "no revision"
        item.flag(
            DENIED,
            f"{shown}: only revision {model.revision} of {model.repo} is in the model catalogue "
            f"(catalog_id {model.id}); other revisions are refused (D64)",
        )
    else:
        catalogue_model(item, model, entry)
        item.conditions.append(f'name it by "catalog_id": "{model.id}"')


def team_trained(
    item: Item,
    entry: dict[str, Any],
    names: list[str],
    denied: list[str],
    team: Item,
    policy: Policy,
) -> None:
    """Weights the team trained itself are the team's own work, covered by its licence (MIT,
    D72); fine-tunes and adapters of a catalogue model name it in base_model (D64) and stay
    subject to that model's licence too (keep its LICENSE and NOTICE: a condition of the item).
    The deny list still applies to every name the entry gives, a declared licence must be MIT, a
    base model outside the catalogue is refused, and a bare name of a model family other than the
    declared base (my-gemma-ft) is review."""
    item.origin = "team-trained"
    catalog = policy.catalog
    if denied:
        item.flag(DENIED, denied[0])
    expression = policy.normalise(item.declared) if item.declared else ""
    other_licence = bool(item.declared) and not policy.team_cleared(expression or "")
    if other_licence:
        reason = f'declared licence "{clean(item.declared, 80)}": team-trained weights are the '
        item.flag(DENIED, reason + f"team's own work, {TEAM_LICENCE} only (D72)")
    bases: list[model_catalog.Model] = []
    base_model = entry.get("base_model")
    if base_model is not None:
        found = catalog.get(base_model) if catalog else None
        if found is None:
            item.flag(DENIED, f"base_model {clean(base_model, 80)} is {OUTSIDE}")
        else:
            bases.append(found)
    for key in ("repo", "url"):
        value = clean(entry.get(key) or "", 400)
        if not value:
            continue
        hub = HF_URL.match(value) if key == "url" else None
        repo = value if key == "repo" else hub.group(1) if hub else ""
        found = catalog.by_repo(repo) if catalog and repo else None
        if found is None:
            item.flag(DENIED, f"names {value}, which is {OUTSIDE}")
        else:
            bases.append(found)
            item.conditions.append(f'name the catalogue base as "base_model": "{found.id}"')
    for name in names:  # a catalogue id given as catalog_id or name: the base, not a third party
        found = catalog.get(name) if catalog else None
        if found is not None and found not in bases:
            bases.append(found)
            if base_model != found.id:
                item.conditions.append(f'name the catalogue base as "base_model": "{found.id}"')
    families = [
        (n, family)
        for n in names
        if "/" not in n
        for pattern, family in policy.name_families
        if matches(n, [pattern])
        and not any(matches(b.repo.lower(), [f"*{pattern.strip('*')}*"]) for b in bases)
    ]
    if families:
        item.flag(REVIEW, f'the name "{families[0][0]}" suggests a {families[0][1]} model')
    for base in bases:  # D72: derived weights stay subject to the catalogue model's licence
        item.conditions.append(
            f"derived from catalogue model {base.id} ({base.license}): the weights stay subject "
            "to its licence as well; keep its LICENSE and NOTICE files with them"
        )
    # The team licence covers only weights that look like the team's own; otherwise the licence
    # column shows only what the entry itself declares.
    third_party_like = bool(denied or families or item.status == DENIED)
    item.license = expression or ("" if other_licence or third_party_like else team.license)
    if denied or other_licence or third_party_like:
        hint = "a model fine-tuned, quantised or converted from a public model outside the "
        item.flag(item.status, hint + "catalogue is not team-trained work that may be used")
    if team.status != ALLOWED:
        item.flag(REVIEW, "covered by the team licence, which is not cleared")
    if not entry.get("path"):
        item.flag(REVIEW, 'team-trained weights need "path" (their file or folder)')


def scan_models(entries: object, root: Path, policy: Policy, team: Item) -> list[Item]:
    items = []
    for entry in map(mapping, entries if isinstance(entries, list) else []):
        repo, label, url = (clean(entry.get(k) or "", 400) for k in ("repo", "name", "url"))
        catalog_id = clean(entry.get("catalog_id") or "", 80)
        hub = HF_URL.match(url)
        names = [n for n in (catalog_id, repo, label, hub.group(1) if hub else "") if n]
        item = Item("bundle", "model", names[0] if names else "(unnamed model)")
        item.version, item.origin = clean(entry.get("revision") or "", 80), "submission.json"
        item.source = f"{repo}@{item.version}" if repo else url or label or catalog_id
        item.declared = clean(entry.get("license")) if isinstance(entry.get("license"), str) else ""
        item.license = policy.normalise(item.declared) or ""
        items.append(item)
        denied = [f"{n}: {r}" for p, r in policy.model_deny for n in names if matches(n, [p])]
        if entry.get("team_trained") is True:
            team_trained(item, entry, names, denied, team, policy)
        elif denied:
            item.flag(DENIED, denied[0])
        elif catalog_id:
            model = policy.catalog.get(catalog_id) if policy.catalog else None
            if model is None:
                item.flag(DENIED, f"catalog_id {catalog_id} is {OUTSIDE}")
            else:
                catalogue_model(item, model, entry)
        elif repo or hub:
            by_repository(item, repo or (hub.group(1) if hub else ""), entry, policy)
        elif not names:
            item.flag(
                DENIED, f'a models[] entry needs "catalog_id" or "team_trained": true ({OUTSIDE})'
            )
        else:
            item.flag(DENIED, OUTSIDE)
        if entry.get("path") and item.status != DENIED:
            model_files(item, root, entry["path"], policy, entry.get("team_trained") is True)
    return items


def undeclared_files(root: Path, entries: object, policy: Policy) -> list[Item]:
    """Weight files and shared libraries in assets/ and participant_module/ that no models[]
    path covers: one review item per folder."""
    paths = [mapping(e).get("path") for e in (entries if isinstance(entries, list) else [])]
    covered = [str(t) for t in (inside(root, p) for p in paths if isinstance(p, str)) if t]
    folders: dict[str, list[str]] = {}
    for top in ("assets", "participant_module"):
        for here, _, files in walk(root / top, lambda d: False):
            for name in files:
                real = os.path.realpath(here / name)
                if any(real == c or real.startswith(c + os.sep) for c in covered):
                    continue
                if name.lower().endswith(policy.weight_suffixes) or LIBRARY.search(name):
                    folders.setdefault(rel(here, root), []).append(name)
    reason = (
        "weight or library files that no submission.json models[] entry covers: public models "
        "are not shipped (name catalogue models by catalog_id), declare your own weights with "
        '"team_trained": true and path; do not ship programs or libraries (LICENSING.md §3.2)'
    )
    items = []
    for folder, names in sorted(folders.items())[:MAX_FOLDERS]:
        detail = f"{len(names)} {reason} ({', '.join(names[:4])})"
        items.append(checker_item("bundle", "asset", folder, REVIEW, detail))
    return items


def dataset_items(declaration: object, problem: str | None, policy: Policy) -> list[Item]:
    sources = mapping(declaration).get("data_sources")
    if not isinstance(sources, list):
        detail = problem or 'has no "data_sources" list'
        reason = f'declaration.json {detail}: declare each source\'s licence ("data_sources": [])'
        return [checker_item("bundle", "dataset", "declaration.json", REVIEW, reason)]
    items = []
    for index, entry in enumerate(map(mapping, sources), 1):
        item = Item("bundle", "dataset", clean(entry.get("source") or f"data source {index}", 200))
        licence = entry.get("license")
        declare(item, licence if isinstance(licence, str) else "", "declaration.json", policy)
        items.append(item)
    return items


def check_bundle(root: Path, policy: Policy, report: Report) -> None:
    team = Item("bundle", "team", "participant_module", origin="submission.json")
    report.items.append(team)
    submission, problem = read_json(root / "submission.json", MAX_TEXT)
    if not isinstance(submission, dict):
        team.flag(DENIED, f"submission.json {problem or 'is not a JSON object'}")
    submission = mapping(submission)
    team_licence(team, submission.get("license"), 'submission.json "license"', policy)
    licence_file = root / "participant_module" / "LICENSE"
    if not present(licence_file):
        team.flag(DENIED, "participant_module/LICENSE is missing or empty: add your licence text")
    elif team.status == ALLOWED:  # "license" is MIT: the file must hold the MIT text (D72)
        team_text(team, licence_file, "participant_module/LICENSE")
    models = submission.get("models", [])
    report.items += scan_models(models, root, policy, team)
    report.items += undeclared_files(root, models, policy)
    lock = root / "requirements.lock"
    requirements = read_requirements(lock, root, set()) if lock.is_file() else []
    # Wheels are read anywhere; an installed environment only inside the two shipped folders
    # (wheels/ holds wheel files, so an environment there is a layout error too).
    delivered = [root / "participant_module", root / "assets"]
    found = find_python(root, lambda f: any(f.is_relative_to(d) for d in delivered))
    report.items += scan_python(root, "bundle", requirements, found, policy)
    reason = (
        "a virtual environment outside participant_module/ and assets/ is not part of the "
        "submission layout (D51, D59): remove it from the upload (not read); every team runs in "
        "the common base and ships no packages (D64)"
    )
    report.items += [
        checker_item("bundle", "layout", rel(e, root), DENIED, reason) for e in found.excluded
    ]
    environment_notes("bundle", root, found, report)
    report.items += dataset_items(*read_json(root / "declaration.json", MAX_TEXT), policy)
    module = root / "participant_module"
    if module.is_dir() and not any(matches(p.name, policy.notices_names) for p in module.iterdir()):
        report.notes.append("bundle: add THIRD_PARTY_NOTICES.md to participant_module/ (--notices)")


# ------------------------------------------------------------------- Stage 2 web system


def npm_licence(manifest: object) -> str | None:
    """The "license" field (or a legacy "licenses" list, read as alternatives)."""
    data = mapping(manifest)
    value = data.get("license")
    value = mapping(value).get("type") if isinstance(value, dict) else value
    if isinstance(value, str) and value.strip():
        return value
    legacy = data.get("licenses")
    legacy = legacy if isinstance(legacy, list) else []
    types = [mapping(e).get("type") if isinstance(e, dict) else e for e in legacy]
    return " OR ".join(t for t in types if isinstance(t, str) and t.strip()) or None


def npm_item(
    name: str, version: str, raw: str | None, origin: str, source: str, policy: Policy
) -> Item:
    item = Item("web", "npm", clean(name, 200), clean(version, 60), source=clean(source, 200))
    declare(item, raw or "", origin, policy)
    apply_rules(item, "npm", policy)
    return item


def manifest_item(folder: Path, web: Path, origin: str, policy: Policy) -> Item:
    data = mapping(read_json(folder / "package.json", MAX_TEXT)[0])
    name, version = str(data.get("name") or folder.name), str(data.get("version") or "")
    source = rel(folder / "package.json", web)
    return npm_item(name, version, npm_licence(data), origin, source, policy)


@dataclass
class WebLayout:
    npm_folders: list[Path] = field(default_factory=list)
    requirement_files: list[Path] = field(default_factory=list)
    unread: list[Path] = field(default_factory=list)
    front_end: dict[Path, list[str]] = field(default_factory=dict)


def walk_web(web: Path, policy: Policy) -> WebLayout:
    layout = WebLayout()
    for here, subdirs, files in walk(web, lambda d: d in policy.skip_dirs):
        if here != web and "pyvenv.cfg" in files:
            subdirs[:] = []
            continue
        names = {f for f in files if not (here / f).is_symlink()}
        if names & NPM_FILES or (here / "node_modules").is_dir():
            layout.npm_folders.append(here)
        requirements = sorted(f for f in names if re.fullmatch(r"requirements.*\.(txt|lock)", f))
        layout.requirement_files += [here / f for f in requirements]
        if not requirements and "PKG-INFO" not in names:
            layout.unread += [here / f for f in sorted(names & policy.unread_files)]
        if scripts := sorted(f for f in names if f.lower().endswith(policy.front_end)):
            layout.front_end[here] = scripts
    return layout


def package_folders(modules: Path) -> list[Path]:
    """node_modules/<name> and node_modules/@scope/<name> (symbolic links included)."""
    found = []
    for child in sorted(modules.iterdir()):
        if child.name.startswith("@") and child.is_dir() and not child.is_symlink():
            found += [c for c in sorted(child.iterdir()) if c.is_dir() or c.is_symlink()]
        elif not child.name.startswith(".") and (child.is_dir() or child.is_symlink()):
            found.append(child)
    return found


def declared_dependency(name: str, spec: object, source: str, reason: str, policy: Policy) -> Item:
    """A dependency named in a package.json that no lock installs: licence not read (review),
    name rules applied; the version only when the range is an exact version."""
    exact = re.fullmatch(r"=?v?(\d+\.\d+\.\d+[-+.\w]*)", str(spec).strip())
    item = Item("web", "npm", clean(name, 200), exact.group(1) if exact else "")
    item.origin, item.source = "package.json", clean(source, 200)
    item.flag(REVIEW, reason)
    apply_rules(item, "npm", policy)
    return item


def marked_project(folder: Path, package: dict[str, Any]) -> bool:
    """The folder says it is an npm project of its own: an npm lock, node_modules/, "private":
    true, "workspaces", or its own yarn, pnpm or bun lock."""
    if any((folder / name).exists() for name in ("node_modules", *NPM_LOCKS)):
        return True
    own_lock = any((folder / name).is_file() for name in OTHER_JS_LOCKS)
    return package.get("private") is True or "workspaces" in package or own_lock


def copied_package(
    web: Path, folder: Path, package: dict[str, Any], locked: list[Path], policy: Policy
) -> bool:
    """A folder below the web root with its own package.json that is not marked as a project of
    its own is a copied third-party package when it names a package of the policy's name rules
    (pdfjs-dist, apexcharts, handsontable: wherever it sits), when it is under a vendor-like
    folder, or when it is inside a project with a lock and declares no dependencies or
    devDependencies of its own (pdfjs-dist declares only optionalDependencies). Otherwise a
    folder that declares dependencies is the team's own sub-project."""
    if folder == web or marked_project(folder, package):
        return False
    if {part.lower() for part in folder.relative_to(web).parts} & policy.vendor_dirs:
        return True
    if policy.named("npm", str(package.get("name") or "")):
        return True
    in_locked = any(project in folder.parents for project in locked)
    own = ("dependencies", "devDependencies")
    return in_locked and not any(mapping(package.get(section)) for section in own)


def copied_dependencies(
    web: Path, folder: Path, package: dict[str, Any], listed: set[str], policy: Policy
) -> list[Item]:
    """The runtime dependencies a copied package declares (not its devDependencies). No lock
    installs them, so each is review with the name rules applied, unless already listed."""
    source = rel(folder / "package.json", web)
    name = clean(package.get("name") or folder.name, 120)
    reason = f"dependency of the copied package {name}, installed by no lock: licence not read"
    items = []
    for section in RUNTIME_SECTIONS:
        for dependency, spec in sorted(mapping(package.get(section)).items()):
            if dependency not in listed:
                listed.add(dependency)
                items.append(declared_dependency(dependency, spec, source, reason, policy))
    return items


def lock_items(
    folder: Path, where: str, lock: Path, packages: dict[str, Any], policy: Policy, team: Item
) -> tuple[list[tuple[Item, bool]], set[Path]]:
    """package-lock.json v2/v3 "packages". Entries outside node_modules/ are workspaces (the
    team's own code, read once); link entries point to them."""
    items, members = [], set()
    for key, entry in sorted((k, mapping(v)) for k, v in packages.items()):
        if not key or entry.get("link"):
            continue
        if "node_modules/" not in key:
            members.add(folder / key)
            if "license" in entry:
                team_licence(team, npm_licence(entry), f'{where}{key} "license"', policy)
            continue
        raw, origin, target = npm_licence(entry), lock.name, inside(folder, key)
        if raw is None and target is not None:
            raw = npm_licence(read_json(target / "package.json", MAX_TEXT)[0])
            origin = "package.json" if raw else origin
        name = str(entry.get("name") or key.rsplit("node_modules/", 1)[1])
        item = npm_item(name, str(entry.get("version") or ""), raw, origin, where + key, policy)
        if target is None:
            item.flag(REVIEW, "lock entry points outside the web directory: not read")
        items.append((item, entry.get("dev") is True))
    return items, members


def scan_npm_project(
    web: Path, folder: Path, policy: Policy, team: Item, report: Report
) -> tuple[list[Item], set[Path]]:
    """One npm project of the team: its lock (or node_modules/, or package.json) and its own
    licence. Returns the items to count and its workspace folders."""
    where = "" if folder == web else rel(folder, web) + "/"
    package = mapping(read_json(folder / "package.json", MAX_TEXT)[0])
    dev_names = set(mapping(package.get("devDependencies")))
    hint = ""
    if folder != web and not marked_project(folder, package):
        hint = " (read as your sub-project; put a copied third-party package under vendor/)"
    licence = npm_licence(package)
    if "license" in package or licence is not None:  # "license", or a legacy "licenses" list
        team_licence(team, licence, f'{where}package.json "license"', policy, hint)
    items: list[tuple[Item, bool]] = []
    own_name = str(package.get("name") or "")
    if own_name and policy.named("npm", own_name):
        # The name rules apply to every folder's own package.json, project or not.
        version, source = str(package.get("version") or ""), where + "package.json"
        own = npm_item(own_name, version, licence, "package.json", source, policy)
        own.flag(own.status, f"{source} names a package the policy lists: a copied package?")
        items.append((own, False))
    members: set[Path] = set()
    lock = next((folder / n for n in NPM_LOCKS if (folder / n).is_file()), None)
    packages = mapping(read_json(lock)[0]).get("packages") if lock else None
    modules = folder / "node_modules"
    if lock is not None and not isinstance(packages, dict):
        report.notes.append(f"web: {where}{lock.name} has no lockfile v2/v3 packages: not read")
    if lock is not None and isinstance(packages, dict):
        items, members = lock_items(folder, where, lock, packages, policy, team)
    elif modules.is_dir() and not modules.is_symlink():
        folders = package_folders(modules)
        for child in (f for f in folders if not f.is_symlink()):
            item = manifest_item(child, web, "package.json", policy)
            items.append((item, item.name in dev_names))
        if links := sum(f.is_symlink() for f in folders):
            reason = f"{links} symbolic links (pnpm, yarn) not followed: add package-lock.json"
            items.append(
                (checker_item("web", "npm", where + "node_modules/", REVIEW, reason), False)
            )
    else:
        reason = f"listed in {where}package.json without a lock or node_modules/: licence not read"
        for section in SECTIONS:
            for name, spec in sorted(mapping(package.get(section)).items()):
                item = declared_dependency(name, spec, where + "package.json", reason, policy)
                items.append((item, section == "devDependencies"))
    for name in OTHER_JS_LOCKS:
        if (folder / name).is_file() and lock is None and not modules.is_dir():
            reason = f"{name} is not read: add package-lock.json (npm install --package-lock-only)"
            items.append((checker_item("web", "npm", where + name, REVIEW, reason), False))
    if dev := [i for i, is_dev in items if is_dev]:
        flagged = [
            f"{i.name} ({i.license or i.declared or '-'}: {i.status})"
            for i in dev
            if i.status != ALLOWED
        ]
        report.notes.append(
            f'web: {where or "./"} {len(dev)} dev-only npm packages (devDependencies, "dev": true) '
            "are not counted, as they are not shipped to browsers; confirm that your build does "
            "not bundle them" + (f". Not allowed: {'; '.join(flagged[:10])}" if flagged else "")
        )
        if flagged:  # a build (Vite, SvelteKit) may bundle a dev dependency after all
            reason = (
                f"{len(flagged)} dev-only npm packages are not allowed ({'; '.join(flagged[:10])}"
                f"{'; ...' if len(flagged) > 10 else ''}): not counted as shipped, but a build "
                "may bundle them; confirm that it does not, or replace them"
            )
            dev_item = checker_item("web", "npm", f"{where}devDependencies", REVIEW, reason)
            items.append((dev_item, False))
    return [i for i, is_dev in items if not is_dev], members


def front_end_items(
    web: Path, layout: WebLayout, projects: list[Path], vendored: list[Path], policy: Policy
) -> list[Item]:
    """Plain .js/.css files carry no package metadata and are never read. One review item per
    folder with minified files (*.min.*), and outside an npm project per vendor-like or build
    output folder; the team's own sources and an npm project's build output are not listed."""
    items: list[Item] = []
    reason = (
        "without package metadata: contents are not read; keep each library's licence file with "
        "it and list it in THIRD_PARTY_NOTICES.md"
    )
    for folder, files in sorted(layout.front_end.items()):
        if any(folder == v or v in folder.parents for v in vendored):
            continue  # a vendored package: its package.json is the item
        project = next((p for p in projects if folder == p or p in folder.parents), None)
        parts = {p.lower() for p in folder.relative_to(project or web).parts}
        minified = [f for f in files if ".min." in f.lower()]
        if project is not None:
            chosen = [] if parts & policy.build_dirs else minified
        else:
            chosen = files if parts & (policy.vendor_dirs | policy.build_dirs) else minified
        if chosen and len(items) < MAX_FOLDERS:
            detail = f"{len(chosen)} copied or minified files ({', '.join(chosen[:3])}) {reason}"
            items.append(checker_item("web", "asset", rel(folder, web), REVIEW, detail))
    return items


def check_web(web: Path, policy: Policy, report: Report) -> None:
    team = Item("web", "team", "web system", origin="LICENSE")
    report.items.append(team)
    licence_files = [name for name in policy.web_licence_files if present(web / name)]
    if not licence_files:
        team.flag(DENIED, f"no LICENSE at the root of the web code: add the {TEAM_LICENCE} text")
    else:  # the web code is the team's own work: MIT only, so the file holds the MIT text (D72)
        name = licence_files[0]
        team_text(team, web / name, f"{name} at the root of the web code")
    layout = walk_web(web, policy)
    projects: list[Path] = []
    locked: list[Path] = []
    vendored: list[Path] = []
    covered: set[Path] = set()
    for folder in sorted(layout.npm_folders, key=lambda p: len(p.parts)):
        if folder in covered:
            continue
        package = mapping(read_json(folder / "package.json", MAX_TEXT)[0])
        if copied_package(web, folder, package, locked, policy):
            vendored.append(folder)  # third-party: its own licence, and what it depends on
            report.items.append(manifest_item(folder, web, "vendored package.json", policy))
            listed = {i.name for i in report.items if i.kind == "npm"}
            report.items += copied_dependencies(web, folder, package, listed, policy)
            continue
        projects.append(folder)
        if any((folder / n).exists() for n in ("node_modules", *NPM_LOCKS)):
            locked.append(folder)
        items, members = scan_npm_project(web, folder, policy, team, report)
        report.items += items
        covered |= members
    seen: set[str] = set()
    requirements = [r for p in layout.requirement_files for r in read_requirements(p, web, seen)]
    reason = (
        "is not read: a web system uses only packages of the common base (D64); leave it out of "
        "the web directory, or list those packages with exact versions (==) in a "
        "requirements.txt next to it"
    )
    for path in (p for p in layout.unread if p.name not in OTHER_JS_LOCKS):
        report.items.append(
            checker_item("web", "lock", rel(path, web), REVIEW, f"{path.name} {reason}")
        )
    found = find_python(web, lambda folder: True)  # the whole web directory is checked
    python_items = scan_python(web, "web", requirements, found, policy)
    outside_base(python_items)
    report.items += python_items
    environment_notes("web", web, found, report)
    report.items += front_end_items(web, layout, projects, vendored, policy)
    if not any(matches(p.name, policy.notices_names) for p in web.iterdir() if p.is_file()):
        report.notes.append("web: no THIRD_PARTY_NOTICES file at the web root (--notices)")


# ------------------------------------------------------------------------------ output


def joined(parts: Iterable[str]) -> str:
    """Reasons and conditions as one "; "-separated line, without a full stop before each ";"."""
    return "; ".join(p[:-1] if p.endswith(".") and not p.endswith("...") else p for p in parts if p)


def table(report: Report) -> str:
    widths = (8, 7, 32, 14, 30)
    headers = ("status", "kind", "component", "version", "licence")
    policy = f"{report.policy.policy_version} {report.policy.version}"
    lines = [f"APCTT licence check - policy {policy} - {NOTICE}", ""]
    rows = [(*headers, "origin / reasons")]
    for i in report.sorted_items():
        details = joined([i.origin, *i.reasons, *i.conditions])
        rows.append((i.status, i.kind, i.name, i.version, i.license or i.declared or "-", details))
    for *cells, details in rows:
        row = "  ".join(clean(c, w).ljust(w) for c, w in zip(cells, widths, strict=True))
        lines.append(f"{row}  {clean(details, 400)}")
    lines += [f"note: {clean(note, 400)}" for note in report.notes]
    count = report.summary()
    totals = f"{count[ALLOWED]} allowed, {count[REVIEW]} review, {count[DENIED]} denied"
    return "\n".join([*lines, f"summary: {totals} -> exit {report.exit_code()}"]) + "\n"


def md(value: str) -> str:
    return clean(value, 300).replace("\\", "\\\\").replace("|", "\\|")


def notices(report: Report) -> str:
    """A THIRD_PARTY_NOTICES draft: shipped third-party components per ecosystem (never denied
    ones, the team's own code or checker items), then everything that is not cleared."""
    sections = [("Python packages", ("wheel", "lock")), ("JavaScript packages", ("npm",))]
    sections += [("Models", ("model",)), ("Data", ("dataset",))]
    out = [
        "# Third-party notices (draft)",
        "",
        f"Drafted by tools/check_licenses.py (policy {report.policy.version}) from declared "
        "metadata only. Check every row against the component's own licence "
        'and notice files, keep them with the component, resolve "Not cleared" before shipping, '
        "and add by hand what the checker does not read (copied files, fonts, icons).",
    ]

    def licence(i: Item) -> str:
        return md(i.license or i.declared or "(none found)")

    for title, kinds in sections:
        rows = [
            f"| {md(i.name)} | {md(i.version)} | {licence(i)} | {md(i.source)} |"
            for i in report.sorted_items()
            if i.kind in kinds and i.status != DENIED and i.origin not in (CHECKER, "team-trained")
        ]
        header = [
            "",
            f"## {title}",
            "",
            "| Component | Version | Licence | Source |",
            "|---|---|---|---|",
        ]
        out += header + list(dict.fromkeys(rows)) if rows else []
    blocked = [i for i in report.sorted_items() if i.status != ALLOWED and i.kind != "team"]
    if blocked:
        out += ["", "## Not cleared", "", "| Component | Version | Licence | Status | Reason |"]
        out.append("|---|---|---|---|---|")
        for i in blocked:
            reasons = md(joined(i.reasons))
            out.append(
                f"| {md(i.name)} | {md(i.version)} | {licence(i)} | {i.status} | {reasons} |"
            )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------- CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(3, f"{self.prog}: error: {message}\n")


def directory(value: str | None, option: str) -> Path | None:
    if value is not None and not Path(value).is_dir():
        raise InputError(f"{option} {value}: not a directory")
    return Path(value) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = Parser(
        prog="check_licenses.py",
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Exit codes: 0 allowed, 1 denied, 2 review only, 3 input error. " + NOTICE,
    )
    parser.add_argument("--bundle", help="Stage 1 submission bundle directory")
    parser.add_argument("--web", help="Stage 2 web system directory")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="licence policy JSON")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--notices", help="write a THIRD_PARTY_NOTICES draft (Markdown) here")
    args = parser.parse_args(argv)
    if args.bundle is None and args.web is None:
        parser.error("give --bundle and/or --web")
    try:
        bundle, web = directory(args.bundle, "--bundle"), directory(args.web, "--web")
        policy = load_policy(Path(args.policy))
    except InputError as exc:
        print(f"check_licenses.py: {exc}", file=sys.stderr)
        return 3
    report = Report(policy, {"bundle": args.bundle, "web": args.web})
    if bundle is not None:
        check_bundle(bundle, policy, report)
    if web is not None:
        check_web(web, policy, report)
    if args.notices:
        try:
            Path(args.notices).parent.mkdir(parents=True, exist_ok=True)
            Path(args.notices).write_text(notices(report), encoding="utf-8")
        except OSError as exc:
            print(f"check_licenses.py: --notices {args.notices}: {exc}", file=sys.stderr)
            return 3
    output = json.dumps(report.as_json(), indent=2, ensure_ascii=False) + "\n"
    sys.stdout.write(output if args.json else table(report))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
