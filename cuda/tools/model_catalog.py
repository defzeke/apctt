"""The public model catalogue (decisions D64, D67): load and check contracts/model-catalog.json.

    python tools/model_catalog.py [CATALOG.json] [--json | --markdown]

The organizer recommends public models for the reference GPU of the event machine (D40: RTX 3070
Laptop GPU, 8 GB VRAM) and provides them: every machine has them in a read-only model store,
<APCTT_MODEL_STORE>/<org>/<name>/<revision>/, and teams download the same files for development
with tools/get_models.py. Only catalogue models (named in submission.json by catalog_id) and
team-trained weights may be used; any other public model is refused, because it is not available
offline and public weights are not shipped. Quantized entries (bitsandbytes-nf4, D67) load the
publisher's bf16 weights in 4 bits on a CUDA GPU with one recipe, quantization_config(), which the
loading snippet and the organizer's GPU measurement share. --markdown prints the catalogue as the
participant document docs/participant/MODEL_CATALOG.md (generated, not edited by hand).

This module is the one loader and validator that every tool uses (get_models.py,
make_submission.py, check_licenses.py and the organizer's tools). A catalogue is valid when it
matches contracts/model-catalog.schema.json and these rules: ids and repositories are unique; the
VRAM estimate's peak is weights + kv_cache_or_activations + overhead; every model fits the budget
(fits_8gb true and peak at most gpu_budget.target_peak_gib, which is at most vram_gib); the loading
library, and the package a quantization method needs, are in the common base
(contracts/base-environment.json); the class is one the library has; file paths are plain relative
paths; at least one weight file is listed, and every weight file has its size and sha256 in
"weights" (the repository's LFS record), which get_models.py checks on download and
make_submission.py uses to refuse catalogue weights in a bundle. The command prints VALID or the
problems.
Exit codes: 0 valid, 2 invalid catalogue, 3 usage error (file missing). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))  # make_env.py, the kit's schema subset
CATALOG = ROOT / "contracts" / "model-catalog.json"
SCHEMA = ROOT / "contracts" / "model-catalog.schema.json"
BASE_RECORD = ROOT / "contracts" / "base-environment.json"
CATALOG_VERSION = "apctt-model-catalog-v1"
STORE_ENV = "APCTT_MODEL_STORE"
STORE_LAYOUT = "<APCTT_MODEL_STORE>/<org>/<name>/<revision>/"
MARKER = ".apctt-model.json"
STORE_RECORD_VERSION = "apctt-model-store-v1"
LIBRARY_PACKAGES = {
    "transformers": "transformers",
    "sentence-transformers": "sentence-transformers",
}
SENTENCE_CLASSES = frozenset({"SentenceTransformer", "CrossEncoder", "SparseEncoder"})
QUANTIZATION_PACKAGES = {"bitsandbytes-nf4": "bitsandbytes", "bitsandbytes-int8": "bitsandbytes"}
DOCUMENT = ROOT / "docs" / "participant" / "MODEL_CATALOG.md"
CASE_LIMIT_SECONDS = 180  # D41: the 3-minute limit per case includes loading the model
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".onnx", ".pt", ".pth", ".ckpt", ".h5")
ENCODER_CATEGORIES = frozenset({"embedding", "reranker", "zero-shot", "encoder"})
PEAK_TOLERANCE_GIB = 0.05
EXIT_VALID, EXIT_INVALID, EXIT_USAGE = 0, 2, 3


class CatalogError(Exception):
    """A catalogue that cannot be read or is not valid; problems lists every finding."""

    def __init__(self, message: str, problems: Iterable[str] = ()) -> None:
        self.problems = list(problems) or [message]
        super().__init__(message if not self.problems else "; ".join(self.problems))


@dataclass(frozen=True)
class Model:
    id: str
    repo: str
    revision: str
    license: str
    category: str
    files: tuple[str, ...]
    download_bytes: int
    load: dict[str, Any]
    vram: dict[str, float]
    raw: dict[str, Any] = field(compare=False, repr=False)
    weights: dict[str, tuple[int, str]] = field(default_factory=dict, compare=False, repr=False)

    @property
    def org(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]

    def relative(self) -> PurePosixPath:
        """The model's folder inside a store: <org>/<name>/<revision>."""
        return PurePosixPath(self.org, self.name, self.revision)

    def directory(self, store: Path) -> Path:
        return store.joinpath(*self.relative().parts)

    @property
    def is_llm(self) -> bool:
        return self.category == "llm" or str(self.load.get("class", "")).endswith("ForCausalLM")


@dataclass(frozen=True)
class Catalog:
    path: Path
    sha256: str
    version: str
    status: str | None
    gpu_budget: dict[str, Any]
    models: tuple[Model, ...]
    notes: tuple[str, ...] = ()
    updated: str | None = None

    def get(self, model_id: object) -> Model | None:
        return next((m for m in self.models if m.id == model_id), None)

    def by_repo(self, repo: object) -> Model | None:
        key = str(repo).lower()
        return next((m for m in self.models if m.repo.lower() == key), None)

    def ids(self) -> list[str]:
        return [m.id for m in self.models]

    def weight_files(self) -> dict[tuple[int, str], tuple[str, str]]:
        """(size, sha256) of every catalogue weight file -> (model id, file name)."""
        return {pair: (m.id, name) for m in self.models for name, pair in m.weights.items()}

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "version": self.version, "status": self.status,
                "sha256": self.sha256}  # fmt: skip


# ------------------------------------------------------------------------------ checking


def base_packages(record: Path = BASE_RECORD) -> set[str]:
    """Package names of the common base (contracts/base-environment.json)."""
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    packages = data.get("packages") if isinstance(data, dict) else None
    return set(packages) if isinstance(packages, dict) else set()


def unsafe_path(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\0" in name
        or re.match(r"^[A-Za-z]:", name) is not None
        or any(part in ("..", ".", "") for part in name.split("/"))
        or pure.as_posix() != name
    )


def _class_problem(library: str, cls: str) -> str | None:
    if library == "transformers" and not re.fullmatch(r"Auto[A-Za-z]+", cls):
        return f"load.class {cls} is not a transformers Auto class (AutoModel...)"
    if library == "sentence-transformers" and cls not in SENTENCE_CLASSES:
        allowed = ", ".join(sorted(SENTENCE_CLASSES))
        return f"load.class {cls} is not a sentence-transformers class ({allowed})"
    return None


def _model_problems(
    model: dict[str, Any], budget: dict[str, Any], base: set[str] | None
) -> list[str]:
    where = f"model {model['id']}"
    problems: list[str] = []
    vram = model["vram_estimate_gib"]
    total = vram["weights"] + vram["kv_cache_or_activations"] + vram["overhead"]
    if abs(vram["peak"] - total) > PEAK_TOLERANCE_GIB:
        problems.append(
            f"{where}: vram_estimate_gib.peak {vram['peak']} is not weights + "
            f"kv_cache_or_activations + overhead ({total:.2f})"
        )
    target = budget["target_peak_gib"]
    if vram["peak"] > target:
        problems.append(
            f"{where}: estimated peak {vram['peak']} GiB is over the 8 GB budget "
            f"(target_peak_gib {target}); it cannot be in the catalogue (D64)"
        )
    if model["fits_8gb"] is not True:
        problems.append(f"{where}: fits_8gb is false; only models that fit the 8 GB GPU (D64)")
    load = model["load"]
    library, cls = load["library"], load["class"]
    package = LIBRARY_PACKAGES[library]
    if base is not None and package not in base:
        problems.append(f"{where}: load.library {library} is not in the common base")
    problem = _class_problem(library, cls)
    if problem:
        problems.append(f"{where}: {problem}")
    quantization = load["quantization"]
    if quantization is not None:
        needed = QUANTIZATION_PACKAGES[quantization["method"]]
        if base is not None and needed not in base:
            problems.append(
                f"{where}: quantization {quantization['method']} needs {needed}, which is not in "
                "the common base, so the model cannot be loaded offline"
            )
    for name in model["files"]:
        if unsafe_path(name):
            problems.append(f"{where}: unsafe file path {name!r}")
    if not any(name.lower().endswith(WEIGHT_SUFFIXES) for name in model["files"]):
        problems.append(f"{where}: files lists no weight file ({', '.join(WEIGHT_SUFFIXES)})")
    weights = model["weights"]
    for name in model["files"]:
        if name.lower().endswith(WEIGHT_SUFFIXES) and name not in weights:
            problems.append(f"{where}: weight file {name} has no size and sha256 in weights")
    for name in sorted(set(weights) - set(model["files"])):
        problems.append(f"{where}: weights names {name}, which is not in files")
    return problems


def problems(data: object, base: set[str] | None = None) -> list[str]:
    """Every problem of a catalogue: the schema, then the rules above. base: package names of
    the common base (default: contracts/base-environment.json)."""
    from make_env import validate  # noqa: PLC0415 - the kit's schema subset, loaded when needed

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    found = validate(data, schema) + _duplicates(data)
    if found:
        return found
    assert isinstance(data, dict)
    base = base_packages() if base is None else base
    budget = data["gpu_budget"]
    if budget["target_peak_gib"] > budget["vram_gib"]:
        found.append("gpu_budget.target_peak_gib is more than vram_gib")
    for model in data["models"]:
        found += _model_problems(model, budget, base)
    return found


def _duplicates(data: object) -> list[str]:
    """Ids and repositories listed twice (read even when the schema is not met)."""
    models = data.get("models") if isinstance(data, dict) else None
    found: list[str] = []
    seen_ids: set[str] = set()
    seen_repos: set[str] = set()
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict):
            continue
        model_id, repo = model.get("id"), model.get("repo")
        if isinstance(model_id, str):
            if model_id in seen_ids:
                found.append(f"id {model_id} is listed twice")
            seen_ids.add(model_id)
        if isinstance(repo, str):
            if repo.lower() in seen_repos:
                found.append(f"repository {repo} is listed twice")
            seen_repos.add(repo.lower())
    return found


def model_of(entry: dict[str, Any]) -> Model:
    return Model(
        id=entry["id"],
        repo=entry["repo"],
        revision=entry["revision"],
        license=entry["license"],
        category=entry["category"],
        files=tuple(entry["files"]),
        download_bytes=int(entry["download_bytes"]),
        load=dict(entry["load"]),
        vram={k: float(v) for k, v in entry["vram_estimate_gib"].items()},
        raw=entry,
        weights={
            str(name): (int(item["size"]), str(item["sha256"]))
            for name, item in entry["weights"].items()
        },
    )


def load(path: Path = CATALOG, base: set[str] | None = None) -> Catalog:
    """The checked catalogue; CatalogError lists every problem."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise CatalogError(f"{path}: not found") from None
    except OSError as exc:
        raise CatalogError(f"{path}: {exc.strerror or exc}") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CatalogError(f"{path}: not readable JSON ({exc})") from None
    found = problems(data, base)
    if found:
        raise CatalogError(
            f"{path}: not a valid model catalogue", [f"{path.name}: {p}" for p in found]
        )
    return Catalog(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        version=data["catalog_version"],
        status=data.get("status"),
        gpu_budget=dict(data["gpu_budget"]),
        models=tuple(model_of(entry) for entry in data["models"]),
        notes=tuple(data.get("notes", [])),
        updated=data.get("updated"),
    )


# ------------------------------------------------------------------------------- loading


def quantization_config(model: Model) -> dict[str, Any] | None:
    """Keyword arguments of transformers.BitsAndBytesConfig for a quantized entry (None for the
    others); the compute dtype is the entry's dtype, by name. bitsandbytes-nf4 is 4-bit NF4 with
    double quantisation, the recipe the catalogue's estimates assume and HQ measures."""
    quantization = model.load["quantization"]
    if quantization is None:
        return None
    if quantization["method"] == "bitsandbytes-int8":
        return {"load_in_8bit": True}
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": model.load["dtype"],
    }


def _config_source(config: dict[str, Any]) -> str:
    parts = []
    for key, value in config.items():
        if key == "bnb_4bit_compute_dtype":
            parts.append(f"{key}=torch.{value}")
        elif isinstance(value, str):
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    return ",\n    ".join(parts)


def loading_snippet(model: Model) -> str:
    """Python that loads the model offline from the store, the way a team module should."""
    relative = model.relative().as_posix()
    head = (
        "import os\n"
        'os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before transformers is imported\n'
        'os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")\n'
        "from pathlib import Path\n\n"
        f'path = Path(os.environ["{STORE_ENV}"]) / "{relative}"\n'
    )
    load = model.load
    if load["library"] == "sentence-transformers":
        cls = load["class"]
        return head + (
            f"from sentence_transformers import {cls}\n\n"
            f'model = {cls}(str(path), device="cpu", local_files_only=True)  # or "cuda"\n'
        )
    dtype = load["dtype"]
    cls = load["class"]
    config = quantization_config(model)
    if config is not None:
        return head + (
            "import torch\n"
            f"from transformers import AutoTokenizer, BitsAndBytesConfig, {cls}\n\n"
            "# Quantized on load: needs a CUDA GPU (bitsandbytes). The store holds the bf16 "
            "weights,\n# which are read and quantised on every load, inside the 3-minute limit "
            "of a case.\n"
            f"quantization = BitsAndBytesConfig(\n    {_config_source(config)},\n)\n"
            "tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)\n"
            f"model = {cls}.from_pretrained(\n"
            f"    path, local_files_only=True, dtype=torch.{dtype},\n"
            '    quantization_config=quantization, device_map={"": 0},\n'
            ").eval()\n"
        )
    return head + (
        "import torch\n"
        f"from transformers import AutoTokenizer, {cls}\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)\n"
        f"model = {cls}.from_pretrained(path, local_files_only=True, dtype=torch.{dtype})\n"
        'model.to("cuda" if torch.cuda.is_available() else "cpu").eval()\n'
    )


# ------------------------------------------------------------------------------ document


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1000 or unit == "GB":
            return f"{int(count)} B" if unit == "B" else f"{count:.2f} {unit}"
        count /= 1000
    return f"{count:.2f} GB"  # pragma: no cover - the loop returns


def _cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _num(value: float) -> str:
    """1.0, 0.92, 7.0: at least one decimal, at most two."""
    text = f"{value:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _vram(model: Model) -> str:
    vram = model.vram
    parts = " + ".join(_num(vram[k]) for k in ("weights", "kv_cache_or_activations", "overhead"))
    return f"{vram['peak']:.2f} GiB ({parts})"


def _kind(model: Model) -> str:
    quantization = model.load["quantization"]
    if quantization is not None:
        return f"{model.category}, 4-bit ({quantization['method']})"
    return f"{model.category}, {model.load['dtype']}"


def markdown(catalog: Catalog) -> str:
    """docs/participant/MODEL_CATALOG.md: the catalogue as a table with its evidence."""
    budget = catalog.gpu_budget
    gpu, vram = budget["reference_gpu"], _num(budget["vram_gib"])
    target, overhead = _num(budget["target_peak_gib"]), _num(budget["overhead_gib"])
    total = sum(m.download_bytes for m in catalog.models)
    quantized = [m for m in catalog.models if m.load["quantization"] is not None]
    large = [m for m in quantized if float(m.raw["params_b"]) >= 6]
    lines = [
        "# Public model catalogue",
        "",
        f"catalogue `{catalog.version}`, status **{catalog.status}**, updated "
        f"{catalog.updated or 'unknown'}",
        "",
        "Generated from `contracts/model-catalog.json` (sha256 "
        f"`{catalog.sha256}`) by `python tools/model_catalog.py --markdown`; do not edit by hand.",
        "",
        "These are the organizer's recommended public models for the **reference GPU: "
        f"{gpu}**, the GPU of the event machine with 8 GB VRAM (`gpu_budget.reference_gpu`; "
        "decisions D40, D64 and D67). The organizer provides every model offline and read-only in "
        "the model store of "
        "each evaluation machine (the worker finds it in `APCTT_MODEL_STORE`). Only these models "
        "and weights your team trained (including fine-tunes and adapters of a catalogue model, "
        "named by `base_model`) may be used; a public model outside the catalogue cannot be used: "
        "each evaluation machine has one shared read-only model store, which cannot hold per-team "
        "models for this many teams, and public weights may not be shipped in your zip (decision "
        "D80). The files of a catalogue model are never in your zip: "
        "`tools/make_submission.py` refuses any file with the bytes of a catalogue weight file "
        "(the sha256 values below), however it is named or declared.",
        "",
        "**VRAM figures are estimates** until the organizer runs "
        "`tools/hq/check_models_gpu.py` on the event GPU (`gpu_budget.measured`: "
        f"{str(budget['measured']).lower()}). Each estimate "
        f"is weights + KV cache or activations + overhead = peak, at most the target of {target} "
        f"GiB of the {vram} GiB card: LLMs at {budget['llm_context_tokens']:,} tokens of context "
        f"(batch 1), encoders at a batch of {budget['encoder_batch']}, with {overhead} GiB of "
        "overhead for the CUDA context and temporary buffers.",
        "",
    ]
    if quantized:
        names = ", ".join(f"`{m.id}`" for m in quantized)
        lines += [
            f"**4-bit entries ({names}) are conditional on the GPU measurement** "
            "(`measured`: false). They load the publisher's bf16 weights in 4 bits with "
            "bitsandbytes on a CUDA GPU; none of them runs on the CPU. The organizer runs "
            "`tools/hq/check_models_gpu.py` on the "
            f"{gpu} before the kit is published and removes any entry that does not load within "
            f"the budget: peak at most {target} GiB and a load well inside the 3-minute limit of "
            "a case (decision D41: the limit includes loading the model).",
            "",
        ]
    for model in large:
        lines += [
            f"**Cold-load warning, `{model.id}`:** every case streams "
            f"{_size(model.download_bytes)} of bf16 weights from the SSD and quantises them on "
            "the GPU before the first token, inside the 3-minute limit (D41). The event machine "
            "has 16 GB of RAM (D76), which cannot keep these weights in the page cache between "
            "cases, so no case starts warm. Time the load on your machine first, keep the "
            "generation short, and have a smaller model as a fallback.",
            "",
        ]
    lines += [
        "Download for development: `python tools/get_models.py --list`, then "
        "`python tools/get_models.py ID [ID ...] --store DIR`; `python tools/get_models.py "
        "--snippet ID` prints the offline loading code. Every catalogue model is Apache-2.0 or "
        "MIT; the tool adds the licence text to a copy whose repository has none.",
        "",
        f"## Summary: {len(catalog.models)} models, {_size(total)} to download",
        "",
        f"All files of every model: {total:,} bytes ({total / 1024**3:.2f} GiB). By category:",
        "",
    ]
    order = ["llm, bf16", "llm, 4-bit (GPU only)", "embedding", "reranker", "zero-shot"]
    order += ["encoder", "other"]
    groups: dict[str, list[Model]] = {}
    for model in catalog.models:
        key = model.category
        if key == "llm":
            key = "llm, 4-bit (GPU only)" if model.load["quantization"] else "llm, bf16"
        groups.setdefault(key, []).append(model)
    for key in sorted(groups, key=order.index):
        models = groups[key]
        size = sum(m.download_bytes for m in models)
        ids = ", ".join(f"`{m.id}`" for m in models)
        lines.append(f"- {key} ({len(models)}, {_size(size)}): {ids}")
    lines += [
        "",
        "## Models",
        "",
        "| id | repository@revision | licence | category | params | download | VRAM estimate "
        "(peak) | CPU | use | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in catalog.models:
        raw = model.raw
        lines.append(
            f"| `{model.id}` | `{model.repo}@{model.revision[:12]}` | {model.license} | "
            f"{_cell(_kind(model))} | {raw['params_b']:g}B | {_size(model.download_bytes)} | "
            f"{_vram(model)} | {_cell(raw['cpu_feasible'])} | {_cell(raw['use'])} | "
            f"{_cell(raw['notes'])} |"
        )
    lines += ["", "## Evidence", ""]
    for model in catalog.models:
        raw = model.raw
        load = model.load
        quantization = load["quantization"]["method"] if load["quantization"] else "none"
        lines += [
            f"### {model.id}",
            "",
            f"- Repository: `{model.repo}`, revision `{model.revision}`; licence {model.license} "
            f"(licence file in the repository: {'yes' if raw['license_file_in_repo'] else 'no'}).",
            f"- Licence evidence: {_cell(raw['license_evidence'])}",
            f"- Task: {_cell(raw['task'])}. Load: {load['library']} `{load['class']}`, dtype "
            f"{load['dtype']}, quantization {quantization}, trust_remote_code false.",
            f"- Files ({len(model.files)}, {model.download_bytes:,} bytes): "
            + ", ".join(f"`{name}`" for name in model.files),
            "- Weight files (size, sha256): "
            + "; ".join(
                f"`{name}` {size:,} bytes `{sha}`" for name, (size, sha) in model.weights.items()
            ),
            f"- Load check with the base: {_cell(raw['verified_load_with_base'])}",
            "",
        ]
    lines += ["## Catalogue notes", ""]
    for note in catalog.notes:
        lines.append(f"- {_cell(note)}")
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------------------ CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="model_catalog.py", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("catalog", nargs="?", type=Path, default=CATALOG)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the result as JSON")
    output.add_argument("--markdown", action="store_true",
                        help="print docs/participant/MODEL_CATALOG.md")  # fmt: skip
    args = parser.parse_args(argv)
    if not args.catalog.is_file():
        print(f"model_catalog.py: {args.catalog}: not found", file=sys.stderr)
        return EXIT_USAGE
    try:
        catalog = load(args.catalog)
    except CatalogError as exc:
        if args.json:
            print(json.dumps({"valid": False, "problems": exc.problems}, indent=2))
        else:
            for problem in exc.problems:
                print(f"PROBLEM {problem}")
            print(f"{args.catalog}: INVALID")
        return EXIT_INVALID
    if args.markdown:
        sys.stdout.write(markdown(catalog))
    elif args.json:
        print(json.dumps({"valid": True, **catalog.identity(), "ids": catalog.ids()}, indent=2))
    else:
        for model in catalog.models:
            print(f"{model.id:28} {model.repo}@{model.revision[:12]} {model.license}")
        print(f"{args.catalog}: VALID ({len(catalog.models)} models, status {catalog.status})")
    return EXIT_VALID


if __name__ == "__main__":
    sys.exit(main())
