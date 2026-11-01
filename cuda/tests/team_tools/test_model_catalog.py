"""tools/model_catalog.py: the public model catalogue of decision D64 (model-catalog.json).

One loader and validator for every tool: the schema (contracts/model-catalog.schema.json, read with
the kit's own schema subset) plus the rules a schema cannot express. The published catalogue is the
organizer's verified selection for the reference GPU (RTX 3070 Laptop GPU, 8 GB; D40, D64), with
4-bit (bitsandbytes nf4) LLMs that the common base can load (D67); its participant document
docs/participant/MODEL_CATALOG.md is generated from it. No network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.team_tools.helpers import ROOT, catalog_data, load_tool, mit_model, tiny_model

model_catalog = load_tool("model_catalog")
make_env = load_tool("make_env")


def problems_of(data: dict[str, Any]) -> str:
    return "\n".join(model_catalog.problems(data))


def test_the_published_catalogue_is_valid() -> None:
    catalog = model_catalog.load()
    assert catalog.version == "apctt-model-catalog-v1"
    assert catalog.path == ROOT / "contracts/model-catalog.json"
    assert len(catalog.sha256) == 64 and catalog.models
    ids = [m.id for m in catalog.models]
    assert len(ids) == len(set(ids))
    for model in catalog.models:
        assert model.license in ("Apache-2.0", "MIT"), model.id
        assert model.raw["fits_8gb"] is True and model.raw["gated"] is False
        assert model.raw["load"]["trust_remote_code"] is False
        assert model.vram["peak"] <= catalog.gpu_budget["target_peak_gib"], model.id
        assert catalog.get(model.id) is model
        assert catalog.by_repo(model.repo.upper()) is model  # repositories are case-insensitive
        org, name = model.repo.split("/")
        assert model.relative().as_posix() == f"{org}/{name}/{model.revision}"
    assert catalog.get("no-such-model") is None
    assert model_catalog.problems(json.loads(catalog.path.read_text())) == []


def test_the_schema_uses_only_what_the_kit_validator_reads() -> None:
    schema = json.loads((ROOT / "contracts/model-catalog.schema.json").read_text())
    used: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                used.add(key)
                if key in ("properties", "$defs"):
                    for child in value.values():
                        walk(child)
                elif key not in ("enum", "const", "required", "examples", "default"):
                    walk(value)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)
    assert used <= make_env.SUPPORTED_KEYWORDS, used - make_env.SUPPORTED_KEYWORDS
    assert schema["properties"]["store_layout"]["const"] == model_catalog.STORE_LAYOUT


def test_a_small_valid_catalogue() -> None:
    assert model_catalog.problems(catalog_data()) == []


@pytest.mark.parametrize(
    ("models", "changes", "expected"),
    [
        ([tiny_model(), tiny_model(repo="test-org/other")], {}, "id tiny-encoder is listed twice"),
        ([tiny_model(), mit_model(repo="Test-Org/Tiny-Encoder")], {}, "listed twice"),
        ([tiny_model(revision="main")], {}, "revision"),
        ([tiny_model(license="GPL-3.0-only")], {}, "license"),
        ([tiny_model(gated=True)], {}, "gated"),
        ([tiny_model(id="Tiny_Encoder")], {}, "id"),
        (
            [
                tiny_model(
                    vram_estimate_gib={
                        "weights": 1.0,
                        "kv_cache_or_activations": 1.0,
                        "overhead": 1.0,
                        "peak": 2.0,
                    }
                )
            ],
            {},
            "weights + kv_cache_or_activations + overhead",
        ),
        (
            [
                tiny_model(
                    vram_estimate_gib={
                        "weights": 6.0,
                        "kv_cache_or_activations": 0.5,
                        "overhead": 1.0,
                        "peak": 7.5,
                    }
                )
            ],
            {},
            "8 GB budget",
        ),
        ([tiny_model(fits_8gb=False)], {}, "fits_8gb"),
        ([tiny_model(measured="no")], {}, "measured"),
        (
            [
                tiny_model(
                    load={
                        "library": "sentence-transformers",
                        "class": "SentenceTransformers",
                        "dtype": "float32",
                        "quantization": None,
                        "trust_remote_code": False,
                    }
                )
            ],
            {},
            "SentenceTransformers",
        ),
        (
            [
                mit_model(
                    load={
                        "library": "transformers",
                        "class": "LlamaForCausalLM",
                        "dtype": "bfloat16",
                        "quantization": None,
                        "trust_remote_code": False,
                    }
                )
            ],
            {},
            "LlamaForCausalLM",
        ),
        (
            [
                tiny_model(
                    load={
                        "library": "sentence-transformers",
                        "class": "SentenceTransformer",
                        "dtype": "float32",
                        "quantization": None,
                        "trust_remote_code": True,
                    }
                )
            ],
            {},
            "trust_remote_code",
        ),
        ([tiny_model(files=["../escape.safetensors"])], {}, "unsafe"),
        ([tiny_model(files=["config.json", "tokenizer.json"])], {}, "no weight file"),
        ([tiny_model(cpu_feasible="maybe")], {}, "cpu_feasible"),
        ([tiny_model(weights={})], {}, "model.safetensors has no size and sha256"),
        (
            [tiny_model(weights={"other.safetensors": {"size": 1, "sha256": "0" * 64}})],
            {},
            "weights names other.safetensors, which is not in files",
        ),
        (
            [tiny_model(weights={"model.safetensors": {"size": 1, "sha256": "ABC"}})],
            {},
            "sha256",
        ),
        ([{k: v for k, v in tiny_model().items() if k != "weights"}], {}, "weights"),
        ([tiny_model()], {"store_layout": "<store>/<repo>"}, "store_layout"),
        ([tiny_model()], {"catalog_version": "v0"}, "catalog_version"),
        ([], {}, "models"),
    ],
)
def test_refusals(models: list[dict[str, Any]], changes: dict[str, Any], expected: str) -> None:
    data = catalog_data(*models, **changes) if models else catalog_data(models=[], **changes)
    found = problems_of(data)
    assert expected in found, found


def test_every_weight_file_has_its_hash_in_the_catalogue() -> None:
    """The size and LFS sha256 of each weight file, which make_submission.py uses to refuse
    catalogue weights in a bundle and get_models.py checks on download (D64)."""
    data = json.loads(model_catalog.CATALOG.read_text(encoding="utf-8"))
    for entry in data["models"]:
        weights = [n for n in entry["files"] if n.lower().endswith(model_catalog.WEIGHT_SUFFIXES)]
        assert sorted(entry["weights"]) == sorted(weights), entry["id"]
    catalog = model_catalog.load()
    index = catalog.weight_files()
    minilm = catalog.get("all-minilm-l6-v2")
    assert minilm is not None
    size, sha = minilm.weights["model.safetensors"]
    assert index[(size, sha)] == ("all-minilm-l6-v2", "model.safetensors")
    assert sha.startswith("53aa5117")  # the bytes of the store's copy (repository LFS record)
    assert len(index) == sum(len(m.weights) for m in catalog.models)


def test_a_target_above_the_card_is_refused() -> None:
    data = catalog_data()
    data["gpu_budget"]["target_peak_gib"] = 9
    assert "target_peak_gib" in problems_of(data)


NF4_LOAD = {
    "library": "transformers",
    "class": "AutoModelForCausalLM",
    "dtype": "bfloat16",
    "quantization": {"method": "bitsandbytes-nf4"},
    "trust_remote_code": False,
}


def test_libraries_and_quantization_come_from_the_base() -> None:
    base = model_catalog.base_packages()
    assert {"transformers", "sentence-transformers", "torch", "bitsandbytes"} <= base  # D67
    data = catalog_data()
    assert "not in the common base" in "\n".join(model_catalog.problems(data, {"torch"}))
    four_bit = catalog_data(tiny_model(), mit_model(load=dict(NF4_LOAD), measured=False))
    assert model_catalog.problems(four_bit) == []
    found = "\n".join(model_catalog.problems(four_bit, base - {"bitsandbytes"}))
    assert "needs bitsandbytes, which is not in the common base" in found


def test_the_published_catalogue_is_the_verified_selection_for_the_reference_gpu() -> None:
    catalog = model_catalog.load()
    raw = json.loads(catalog.path.read_text(encoding="utf-8"))
    assert catalog.status == "verified" and "author" not in raw  # no personal attribution (D71)
    budget = catalog.gpu_budget
    assert "RTX 3070 Laptop GPU" in budget["reference_gpu"] and budget["vram_gib"] == 8
    # D76 (2026-10-13): the final event machine, named where the reference GPU is.
    assert "Asus TUF Dash F15 FX516PR-HN002" in budget["reference_gpu"]
    assert "D76" in budget["reference_gpu"] and catalog.updated == "2026-10-13"
    assert budget["target_peak_gib"] == 7.0 and budget["measured"] is False
    assert "RTX 3070 Laptop GPU with 8 GB VRAM" in raw["notes"][0]  # the reference GPU (e)
    categories = {m.category for m in catalog.models}
    assert {"llm", "embedding", "reranker", "zero-shot", "encoder"} <= categories
    quantized = [m for m in catalog.models if m.load["quantization"] is not None]
    assert len(quantized) >= 2 and catalog.get("qwen3-4b-instruct-2507-nf4") in quantized
    for model in quantized:  # conditional on HQ's GPU measurement (D67)
        assert model.load["quantization"]["method"] == "bitsandbytes-nf4", model.id
        assert model.raw["measured"] is False, model.id
        assert model.raw["cpu_feasible"].startswith("no"), model.id
        assert "check_models_gpu.py" in model.raw["notes"], model.id
        if float(model.raw["params_b"]) >= 6:  # 7-8B: the cold-load warning
            assert "COLD-LOAD WARNING" in model.raw["notes"], model.id
            assert "16 GB of RAM" in model.raw["notes"], model.id  # D76: streamed every case
    for model in catalog.models:
        if model.load["quantization"] is None:
            assert model.raw.get("measured", False) is False, model.id
    # Every category has a model that runs on a CPU (D57: CPU-only modules are allowed).
    for category in categories:
        assert any(
            m.category == category and m.raw["cpu_feasible"].startswith("yes")
            for m in catalog.models
        ), category


def test_quantization_recipe_and_its_loading_snippet() -> None:
    catalog = model_catalog.load()
    model = catalog.get("qwen3-4b-instruct-2507-nf4")
    assert model_catalog.quantization_config(model) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert model_catalog.quantization_config(catalog.get("qwen2.5-0.5b-instruct")) is None
    int8 = model_catalog.model_of(
        mit_model(load={**NF4_LOAD, "quantization": {"method": "bitsandbytes-int8"}})
    )
    assert model_catalog.quantization_config(int8) == {"load_in_8bit": True}
    snippet = model_catalog.loading_snippet(model)
    compile(snippet, "<nf4 snippet>", "exec")
    for phrase in (
        "BitsAndBytesConfig(",
        "load_in_4bit=True",
        'bnb_4bit_quant_type="nf4"',
        "bnb_4bit_use_double_quant=True",
        "bnb_4bit_compute_dtype=torch.bfloat16",
        'device_map={"": 0}',
        "local_files_only=True",
        "Qwen/Qwen3-4B-Instruct-2507/cdbee75f17c01a7cc42f958dc650907174af0554",
    ):
        assert phrase in snippet, phrase
    assert ".to(" not in snippet  # a quantized model is placed by device_map, never moved


def test_the_participant_document_is_generated_from_the_catalogue() -> None:
    catalog = model_catalog.load()
    document = (ROOT / "docs/participant/MODEL_CATALOG.md").read_text(encoding="utf-8")
    assert document == model_catalog.markdown(catalog)  # regenerate with --markdown
    assert catalog.sha256 in document
    header = document.splitlines()[2]
    assert header.startswith("catalogue `") and "Author" not in header  # no author line (D71)
    for model in catalog.models:
        assert f"| `{model.id}` | `{model.repo}@{model.revision[:12]}` |" in document, model.id
    flat = " ".join(document.split())
    for phrase in (
        "reference GPU: RTX 3070 Laptop GPU",
        "Asus TUF Dash F15 FX516PR-HN002",
        "VRAM figures are estimates",
        "tools/hq/check_models_gpu.py",
        "conditional on the GPU measurement",
        "Cold-load warning",
        "3-minute limit",
        "16 GB of RAM",  # D76: the 7-8B weights are streamed from the SSD in every case
        "streams",
    ):
        assert phrase in flat, phrase
    tool = [sys.executable, str(ROOT / "tools/model_catalog.py"), "--markdown"]
    printed = subprocess.run(tool, capture_output=True, text=True, timeout=60, check=False)
    assert printed.returncode == 0 and printed.stdout == document


def test_load_raises_with_every_problem(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog_data(tiny_model(revision="x"), tiny_model())))
    with pytest.raises(model_catalog.CatalogError) as caught:
        model_catalog.load(path)
    assert "revision" in str(caught.value) and "twice" in str(caught.value)
    path.write_text("{not json")
    with pytest.raises(model_catalog.CatalogError, match="JSON"):
        model_catalog.load(path)
    with pytest.raises(model_catalog.CatalogError, match="not found"):
        model_catalog.load(tmp_path / "missing.json")


def test_loading_snippets_are_offline_and_valid_python(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog_data()))
    catalog = model_catalog.load(path)
    for model in catalog.models:
        snippet = model_catalog.loading_snippet(model)
        compile(snippet, f"<snippet {model.id}>", "exec")
        offline = snippet.index('os.environ.setdefault("HF_HUB_OFFLINE", "1")')
        imports = [snippet.index(m) for m in ("from transformers", "from sentence_transformers")
                   if m in snippet]  # fmt: skip
        assert imports and all(offline < where for where in imports), snippet
        assert 'os.environ["APCTT_MODEL_STORE"]' in snippet
        assert model.relative().as_posix() in snippet
        assert "local_files_only=True" in snippet
    encoder = model_catalog.loading_snippet(catalog.get("tiny-encoder"))
    assert "SentenceTransformer(" in encoder
    llm = model_catalog.loading_snippet(catalog.get("tiny-mit"))
    assert "AutoModelForCausalLM.from_pretrained(" in llm and "torch.bfloat16" in llm


def test_cli_validates_a_catalogue(tmp_path: Path) -> None:
    tool = [sys.executable, str(ROOT / "tools/model_catalog.py")]
    ok = subprocess.run(tool, capture_output=True, text=True, timeout=60, check=False)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "VALID" in ok.stdout and "qwen2.5-1.5b-instruct" in ok.stdout
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(catalog_data(tiny_model(gated=True))))
    refused = subprocess.run(
        [*tool, str(bad)], capture_output=True, text=True, timeout=60, check=False
    )
    assert refused.returncode == 2 and "gated" in refused.stdout
    missing = subprocess.run(
        [*tool, str(tmp_path / "missing.json")], capture_output=True, timeout=60, check=False
    )
    assert missing.returncode == 3
