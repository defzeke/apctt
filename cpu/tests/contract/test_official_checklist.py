"""The official checklist (decision D74): contracts/checklist.official.json, shipped in both kits.

The organizer made the C01-C12 draft official on 2026-10-13. It must load as a checklist, pass the
evaluation profile's checklist gates (official, not DEMO, not a draft) and be the version that
the public sample cases carry. It is the default of the kit's configurations (config/dev.json and
config/cuda-example.json) and of the example request; the DEMO checklist stays for the protocol
tests and for the exercise with a changed checklist (README step 9).
"""

from __future__ import annotations

import json
from pathlib import Path

from apctt_contracts import DEMO_CHECKLIST_VERSION
from apctt_contracts.config import Checklist, load_config, startup_problems

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "contracts/checklist.official.json"


def test_the_official_checklist_is_the_c01_to_c12_list() -> None:
    raw = json.loads(OFFICIAL.read_text())
    checklist = Checklist.model_validate(raw)
    assert checklist.version == "APCTT-2026-CHECKLIST-v1"
    assert checklist.official is True and "DRAFT" not in checklist.version.upper()
    ids = [c.criterion_id for c in checklist.criteria]
    assert ids == [f"APCTT-C{n:02d}" for n in range(1, 13)]
    for number, criterion in enumerate(checklist.criteria[:9], start=1):
        assert f"(TRL {number})" in criterion.description, criterion.criterion_id
    public = json.loads((ROOT / "contracts/public-cases.json").read_text())
    assert public["checklist_version"] == checklist.version


def test_the_evaluation_profile_accepts_the_official_checklist(tmp_path: Path) -> None:
    config = json.loads((ROOT / "config/dev.json").read_text())
    config.update(
        profile="evaluation",
        schema_version="apctt-2026-v1",
        checklist_path=str(OFFICIAL),
        data_dir=str(tmp_path / "server_data"),
        worker_tmp_dir=str(tmp_path / "worker_tmp"),
    )
    config["module"].update(
        module_root=str(ROOT),
        package_root=str(ROOT / "participant_example"),
        asset_root=str(ROOT / "participant_example"),
    )
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(config))
    problems = startup_problems(load_config(path, profile="evaluation"))
    assert not [p for p in problems if p.startswith("EVALUATION_CHECKLIST")], problems
    assert "EVALUATION_DEV_SCHEMA" not in problems


def test_the_official_checklist_is_the_default_of_every_example_configuration() -> None:
    """D74: whoever starts the kit's server with a shipped configuration gets the official
    checklist, and the example request is for it. The CUDA configuration ships only in the CUDA
    kit; in a kit without it, it is not read."""
    ids = [f"APCTT-C{n:02d}" for n in range(1, 13)]
    for relative in ("config/dev.json", "config/cuda-example.json"):
        if not (ROOT / relative).exists():
            continue
        config = json.loads((ROOT / relative).read_text())
        assert config["checklist_path"] == "../contracts/checklist.official.json", relative
        assert list(config["module"]["settings"]["criteria_phrases"]) == ids, relative
        loaded = load_config(ROOT / relative, profile="dev")
        assert loaded.checklist.version == "APCTT-2026-CHECKLIST-v1" and loaded.checklist.official
    request = json.loads((ROOT / "docs/EXAMPLE_REQUEST.json").read_text())
    assert request["case"]["checklist_version"] == "APCTT-2026-CHECKLIST-v1"
    response = json.loads((ROOT / "docs/EXAMPLE_RESPONSE.json").read_text())
    assert list(response["result"]["criterion_status"]) == ids
    demo = json.loads((ROOT / "contracts/checklist.demo.json").read_text())
    assert demo["version"] == DEMO_CHECKLIST_VERSION and demo["official"] is False
