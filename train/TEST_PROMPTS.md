# Testing the fine-tuned model

Two things can be tested:

1. the **adapter alone** (TRL prediction from report excerpts), and
2. the **shipped module** (rules plus optional adapter, criteria and evidence).

Use the exact prompt format below. The adapter was trained on that format; a loosely
worded prompt gives worse and less comparable answers.

## 1. One prompt at a time

From Git Bash in the repository root (inside WSL, drop the `wsl ... -lc '` wrapper):

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "$(cygpath -w "$PWD")" -- bash -lc '\
  ~/.venvs/apctt-train/bin/python train/infer_check.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/test_prompts.jsonl --index 0'
```

`--index` selects the record: 0 = USGS-SDG-076, 1 = 061, 2 = 040, 3 = 012, 4 = 096,
5 = 100. The output shows the generated text, the parsed JSON, whether the TRL matches,
generation seconds and peak VRAM.

Print a prompt without running the model:

```bash
~/.venvs/apctt-train/bin/python - <<'PY'
import json
rows = [json.loads(line) for line in open("train/test_prompts.jsonl", encoding="utf-8")]
print(rows[2]["messages"][1]["content"])
PY
```

## 2. All six prompts

```bash
~/.venvs/apctt-train/bin/python train/evaluate.py \
  --adapter my_submission/assets/smollm3-trl-lora \
  --data train/test_prompts.jsonl --out train/out/test-prompts-eval.json
```

It loads the base and adapter once, then scores every record: exact = 1.0, off by one =
0.5, off by two or more = 0.0, invalid JSON = 0.

Measured on 2026-11-02 (RTX 3050, about 90 s for the whole set):

| case | label | predicted | score | note |
|---|---|---|---|---|
| USGS-SDG-076 | 6 | 6 | 1.0 | held out (SDG 14) |
| USGS-SDG-061 | 6 | 6 | 1.0 | in training |
| USGS-SDG-040 | 7 | 7 | 1.0 | in training |
| USGS-SDG-012 | 4 | 6 | 0.0 | early-stage report, adapter over-predicts |
| USGS-SDG-096 | 8 | 6 | 0.0 | mature system, adapter under-predicts |
| USGS-SDG-100 | 7 | 6 | 0.5 | |

Summary: mean 0.5833, 3 exact, 1 off-by-one, 2 off-by-two, 0 invalid, 13.7 s per case
after load, 2.08 GiB peak VRAM. This is the known TRL-6 collapse, not a loading problem.

## 3. Through the shipped module

```bash
~/.venvs/apctt-train/bin/python train/module_eval.py --llm \
  --only USGS-SDG-076,USGS-SDG-061,USGS-SDG-040,USGS-SDG-012,USGS-SDG-096,USGS-SDG-100
```

This path extracts excerpts from the PDF itself, runs the rules first, then lets the
adapter adjust the TRL. It is the closest test to the real 180 s case path.

The default submission disables the adapter (`settings.use_llm = false`); add `--llm` only
to test this path.

## 4. Prompt format

The training prompt is a system message plus one user message:

```text
You are a technology-readiness assessor. Read the report excerpts and reply with one JSON object only: {"trl": <integer 1-9>, "rationale": "<one sentence>"}.
```

```text
Assessment target: <target description> (type: <target type>)
Intended application: <SDG target text>
Report: <report title> (<year>)

Report excerpts:
[page N] <about 600 characters>
[page N] <about 600 characters>
[page N] <about 600 characters>
[page N] <about 600 characters>

Which TRL does the evidence support?
```

Excerpts come from 4 pages, ranked by readiness keywords (`prototype`, `demonstrat`,
`field`, `validat`, `test`, `deploy`, `operat`, `calibrat`, `monitor`, `pilot`, ...), about
2,400 characters in total. The model was trained on about 780 tokens per prompt.

Full example (USGS-SDG-061, label 6, predicted 6):

```text
Assessment target: Rio Hondo PRMS prefire/postfire hydrologic model (type: Hydrologic models)
Intended application: 13.1 Climate-disaster resilience
Report: Calibration of Precipitation-Runoff Modeling System (PRMS) to simulate prefire and postfire hydrologic response in the upper Rio Hondo Basin, New Mexico (2019)

Report excerpts:
[page 14] 6 Calibration of Precipitation-Runoff Modeling System precipitation, climate, and land use on streamflow and general basin hydrology on a daily time step. The national-scale framework of the NHM allows for subcatchment modeling by using predefined, spatially distributed hydrologic response units (HRUs). HRUs partition the model domain into areas with relatively uniform hydrologic response that are based on watershed topography, vegetation, soil, and climate. The model input data include precipitation, minimum temperature, and maximum temperature. The PRMS model represents the hydrologic cycle 

[page 16] 8 Calibration of Precipitation-Runoff Modeling System Fourier Amplitude Sensitivity Test A national-level sensitivity analysis was performed by Markstrom and others (2016) to better understand interactions between PRMS calibration parameters and process variables. They used a Fourier amplitude sensitivity test (FAST) procedure to analyze 110,000 subregions, or HRUs, of the coterminous United States for an 11-year evaluation period (1990–2000). The FAST analysis used seven fundamental daily streamflow statistics, where each statistic represented nonredundant information on hydrologic response (

[page 23] PRMS Model Calibration for the North Fork Eagle Creek Subwatershed 15 rad_trnc = 1.0652 [covden_xxx] + 0.86557, R2 =0.94 (3) snow_intcp = 0.07 [covden_xxx] + 0.0417, R2 =0.93 (4a) where covden_xxx is either covden_sum or covden_win, depending on the month. Despite a good fit (R2 = 0.93), the adjusted values for snow_intcp were very low, with a maximum of less than 0.039 in., which is unrealistic. The snow_intcp equation slope was increased by an order of magnitude (eq. 4b) to give maximum capable wrain_intcp (within PRMS), which was still very low according to prior studies (Moeser and others,

[page 30] 22 Calibration of Precipitation-Runoff Modeling System Summary and Conclusions The Precipitation-Runoff Modeling System (PRMS) is widely used to simulate the effects of climate, topography, land cover, and soils on landscape-level hydrologic responses and streamflow. The U.S. Geological Survey, in cooperation with the New Mexico Department of Homeland Security and Emergency Management, developed procedures to apply the PRMS model to simulate the effects of fire on hydrologic responses. A platform was created to provide a quasi-graphical user interface to easily query the Fourier amplitude sens

Which TRL does the evidence support?
```

## 5. Add your own test

Keep the five labelled lines and the four excerpt blocks. Replace the excerpts with text
from your report; do not include the expected answer. Append a record to a copy of the
JSONL, then run `infer_check.py --data` on it:

```python
import json

prompt = """Assessment target: <target> (type: technology)
Intended application: <application>
Report: <title> (<year>)

Report excerpts:
[page 1] <excerpt>
[page 2] <excerpt>
[page 3] <excerpt>
[page 4] <excerpt>

Which TRL does the evidence support?"""

record = {
    "case_id": "MY-CASE",
    "messages": [
        {"role": "system", "content": (
            "You are a technology-readiness assessor. Read the report excerpts and reply "
            "with one JSON object only: {\"trl\": <integer 1-9>, \"rationale\": \"<one sentence>\"}."
        )},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "{\"trl\": 6, \"rationale\": \"placeholder\"}"},
    ],
}
with open("train/my_prompts.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
```

To build in-distribution excerpts from a PDF, reuse the training builder:

```python
import sys
sys.path.insert(0, "train")
import build_data as bd
excerpts = bd.pick_pages(bd.page_texts("train/data/pdfs/USGS-SDG-061.pdf"), 2400, 4)
```

## 6. How to read a result

- Output must be one JSON object with an integer `trl` from 1 to 9. If it is not, the parse
  failed or the adapter did not load.
- A rationale that mentions the report, the model or a test is a good sign; a generic
  sentence repeated across cases is the 6-collapse.
- On the RTX 3050 (about 2 GiB VRAM) a single cold `infer_check.py` run takes 20 to 35
  seconds; `evaluate.py` averages about 14 seconds per case after the one-time load.
  Minutes per answer means it is running on CPU or swapping.
- Cases the adapter saw in training (012, 040, 061) are optimistic. Held-out cases are
  076, 096 and 100. Judge generalisation on those.
- The adapter predicts 6 most of the time and rarely reaches 4, 5, 8 or 9. That is why the
  submission uses rules-only by default.

## 7. If it fails

| symptom | cause | fix |
|---|---|---|
| `No module named peft` / `transformers` | wrong Python | use `~/.venvs/apctt-train/bin/python` inside WSL |
| `torch.cuda.is_available() == False` | run on Windows Python | run in WSL2 |
| CUDA out of memory | other GPU users, or long prompt | close other GPU apps, keep prompts under 900 tokens |
| invalid JSON / empty output | wrong chat template | `enable_thinking=False`, `add_generation_prompt=True` |
| very slow | CPU or swap | check `nvidia-smi`; free Windows RAM, `wsl --shutdown` if needed |
