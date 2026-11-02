# SKILLS.md — documentation sync for the fine-tune

This is a required behaviour, not optional prose. Any change to the fine-tuned model, its
data, its evaluation, the assessment module or the submission package must update the
project documentation in the same change. Stale numbers and stale hashes are treated as
bugs.

## 1. When this applies

Run this checklist if you touch any of:

- `my_submission/assets/smollm3-trl-lora/` (weights, config, manifest, model card);
- `train/build_data.py`, `train/train_qlora.py`, or `train/data/*.jsonl`;
- `train/evaluate.py`, `train/infer_check.py`, `train/module_eval.py`;
- `my_submission/participant_module/` (rules, extraction, TRL policy, settings);
- `my_submission/submission.json`, `my_submission/declaration.json`, `my_submission/README.md`;
- the README or handoff files themselves.

## 2. What to update, and where

| change | files that must change |
|---|---|
| adapter retrained or replaced | `my_submission/assets/smollm3-trl-lora/training_manifest.json`, `.../README.md`, `train/HANDOFF.md`, `train/README.md`, `train/USING_THE_ADAPTER.md`, `my_submission/README.md`, `AGENTS.md`, `submission.json`, `declaration.json` |
| training data or split changed | `train/HANDOFF.md` (dataset hash, record counts), `train/README.md`, `AGENTS.md`, `training_manifest.json` |
| new evaluation result | `train/HANDOFF.md`, `train/README.md` (status line), `train/USING_THE_ADAPTER.md` (measured-state table), `my_submission/README.md`, `AGENTS.md` |
| module behaviour or TRL policy changed | `my_submission/README.md`, `AGENTS.md` §4, `train/HANDOFF.md` §12, the public-case result table |
| `use_llm`, device or `max_pages` changed | `submission.json` settings, `my_submission/README.md`, `AGENTS.md` §4-5, `declaration.json` notes |
| anything under `participant_module/` or `assets/` | repack, then refresh every hash in §3 |
| licence or data source changed | `declaration.json`, `participant_module/THIRD_PARTY_NOTICES.md`, `AGENTS.md` §5, `train/USING_THE_ADAPTER.md` §7 |

Do not leave a number in one file and a different number in another. If two tools disagree
(`evaluate.py` and `module_eval.py` do), report both with the tool name.

## 3. Hash refresh, in the right order

1. Finish all code, data and asset edits first.
2. Run the evaluation you intend to quote and save the JSON under `train/out/`.
3. Check line endings: `git ls-files --eol my_submission | grep w/crlf` must print nothing.
   The package hash reads raw bytes, so a CRLF checkout changes it.
4. Repack as the last write to the bundle. From the repository root (the packer needs only
   the standard library, so Windows Python works too):

```bash
python cuda/tools/make_submission.py my_submission --output train/out
```

5. Copy the printed `zip_sha256` and `declared_package_sha256` into:
   - `AGENTS.md` §5,
   - `train/HANDOFF.md` §12,
   - `train/USING_THE_ADAPTER.md` §7,
   - any other place that quotes them. Search for the first 8 characters of the *previous*
     hashes before you overwrite them, e.g. `rg "fd4bd882|6d1e8831"` for the current pair.
6. Rebuild the adapter-only handoff zip with `python train/pack_adapter.py` if the adapter
   folder, `train/evaluate.py`, `train/infer_check.py` or `train/data/val.jsonl` changed, and
   update its hash in `AGENTS.md` §5 and `train/USING_THE_ADAPTER.md` §7.
7. Run `make_submission.py --check` on the final zip and record the result.

Hard rule: never edit a file under `my_submission/participant_module/` or
`my_submission/assets/` after the final repack. Editing only `my_submission/README.md`
changes the zip hash but not the package hash; repack and refresh the zip hash anyway.

## 4. Status wording rules

- The submission is **rules-only** by default (`use_llm = false`) because the adapter scored
  0.68 on the SDG 14 holdout under `module_eval.py` versus 0.70 for the rules. Do not write
  that the adapter is enabled unless `submission.json` says so.
- Quote part-A results with the tool: "0.72 with `evaluate.py`" and "0.68 with
  `module_eval.py`" on the same split. Never quote 0.72 as the module's result.
- The public labels are the organizer's estimates, not expert gold. Say so when quoting the
  99-case score.
- `VALID_WITH_REVIEW` is the expected intake result; name the two USGS public-domain licence
  strings as the only review items.

## 5. Verification before you finish

- [ ] `rg` for the old hash finds no stale copy outside `train/out/`.
- [ ] `train/HANDOFF.md` status, result tables, checklist and §12 agree.
- [ ] `train/README.md` status date and score match the latest evaluation JSON.
- [ ] `train/USING_THE_ADAPTER.md` measured-state table, file hashes and limits match the
      current adapter folder.
- [ ] `my_submission/README.md` result table matches `train/out/module-eval.json`
      (git-ignored: after a fresh clone, fetch the PDFs and rerun `module_eval.py` first).
- [ ] `AGENTS.md` §3, §5 numbers match the same JSON and the repacked zip.
- [ ] `submission.json` and `declaration.json` carry the latest package hash written by the
      tool, and their settings/training blocks match the shipped state.
- [ ] `make_submission.py --check train/out/CICA_submission.zip` is `VALID` or
      `VALID_WITH_REVIEW` with only the known licence review items.
- [ ] If the adapter folder changed, `training_manifest.json` records the new dataset hash,
      records, seed and runtime.

## 6. What not to do

- Do not hand-edit `declared_package_sha256`; let `make_submission.py` write it.
- Do not update a score in one doc only.
- Do not claim a model file is shipped when only the adapter is shipped.
- Do not enable the adapter by default without a new held-out result that beats the rules
  prior; if you do, update every status section listed in §2.

## 7. Commit messages

Format: `<type>(<scope>): <summary>` — imperative, lowercase, no trailing period, at most
72 characters.

| type | use for |
|---|---|
| `feat` | module behaviour, rules, extraction, packaging |
| `fix` | a measured regression or a broken contract |
| `eval` | evaluation harness or a new result JSON |
| `data` | training data, splits, labels |
| `docs` | docs only (AGENTS, HANDOFF, README, SKILLS) |
| `chore` | repo hygiene, artifacts, gitignore |

- One commit per iteration. Code, docs, hashes and result numbers land in the same commit.
- The summary states the outcome, not the files touched. The body lists the behaviour
  change, the measured score and the package/zip hashes when the bundle changed.
- Never split a repack from the code change it contains.
- Do not commit `train/out/`, `__pycache__` or `train/data/pdfs/`; they are build artifacts.
  If they are already tracked, untrack them in a separate `chore(repo):` commit.
- Stage only source and docs. Result JSON stays in the git-ignored `train/out/`; its numbers
  go into the docs and the commit body.

Current iteration (2026-11-03, doc sync, LF line endings and re-pack). Two commits:

```
chore(repo): untrack build artifacts and force lf line endings

- add .gitattributes (eol=lf) and a root .gitignore for caches
- untrack train/out/smoke, team __pycache__ and the stray public PDF
```

```
fix(module): keep the adapter off when use_llm is not set

- AssessmentModule defaulted use_llm to true; it now defaults to false
- rules path unchanged; 99 public cases stay A 0.7525 (54 exact)
- add train/pack_adapter.py (deterministic adapter bundle)
- sync docs: repo-root commands, r=8, licence files, versions, hashes
- package fd4bd882315c4b3647d150f80f8f98be8eb3fd054c723445bd8d42bd0f63fe9c
- zip 6d1e8831d33ba2c324137724412af504a8fbae4e99c624403b061e4a45d35cb4
```
