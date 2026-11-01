# Submission, data declaration and handover

kit 0.5.0

This page lists what you send to the organizer and when: up to two submissions, the zip, the data
declaration and the handover items. The submission channel and the deadlines are published on the
official competition website, with the dates (decision D81); they are not part of this kit. The time of receipt of a
submission is the one the submission channel records (decision D80). The official checklist and the public sample
cases are in the kit ([PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §5). The challenge as a whole,
including how submissions are run and scored: [PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md).

## 1. What you send, and what comes back

### 1.1 Up to two submissions

The organizer decided on 2026-10-13 (decision D73): **"Each team may submit its Stage 1 model up
to two times. The latest valid submission received by the deadline will be treated as the final
model."**

| | What happens |
|---|---|
| Every submission (the first and the second) | Checked in full at intake with the same published rules for every team. You get back the intake receipt, **VALID** or **REFUSED** with the reasons (and any licence review items), the time it was received, the hashes of your zip and package, the common base (`base_id`) and the models; for a refused zip the package hash, base and models are given as far as they could be read from it ("not available" otherwise, for example for a zip that cannot be opened). For a valid submission you also get a diagnostic run of 3 public sample cases: per case the status, TRL, error code, the server's fixed failure sentence, output issues (path and rule), warning codes and timings. **No score.** |
| Your final model | The **latest valid submission received by the deadline**. It is run once on the frozen hidden cases (and 10 more for finalists) and **scored**. If your second submission is refused, your first valid one stays your final model. |
| No valid submission received by the deadline | Your team is **not scored and not ranked**. |

Each submission is the same kind of zip, made with `tools/make_submission.py` (§1.2). Valid means
that the intake accepted it: the same checks as `tools/make_submission.py --check`. A failed
sample case does not make a submission invalid; there are no exceptions. One valid submission is enough; between your two submissions you may
change your code, rules and catalogue models. A submission received after the deadline is not
considered, and there is no third submission: a late or third upload is answered that it cannot be
your final model and is not run (decision D81). The time of receipt is the one the submission
channel records (decision D80); the dates, the channels and when the reply comes back are
published on the official competition website (decision D81). After the deadline the organizer tells each team
which submission is its final model (decision D81).

### 1.2 The submission zip

You submit **no environment**. Every team's module runs in the common base environment (the same
for everyone, read-only, identified by `base_id`) and reads public models from the organizer's
read-only model store ([PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §7–8).
`tools/make_submission.py` packs `<team>_submission.zip` from your submission folder. It takes only
these entries and leaves everything else out:

| Entry | Contents |
|---|---|
| `participant_module/` | Your code and rules: `__init__.py`, `adapter.py` (entry point `participant_module.adapter:AssessmentModule`), `LICENSE` (the MIT text, §5), `THIRD_PARTY_NOTICES.md` ([LICENSING.md](LICENSING.md) §2), and any third-party pure-Python modules you copied in (§5) |
| `assets/` | Only weights and rules your team trained or wrote (may be empty); passed to `load()` as `asset_root` |
| `submission.json` | `submission_version`, `team`, `package_id`, `package_version`, `base_id`, `declared_package_sha256`, `license`, `compute`, `settings` and `models` (`contracts/submission.schema.json`) |
| `declaration.json` | The data declaration (§3) |
| `README.md` | What your module does and how it was built |

- `base_id` is the common base of this kit (`contracts/base-environment.json`); another value is
  refused.
- `compute.device` is `cpu` or `cuda:0` (CPU-only is fully allowed); `settings` is passed to
  `load()` as `config["module"]["settings"]`.
- `models` lists every public model your module loads, by `catalog_id` from the model catalogue
  (`contracts/model-catalog.json`, `python tools/get_models.py --list`): the organizer provides
  these in the model store, so a catalogue entry has no `path` and its weights are never in your
  zip. A model outside the catalogue is refused, whatever its licence.
- `models` also lists the weights your team trained and ships in `assets/` or
  `participant_module/`: `{"name", "path", "team_trained": true}`, optionally with `license`
  (MIT), `sha256` and `description`. A fine-tune or adapter of a catalogue model names it in
  `base_model` (a catalogue id) and stays subject to that model's licence as well: keep the
  model's `LICENSE` and `NOTICE` files with your weights. Every weight file over 50 MiB must be covered by such
  an entry. A copy of a catalogue weight file is refused even when declared here: the catalogue
  records the size and sha256 of every catalogue weight file, and the tool compares every file of
  the bundle with them.
- `declared_package_sha256` is the package hash of `participant_module/` and `assets/`, the same
  value as `verified_package_sha256` of the server ([CONFIGURATION.md](CONFIGURATION.md) §5). The
  tool computes it and writes it into `submission.json` (and into `declaration.json` when that file
  has the key); leave it `null` until then.

### 1.3 Size and what is refused

The zip may be at most **2 GB** (2,000,000,000 bytes), the same fixed cap for every team.
Catalogue models do not count: they are in the model store. Refused: other top-level entries;
symbolic links; compiled extensions and binaries (`.so`, `.pyd`, `.dylib`, `.dll`, `.exe`), wheels,
virtual environments, `site-packages`, installed package metadata, `.git` and `node_modules`;
copies of base packages (a folder named like a base package's import name); any file with the bytes
of a catalogue weight file, whatever its name or declaration; undeclared weight files over 50 MiB
(an undeclared weight file under 50 MiB that looks like a public model is a licence review item,
decided by the organizer after the submission); a missing required file; a `submission.json` that does not follow the schema; a
`declaration.json` whose team or package differs; a `base_id` other than the kit's; a model outside
the catalogue; a package hash that differs from the declared one; and anything the licence checker
refuses. The organizer's intake runs the same checks and recomputes the hash.

```bash
.venv/bin/python tools/make_submission.py ../my_submission
.venv/bin/python tools/make_submission.py --check ../MY-TEAM_submission.zip
```

Exit codes: 0 `VALID`, 1 `REFUSED` (each `ERROR` line names a problem; the zip is deleted), 2
`VALID_WITH_REVIEW` (licence items the organizer decides after the submission), 3 usage error.
Every pack and check also prints a `REMINDER` line about third-party code in your package (§5).
How to: [README.md](../../README.md) step 10.

## 2. Using NASA TechPort

Using TechPort is optional and non-exclusive. If you use it:

- Review the related datasets, technical reports, papers and test and verification records that
  are accessible to you, not only the project summary or its TRL fields.
- Do not mix up start, current and target TRL.
- Do not copy a project-level TRL onto every document, subsystem or component of that project.
- Record where each item came from (§3).

Data volume is not scored, training is not mandatory and exhaustive collection is not expected.
The assessment server never crawls or queries TechPort; everything your module needs must be in your
package or assets, the common base environment or the catalogue models of the model store.

## 3. Data declaration

For each data source, declare its source, version, date, role, label origin and linked project, and
its licence or terms of use.
If your team did not train anything, declare `no_team_training`. Example structure
([SUBMISSION_DECLARATION.example.json](SUBMISSION_DECLARATION.example.json)):

```json
{
  "declaration_version": "apctt-declaration-example-v1",
  "example_only": true,
  "team": "EXAMPLE-TEAM",
  "package_id": "example-assessment-package",
  "package_version": "0.1.0",
  "declared_package_sha256": null,
  "team_training": "no_team_training",
  "training": null,
  "data_sources": [
    {
      "source": "NASA TechPort (example entry)",
      "version": "public export used by the team",
      "date": "YYYY-MM-DD",
      "role": "retrieval reference for similar projects",
      "label_origin": "project-level start/current/target TRL as published; not used as component labels",
      "linked_project": "TechPort project ID of the referenced project",
      "license": "licence or terms of use as published by the source (name, SPDX identifier or URL)"
    }
  ],
  "inference_assets": {
    "source_code": "participant_module/ (commit or archive hash)",
    "weights": [],
    "prompts_or_rules": ["rules/criteria.yaml"],
    "reference_assets": [],
    "dependency_declarations": ["the common base environment (base_id in submission.json)", "catalogue models listed in submission.json models[]"]
  },
  "notes": "Example structure only. Replace every value; keep keys you use and state what does not apply."
}
```

| Field | What to write |
|---|---|
| `team`, `package_id`, `package_version` | Your team and package identity; `package_id` as in your configuration. |
| `declared_package_sha256` | The package hash; `tools/make_submission.py` fills it in when the key is present. |
| `team_training` | `no_team_training` if you trained nothing; otherwise describe it in `training`. |
| `training` | If you trained: training code location, configuration, data sources used, seeds, hardware. |
| `data_sources[].source` | Name of the source (dataset, report collection, database). |
| `data_sources[].version` | Release, export or access version. |
| `data_sources[].date` | Date you obtained it. |
| `data_sources[].role` | How you used it: training, validation, retrieval reference, rules, examples. |
| `data_sources[].label_origin` | Where any labels came from and what they mean (e.g. project-level current TRL). |
| `data_sources[].linked_project` | The project the material belongs to, if any. |
| `data_sources[].license` | The licence or terms of use of the source: an SPDX identifier, a licence name or the URL of the terms ([LICENSING.md](LICENSING.md) §5). |
| `inference_assets` | Everything needed to run inference (§4). |

The example is a structure, not a fixed schema (decision D81): the checks read `example_only`,
`team`, `package_id`, `package_version`, `declared_package_sha256` and `data_sources[].license`;
write the rest so that a reader can follow where your data came from.

If you used the public sample cases (§5.2 of the problem description) to develop, tune or check
your module, declare them like any other source: `"source"` "APCTT public sample cases
(contracts/public-cases.json, USGS reports)", the kit version as `"version"`, your `"role"`,
`"label_origin"` "organizer's estimated TRL per report; not expert labels" if you used
`--labels`, and `"license"` "U.S. public domain (USGS); some figures third-party, see the
manifest's licence_note".

## 4. Handover

Link your execution package and follow-on development materials by version and hash:

- inference source code;
- weights;
- prompts or rules;
- reference assets;
- dependency declarations (the `base_id` of the common base and the catalogue models you use, by
  `catalog_id`; your submission contains no packages);
- if your team trained a model: training code, configuration and data information; otherwise
  `no_team_training`.

The Stage 2 application uses the same frozen assessment pipeline, so the package you hand over must
be the one that produced your assessment results. You develop the Stage 2 web system on site,
during the Stage 2 event, and merge it with this package on the event computer itself; there is no
advance upload of web code (decision D80, [PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) §14).

## 5. Licence of your work

The kit's MIT licence covers the kit only. Your module, weights, rules and data remain yours. So that
the organizer can run, demonstrate, keep and hand over your work (§4), these rules apply; the
organizer confirmed them on 2026-10-10 (decision D53) and made your own work MIT only on
2026-10-13 (decision D72), and the official participation terms will carry them:

- License your own code, rules and trained weights (adapters and fine-tunes included) under
  **MIT**: put the MIT text in `participant_module/LICENSE` and `"license": "MIT"` in
  `submission.json`. The same rule applies to the code of your Stage 2 web system (`LICENSE` at its
  root, and `"license"` in your `package.json` if you set it). Any other licence, Apache-2.0
  included (earlier versions of the kit allowed it; decision D72 of 2026-10-13 made it MIT only so
  that APCTT can use your work later without a licence problem), or a missing declaration or
  licence file, is refused. Weights derived from a catalogue model also stay subject to that
  model's licence: keep its `LICENSE` and `NOTICE` files with them.
- Use only third-party packages, front-end libraries, fonts and models whose licences the policy
  allows (third-party components keep their own licences: Apache-2.0 and MIT components stay
  allowed), and list them with version and licence in `participant_module/THIRD_PARTY_NOTICES.md`.
  Strong copyleft (AGPL, GPL, SSPL, EUPL, OSL, RPL), non-commercial, research-only,
  revenue-capped and commercial-only components are refused; weak copyleft (LGPL, EPL, CDDL) is
  reviewed; MPL-2.0 is allowed.
- Use public models only from the model catalogue (`contracts/model-catalog.json`, named in
  `submission.json` by `catalog_id`; any other public model is refused and public model weights
  are never shipped: each evaluation machine has one shared read-only model store, which cannot
  hold per-team models for this many teams, decision D80), and declare the licence of every data
  source (§3).
- You may copy third-party pure-Python modules into `participant_module/`, each listed with its
  licence in `participant_module/THIRD_PARTY_NOTICES.md`; compiled extensions are refused. If a
  licence or security problem with code you included arises later, your team bears the legal
  responsibility and the award is revoked (decision D66, 2026-10-10; [LICENSING.md](LICENSING.md)
  §2.2).
- Ship no packages at all: PyTorch, NVIDIA CUDA packages and every other package come from the
  common base environment, which the organizer installs.

You keep ownership of your work. By submitting it under MIT you submit it as open source, and you
do not assert, and will not assert, patent, copyright, trade-secret or other
intellectual-property claims over the submitted work, including its follow-on development, against
the organizer, APCTT or anyone who uses it under that licence; you warrant that you have the right
to submit everything in your submission; third-party components keep their own licences. The
organizer decided this on 2026-10-10 (decision D54). The clause, and what MIT already provides
(it has no express patent licence, which the covenant supplies): [LICENSING.md](LICENSING.md)
§2.1.

The licence checker `tools/check_licenses.py` flags components from their declared metadata; HQ
decides flagged items after submission.
What is allowed, what is refused, permissive alternatives, what the checker does and does not
check, and how to run it: [LICENSING.md](LICENSING.md).

## 6. Final check

Work through the list in [MODULE_GUIDE.md](MODULE_GUIDE.md) §9, then confirm:

- [ ] `tools/make_submission.py ../my_submission` ends with `VALID`, or with `VALID_WITH_REVIEW`
      and you can explain each review item ([LICENSING.md](LICENSING.md) §7).
- [ ] `submission.json` names the kit's `base_id`, your device, and every catalogue model and
      team-trained weight your module loads.
- [ ] `GET /v1/package` shows your `package_id` and `verification_status: verified`, and its
      `verified_package_sha256` equals the `declared_package_sha256` the tool wrote.
- [ ] Your declaration lists every data source, its licence and the label origin of any labels you
      used.
- [ ] Your handover links match the exact files you submit.
- [ ] `submission.json` says `"license": "MIT"` and `participant_module/LICENSE` holds the MIT text.
- [ ] Your final model is uploaded before the deadline: the latest valid submission received by
      the deadline counts, and each team has at most two submissions.
