# Licences: what you may use and what you declare (Stage 1 and Stage 2)

kit 0.5.0 | licence policy 0.4.0 (`contracts/license-policy.json`)

**Status: confirmed by the organizer on 2026-10-10 (decision D53); your own work is MIT only since
2026-10-13 (decision D72).** The rules for your own work (§2), third-party components (§3), public
models (§4) and the common CUDA base (§6) are confirmed, and the participation terms carry them.
The data rules (§5) are rules too (decision D81, 2026-10-13). The organizer also decided on 2026-10-10 that
submissions are open source and that teams assert no patent or other intellectual-property rights
over them (decision D54, §2.1). The licence checker flags components from their declared metadata
and the policy's name rules; HQ decides flagged items after submission (SPEC-14). Nothing is
approved in advance and there are no exceptions (decision D63).

Why this matters: the organizer runs your Stage 1 bundle, tests and demonstrates your Stage 2 web
system, keeps both, and hands them over to APCTT for follow-on development (the handover in
[SUBMISSION.md](SUBMISSION.md) §4). Every component you ship must allow all of that.

## 1. Summary

| What | Rule | Where |
|---|---|---|
| Your own code, rules and trained weights (Stage 1) | MIT; anything else, Apache-2.0 included, is refused (D72) | `participant_module/LICENSE` and `"license"` in `submission.json` (§2) |
| Your Stage 2 web code | MIT; anything else, Apache-2.0 included, is refused (D72) | `LICENSE` at the root of the web code, `"license"` in your `package.json` files (§2) |
| Your rights in your work | You keep ownership; you submit it as open source and assert no patent, copyright, trade-secret or other intellectual-property claim over it against the organizer, APCTT or anyone who uses it under that licence (§2.1) | The official participation terms |
| Third-party packages and front-end libraries | Strong copyleft and use-restricted licences are refused; weak copyleft is reviewed after the submission; MPL-2.0 is allowed (§3). Your Stage 1 zip contains no packages: Python packages come from the common base | `package-lock.json` of your web system, your notices file |
| Third-party pure-Python code you copy into your package | Allowed, listed with its licence; compiled extensions are refused. If a licence or security problem arises later, your team bears the legal responsibility and the award is revoked (D66, §2.2) | `participant_module/`, `participant_module/THIRD_PARTY_NOTICES.md` |
| Public models | Only the models of the model catalogue, by `catalog_id`, and weights your team trained (§4); weights derived from a catalogue model also keep that model's licence | `"models"` in `submission.json` |
| Data and documents | Declare the licence or terms of every source (§5) | `data_sources[].license` in your data declaration |
| PyTorch and NVIDIA CUDA libraries | Part of the common base, installed by the organizer; never shipped by you (§6) | Nowhere in your zip or your web system |

Run [`tools/check_licenses.py`](../../tools/check_licenses.py) before you submit (§7).

## 2. Your own work

License everything your team wrote or trained under **MIT**: the module code, rules, prompts,
configuration and any weights you trained yourselves, adapters and fine-tunes included. The
organizer decided this on 2026-10-13 (decision D72), so that APCTT can use, combine and
redistribute your work later without a licence problem; until then Apache-2.0 was allowed as well.
Anything else is refused: another licence (Apache-2.0 and other permissive ones such as
BSD-3-Clause included), MIT together with another licence (`MIT OR Apache-2.0`), MIT with an
exception or added terms (`MIT WITH Commons-Clause`: another licence), no declaration, a missing or
empty licence file, or a missing or unreadable `submission.json`. The checker reads the identifier you
declare and checks that your licence file (`participant_module/LICENSE`; for Stage 2 the `LICENSE`
at the root of the web code) holds the MIT text: its permission, condition and warranty
paragraphs, compared without regard to case, quotes, punctuation and line breaks. Another text
there, such as the Apache-2.0 text, is refused with a message that says so (D72); the MIT text
followed by another licence text is a review item. Copy `licenses/MIT.txt` of the kit, put your
team in the copyright line and save it as UTF-8 text (a file saved as UTF-16 with a byte order
mark, such as Notepad's "Unicode", is read too).

Stage 1 bundle:

- `participant_module/LICENSE`: the full licence text, with your team or organisation in the
  copyright line. It sits inside your package, so the package hash covers it. Make it a regular
  file: the checker does not follow a symbolic link, so a linked `LICENSE` counts as missing.
- `submission.json`: `"license": "MIT"`, the licence of the file.
- `participant_module/THIRD_PARTY_NOTICES.md`: every third-party component you ship, with version
  and licence (`--notices` drafts it, §7). Add both files before you compute your final package hash.

Stage 2 web system:

- `LICENSE` (or `LICENSE.md`, `LICENSE.txt`) at the root of the web code, holding the MIT text.
- `"license"` in your own `package.json` files, where you set it (the root, sub-projects such as
  `frontend/package.json`, npm workspaces): `MIT` (not `UNLICENSED`; leave the
  field out rather than use another value; a legacy `"licenses"` list counts the same way). A
  folder with its own `package.json` counts as your sub-project when it has `package-lock.json`,
  `node_modules/`, `"private": true`, `"workspaces"` or its own `yarn.lock`, `pnpm-lock.yaml` or
  `bun.lock`, or when it declares `dependencies` or `devDependencies`, is not under a
  `vendor/`-like folder and is not a package the policy names (§3.4).
- `THIRD_PARTY_NOTICES.md` at the root, and an About or Licences page in the web system that shows
  it (§3.4).

The licence-related part of `submission.json` (the other fields are in [SUBMISSION.md](SUBMISSION.md) §1.2
and `contracts/submission.schema.json`):

<!-- example: submission-licence -->
```json
{
  "license": "MIT",
  "models": [
    {"catalog_id": "qwen2.5-1.5b-instruct"},
    {
      "name": "our-trl-classifier",
      "team_trained": true,
      "base_model": "bge-base-en-v1.5",
      "path": "assets/models/trl-classifier"
    }
  ]
}
```

| Field | What to write |
|---|---|
| `license` | `MIT`: the licence of your own work (D72). |
| `models[].catalog_id` | The id of each public model you use, from the model catalogue `contracts/model-catalog.json` (`python tools/get_models.py --list`). Only catalogue models may be used (D64); the organizer provides them, so a catalogue entry has no `path` and their weights are never in your bundle. |
| `models[].team_trained` | `true` for weights your team trained itself, with `name` and `path`; they are covered by your `license`. A fine-tune or adapter of a catalogue model is your own work too: name its base in `base_model` (a catalogue id); the weights also stay subject to the base model's licence, so keep its `LICENSE` and `NOTICE` files with them (the checker lists this as a condition). A base model, `repo` or `url` outside the catalogue is refused. The model deny list (§3.3) still applies to a team-trained entry's names, a `license` you declare for it must be MIT (anything else is refused), and a `name` that suggests a model family other than its catalogue `base_model` (Gemma, Llama, EXAONE, Qwen and the others in the policy's `models.name_families`, for example `my-gemma-ft`) goes to review. |

Weight files (`*.safetensors`, `*.bin`, `*.gguf`, `*.onnx`, `*.pt`, …) and shared libraries or
executables (`*.so`, `*.dll`, `*.dylib`, `*.exe`) in `assets/` or `participant_module/` that no
`models[]` entry covers by its `path` are reported as review, one item per folder.

### 2.1 Open source and non-assertion of intellectual-property rights (D54)

The organizer decided on 2026-10-10 that submissions are open source and that teams assert no
patent or other intellectual-property rights over them (decision D54). The clause below is the
clause of the participation terms. In short: you keep ownership
of your work; you license it under MIT; you promise not to bring patent, copyright,
trade-secret or other intellectual-property claims over it, or over its follow-on development,
against the organizer, APCTT or anyone who uses it under that licence; and you confirm that you
have the right to submit everything in your submission. Nothing in the clause changes what §2
asks you to put in your bundle or web code.

> **Open-source submission and non-assertion of intellectual-property rights (decision D54)**
>
> In this clause, "submitted work" means everything the team itself wrote or trained and submits
> for the hackathon: the Stage 1 bundle (the code, rules, prompts and configuration in
> `participant_module/`, and the weights the team trained in `assets/`), the Stage 2 web system,
> and the handover materials that go with them. It does not include third-party components
> (item 5). "Declared licence" means the MIT License, which the team declares in its submission
> (decision D72).
>
> 1. Open source. The team submits the submitted work as open-source software under the declared
>    licence (MIT, decisions D53 and D72). The declared licence is granted to the organizer,
>    to APCTT and to everyone who receives the submitted work under it.
> 2. Non-assertion. With respect to the submitted work, including its follow-on development, the
>    team does not assert, and will not assert, any patent, copyright, trade-secret or other
>    intellectual-property claim against the organizer, APCTT or anyone who uses, copies,
>    modifies, distributes or runs the submitted work under the declared licence. The covenant
>    covers rights the team or its members own or control now or later, binds the team, its
>    members and their successors, and reaches claims that the submitted work, alone or combined
>    with a modification or follow-on development of it, infringes such a right; it does not
>    reach a claim that concerns only material that others added, or use outside the declared
>    licence. The MIT License contains no express patent licence; the covenant supplies the
>    patent commitment that the licence text lacks.
> 3. Warranty of rights. The team warrants that it has the right to submit everything in its
>    submission under the declared licence: the submitted work is the team's own; every team
>    member, and every employer, university or other organisation whose consent is needed, has
>    agreed; third-party components are included only as their own licences allow (item 5); and,
>    to the team's knowledge, the submission infringes no right of a third party.
> 4. Ownership. The team keeps ownership of the submitted work. This clause is a licence and a
>    non-assertion covenant, not an assignment or a transfer of ownership: the copyright in the
>    team's code, rules and weights, and any patent or other right the team holds, remain the
>    team's. The organizer and APCTT receive the rights the declared licence gives everyone, plus
>    this covenant; they claim no ownership of the submitted work.
> 5. Third-party components. Third-party packages, front-end libraries, fonts, icons, models and
>    data in the submission are governed by their own licences; the team's declared licence and
>    this covenant do not cover them, and the team's warranty for them is limited to the right to
>    include them as shipped (item 3).

What the MIT licence already provides, and what the clause adds:

- MIT contains no express patent grant. It is a copyright permission ("to deal in the Software
  without restriction, including ... to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies"), and whether that wording implies a patent licence is unsettled. Item 2
  closes that gap: it is the team's patent commitment, and it also covers copyright, trade-secret
  and other claims and reaches follow-on development. (Apache-2.0, which teams could choose until
  decision D72, has an express patent licence in its section 3; with MIT only, item 2 is the
  team's whole patent commitment.)
- MIT does not mention trade secrets, and it disclaims every warranty, including non-infringement
  and title (its "AS IS" paragraph). Item 3 is therefore a warranty the licence itself does not
  give; it is given to the organizer and APCTT under the participation terms.
- Items 4 and 5 restate what the licence already implies: it is a non-exclusive licence, not an
  assignment, and it covers only the licensor's own work.

### 2.2 Third-party pure-Python code in your package (D66)

The organizer decided on 2026-10-10 (decision D66): you may copy third-party pure-Python modules
(source files that need no compiling) into `participant_module/`. List each one in
`participant_module/THIRD_PARTY_NOTICES.md` with its name, source, version and licence, keep its
licence text with it, and use only code whose licence the policy allows (§3). Compiled extensions
and binaries (`.so`, `.pyd`, `.dylib`, `.dll`, `.exe`), wheels, virtual environments,
`site-packages`, installed package metadata (`*.dist-info`) and copies of packages of the common
base are refused. **If a licence problem or a security problem with code you included arises
later, your team bears the legal responsibility and the award is revoked.** The checks of this kit
flag components from their declared metadata. `tools/make_submission.py` prints this rule as a
one-line reminder every time it packs or checks your zip.

## 3. Third-party components

### 3.1 Licence lists

| Status | Licences |
|---|---|
| Allowed | MIT, MIT-0, MIT-CMU, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC, PSF-2.0, Python-2.0, CNRI-Python (used by `regex`, a `transformers` dependency), Zlib, 0BSD, Unlicense, CC0-1.0, BSL-1.0, HPND, CDLA-Permissive-2.0, BlueOak-1.0.0, and the classifiers "BSD License" and "Apache Software License" that name no version (keep the package's licence file) |
| Allowed with conditions | MPL-2.0 (weak, file-level copyleft: ship the MPL-covered files with the licence text and say where their source is; your changes to those files stay MPL-2.0); OFL-1.1 (fonts only; ship the OFL text with the fonts); CC-BY-4.0 and CC-BY-3.0 (icons, images, documents and data only; keep the attribution) |
| Review: the organizer decides after the submission | Weak copyleft: LGPL (every version), EPL, CDDL, MPL-1.0 and MPL-1.1, CeCILL; CC-BY-SA; CC-BY-ND (it forbids modified versions); Artistic; BSD-4-Clause; Elastic-2.0 (it bars offering the software to others as a hosted or managed service; it stays review, decided by HQ after the submission, decision D81); proprietary, public-domain and custom licences (`LicenseRef-…`); any identifier that is not in these lists; an unrecognised licence string (§3.2); no licence declared |
| Denied (no exceptions, D63) | Strong copyleft: AGPL, GPL (every version, `-only` or `-or-later`), SSPL, EUPL, OSL, RPL, and dual "GPL or commercial" licences that offer no permissive alternative. Use-restricted: non-commercial (CC-BY-NC…, PolyForm Noncommercial), research- or evaluation-only, revenue- or funding-capped (OpenRAIL-M variants with revenue caps, PolyForm Small Business), production-use restricted (BUSL), commercial-only licences and SDKs, Commons Clause. Gated custom model licences (Llama, Gemma) |

How expressions count:

- `OR`: you may choose, so the most permissive allowed alternative counts (`MIT OR GPL-3.0-only`
  is allowed). A paid or proprietary alternative (`LicenseRef-Commercial`, `LicenseRef-Proprietary`
  and any `LicenseRef-…Commercial…` or `LicenseRef-…Proprietary…`) is not a licence offered to
  everyone and never counts as a choice, so `AGPL-3.0-only OR LicenseRef-Commercial` (PyMuPDF's
  "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License") is refused.
- `AND`: all parts apply, so the most restrictive counts.
- `WITH`: only these listed exceptions change the result: `LLVM-exception` keeps the licence's
  own status (`Apache-2.0 WITH LLVM-exception` is allowed for a third-party component);
  `Commons-Clause` is refused, and so is any expression that names it as a plain identifier
  (`Apache-2.0 AND Commons-Clause`, `MIT OR Commons-Clause`: Commons Clause is a restriction, not
  an alternative licence). `GCC-exception-3.1` (the GCC runtime library exception) and
  `Font-exception-2.0` (a font exception, such as the Liberation fonts in `pdfjs-dist`, §3.4) make
  a GPL-2.0 or GPL-3.0 licence allowed with their condition, and `Classpath-exception-2.0` makes
  GPL-2.0 review. These GPL carve-outs were added by the kit builder and are rules since
  decision D81; on any other licence they change nothing except that the result is at
  least review, so `AGPL-3.0-only WITH GCC-exception-3.1` stays refused. Any other exception makes
  the result at least review. Your own licence (§2) takes no exception at all.
- `+` and `-or-later` stay in the same family: `GPL-2.0+` is GPL (refused), `LGPL-2.1+` is LGPL
  (review). Deprecated identifiers count as their modern form: `GPL-2.0-with-classpath-exception`
  is `GPL-2.0-only WITH Classpath-exception-2.0` (review), `GPL-2.0-with-GCC-exception` is
  `GPL-2.0-only WITH GCC-exception-2.0` (an exception that is not listed, so refused).
- Several `License ::` classifiers on one Python package count as alternatives; when they have
  different statuses the package goes to review (unless the result is refused, as for an AGPL
  classifier next to a proprietary one).

Allowed components still carry obligations: keep their licence and NOTICE files when you ship
them.

### 3.2 How the checker decides, and what it does not check

The checker decides each component's licence **only from its declared metadata**:

- Python packages: `License-Expression` first; without it the `License ::` classifiers (through a
  fixed table); without those the `License` field.
- npm packages: the `"license"` field (or a legacy `"licenses"` list) in `package-lock.json`, in
  `node_modules/*/package.json`, or in the `package.json` of a vendored folder.
- Models: `"license"` in `submission.json`; data: `data_sources[].license`.

A declared string counts when it is an SPDX expression (operators `AND`, `OR`, `WITH` in upper
case) or one of the common names in the exact table of the policy (`license_strings`, matched
without regard to case or extra spaces): for example "MIT License", "Apache 2.0 License",
"Apache Software License", "BSD License", "GPLv3", "GPL v2 or later" (GPL-2.0-or-later, never
read as an `OR`) and "LGPLv2.1+". Anything else is review with the reason "unrecognised licence
string": for example `pypdfium2`'s "BSD-3-Clause, Apache-2.0, dependency licenses", npm's "SEE
LICENSE IN LICENSE.md", or a whole licence text pasted into the `License` field.

The checker **does not read licence files or file contents**, except your own licence file, which
must hold the MIT text (§2). Words in a package's LICENSE or
NOTICE files (a bundled non-commercial component, a Commons Clause, extra terms) do not change its
result, and npm licence files are not read either: npm packages are judged by their declared
licence and the name rules below. Packages whose metadata looks permissive but whose terms are not
are caught by **name rules** in the policy (exact names, npm scopes such as `@ckeditor/`, and
version rules such as `apexcharts` 5.0.0 and later). It does not check fonts, images, icons, code
you copied into your own files, or the contents of copied front-end files (§3.4, §7).

So a result of "allowed" means the declared metadata and the name rules raise no flag. Read the
licence files of what you ship; HQ decides flagged items after submission.

### 3.3 Common components that are denied, and permissive alternatives

Checked on 2026-10-09 and 2026-10-10 against PyPI, npm and Hugging Face metadata and licence
files. The policy lists these by name, so the checker refuses them even when their metadata looks
permissive.

| Component | Why | Use instead |
|---|---|---|
| PyMuPDF (`pymupdf`, `pymupdfb`, `pymupdf4llm`) | AGPL-3.0 or a paid Artifex commercial licence. AGPL-3.0 section 13 can require offering the source code to every user of a network service built with it, such as a Stage 2 web system. | `pypdf` (BSD-3-Clause, used by this kit), `pdfplumber` (MIT), `pdfminer.six` (MIT), `pypdfium2` (BSD-3-Clause and Apache-2.0; its metadata also names the licences of the bundled PDFium dependencies in a form the checker does not recognise, so it goes to review) |
| `ultralytics`, `doclayout-yolo`, YOLO weights from `Ultralytics/*` | AGPL-3.0 | `docling` (MIT) for layout, `rapidocr` (Apache-2.0) for OCR |
| `magic-pdf`, `borb` | AGPL-3.0 | `docling` (MIT), `pypdf` (BSD-3-Clause) |
| `mineru` | The MinerU Open Source License (`LicenseRef-MinerU-Open-Source-License`; read in mineru 4.0.11): Apache-2.0 with additional terms. A licensee whose group exceeds 100 million monthly active users or USD 20 million total monthly revenue needs a separate commercial licence, online services built on it must state that MinerU is used, and the licence ends automatically when either term is not met. A revenue-capped licence. | `docling` (MIT), `pypdf` (BSD-3-Clause) |
| `surya-ocr`, `marker-pdf` | The code says Apache-2.0, but the model weights they use (for example `datalab-to/surya-ocr-2`) are under a modified OpenRAIL-M licence that excludes use by organisations with more than USD 5 million in gross revenue or in equity or debt funding raised, except personal or research use. | `docling` (MIT), `rapidocr` (Apache-2.0), `easyocr` (Apache-2.0), `paddleocr` (Apache-2.0) |
| `konlpy`, `soynlp` | GPL-3.0 | `kiwipiepy` (Apache-2.0) |
| `unidecode` | GPL-2.0-or-later | `unicodedata` from the Python standard library |
| `pyqt6` | GPL-3.0 or a commercial licence | A web front-end instead of a desktop toolkit |
| `ghostscript` (the Python binding) | GPL-3.0-or-later; Ghostscript itself is AGPL-3.0 or a commercial licence | `pypdf` (BSD-3-Clause), `pypdfium2` |
| `tinymce` 7.0.0 and later, `ckeditor5` and every `@ckeditor/` package | GPL-2.0-or-later or a commercial licence (npm metadata: "SEE LICENSE IN …") | `tinymce` 6.8.5 (MIT), `quill` (BSD-3-Clause), `@tiptap/core` (MIT) |
| `highcharts`, `handsontable` (every version: 7.0.0 and later are commercial; 6.2.2 is MIT, but the confirmed rule names Handsontable without a version, so it is refused too; there are no exceptions, D63), `ag-grid-enterprise`, `@mui/x-*-pro` and `-premium`, Font Awesome Pro (`@fortawesome/*-pro`, `@fortawesome/pro-*`) | Commercial licences or licences free only for non-commercial use | `chart.js` (MIT), `echarts` (Apache-2.0), `ag-grid-community` (MIT), `@mui/x-data-grid` (MIT), `@fortawesome/fontawesome-free` |
| `@react-pdf-viewer/` packages, `@pdftron/webviewer`, `pspdfkit`, `@nutrient-sdk/` packages | Commercial or custom PDF viewer licences | `pdfjs-dist` (Apache-2.0) with `react-pdf` (MIT) |
| `mapbox-gl` 2.0.0 and later | The Mapbox Web SDK terms (use only with Mapbox services). The LICENSE.txt of 1.13.3 is BSD-3-Clause, but its `package.json` says "SEE LICENSE IN LICENSE.txt", so the checker reports it for review | `maplibre-gl` (BSD-3-Clause) |
| `apexcharts` 5.0.0 and later | The LICENSE file is the ApexCharts licence (free only below USD 2 million revenue; an OEM licence for products used by others), even in 5.0.0 whose `package.json` still says MIT | `apexcharts` 4.7.0 (MIT), `chart.js`, `echarts` |
| `primevue` and `@primevue/*` 5.0.0 and later, and the PrimeUI packages they use: `@primeuix/styles` and `@primeuix/themes` 3.0.0 and later, `@primeuix/styled` and `@primeuix/motion` 1.0.0 and later, `@primeuix/utils` 0.8.0 and later, `primeicons` and `@primeicons/*` 8.0.0 and later, `@primeui/*` | The PrimeUI licence (npm metadata: "SEE LICENSE IN LICENSE.md"): free only for organisations under USD 1 million revenue, fewer than 5 developers and 10 employees and under USD 3 million funding, otherwise a paid licence with a licence key | `primevue` 4.5.5 (MIT) with its MIT dependencies (`@primeuix/styles` 2.x, `@primeuix/styled` 0.7.x, `@primeuix/utils` 0.6.x), `primeicons` 7.0.0, `vuetify` (MIT), `element-plus` (MIT) |
| Models: Llama (every `meta-llama/` repository, and copies or derivatives by other publishers whose repository name says Llama, for example `unsloth/Llama-3.2-1B-Instruct` or `NousResearch/Hermes-3-Llama-3.1-8B`), Gemma (every repository with "gemma" in its name, for example `google/embeddinggemma-300m` and fine-tunes by other publishers), `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, EXAONE, HyperCLOVAX-SEED, LayoutLMv2, LayoutLMv3, LayoutXLM, `facebook/nougat-base`, `jinaai/jina-embeddings-v3`, `datalab-to/*` | Gated or custom licences, non-commercial terms (Qwen Research Licence, EXAONE "NC", CC-BY-NC, CC-BY-NC-SA) or revenue caps | The catalogue models in §4 |

Do not put executables or shared libraries (for example OCR or PDF programs) in `assets/`: compiled
code is refused (§2.2), such programs often carry their own copyleft licences, and programs outside
the common base environment are not provided on the evaluation machines. Use the packages of the
base instead (for example `rapidocr-onnxruntime` for OCR and `pypdfium2` for rendering pages).

### 3.4 Web systems (Stage 2)

- **Node.js is available on the event computers** (decision D81), so a Node.js-based front-end
  build may run on site. The event computers have no network, so nothing can be downloaded during
  the event: bring every dependency as files (vendored in your web directory, for example
  `node_modules/` with `package.json` and `package-lock.json`) and check them with
  `tools/check_licenses.py --web`.
- Everything the browser downloads (JavaScript, CSS, fonts, icons, images) is distributed to every
  user. Because the demonstration runs without network access, you bundle these files with your web
  code; ship each one's licence file with it.
- Minified bundles lose licence comments easily. Use your bundler's option or plugin that collects
  them into a third-party licence file, and serve that file.
- Keep your front-end sources with `package.json` and `package-lock.json` in your web directory,
  not only the build output: the checker judges npm packages by the licence declared in the lock (or in
  `node_modules/`) and by the name rules. Packages only in `devDependencies` (or marked
  `"dev": true` in the lock) are listed in a note, not counted, because they are not shipped to
  browsers; make sure your build does not bundle them. Dev-only packages that are not allowed
  (a refused licence or name rule, or a licence that needs review) make one review item per
  project, because a build such as Vite or SvelteKit may bundle them after all: the organizer
  confirms that they are not shipped. Packages marked `"devOptional": true` (a
  dev dependency that is also an optional dependency of a shipped package, such as `typescript`
  for `vue`) are counted, because they may be installed for production. npm workspaces are your
  own code and are read once from the root lock.
- Prefer npm packages to copied files. A library you copy into your code (for example
  `static/js/chart.umd.min.js`) has no package metadata, so its licence is not read: keep its
  licence file in the same folder and list it by hand in `THIRD_PARTY_NOTICES.md`. The checker
  reports one review item per folder of copied files: folders with minified files (`*.min.js`,
  `*.min.css`) and, outside an npm project, folders named like `vendor/`, `lib/` or
  `third_party/` and build output (`dist/`, `build/` …) without `package.json`. Your own
  sources and the build output of an npm project are not listed. A copied folder that keeps its
  own `package.json` and is installed by no lock is read as a third-party package: its declared
  licence and the name rules, plus each runtime dependency it declares that is not listed yet
  (review, with the name rules). That holds for a package the policy names wherever it sits (for
  example `pdfjs-dist`, `apexcharts` or `handsontable` copied to `public/` or `static/js/`), for
  anything under a `vendor/`-like folder, and, inside a project with `package-lock.json`, for a
  folder whose `package.json` declares no `dependencies` or `devDependencies` (only
  `optionalDependencies` or `peerDependencies`, as `pdfjs-dist` 6.4.299 does). Otherwise a folder
  whose `package.json` declares dependencies is your sub-project (§2), and its licence must be
  MIT: put a copied package with a licence such as ISC, BSD or Apache-2.0 under `vendor/` so that
  it is read as a third-party package. The name rules also apply to the `package.json` of every
  folder that is read as a project.
- `pdfjs-dist` (Apache-2.0), the usual viewer for showing the evidence page of the original PDF,
  is allowed, and the policy records it with a condition: keep every licence file in the package
  when you ship or vendor it (version 6.4.299 has eleven: `LICENSE`, `cmaps/LICENSE`,
  `iccs/LICENSE`, `standard_fonts/LICENSE_FOXIT`, `standard_fonts/LICENSE_LIBERATION` and six
  `wasm/LICENSE_*` files). The Liberation fonts in `standard_fonts/` are GPL-2.0 with Red Hat's
  font exception: a listed exception (`Font-exception-2.0`) to the GPL rule of §3.1, valid for the
  unmodified fonts; do not modify or subset them. `react-pdf` (MIT) uses `pdfjs-dist`, so the same
  applies.
- Fonts: Noto Sans, Noto Sans KR, Roboto (`@fontsource` packages) and Pretendard are OFL-1.1. Ship
  the OFL text next to the font files and use the official files; a modified or subset font must
  not use a Reserved Font Name. The checker does not check font files.
- Icons: `@fortawesome/fontawesome-free` is `CC-BY-4.0 AND OFL-1.1 AND MIT`; keep its
  `LICENSE.txt` and the attribution comments in its files.
- Give the web system an About or Licences page that shows your licence and
  `THIRD_PARTY_NOTICES.md`.

## 4. Public models

Decision D64 (2026-10-10): only the models of the model catalogue `contracts/model-catalog.json`,
named in `submission.json` by `catalog_id`, and weights your team trained itself (including
fine-tunes and adapters of a catalogue model, which name it in `base_model`) may be used. The
organizer selected these models for the 8 GB GPU of the event machine and provides them offline in
a read-only model store; the worker finds it in the environment variable `APCTT_MODEL_STORE`.
Download the same files for development with `python tools/get_models.py ID --store DIR`
(`--list` shows the catalogue with sizes, VRAM estimates and CPU feasibility). Every catalogue model
is Apache-2.0 or MIT, and the tool adds the licence text to a copy whose repository has none. Any
other public model is refused, whatever its licence, and public model weights are never shipped in
`assets/`; there are no exceptions (D63). A fine-tune or adapter is your own work under MIT (D72)
and keeps the licence terms of its catalogue base model for the parts that come from it: keep the
base model's `LICENSE` and `NOTICE` files with your weights.

## 5. Data and documents

- Declare the licence or terms of every data source in your data declaration
  (`data_sources[].license`, next to source, version, date and role; [SUBMISSION.md](SUBMISSION.md)
  §3). Use an SPDX identifier (`CC-BY-4.0`) or a common name from the policy table (`CC BY 4.0`,
  "Creative Commons Attribution 4.0 International"); anything else, including a URL or words
  added after a licence name ("CC BY 4.0 for research"), goes to review. Use `"data_sources": []`
  if you used no data; a missing `declaration.json` goes to review.
- Data under non-commercial or research-only terms (for example `CC-BY-NC-4.0`) is refused (no
  exceptions, D63); share-alike terms (CC-BY-SA) and terms the checker does not recognise
  go to review. These data rules are rules (decision D81).
- PDFs you show in a Stage 2 demonstration must be ones you may show: the organizer's samples, or
  documents in the public domain or under a licence that allows it (see §9).

## 6. PyTorch and NVIDIA CUDA libraries

The organizer installs the common base environment on its own machines, with torch 2.11.0+cu128
and its dependencies from the official PyTorch index (versions in `requirements-base.lock`). The
organizer does not redistribute these wheels.

- Do not put any of them anywhere in your bundle: your bundle contains no packages at all
  (`tools/make_submission.py` refuses wheels, virtual environments and copies of base packages).
  In a bundle the checker refuses every one of them too (status `denied`, decision D52). That
  includes torch's pure-Python dependencies, not only the CUDA packages: leave them out even when
  your own code imports them, because the common base provides them. These packages are `torch`, `triton`, `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit`, the
  `nvidia-*-cu12` CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`,
  `nvidia-cudnn-cu12` and the others), `filelock`, `fsspec`, `jinja2`, `markupsafe`, `mpmath`,
  `networkx`, `setuptools`, `sympy` and `typing-extensions`. The complete list is
  `organizer_base.packages` in `contracts/license-policy.json`; the versions are in
  `requirements-cuda-example.lock` (CUDA kit).
- In a web system and in a bundle alike, PyTorch, triton, `cuda-*` and the NVIDIA CUDA library
  wheels are refused: the CUDA 12 ones carry a `-cu12` suffix, the newer ones have none
  (`nvidia-cublas`, `nvidia-cuda-runtime`, …), and any other `nvidia-*` package that declares
  NVIDIA's proprietary licence counts too. Your web system, which runs in the common base, may use
  the open-source base packages of the list (for example `jinja2`). Other `nvidia-*` packages, such as the BSD-3-Clause
  NVML binding `nvidia-ml-py` for a GPU dashboard, are checked like any other package.
- When you install them for development (`tools/make_env.py` installs them with the CUDA base, or
  the CUDA example's own environment), you accept their licences yourself, including NVIDIA's terms: the CUDA Toolkit licence agreement in most `nvidia-*-cu12`
  wheels, the NVIDIA SDK licence with its cuDNN and cuSPARSELt supplements, and the NVIDIA Software
  License of `cuda-bindings`. Among other conditions they cover use with NVIDIA GPUs, U.S. export
  rules and who may accept them. What two wheels of the base state, exactly: `nvidia-nccl-cu12`
  2.28.9 declares `License-Expression: LicenseRef-NVIDIA-Proprietary`, while its only licence file
  (`.dist-info/licenses/License.txt`) is a BSD-3-Clause-style text from NVIDIA that also points to
  the NVTX project for files included from it; `nvidia-nvtx-cu12` 12.8.90 declares
  `License: Apache 2.0` with the classifier `License :: Other/Proprietary License`, and its licence
  file is the Apache License 2.0 with LLVM exceptions. Metadata and files disagree; the organizer
  treats both as part of NVIDIA's base, which you do not ship. `cuda-pathfinder` declares
  Apache-2.0 and `cuda-toolkit` states no licence. Read the licence file in each wheel's
  `.dist-info` folder.
- Do not copy NVIDIA libraries into your bundle or web system. A wheel that carries them (for
  example a `libcudart` file inside another package) is flagged for review.

## 7. Run the checker

Standard library only, Python 3.10 or newer, no network access. `tools/make_submission.py` runs
the checker (`--bundle`) on every zip it packs or checks, with the policy the organizer's intake
uses, so a `VALID` zip has passed it. Run the checker yourself for the full report, the JSON or a
draft notices file: `--bundle` on the folder you pack (the five entries of the minimal bundle,
[SUBMISSION.md](SUBMISSION.md) §1.2) and `--web` on your Stage 2 web system. From the kit
directory:

```bash
python tools/check_licenses.py --bundle ../my_submission
python tools/check_licenses.py --web ../my_web_system
python tools/check_licenses.py --bundle ../my_submission --json > licence-report.json
python tools/check_licenses.py --bundle ../my_submission --notices third-party-notices-draft.md
```

`--bundle` and `--web` can be given together. `--notices` writes a draft notices file listing every
component, version and licence it found (refused components only under "Not cleared"); check it
against the components' own licence files, then save it as `participant_module/THIRD_PARTY_NOTICES.md`
in your module before you compute the package hash. `--policy` selects another policy
file (default: `contracts/license-policy.json`).

What `--bundle` reads: `submission.json` (`"license"`, `"models"`), the text of
`participant_module/LICENSE` (it must hold the MIT text, §2), `declaration.json` (`data_sources[].license`), weight and
library files in `assets/` and `participant_module/`, and every Python package it finds: every
`*.whl` file and every installed or vendored package, read from its `.dist-info/METADATA`,
`.egg-info/PKG-INFO` or the `PKG-INFO` of an unpacked source package. The minimal bundle contains
none of these (`tools/make_submission.py` refuses wheels, a `wheels/` folder, virtual environments
and installed package metadata). Third-party code with package metadata inside
`participant_module/` is judged by its own metadata and goes to review; copy a third-party
pure-Python module as plain source files and list it by hand in `THIRD_PARTY_NOTICES.md` instead
(§2.2). A virtual environment or `site-packages` folder outside `participant_module/` and `assets/`
is not part of the submission layout: it is reported as a refused layout item and not read. A
symbolic link to a wheel is not followed and is review. A leftover `requirements.lock` is read as
well (each package in it without a wheel is review); the minimal bundle has none.

What `--web` reads, in the whole web directory (sub-folders such as `frontend/` and `backend/`
included, but not `node_modules/` below a project, virtual environments or `.git`):

- npm, in every project folder (the root and each sub-project, §2): `package-lock.json` or
  `npm-shrinkwrap.json` (version 2 or 3); without one, `node_modules/.package-lock.json`; without
  that, `node_modules/*/package.json` (direct `devDependencies` are the dev-only ones); without
  that, the names in `package.json` (licences not read: review; name rules applied).
  `yarn.lock`, `pnpm-lock.yaml` and `bun.lock` are not read: without `node_modules/` they are
  review, and pnpm's symbolic links in `node_modules/` are not followed (review). Add
  `package-lock.json` (`npm install --package-lock-only`) to each project.
- Python: your web system runs in the common base environment and uses only its packages
  (fastapi, uvicorn, jinja2, python-multipart, httpx, pydantic), so it needs no Python dependency
  files. If the web directory has `requirements*.txt` or `requirements*.lock` files, every line is read (a
  requirement that is not pinned with `==` is review: "not pinned"; a package without a wheel is
  review, because its licence is not read), and so is every wheel and every installed package in
  the web directory (a `.venv`, `site-packages`, a `.dist-info` folder), because the whole web directory is
  checked; keep your development environment outside it. `pyproject.toml`, `uv.lock`,
  `poetry.lock`, `Pipfile`, `Pipfile.lock`, `setup.py`, `setup.cfg` and `environment.yml` are not
  read: each is review unless a `requirements*.txt` sits next to it. Leave these files out of the
  web directory: the web code needs no Python dependency files. A Python package of the web directory that is not
  in the common base (`contracts/base-environment.json`) is review, because the web system runs in
  the common base with its packages only (D64).
- Your `LICENSE` at the root (it must hold the MIT text, §2), the `"license"` of your own
  `package.json` files, and folders of
  copied or minified front-end files (§3.4), including build output (`dist/`, `build/` …) that has
  no `package.json`.

What it cannot see: licence files other than your own and the contents of files (JavaScript, CSS, images, icons,
fonts), code pasted into your own files, and libraries copied as plain source files without
package metadata: list them by hand, with version and licence, in `THIRD_PARTY_NOTICES.md`. Exit
code 0 means that the declared metadata of everything the checker found raises no flag, not that
nothing else is there or that a licence has been approved.

| Exit code | Meaning |
|---|---|
| 0 | Everything the checker found is allowed (see above for what it cannot see) |
| 1 | Something is refused (`denied`), including a package of the common base (§6) or a layout item |
| 2 | Nothing refused, but some items need the organizer's review (after the submission) |
| 3 | Usage or input error (for example a missing directory or an unreadable policy file) |

Each line of the report gives the status (`allowed`, `review`, `denied`), the kind (`team`,
`layout`, `wheel` for a wheel or an installed package, `lock`, `npm`, `model`, `dataset`,
`asset`), the component, version, licence, where the licence came from and why. Review items are
not failures: the organizer decides them after the submission, and there are no advance rulings
(D63). Fix denied items before you submit.

## 8. Checklist

- [ ] `participant_module/LICENSE` with the MIT text and `"license": "MIT"` in `submission.json` (D72).
- [ ] Every catalogue model you use listed under `"models"` by `catalog_id` (no weights of it in
      your zip); your own trained weights listed with `"team_trained": true` and `path`, and
      `base_model` (a catalogue id) for a fine-tune or adapter.
- [ ] Every third-party pure-Python module you copied in listed with its licence in
      `participant_module/THIRD_PARTY_NOTICES.md`; your team answers for it (D66, §2.2).
- [ ] Every data source in your declaration has a `license`.
- [ ] No package anywhere in your bundle (no wheels, no virtual environment, no copy of a base
      package): torch, triton, `cuda-*`, `nvidia-*` CUDA libraries, torch's pure-Python
      dependencies filelock, fsspec, jinja2, markupsafe, mpmath, networkx, setuptools, sympy and
      typing-extensions, and every other package come from the common base (§6).
- [ ] `participant_module/THIRD_PARTY_NOTICES.md` written, then the package hash computed.
- [ ] Stage 2: `LICENSE` and `THIRD_PARTY_NOTICES.md` at the root of the web code, the front-end
      sources with `package.json` and `package-lock.json` (not only the build output), licence
      files next to bundled fonts and assets (all of `pdfjs-dist`'s), Python code that uses only
      packages of the common base (no virtual environment in the web directory), copied front-end files
      listed by hand, and an About or Licences page.
- [ ] `tools/make_submission.py` (which runs `tools/check_licenses.py --bundle`) ends with `VALID`,
      or with `VALID_WITH_REVIEW` and you have explained each review item; `--web` on the web
      system exits with 0, or with 2 and you have explained each review item.

## 9. Participation terms

The participation terms state these points; the organizer settled the last open questions on
2026-10-13 (decision D81).

- Rights: your team grants the organizer and APCTT the rights to run, evaluate, demonstrate, keep,
  hand over and further develop your submission; your work is under the licence rule of §2; and
  you assure that you have the right to everything you submit.
- The non-assertion clause of §2.1 (D54) applies as written there. Your assurance of rights is
  given to the organizer and APCTT and is separate from the warranty disclaimer of the MIT
  licence.
- The common CUDA base is installed by the organizer on its own machines and is not
  redistributed; on the event computers you use it as installed. If you install the NVIDIA
  packages yourself for development, you accept their terms, U.S. export rules included (§6).
- Each member who signs the registration confirms that he or she is of age and authorised to
  bind the team, or that a parent, guardian or authorised representative has signed.
- PDFs you show in a Stage 2 demonstration must be ones you may show (§5).
- Revocation under D66 (§2.2) follows the procedure of PROBLEM_DESCRIPTION.md §16: written notice
  with the reason and the evidence, 48 hours to respond, and a decision by a panel of at least
  three organizer members without a conflict of interest.
- Any question that the terms do not answer is decided by APCTT, which publishes the answer to
  every team (decision D81).
