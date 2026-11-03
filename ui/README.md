# Local test UI (development only, never shipped)

One page for trying the assessment module without the CLI. Run it in WSL from the
repository root:

```bash
~/.venvs/apctt-train/bin/python ui/server.py            # --port 3000 by default
```

Then open **http://127.0.0.1:3000** in the Windows browser. Use `127.0.0.1`: if another app
on Windows holds `localhost:3000` (for example a Next.js dev server), `localhost` reaches that
app instead. Use `--port 3001` to avoid the clash entirely.

Every input is a dropdown. The options come from `cuda/contracts/public-cases.json` (99 cases)
and the organizer sheet `USGS_SDG_open_data_set_corrected.xlsx` (100 rows, the same file as
the team's Google Drive copy; read with the standard library). The sheet adds the left-out
report's assessment unit, and short categories under each SDG target. Picking a public case
fills the other fields, and you can change any of them afterwards. The organizer TRL is
shown only when a public case runs on its own PDF. "Upload a new PDF…" at the end of the
PDF list opens the file picker.

What it does:

- runs `participant_module` rules-only on a public case (`train/data/pdfs/`) or an uploaded
  PDF (max 50 MiB, saved under `train/out/ui-uploads/`);
- checks the result with the kit's contract checker (`apctt_contracts.validate_result`);
- shows each criterion's status, evidence pages (link opens the PDF at that page) and spans;
- with "Show ranker", adds the top 3 sentences of `train/out/criteria-ranker` per criterion.

Needs the training venv (`pypdfium2`, `pydantic`; `torch` and `transformers` only for the
ranker). `server.py` uses only the standard library and binds to 127.0.0.1.
