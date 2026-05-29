# EMP — Evals Mini Project (tooling & outputs)

Self-contained home for the Evals Mini Project's code and analysis outputs.
The **canonical write-up and method** live in `reference/career-build/emp.md`
(outside this repo); this folder is just the tools and their results.

## Layout
- `src/` — analysis tools (read-only over session data; run from the repo root):
  - `count.py` — tally the hand-coded `axial-labels.json` into the failure-mode pivot. `--html` writes `results/pivot.html`.
  - `gap_analysis.py` — Mode 3 missed-speech sweep (TMAS-44).
  - `timestamp_drift_analysis.py` — Mode 7 timestamp-drift sweep (TMAS-45).
  - `populate_mode9.py` / `populate_mode10.py` — write Mode 9 / Mode 10 codes back into `axial-labels.json`.
- `results/` — outputs:
  - `pivot.html` — the failure-mode pivot (counts only; committed).
  - `gap-analysis/summary.md`, `timestamp-drift/summary.md` — cross-session sweep summaries (committed).
  - `visuals/<id>/*.html` — per-session visual breakdowns (gitignored; contain transcript text).

## What stays in `sessions/<id>/` (not here)
`axial-labels.json` and `validation-notes.json` are the validator app's live
read/write store (served by `api/routes/`), co-located with the transcript they
annotate — durable session data, kept there beyond the EMP.

## Run (from repo root)
```
python3 emp/src/count.py --html emp/results/pivot.html
python3 emp/src/gap_analysis.py
python3 emp/src/timestamp_drift_analysis.py
```
