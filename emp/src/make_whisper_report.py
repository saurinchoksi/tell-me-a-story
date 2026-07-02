#!/usr/bin/env python3
"""Build the Whisper-with-context report — a self-contained, mobile-friendly HTML for Choksi.

Reads the experiment result files (score.json / clips.json / precision.json) and renders the
frontier (baseline -> world -> world+cast -> clip-level), the winning run's per-name detail,
and the plain-English findings. Run after the experiments complete.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import importlib.util

_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

SID = "20260211-210718"
D = ROOT / "emp/results/visuals/whisper-context" / SID


def esc(x):
    return html.escape(str(x))


def baseline():
    sd = ROOT / "sessions" / SID
    key = svk.load_key(sd)
    t = json.loads((sd / "transcript-rich.json").read_text())
    return svk.score(key, svk.flatten_words(t))


def load(name, kind):
    p = D / f"{name}.{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


# (label, file, kind, note)
ROWS = [
    ("B0 baseline — blind Whisper", None, None, "the floor; all recoveries are sound-alike, 0 exact. 14 of 19 misses are Pandavas heard as Fondos/Bondos."),
    ("E1 · world only (full file)", "E1_world_only", "score", "first 2 Pandavas → exact, then decays — a single upfront prompt doesn't reach minute 9."),
    ("E2 · world + cast (full file)", "E2_world_cast", "score", "plateaus; heavier prompt even forced Bhishma onto a Bhima spot."),
    ("E4 · CLIP + world only", "E4_clip_world", "clips", "control — names matter (worse than baseline)."),
    ("E4 · CLIP + no prompt", "E4_clip_none", "clips", "control — windowing ALONE hurts."),
    ("E6 · CLIP + combined cast", "E6_clip_combined", "clips", "Qwen∪Gemma casts with the groups restored."),
    ("E6 · CLIP + lean prompt", "E6_clip_lean", "clips", "minimal prose (leak-reduction attempt)."),
    ("E5 · CLIP + cast, wide window", "E5_clip_cast_wide", "clips", "wider window drifts back toward dilution."),
    ("E7 · CLIP + cast (repro)", "E7_clip_cast_repro", "clips", "reproducibility of the winner."),
    ("E4 · CLIP + CAST ★", "E4_clip_cast", "clips", "THE WIN — fresh cast prompt per occurrence."),
]


def recall_of(name, kind):
    d = load(name, kind)
    return d if d else None


def main():
    b = baseline()
    rows = []
    for label, name, kind, note in ROWS:
        if name is None:
            rows.append((label, b["recall"], b["hits"], b["scoreable"], note))
        else:
            d = load(name, kind)
            if d:
                rows.append((label, d["recall"], d["hits"], d["scoreable"], note))
    rows.sort(key=lambda r: r[1])
    maxr = max(r[1] for r in rows)

    # winning run per-name detail
    win = load("E4_clip_cast", "clips")
    prec = json.loads((D / "precision.json").read_text()) if (D / "precision.json").exists() else None

    bars = ""
    for label, rec, hits, scoreable, note in rows:
        pct = int(rec / maxr * 100)
        star = "win" if "★" in label else ""
        bars += (f'<div class="row {star}"><div class="lab">{esc(label)}</div>'
                 f'<div class="track"><div class="bar" style="width:{pct}%"></div>'
                 f'<span class="val">{rec:.2f} · {hits}/{scoreable}</span></div>'
                 f'<div class="note">{esc(note)}</div></div>')

    detail = ""
    if win:
        for r in sorted(win["rows"], key=lambda x: x["word_start"]):
            if r["match"] == "unknowable":
                continue
            cls = {"exact": "ok", "dm": "oks", "miss": "no"}[r["match"]]
            tag = {"exact": "recovered (exact)", "dm": "recovered (sound-alike)", "miss": "missed"}[r["match"]]
            detail += (f'<tr class="{cls}"><td>{r["word_start"]:.0f}s</td><td class="n">{esc(r["answer"])}</td>'
                       f'<td>{esc(r.get("matched_word") or "—")}</td><td class="c">{esc(r.get("clip_words",""))[:90]}</td>'
                       f'<td class="t">{tag}</td></tr>')

    prec_html = ""
    if prec:
        prec_html = (f'<p>On {prec["n_controls"]} ordinary-word control spots, the cast prompt forced a '
                     f'name in <b>{prec["forced_cast"]}</b> ({prec["force_rate_cast"]:.0%}) vs '
                     f'<b>{prec["forced_none"]}</b> ({prec["force_rate_none"]:.0%}) with no prompt — '
                     f'<b>net +{prec["forced_cast"]-prec["forced_none"]}</b> forced by the prompt, including a '
                     f'verbatim prompt-leak. So this is applied only where a name is already suspected, never blanket.</p>')

    CSS = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:820px;margin:0 auto;padding:22px;color:#1a1a1a;line-height:1.55}
    h1{font-size:1.45rem;margin:.1em 0}h2{font-size:1.12rem;margin-top:1.7em;border-bottom:2px solid #eee;padding-bottom:.2em}
    .sub{color:#666;font-size:.92rem}.big{font-size:2.1rem;font-weight:700;color:#1a7a1a}
    .row{margin:.5em 0}.lab{font-size:.86rem;font-weight:600}.row.win .lab{color:#1a7a1a}
    .track{position:relative;background:#f0f0f2;border-radius:6px;height:22px;margin:2px 0}
    .bar{background:#9aa7b4;height:100%;border-radius:6px}.row.win .bar{background:#2f9e44}
    .val{position:absolute;right:8px;top:1px;font-size:.76rem;color:#333}
    .note{font-size:.76rem;color:#777}
    table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.6em 0}
    td,th{padding:4px 7px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
    tr.ok{background:#eafbea}tr.oks{background:#f2faf2}tr.no{background:#fdecec}
    td.n{font-weight:600}td.c{color:#555;font-style:italic}td.t{white-space:nowrap;font-size:.74rem}
    .k{background:#f0f6ff;border-left:4px solid #4a90d9;padding:9px 13px;border-radius:6px;margin:.9em 0}
    .k.warn{background:#fff6f0;border-color:#e08a4a}
    """
    H = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whisper-with-context — telling Whisper the world</title><style>{CSS}</style></head><body>
<h1>Can we just tell Whisper the world?</h1>
<p class="sub">Re-transcribing the Mahabharata session with the story's world (and cast) as context, scored against your 36-item by-ear key. Autonomous run, 2026-07-01.</p>

<div class="k"><b>Headline:</b> blind Whisper hits <b>{int(b['hits'])}/{b['scoreable']}</b> of your key names on its own.
Telling it the world and its cast <b>per name occurrence</b> (short clips) lifts that to <span class="big">{int(win['hits']) if win else '?'}/{win['scoreable'] if win else '?'}</span>
— and it turns the child's <i>Fondos</i> back into <b>Pandavas</b>, which no text-based fixer could ever reach.</div>

<h2>The frontier</h2>
{bars}

<h2>Why it works — and its limit</h2>
<div class="k"><b>The bias decays over a long file.</b> A single prompt at the start recovered "Pandavas" only for the first minute; by minute 9 it was back to <i>Fondos</i>. Re-prompting a short clip around <i>each</i> occurrence keeps the context fresh — that's the whole gain (the two controls above prove it's the cast names, not the clipping: windowing alone actually hurts).</div>
<div class="k"><b>The cast must include the groups.</b> Both Qwen and Gemma, asked for "characters," list the five Pandava brothers but NOT the group name "Pandavas" or "Kauravas" — which is 14 of your 36 items. A cast built for name-correction has to carry the story's families/groups, the opposite of a tidy character list.</div>
<div class="k warn"><b>It's not free — it can force names.</b> {prec_html}</div>
<div class="k"><b>The floor is honest.</b> The Pandavas it still misses no longer read <i>Fondos</i> — they read <b>Bandhas / Bando</b>, which is how your daughter actually said the word. At that point Whisper is decoding what's really on the tape; that's the truth, not an error to fix.</div>

<h2>The winning run, name by name</h2>
<table><tr><th>time</th><th>answer</th><th>Whisper wrote</th><th>the clip it decoded</th><th>result</th></tr>
{detail}</table>

<p class="sub">Scripts: emp/src/retranscribe.py · clip_retranscribe.py · score_vs_key.py · precision_check.py · gemma_cast.py. Full log: emp/results/visuals/whisper-context/notes.md.</p>
</body></html>"""
    out = D / "whisper-context-report.html"
    out.write_text(H)
    print(out)


if __name__ == "__main__":
    main()
