#!/usr/bin/env python3
"""Detector groupings vs. the human truth, for one session.  READ-ONLY.

Overlays the name-truth review (emp/src/name_truth.py -> name-truth.json) on what
the two name detectors actually grouped, to make the detector's three failure
shapes visible at a glance:

  - MERGE  — the detector put two different true names in one cluster
             (Jiraki + the place York; the canon James + the made-up Jammus).
  - SPLIT  — one true name's spellings the detector never linked, because they
             share no Double Metaphone code (Pataki/Bacchus; Biffy/Bibi/Bessie).
  - SUBSTITUTION — an improvised name Whisper wrote as a valid name, so nothing
             flagged it at all (Jammus heard as the canon "Thomas").

Private (real names) — written under the gitignored emp/results/visuals/<id>/.

    python emp/src/show_m9_groupings.py 20260129-204404
"""
import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metaphone import doublemetaphone  # noqa: E402

from detectors.family_names import FamilyNameDetector  # noqa: E402
from detectors.name_consistency import NameConsistencyDetector  # noqa: E402
from detectors.phonetics import clean  # noqa: E402

# category value -> (display mode, css class)
CAT = {
    "roster": ("M9a roster", "roster"),
    "tv-canon": ("M9c canon", "canon"),
    "improvised": ("M9b improvised", "improvised"),
    "place-other": ("place / word", "place"),
    "not-a-name": ("not a name", "nota"),
    "": ("unreviewed", "unknown"),
}
NAME_CATS = {"roster", "tv-canon", "improvised"}  # the categories that are real names


def load_truth(sid):
    path = ROOT / "emp" / "results" / "visuals" / sid / "name-truth.json"
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text())
    items = {k: v for k, v in raw.get("items", {}).items()}
    occ = {k: v.get("true_name", "") for k, v in raw.get("occurrences", {}).items()}
    return items, occ


def build(sid):
    sd = ROOT / "sessions" / sid
    rich = json.loads((sd / "transcript-rich.json").read_text())
    word_at = {}  # "seg:wi" -> cleaned token
    for s in rich["segments"]:
        for wi, w in enumerate(s.get("words", [])):
            word_at[f"{s['id']}:{wi}"] = clean(w["word"])

    items, occ = load_truth(sid)
    truename = {c: (items[c].get("true_spelling") or c) for c in items}
    category = {c: items[c].get("category", "") for c in items}
    known_names = {truename[c] for c in items if category.get(c) in NAME_CATS}

    m9a = FamilyNameDetector().run(sd)
    m9b = NameConsistencyDetector().run(sd)
    m9a_tokens = {f["cleaned"] for f in m9a["flags"]}

    clusters = defaultdict(lambda: {"members": set(), "n": 0})
    for f in m9b["flags"]:
        clusters[f["cluster_id"]]["members"].add(f["cleaned"])
        clusters[f["cluster_id"]]["n"] += 1

    # ---- Section A: detector clusters, truth-annotated ----
    cluster_cards = []
    for cid, c in sorted(clusters.items()):
        members = sorted(c["members"])
        true_in = {truename.get(m, m) for m in members if category.get(m) in NAME_CATS}
        has_roster = any(category.get(m) == "roster" for m in members)
        has_nonname = any(category.get(m) in ("place-other", "not-a-name") for m in members)
        if has_roster:
            verdict, vcls = "roster name — M9a should own this", "roster"
        elif len(true_in) > 1:
            verdict, vcls = f"MERGED {len(true_in)} names: {' + '.join(sorted(true_in))}", "merge"
        elif has_nonname:
            verdict, vcls = "MERGED a non-name into a name", "merge"
        else:
            verdict, vcls = "one true name — clean", "clean"
        rows = ""
        for m in members:
            mode, mcls = CAT.get(category.get(m, ""), CAT[""])
            tn = truename.get(m, "")
            tn_html = f' &rarr; <b>{html.escape(tn)}</b>' if tn else ""
            star = " ✦M9a" if m in m9a_tokens else ""
            rows += (f'<div class="row"><span class="form">{html.escape(m)}</span>'
                     f'<span class="badge {mcls}">{mode}</span>'
                     f'<span class="tn">{tn_html}{star}</span></div>')
        cluster_cards.append(
            f'<div class="card {vcls}"><div class="chead"><span class="cid">cluster · {html.escape(cid)}</span>'
            f'<span class="verdict {vcls}">{verdict}</span><span class="count">{c["n"]}×</span></div>'
            f'{rows}</div>')

    # ---- Section B: each true name, and how the detector handled it ----
    by_true = defaultdict(list)  # true name -> [cleaned spellings]
    for cln, cat in category.items():
        if cat in NAME_CATS:
            by_true[truename[cln]].append(cln)
    cln_cluster = {m: cid for cid, c in clusters.items() for m in c["members"]}

    true_rows = ""
    for tn in sorted(by_true):
        spellings = sorted(by_true[tn])
        cids = [cln_cluster.get(s) for s in spellings]
        nonnull = {c for c in cids if c}
        if len(spellings) == 1:
            verdict, vcls = ("single spelling — consistent", "clean")
        elif len(nonnull) == 1 and None not in cids:
            # all spellings landed in exactly one cluster; was it mixed with another name?
            others = {truename.get(m, m) for m in clusters[next(iter(nonnull))]["members"]
                      if category.get(m) in NAME_CATS} - {tn}
            verdict, vcls = ((f"clustered, but MERGED with {', '.join(sorted(others))}", "merge")
                             if others else ("clustered correctly", "ok"))
        else:
            # >1 spelling, not all in one cluster -> the detector never linked them
            verdict, vcls = ("SPLIT — one name, but the detector never linked these spellings", "split")
        mode, mcls = CAT.get(category.get(spellings[0], ""), CAT[""])
        true_rows += (f'<div class="trow"><span class="tname badge {mcls}">{html.escape(tn)}</span>'
                      f'<span class="spell">{" · ".join(html.escape(s) for s in spellings)}</span>'
                      f'<span class="verdict {vcls}">{verdict}</span></div>')

    # ---- Section C: sound-swaps ----
    # A swap = an occurrence whose WRITTEN word sounds different (different DM primary
    # code) from its true name. That means Whisper wrote a different-sounding, often
    # valid, word (Bacchus for Pataki, Thomas for Jammus) — the silent failure no text
    # check sees. Counting per occurrence catches EVERY instance: the spelling-level
    # truth (every Bacchus is Pataki) and any per-occurrence override (one is Jiraki).
    def pc(x):
        return doublemetaphone(x)[0]
    flagged_forms = m9a_tokens | set(cln_cluster)  # forms a detector already surfaces
    swaps = defaultdict(lambda: defaultdict(int))  # written form -> true name -> count
    for s in rich["segments"]:
        for wi, w in enumerate(s.get("words", [])):
            c = clean(w["word"])
            if category.get(c) not in NAME_CATS or c in flagged_forms:
                continue  # only the truly silent swaps: a detector never flags this form
            ov = occ.get(f"{s['id']}:{wi}", "")
            tv = ov if ov in known_names else truename.get(c, "")
            if tv and pc(c) != pc(tv):  # sounds different -> a swap, not a spelling variant
                swaps[c][tv] += 1
    sub_rows = ""
    for written in sorted(swaps, key=lambda x: -sum(swaps[x].values())):
        total = sum(swaps[written].values())
        parts = " · ".join(f"<b>{html.escape(tn)}</b> {n}×"
                           for tn, n in sorted(swaps[written].items(), key=lambda x: -x[1]))
        sub_rows += (f'<div class="trow"><span class="spell">Whisper wrote “<b>{html.escape(written)}</b>” '
                     f'{total}×</span><span class="verdict merge">really {parts} — no detector sees this</span></div>')
    if not sub_rows:
        sub_rows = '<p class="empty">none recorded</p>'

    return PAGE.format(sid=html.escape(sid), clusters="".join(cluster_cards),
                       truths=true_rows, subs=sub_rows)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Groupings vs truth · {sid}</title>
<style>
  :root {{ --ink:#1a1d24; --muted:#6b7280; --line:#e6e8ec; --bg:#fbfcfd;
    --roster:#7c5cff; --canon:#0369a1; --improvised:#0d7a4f; --place:#c2410c; --nota:#9aa0ab;
    --merge:#c2410c; --split:#b45309; --ok:#0d7a4f; }}
  * {{ box-sizing:border-box; }}
  body {{ font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink);
    background:var(--bg); margin:0; padding:2rem clamp(1rem,5vw,4rem); max-width:62rem; }}
  h1 {{ font-size:1.45rem; margin:0 0 .2rem; letter-spacing:-.02em; }}
  .lede {{ color:var(--muted); margin:0 0 1.4rem; font-size:.9rem; }}
  h2 {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
    margin:2.2rem 0 .9rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }}
  .card {{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:.8rem 1rem; margin-bottom:.7rem; }}
  .card.merge {{ border-left-color:var(--merge); background:#fffaf6; }}
  .card.roster {{ border-left-color:var(--roster); }}
  .card.clean {{ border-left-color:var(--ok); }}
  .chead {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin-bottom:.5rem; }}
  .cid {{ font-weight:700; font-family:ui-monospace,Menlo,monospace; font-size:.85rem; }}
  .count {{ margin-left:auto; color:var(--muted); font-size:.8rem; }}
  .verdict {{ font-size:.74rem; font-weight:700; }}
  .verdict.merge, .verdict.split {{ color:var(--merge); }}
  .verdict.ok, .verdict.clean {{ color:var(--ok); }}
  .verdict.roster {{ color:var(--roster); }}
  .row {{ display:flex; align-items:baseline; gap:.6rem; padding:.18rem 0; font-size:.9rem; }}
  .form {{ font-weight:600; min-width:7rem; font-family:ui-monospace,monospace; font-size:.82rem; }}
  .tn {{ color:var(--muted); }}
  .badge {{ font-size:.66rem; font-weight:700; padding:.08rem .45rem; border-radius:999px;
    text-transform:uppercase; letter-spacing:.03em; white-space:nowrap; }}
  .badge.roster {{ background:#efeaff; color:var(--roster); }}
  .badge.canon {{ background:#e0f2fe; color:var(--canon); }}
  .badge.improvised {{ background:#e3f5ec; color:var(--improvised); }}
  .badge.place {{ background:#ffe8d9; color:var(--place); }}
  .badge.nota {{ background:#f0f1f3; color:var(--nota); }}
  .badge.unknown {{ background:#f0f1f3; color:var(--nota); }}
  .trow {{ display:flex; align-items:baseline; gap:.7rem; padding:.4rem .2rem; border-bottom:1px solid var(--line);
    font-size:.88rem; flex-wrap:wrap; }}
  .tname {{ min-width:6rem; text-align:center; }}
  .spell {{ color:var(--muted); flex:1; min-width:10rem; }}
  .empty {{ color:var(--muted); font-style:italic; }}
  .legend {{ font-size:.78rem; color:var(--muted); margin-top:.6rem; }}
</style></head>
<body>
  <h1>Detector groupings vs. your truth — {sid}</h1>
  <p class="lede">What the code grouped, checked against the by-ear review. Orange = the detector
    got the grouping wrong (merged two names, or split one name it couldn't hear was the same).</p>

  <h2>What the detector clustered (with your truth overlaid)</h2>
  {clusters}

  <h2>Each true name — and how the detector handled it</h2>
  {truths}
  <p class="legend">SPLIT = the spellings are one name but share no phonetic code, so clustering
    never linked them — the recall hole no clustering tweak can fix.</p>

  <h2>Substitutions — improvised name written as a valid name (undetectable)</h2>
  {subs}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    args = ap.parse_args()
    out = ROOT / "emp" / "results" / "visuals" / args.session_id / "groupings.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.session_id))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
