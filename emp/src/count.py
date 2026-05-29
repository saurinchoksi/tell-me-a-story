#!/usr/bin/env python3
"""EMP Count (step 5): tally hand-coded axial-labels into the failure-mode pivot,
and dump the data needed for the two bookkeeping debts.

axial-labels.json schema: {"labels":[{segmentId, codes:[...], createdAt, updatedAt}]}.
`codes` is a list of mode tags ("M1".."M10","NotA"). Read-only on session data.

Run:  python3 emp/src/count.py                              # pivot + debt data to stdout
      python3 emp/src/count.py --html emp/results/pivot.html  # also write the HTML pivot
"""
import json, collections, re, argparse, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # emp/src/count.py -> repo root

S = [("20251207-195607", "Moon"), ("20251207-202105", "Cruel Baby"),
     ("20251210-203654", "Rubber Ducky"), ("20260117-202237", "Pandavas"),
     ("20260129-204404", "Portal")]
NM = [n for _, n in S]
MODES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "NotA"]
MODE_LABELS = {
    "M1": "Wrong words on real speech", "M2": "Words on silence or noise",
    "M3": "Missed real speech", "M4": "Wrong speaker", "M5": "Overlapping speech",
    "M6": "Wrong segment boundaries", "M7": "Word at the wrong time",
    "M8": "Non-speech marked unintelligible", "M9": "Name mistranscription",
    "M10": "Broken-Whisper filler loop", "NotA": "Not a failure (clean segment)",
}
# mechanical floors from the committed sweep summaries (not plain segment counts)
M3_FLOOR_TOTAL = 39  # TMAS-44 gap sweep, story-scope >=1.0s (Portal 29, Moon 4, CB 4, RD 2, Pand 0)
M7_NOTE = ("a distribution, not a count: ~12% of words start >=0.1s late, ~8% >=0.25s, "
           "~3% >=0.5s; fix is forced alignment (TMAS-50)")

def jload(p): return json.load(open(p))
def labels(sid): return jload(ROOT / "sessions" / sid / "axial-labels.json")["labels"]

def seg_text(sid):
    """segmentId -> transcript text, from transcript-rich.json (schema-tolerant)."""
    tr = jload(ROOT / "sessions" / sid / "transcript-rich.json")
    segs = tr["segments"] if isinstance(tr, dict) and "segments" in tr else (tr if isinstance(tr, list) else [])
    out = {}
    for i, s in enumerate(segs):
        txt = s.get("text", "") if isinstance(s, dict) else ""
        if not txt and isinstance(s, dict) and "words" in s:
            txt = "".join(w.get("word", w.get("text", "")) for w in s["words"])
        out[i] = txt
    return out

# ---------- name list (dictionary cross-ref for the Mode 9 / Mode 1 check) ----------
md = jload(ROOT / "data" / "mahabharata.json")
names = []
for e in md.get("entries", []):
    for fld in ("canonical", "name"):
        if isinstance(e.get(fld), str): names.append(e[fld])
    for fld in ("variants", "aliases"):
        if isinstance(e.get(fld), list): names += [x for x in e[fld] if isinstance(x, str)]
names_l = sorted({n.lower() for n in names if n and len(n) > 2})
def is_name(t):
    t = (t or "").lower()
    return any(re.search(r"\b" + re.escape(n) + r"\b", t) for n in names_l)

def compute():
    piv = {m: collections.Counter() for m in MODES}
    multi = []; recs = 0; raw = collections.Counter()
    for sid, name in S:
        for r in labels(sid):
            recs += 1
            codes = [c.strip() for c in r.get("codes", []) if str(c).strip()]
            for c in codes: raw[c] += 1
            nn = sorted({c for c in codes if c != "NotA"})
            if len(nn) > 1: multi.append((name, r.get("segmentId"), codes))
            for c in set(codes): piv.setdefault(c, collections.Counter())[name] += 1
    return piv, multi, recs, raw

def render_html(piv, recs):
    fail_vals = [piv[m][n] for m in MODES if m != "NotA" for n in NM]
    mx = max(fail_vals) if fail_vals else 1
    rows = []
    for m in MODES:
        rc = piv[m]; tot = sum(rc.values())
        cells = []
        for n in NM:
            v = rc[n]
            if m == "NotA":
                bg = "#f3f4f6"
            elif v:
                a = 0.08 + 0.72 * (v / mx)
                bg = f"rgba(190,40,40,{a:.3f})"
            else:
                bg = "transparent"
            cells.append(f'<td style="background:{bg}">{v}</td>')
        star = ' <span class="star">*</span>' if m == "M3" else ' <span class="star">**</span>' if m == "M7" else ""
        cls = ' class="nota"' if m == "NotA" else ""
        rows.append(f'<tr{cls}><th class="mode">{m}</th><td class="lbl">{html.escape(MODE_LABELS.get(m, m))}{star}</td>{"".join(cells)}<td class="tot">{tot}</td></tr>')
    head = "".join(f"<th>{html.escape(n)}</th>" for n in NM)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>EMP failure-mode pivot</title>
<style>
 :root{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
 body{{margin:2.5rem auto;max-width:920px;color:#1a1a1a;padding:0 1rem}}
 h1{{font-size:1.4rem;margin:0 0 .2rem}}
 .sub{{color:#666;font-size:.85rem;margin:0 0 1.4rem;line-height:1.45}}
 table{{border-collapse:collapse;width:100%;font-size:.85rem}}
 th,td{{padding:.4rem .55rem;text-align:center;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums}}
 th.mode{{text-align:left;color:#999;font-weight:600;width:3rem}}
 td.lbl{{text-align:left;white-space:nowrap}}
 td.tot{{font-weight:700;border-left:1px solid #ddd}}
 thead th{{border-bottom:2px solid #333;font-size:.8rem}}
 tr.nota td,tr.nota th{{color:#999}}
 .star{{color:#be2828;font-weight:700}}
 .notes{{font-size:.8rem;color:#444;margin-top:1.25rem;line-height:1.55}}
</style></head><body>
<h1>EMP &mdash; failure-mode pivot</h1>
<p class="sub">Hand-coded segments across the five story sessions ({recs} coded). Each cell is the number of segments carrying that mode; a segment can carry more than one. Generated by <code>emp/src/count.py --html</code>.</p>
<table>
<thead><tr><th class="mode">&nbsp;</th><th class="lbl">Failure mode</th>{head}<th class="tot">Total</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
<p class="notes">
<span class="star">*</span> <b>Mode 3</b> shows only segments a coder could see; the no-trace floor (TMAS-44 gap sweep, story-scope &ge;1.0s) is <b>{M3_FLOOR_TOTAL}</b> (Portal 29, Moon 4, Cruel Baby 4, Rubber Ducky 2, Pandavas 0).<br>
<span class="star">**</span> <b>Mode 7</b> is {html.escape(M7_NOTE)} &mdash; the tagged figure understates it.
</p>
</body></html>"""

def main():
    ap = argparse.ArgumentParser(description="EMP failure-mode count / pivot")
    ap.add_argument("--html", metavar="PATH", help="also write the pivot as a self-contained HTML artifact")
    args = ap.parse_args()
    piv, multi, recs, raw = compute()

    print("### PIVOT - hand-coded mode instances (distinct code per segment)")
    print("mode," + ",".join(NM) + ",TOTAL")
    for m in MODES:
        rc = piv[m]; print(f"{m}," + ",".join(str(rc[n]) for n in NM) + f",{sum(rc.values())}")
    for k in [x for x in piv if x not in MODES]:
        rc = piv[k]; print(f"!{k}," + ",".join(str(rc[n]) for n in NM) + f",{sum(rc.values())}")
    print(f"records={recs} distinct_codes={sorted(raw)} dict_names={len(names_l)}")
    print("multicoded=%d: " % len(multi) + "; ".join(f"{a}/s{b}/{c}" for a, b, c in multi))

    print("\n### DEBT1 - every M1-coded segment, with its transcript text + name flag")
    for sid, name in S:
        txt = seg_text(sid)
        for r in labels(sid):
            if "M1" not in r.get("codes", []): continue
            sg = r.get("segmentId"); t = txt.get(sg, "")
            print(f"  {name}|s{sg}|{r.get('codes')}|{'NAME' if is_name(t) else 'common'}|{t[:80]!r}")

    print("\n### DEBT2 - every M2-coded segment, with its transcript text")
    for sid, name in S:
        txt = seg_text(sid)
        for r in labels(sid):
            if "M2" not in r.get("codes", []): continue
            sg = r.get("segmentId")
            print(f"  {name}|s{sg}|{r.get('codes')}|{txt.get(sg, '')[:80]!r}")

    print("\n### M9 hand-coded per session")
    for sid, name in S:
        print(f"  {name}: " + str(sum(1 for r in labels(sid) if 'M9' in r.get('codes', []))))

    if args.html:
        outp = Path(args.html)
        if not outp.is_absolute(): outp = ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(render_html(piv, recs))
        print(f"\nwrote HTML pivot -> {outp.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
