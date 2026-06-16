#!/usr/bin/env python3
"""Per-session AGREEMENT VIEW for the Stage-1 name auditor — human-checkable, with audio.

Read-only. For one session it lays each reviewed name beside (a) the by-ear GOLD class
and (b) the three architectures' verdicts, colours the agreement, and gives a ▶ to hear
each occurrence — so the off-diagonals (a miss, a false positive, the James/Jammus
collision) can be spot-checked by ear, the same ear-pass that built the ground truth.

Reuses name_truth.py's audio range-server + clip windows, and score_names.py's gold and
prediction logic (so the view can never disagree with the score). PRIVATE: echoes name
variants + plays family audio -> gitignored under emp/results/visuals/<id>/.

    ./venv/bin/python emp/src/name_audit_view.py 20260129-204404 --serve   # audio + live
    ./venv/bin/python emp/src/name_audit_view.py                           # static HTML, all 5
"""
import argparse
import html
import json

from audit_common import SESSIONS, ROOT, load_rich
from name_truth import audio_file, clip_window, fmt_time
from score_names import build_gold, gold_map, load_audit, ARCHS, _detcol

GOLD_LABEL = {"M9c": "M9c canon-wrong", "M9b": "M9b inconsistent", "M9d": "M9d substitution (audio-only)",
              "M9a": "M9a family name", "no-error": "correct / not-a-name"}


def word_times(rich):
    """{(seg_id, wi) -> (w_start, w_end, seg_start, seg_end)} for clip windows."""
    out = {}
    for s in rich["segments"]:
        for wi, w in enumerate(s.get("words", [])):
            out[(s["id"], wi)] = (w.get("start"), w.get("end"), s.get("start"), s.get("end"))
    return out


def pred_detail(audit):
    """{(story_idx, cleaned) -> {case, canonical}} from an audit's flags."""
    out = {}
    if not audit:
        return out
    for st in audit.get("stories", []):
        sidx = st["story"]["idx"]
        for f in st.get("flags", []):
            for c in f.get("wrong_cleaned", []):
                out.setdefault((sidx, c), {"case": f["case"], "canonical": f.get("canonical", "")})
    return out


def verdict_cell(gold, det):
    """(text, css-class) for a detector verdict against the gold class."""
    case = _detcol(det["case"]) if det else "none"
    if gold in ("M9b", "M9c"):
        if case == gold:
            return (det["case"] + (f"→{det['canonical']}" if det.get("canonical") else ""), "hit")
        if case != "none":
            return (det["case"] + f" (gold {gold})", "warn")   # caught, wrong case
        return ("— missed —", "miss")
    if gold == "no-error":
        if case != "none":
            return (det["case"] + " (false +)", "miss")
        return ("ok", "hit")
    # M9a / M9d (informational rows)
    if case != "none":
        return (det["case"] + (f"→{det['canonical']}" if det.get("canonical") else ""), "info")
    return ("—", "")


def build_rows(sid):
    g = build_gold(sid)
    gm, _ = gold_map(sid)
    rich = load_rich(sid)
    wt = word_times(rich)
    audits = {a: load_audit(sid, a) for a in ARCHS}
    preds = {a: pred_detail(audits[a]) for a in ARCHS}

    # one row per (story, cleaned) gold unit (skip phrase-member folds)
    by_story = {}
    for u in g["units"]:
        if u["gold"] == "phrase-member":
            continue
        st = u["story"]
        occ_clips = []
        for (seg_id, wi) in u["occs"][:8]:
            ws, we, ss, se = wt.get((seg_id, wi), (None, None, None, None))
            win = clip_window(ss, se, ws, we)
            occ_clips.append({"t": ws, "win": win})
        by_story.setdefault(st, []).append({
            "cleaned": u["cleaned"], "gold": u["gold"], "true": u["true"],
            "n": u["n_occ"], "in_story": u["in_story"], "clips": occ_clips,
            "cells": [verdict_cell(u["gold"], preds[a].get((st, u["cleaned"]))) for a in ARCHS],
            "phantom": False,
        })
    # phantom rows: anything an architecture flagged that is NOT a gold unit
    seen = {(u["story"], u["cleaned"]) for u in g["units"]}
    for a in ARCHS:
        for (st, c), det in preds[a].items():
            if (st, c) in seen:
                continue
            seen.add((st, c))
            cells = []
            for aa in ARCHS:
                d = preds[aa].get((st, c))
                cells.append((d["case"] + " (false +)", "miss") if d else ("—", ""))
            by_story.setdefault(st, []).append({
                "cleaned": c, "gold": "phantom", "true": "", "n": 0, "in_story": st is not None,
                "clips": [], "cells": cells, "phantom": True})
    return g, by_story


def render(sid):
    g, by_story = build_rows(sid)
    has_audio = audio_file(sid) is not None
    regions = {r["idx"]: r for r in g["regions"]}
    sections = []
    order = sorted(by_story, key=lambda x: (x is None, x))
    for st in order:
        if st is None:
            head = "Out-of-story names (segmenter excluded — not audited)"
        else:
            r = regions.get(st, {})
            head = f"Story {st}: segs {r.get('start_id')}–{r.get('end_id')} · world: {html.escape(str(r.get('world','')))}"
        rows = []
        for row in sorted(by_story[st], key=lambda x: (x["gold"], x["cleaned"])):
            clips = "".join(
                f'<button class="play" data-start="{c["win"][0]}" data-end="{c["win"][1]}">▶</button>'
                if (c["win"] and has_audio) else ""
                for c in row["clips"])
            cells = "".join(f'<td class="v {cls}">{html.escape(txt)}</td>' for txt, cls in row["cells"])
            true = f' → <b>{html.escape(row["true"])}</b>' if row["true"] else ""
            goldlab = GOLD_LABEL.get(row["gold"], row["gold"])
            rows.append(
                f'<tr class="{"phantom" if row["phantom"] else ""}"><td class="nm">{html.escape(row["cleaned"])}{true}'
                f'<span class="ct">{row["n"]}×</span></td>'
                f'<td class="g g-{row["gold"].split("-")[0]}">{html.escape(goldlab)}</td>{cells}'
                f'<td class="clips">{clips}</td></tr>')
        sections.append(f'<h2>{html.escape(head)}</h2><table>'
                        f'<tr><th>name</th><th>by-ear gold</th>'
                        + "".join(f"<th>{a}</th>" for a in ARCHS) + "<th>hear</th></tr>"
                        + "".join(rows) + "</table>")
    audio_el = '<audio id="player" src="/audio" preload="metadata"></audio>' if has_audio else ''
    note = "" if has_audio else '<p class="warn-note">no audio file — play disabled</p>'
    return PAGE.format(sid=html.escape(sid), name=html.escape(SESSIONS.get(sid, sid)),
                       sections="".join(sections), audio_el=audio_el, note=note, script=SCRIPT)


SCRIPT = r"""
var p=document.getElementById('player'),stop=null,cur=null;
if(p){p.addEventListener('timeupdate',function(){if(stop!=null&&p.currentTime>=stop)p.pause();});
 function rs(){stop=null;if(cur){cur.textContent='▶';cur=null;}}
 p.addEventListener('pause',rs);p.addEventListener('ended',rs);
 document.querySelectorAll('.play[data-start]').forEach(function(b){b.addEventListener('click',function(){
  var s=parseFloat(b.getAttribute('data-start')),e=parseFloat(b.getAttribute('data-end'));
  if(cur===b){p.pause();return;}if(cur)cur.textContent='▶';
  stop=e+0.4;p.currentTime=Math.max(0,s-0.5);p.play().then(function(){cur=b;b.textContent='❚❚';}).catch(rs);});});}
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Name audit · {name}</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1a1d24;background:#fbfcfd;margin:0;padding:1.5rem clamp(1rem,4vw,3rem);max-width:72rem}}
 h1{{font-size:1.4rem;margin:0 0 .2rem}} h2{{font-size:.95rem;margin:1.6rem 0 .5rem;color:#374151}}
 .lede{{color:#6b7280;margin:.2rem 0 1rem;font-size:.9rem}}
 table{{border-collapse:collapse;width:100%;margin-bottom:.6rem;font-size:.83rem}}
 th{{text-align:left;font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;color:#9aa0ab;border-bottom:1px solid #e6e8ec;padding:.3rem .5rem}}
 td{{padding:.3rem .5rem;border-bottom:1px solid #f0f1f3;vertical-align:top}}
 .nm{{font-weight:600}} .nm .ct{{color:#9aa0ab;font-weight:400;font-size:.74rem;margin-left:.4rem}}
 .g{{font-size:.76rem;white-space:nowrap}} .g-M9c{{color:#0369a1}} .g-M9b{{color:#0d7a4f}} .g-M9d{{color:#c2410c}} .g-M9a{{color:#7c5cff}} .g-no{{color:#9aa0ab}} .g-phantom{{color:#cbb}}
 td.v{{font-size:.78rem}} .v.hit{{color:#0d7a4f}} .v.miss{{color:#c2185b;font-weight:600}} .v.warn{{color:#b7791f}} .v.info{{color:#7c5cff}}
 tr.phantom{{background:#fff7f7}}
 .play{{width:1.5rem;height:1.5rem;border-radius:50%;border:1px solid #e6e8ec;background:#fff;color:#0369a1;cursor:pointer;font-size:.62rem;margin-right:.15rem}}
 .play:hover{{background:#f0f7ff}} .warn-note{{color:#c2410c}}
 .legend{{font-size:.76rem;color:#6b7280;margin:.4rem 0 1rem}} .legend b{{color:#1a1d24}}
</style></head><body>
{audio_el}
<h1>Name audit — {name} <span style="color:#9aa0ab;font-weight:400;font-size:1rem">({sid})</span></h1>
<p class="lede">By-ear gold vs the three auditor architectures, per story. Green = agree, pink = miss / false positive, amber = right name wrong case. ▶ hears the occurrence.</p>
<p class="legend"><b>worksheet</b> = name cards + examples · <b>fulltext</b> = read the windowed story · <b>hybrid</b> = cards + sampled context. M9d = substitution onto a valid word (audio-only ceiling); M9a = family name (separate detector).</p>
{note}
{sections}
<script>{script}</script>
</body></html>"""


def serve(sid, port):
    import http.server
    ap = audio_file(sid)

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(data)

        def _audio(self):
            if not ap or not ap.exists():
                self._send(404, "no audio", "text/plain"); return
            size = ap.stat().st_size
            ctype = "audio/mp4" if ap.suffix == ".m4a" else "audio/mpeg"
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                s = int(a) if a else 0
                e = min(int(b) if b else size - 1, size - 1)
                with open(ap, "rb") as f:
                    f.seek(s); chunk = f.read(e - s + 1)
                self.send_response(206); self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {s}-{e}/{size}")
                self.send_header("Content-Length", str(len(chunk))); self.end_headers(); self.wfile.write(chunk)
            else:
                self._send(200, ap.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, render(sid))
            elif path == "/audio":
                self._audio()
            else:
                self._send(404, "not found", "text/plain")

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), H)
    print(f"name audit  ->  http://127.0.0.1:{port}/   ({SESSIONS.get(sid, sid)})")
    print(f"audio       ->  {ap.name if ap else 'NONE'}   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=5057)
    args = ap.parse_args()
    if args.serve:
        if not args.session:
            ap.error("--serve needs a session id")
        serve(args.session, args.port)
        return
    sids = [args.session] if args.session else list(SESSIONS)
    for sid in sids:
        out = ROOT / "emp" / "results" / "visuals" / sid / "name-audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(sid))
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
