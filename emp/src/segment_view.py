#!/usr/bin/env python3
"""Per-session view: the segmenter's predicted stories vs. the human truth.

A read-only companion to the ground-truth tool — two ribbons down the transcript,
TRUTH on the left and PREDICTED on the right, so a mismatch (a start placed a few
lines off, an over-split, a missed story) is visible at a glance. The model's
evidence quotes sit at each predicted boundary, and ▶ plays any line so a fuzzy
start can be checked by ear. This is the EMP's human-checkable artifact for
Stage 0; the numbers live in score_segmentation.py.

    ./venv/bin/python emp/src/segment_view.py            # write static HTML (no audio)
    ./venv/bin/python emp/src/segment_view.py --serve    # serve with audio + range requests

PRIVATE (transcript text) -> gitignored emp/results/visuals/.
"""
import argparse
import html
import json
from pathlib import Path

from segment import SESSIONS, load_segments, load_truth, PRED_OUT, ROOT

HTML_OUT = ROOT / "emp" / "results" / "visuals" / "segmentation-view.html"
GT_COLORS = ["#7fb0a3", "#8fb27a", "#6fa3b0"]      # truth = cool greens/teals
PR_COLORS = ["#e0876f", "#e6b566", "#d98a9e"]      # predicted = warm corals/golds


def audio_path(sid):
    return next((ROOT / "sessions" / sid).glob("audio.*"), None)


def fmt(t):
    return "—" if t is None else f"{int(t // 60)}:{int(t % 60):02d}"


def bars_for(stories, order, start_key, end_key):
    """position -> (story_idx, is_start, is_end) for each covered position."""
    pos = {x: i for i, x in enumerate(order)}
    cover = {}
    for k, st in enumerate(stories):
        s, e = pos.get(st[start_key]), pos.get(st[end_key])
        if s is None or e is None:  # stale story ids vs current transcript — surface it
            print(f"WARNING: story ids {st.get(start_key)}-{st.get(end_key)} absent from the transcript — skipped")
            continue
        s, e = min(s, e), max(s, e)
        for p in range(s, e + 1):
            cover[p] = (k, p == s, p == e)
    return cover


def render(current=None):
    truth = load_truth()
    pred_all = json.loads(PRED_OUT.read_text())["sessions"] if PRED_OUT.exists() else {}
    current = current if current in SESSIONS else next(iter(SESSIONS))
    segs = load_segments(current)
    order = [s["id"] for s in segs]
    has_audio = audio_path(current) is not None

    gt = truth[current]
    pr = pred_all.get(current, {}).get("stories", [])
    gt_cov = bars_for(gt, order, "start", "end")
    pr_cov = bars_for(pr, order, "start_id", "end_id")

    pills = ""
    for sid, name in SESSIONS.items():
        ng, npd = len(truth[sid]), len(pred_all.get(sid, {}).get("stories", []))
        ok = ng == npd
        cls = "pill" + (" pill--on" if sid == current else "") + (" pill--ok" if ok else " pill--bad")
        pills += (f'<a class="{cls}" href="/?session={sid}"><span class="pn">{html.escape(name)}</span>'
                  f'<span class="pm">truth {ng} · pred {npd}{" ✓" if ok else " ✗"}</span></a>')

    # legend: truth stories vs predicted stories
    def card_set(stories, colors, label, sk, ek, wk, tk):
        cards = ""
        for k, st in enumerate(stories):
            c = colors[k % len(colors)]
            world = html.escape(st.get(wk, "") or "—")
            title = html.escape(st.get(tk, "") or "")
            cards += (f'<div class="card" style="--c:{c}"><div class="ct"><span class="cdot"></span>'
                      f'{label} {k+1}<span class="cspan">ids {html.escape(str(st[sk]))}–{html.escape(str(st[ek]))}</span></div>'
                      f'<div class="cw">{world}</div>'
                      + (f'<div class="cti">{title}</div>' if title else "") + "</div>")
        return cards or '<div class="card card--empty">none</div>'

    gt_cards = card_set(gt, GT_COLORS, "Truth", "start", "end", "world", "title")
    pr_cards = card_set(pr, PR_COLORS, "Pred", "start_id", "end_id", "world", "title")

    rows = ""
    for i, s in enumerate(segs):
        play = (f'<button class="play" data-start="{s["start"]}" data-end="{s["end"]}">▶</button>'
                if has_audio and s["start"] is not None else '<span class="play play--off">·</span>')
        g, p = gt_cov.get(i), pr_cov.get(i)
        gstyle = f'background:{GT_COLORS[g[0] % len(GT_COLORS)]}' if g else ""
        pstyle = f'background:{PR_COLORS[p[0] % len(PR_COLORS)]}' if p else ""
        lab = ""
        if g and g[1]:
            lab += f'<span class="lab lab--gt" style="--c:{GT_COLORS[g[0]%len(GT_COLORS)]}">◆ truth {g[0]+1} start</span>'
        if g and g[2]:
            lab += f'<span class="lab lab--gt" style="--c:{GT_COLORS[g[0]%len(GT_COLORS)]}">◆ truth {g[0]+1} end</span>'
        if p and p[1]:
            ev = html.escape((pr[p[0]].get("evidence", {}) or {}).get("start_quote", "") or "")
            lab += f'<span class="lab lab--pr" style="--c:{PR_COLORS[p[0]%len(PR_COLORS)]}">◇ pred {p[0]+1} start · {html.escape(pr[p[0]].get("world","") or "")}</span>'
        if p and p[2]:
            lab += f'<span class="lab lab--pr" style="--c:{PR_COLORS[p[0]%len(PR_COLORS)]}">◇ pred {p[0]+1} end</span>'
        dim = "" if (g or p) else " row--non"
        txt = html.escape(s["text"]) if s["text"] else '<span class="muted">[no text]</span>'
        rows += (f'<div class="row{dim}">{play}'
                 f'<span class="gt-rib" style="{gstyle}"></span><span class="pr-rib" style="{pstyle}"></span>'
                 f'<span class="sid">{html.escape(str(s["id"]))}</span><span class="st">{fmt(s["start"])}</span>'
                 f'<span class="tx">{txt}{lab}</span></div>')

    audio_el = f'<audio id="player" src="/audio?session={current}" preload="metadata"></audio>' if has_audio else ""
    return PAGE.format(name=html.escape(SESSIONS[current]), current=current, pills=pills,
                       gt_cards=gt_cards, pr_cards=pr_cards, rows=rows, audio_el=audio_el,
                       n_gt=len(gt), n_pred=len(pr), script=SCRIPT)


SCRIPT = r"""
(function(){
  var player=document.getElementById('player'), stopAt=null, cur=null;
  if(player){
    player.addEventListener('timeupdate', function(){ if(stopAt!=null && player.currentTime>=stopAt) player.pause(); });
    function reset(){ stopAt=null; if(cur){ cur.textContent='▶'; cur=null; } }
    player.addEventListener('pause', reset); player.addEventListener('ended', reset);
    document.querySelectorAll('.play[data-start]').forEach(function(b){
      b.addEventListener('click', function(){
        var s=parseFloat(b.getAttribute('data-start')), en=parseFloat(b.getAttribute('data-end'));
        if(cur===b){ player.pause(); return; }
        if(cur) cur.textContent='▶';
        stopAt=((en && en-s<14)?en:s+5)+0.4; player.currentTime=Math.max(0,s-0.4);
        player.play().then(function(){ cur=b; b.textContent='❚❚'; }).catch(reset);
      });
    });
  }
})();
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Segmenter vs truth · {name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Newsreader:wght@400..600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
 :root{{--night:#13111f;--panel:#1d1934;--panel2:#221d3d;--ink:#f1e9da;--soft:#bcb0c4;--faint:#7a7090;--line:#322b4e;--gold:#e9b873;--rule:#3a3358}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--night);color:var(--ink);font-family:"Newsreader",Georgia,serif;font-size:16px;line-height:1.5;
   background-image:radial-gradient(120% 70% at 50% -10%,rgba(233,184,115,.10),transparent 60%);background-attachment:fixed}}
 .wrap{{max-width:64rem;margin:0 auto;padding:2rem clamp(1rem,4vw,3rem) 5rem}}
 h1{{font-family:"Fraunces",serif;font-weight:500;font-size:2rem;margin:0}} h1 .m{{color:var(--gold)}}
 .lede{{color:var(--soft);font-style:italic;margin:.4rem 0 1.3rem;max-width:46rem}}
 .lede b{{color:var(--gold);font-style:normal}}
 .pills{{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.2rem}}
 .pill{{text-decoration:none;color:var(--soft);background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:.45rem .75rem;display:flex;flex-direction:column;min-width:7rem}}
 .pill--on{{background:var(--panel2);border-color:var(--gold);color:var(--ink)}}
 .pn{{font-family:"Fraunces",serif;font-size:1rem}} .pm{{font-family:"JetBrains Mono",monospace;font-size:.62rem;color:var(--faint)}}
 .pill--ok .pm{{color:#8fc0a8}} .pill--bad .pm{{color:#e89}}
 .legend{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.4rem}}
 .legcol h3{{font-family:"JetBrains Mono",monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin:.2rem 0 .5rem}}
 .card{{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-left:4px solid var(--c);border-radius:10px;padding:.55rem .7rem;margin-bottom:.5rem}}
 .card--empty{{border-left-color:var(--rule);color:var(--faint);font-style:italic}}
 .ct{{display:flex;align-items:center;gap:.4rem;font-family:"Fraunces",serif;font-weight:600;font-size:.85rem}}
 .cdot{{width:.55rem;height:.55rem;border-radius:50%;background:var(--c)}} .cspan{{margin-left:auto;font-family:"JetBrains Mono",monospace;font-size:.6rem;color:var(--faint)}}
 .cw{{font-style:italic;color:var(--soft);font-size:.82rem}} .cti{{font-size:.8rem;color:var(--ink)}}
 .transcript{{border-top:1px solid var(--line)}}
 .row{{display:grid;grid-template-columns:1.3rem .5rem .5rem 2.6rem 2.2rem 1fr;align-items:baseline;gap:.5rem;padding:.16rem .3rem;}}
 .row:hover{{background:rgba(255,255,255,.025)}} .row--non{{opacity:.36}} .row--non:hover{{opacity:.7}}
 .gt-rib,.pr-rib{{align-self:stretch;width:.42rem;border-radius:2px;min-height:1.1rem}}
 .sid{{font-family:"JetBrains Mono",monospace;font-size:.6rem;color:var(--faint);text-align:right}}
 .st{{font-family:"JetBrains Mono",monospace;font-size:.6rem;color:var(--faint)}} .tx{{font-size:.96rem}}
 .muted{{color:var(--faint)}}
 .play{{background:transparent;border:1px solid var(--rule);color:var(--soft);cursor:pointer;border-radius:50%;width:1.25rem;height:1.25rem;font-size:.5rem;padding:0}}
 .play:hover{{border-color:var(--gold);color:var(--gold)}} .play--off{{border:none;color:var(--line)}}
 .lab{{display:inline-block;font-family:"JetBrains Mono",monospace;font-size:.6rem;color:var(--c);border:1px solid var(--c);border-radius:5px;padding:0 .35rem;margin-left:.5rem;white-space:nowrap}}
 .legend-foot{{font-family:"JetBrains Mono",monospace;font-size:.66rem;color:var(--faint);margin:.2rem 0 1rem}}
 .legend-foot b{{color:#7fb0a3}} .legend-foot i{{color:#e0876f;font-style:normal}}
</style></head><body>
 {audio_el}
 <div class="wrap">
  <h1>Segmenter <span class="m">vs</span> Truth — {name}</h1>
  <p class="lede">Left ribbon is the <b>hand-marked truth</b>, right ribbon is the <b>segmenter's prediction</b>. A start placed a few lines off, an over-split, or a missed story shows as the two ribbons disagreeing. ▶ to hear any line.</p>
  <nav class="pills">{pills}</nav>
  <p class="legend-foot"><b>◆ teal = truth</b> &nbsp; <i>◇ coral = predicted</i> &nbsp;·&nbsp; this session: truth {n_gt} / predicted {n_pred} stories</p>
  <div class="legend"><div class="legcol"><h3>Truth stories</h3>{gt_cards}</div><div class="legcol"><h3>Predicted stories</h3>{pr_cards}</div></div>
  <div class="transcript">{rows}</div>
 </div>
 <script>{script}</script>
</body></html>"""


def serve(port):
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(data)

        def _audio(self, sid):
            ap = audio_path(sid)
            if not ap:
                self._send(404, "no audio", "text/plain"); return
            size = ap.stat().st_size; rng = self.headers.get("Range", "")
            ct = "audio/mp4" if ap.suffix == ".m4a" else "audio/mpeg"
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                a = int(a) if a else 0; b = int(b) if b else size - 1; b = min(b, size - 1)
                with open(ap, "rb") as f:
                    f.seek(a); chunk = f.read(b - a + 1)
                self.send_response(206); self.send_header("Content-Type", ct); self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}"); self.send_header("Content-Length", str(len(chunk)))
                self.end_headers(); self.wfile.write(chunk)
            else:
                self._send(200, ap.read_bytes(), ct, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(200, render(q.get("session", [None])[0]))
            elif u.path == "/audio":
                self._audio(q.get("session", [None])[0])
            else:
                self._send(404, "not found", "text/plain")

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), H)
    print(f"segmenter view -> http://127.0.0.1:{port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=5058)
    args = ap.parse_args()
    if args.serve:
        serve(args.port)
    else:
        HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
        HTML_OUT.write_text(render())
        print(f"wrote {HTML_OUT}  (open with --serve for audio)")


if __name__ == "__main__":
    main()
