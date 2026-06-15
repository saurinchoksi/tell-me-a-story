#!/usr/bin/env python3
"""Story-segmentation ground truth — draggable story bars over a nocturnal transcript.

Locks the human truth for "where does each story start and stop" across the EMP
sessions, so a segmenter can be validated against it. A story is a BAR with a
start and an end you drag to the right lines; everything outside every bar is
non-story (preamble, negotiation, milk runs, wind-down) and is excluded for free.

Direct manipulation: grab a bar's top handle to move its start, its bottom handle
to move its end. Drag to fine-tune; ＋ new story drops a bar to size. No cut-points,
no "mark non-story" — uncovered = non-story.

Same proven serve shape as before: a generator that bakes the current truth into
the HTML (reads on file://) + a --serve bridge that writes edits to a JSON sidecar
one op at a time, with a per-session undo stack, and streams session audio
(range-aware) so a boundary can be checked by ear.

    python emp/src/segment_truth.py            # write the static HTML
    python emp/src/segment_truth.py --serve     # serve with audio + live saving

PRIVATE (transcript text, family content) → gitignored emp/results/visuals/.
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SESSIONS = {
    "20251207-195607": "Moon",
    "20251207-202105": "Cruel Baby",
    "20251210-203654": "Rubber Ducky",
    "20260117-202237": "Pandavas",
    "20260129-204404": "Portal",
}
RIBBONS = ["#e0876f", "#7fb0a3", "#e6b566", "#b196c9", "#d98a9e", "#8fb27a"]

# Proposed stories to verify, as (start_id, end_id, title, world). end "LAST" = the
# session's final segment. Only used to seed a session never edited before; an
# existing sidecar (incl. the older boundary format) is migrated, not overwritten.
DEFAULT_SEED = {
    "20251207-195607": [(0, "LAST", "Moon", "original")],
    "20251207-202105": [(0, 277, "Cruel Baby", "original")],
    "20251210-203654": [(0, "LAST", "Rubber Ducky", "original")],
    "20260117-202237": [(0, "LAST", "Pandavas", "Mahabharata")],
    "20260129-204404": [
        (6, 124, "the scooter & park story", "original"),
        (129, 279, "the portal-jumping story", "original"),
        (339, "LAST", "the engine story", "Thomas & Friends + invented engines"),
    ],
}

SIDECAR = ROOT / "emp" / "results" / "visuals" / "segmentation-truth.json"
HTML_OUT = ROOT / "emp" / "results" / "visuals" / "segmentation-truth.html"
ABOUT = ("Human ground truth for story segmentation. Per session: stories = list of "
         "{start, end (segment ids, inclusive), title, world}; uncovered segments are "
         "non-story and excluded from the story-name passes. Edited live in "
         "segment_truth.py --serve (drag the bar handles).")


def session_segments(sid):
    segs = json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())["segments"]
    return [{"id": s["id"], "start": s.get("start"), "end": s.get("end"),
             "text": (s.get("text") or "").strip(), "gap": not isinstance(s["id"], int)}
            for s in segs]


def audio_path(sid):
    return next((ROOT / "sessions" / sid).glob("audio.*"), None)


def _ordered_ids(segs):
    return [s["id"] for s in segs]


def _snap(int_ids, desired):
    if desired == "LAST":
        return int_ids[-1]
    ge = [i for i in int_ids if i >= desired]
    return ge[0] if ge else (int_ids[-1] if int_ids else desired)


def _seed_stories(sid, segs):
    int_ids = sorted(i for i in _ordered_ids(segs) if isinstance(i, int))
    out = []
    for s, e, title, world in DEFAULT_SEED.get(sid, [(0, "LAST", SESSIONS[sid], "original")]):
        out.append({"start": _snap(int_ids, s), "end": _snap(int_ids, e), "title": title, "world": world})
    return out


def _migrate_boundaries(segs, boundaries):
    """Old format → regions: each story boundary becomes a region ending the segment
    before the next boundary (any kind), or the last segment."""
    order = _ordered_ids(segs)
    pos = {sid: i for i, sid in enumerate(order)}
    bs = sorted((b for b in boundaries if b.get("segment_id") in pos), key=lambda b: pos[b["segment_id"]])
    stories = []
    for b in bs:
        if b.get("kind") != "story":
            continue
        sp = pos[b["segment_id"]]
        laters = [pos[x["segment_id"]] for x in bs if pos[x["segment_id"]] > sp]
        ep = (min(laters) - 1) if laters else len(order) - 1
        stories.append({"start": order[sp], "end": order[max(ep, sp)],
                        "title": b.get("title", ""), "world": b.get("world", "")})
    return stories


def _coerce_stories(segs, stories):
    order = _ordered_ids(segs)
    pos = {sid: i for i, sid in enumerate(order)}
    out = []
    for st in stories:
        if st.get("start") in pos and st.get("end") in pos:
            s, e = pos[st["start"]], pos[st["end"]]
            out.append({"start": order[min(s, e)], "end": order[max(s, e)],
                        "title": st.get("title", ""), "world": st.get("world", "")})
    out.sort(key=lambda st: pos[st["start"]])
    return out


def load_truth():
    raw = json.loads(SIDECAR.read_text()) if SIDECAR.exists() else {}
    data = {k: v for k, v in raw.items() if k.startswith("_")}
    data.setdefault("_about", ABOUT)
    data.setdefault("_undo", {})
    sess = raw.get("sessions", {}) if isinstance(raw.get("sessions"), dict) else {}
    data["sessions"] = {}
    for sid in SESSIONS:
        segs = session_segments(sid)
        cur = sess.get(sid, {})
        if isinstance(cur.get("stories"), list):
            stories = _coerce_stories(segs, cur["stories"])
        elif isinstance(cur.get("boundaries"), list) and cur["boundaries"]:
            stories = _migrate_boundaries(segs, cur["boundaries"])  # old format → regions
        else:
            stories = _seed_stories(sid, segs)
        data["sessions"][sid] = {"verified": bool(cur.get("verified")), "stories": stories}
    return data


def save_truth(data):
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _push_undo(data, sid):
    h = data.setdefault("_undo", {}).setdefault(sid, [])
    h.append([dict(st) for st in data["sessions"][sid]["stories"]])
    del h[:-25]


def fmt(t):
    return "—" if t is None else f"{int(t // 60)}:{int(t % 60):02d}"


def render_html(current=None):
    truth = load_truth()
    current = current if current in SESSIONS else next(iter(SESSIONS))
    has_audio = audio_path(current) is not None
    segs = session_segments(current)
    order = _ordered_ids(segs)
    pos = {sid: i for i, sid in enumerate(order)}
    stories = truth["sessions"][current]["stories"]
    verified = truth["sessions"][current]["verified"]

    # session pills
    pills = ""
    for sid, name in SESSIONS.items():
        n = len(truth["sessions"][sid]["stories"])
        ver = truth["sessions"][sid]["verified"]
        cls = "pill" + (" pill--on" if sid == current else "") + (" pill--ver" if ver else "")
        pills += (f'<a class="{cls}" href="/?session={sid}"><span class="pill-name">{html.escape(name)}</span>'
                  f'<span class="pill-meta">{n} stor{"y" if n==1 else "ies"}{" · ✓" if ver else ""}</span></a>')

    # legend cards (title/world editing; story identified by index)
    cards = ""
    for i, st in enumerate(stories):
        color = RIBBONS[i % len(RIBBONS)]
        sp, ep = pos.get(st["start"]), pos.get(st["end"])
        a, b = (segs[sp]["start"] if sp is not None else None), (segs[ep]["end"] if ep is not None else None)
        nseg = (ep - sp + 1) if (sp is not None and ep is not None) else 0
        cards += f"""
        <div class="story-card" style="--rib:{color}">
          <div class="sc-top"><span class="sc-dot"></span><span class="sc-n">Story {i+1}</span>
            <span class="sc-span">segs {html.escape(str(st['start']))}–{html.escape(str(st['end']))} · {nseg} · {fmt(a)}–{fmt(b)}</span>
            <button class="sc-del" data-story="{i}" title="delete this story">✕</button></div>
          <input class="sc-title" data-story="{i}" data-field="title" value="{html.escape(st.get('title',''))}"
                 placeholder="untitled story" spellcheck="false">
          <input class="sc-world" data-story="{i}" data-field="world" value="{html.escape(st.get('world',''))}"
                 placeholder="world / canon (free text — inferred from the story)" spellcheck="false">
        </div>"""
    if not cards:
        cards = '<p class="empty">No stories yet — hit <b>＋ new story</b>, then drag its handles.</p>'

    # transcript rows (plain; the client paints membership + handles from STORIES)
    rows = ""
    for i, s in enumerate(segs):
        play = (f'<button class="play" data-start="{s["start"]}" data-end="{s["end"]}" title="hear it">▶</button>'
                if has_audio and s["start"] is not None else '<span class="play play--off">·</span>')
        txt = html.escape(s["text"]) if s["text"] else '<span class="muted">[no text]</span>'
        rows += (f'<div class="row" data-pos="{i}" data-seg="{html.escape(str(s["id"]))}">'
                 f'<span class="rib"></span>{play}'
                 f'<span class="seg-id">{html.escape(str(s["id"]))}</span><span class="seg-t">{fmt(s["start"])}</span>'
                 f'<span class="seg-tx">{txt}</span></div>')

    # client state for drag
    cstories = [{"s": pos[st["start"]], "e": pos[st["end"]], "title": st.get("title", "")}
                for st in stories if st["start"] in pos and st["end"] in pos]
    state = json.dumps({"stories": cstories, "order": order, "colors": RIBBONS, "session": current})

    audio_el = f'<audio id="player" src="/audio?session={current}" preload="metadata"></audio>' if has_audio else ""
    return PAGE.format(current=current, name=html.escape(SESSIONS[current]), pills=pills, cards=cards,
                       rows=rows, audio_el=audio_el, n_seg=len(segs), n_story=len(stories),
                       ver_cls="lock lock--on" if verified else "lock",
                       ver_txt="✓ locked" if verified else "lock this session",
                       state=state, script=SCRIPT)


SCRIPT = r"""
(function(){
  var served = location.protocol.indexOf('http')===0;
  var stat = document.getElementById('stat');
  var ST = window.__STATE__, SID = ST.session;
  var ROWS = Array.prototype.slice.call(document.querySelectorAll('.row'));
  function flash(t,ok){ if(stat){ stat.textContent=t; stat.className='stat'+(ok===false?' bad':ok?' ok':''); } }
  function post(body, reload){
    if(!served){ flash('run with --serve to save', false); return Promise.resolve(); }
    flash('saving…');
    return fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(function(r){ if(!r.ok){ flash('save failed', false); return; }
        if(reload){ sessionStorage.setItem('sy', String(window.scrollY)); location.reload(); } else flash('saved ✓', true); })
      .catch(function(){ flash('save failed', false); });
  }
  var sy = sessionStorage.getItem('sy'); if(sy){ window.scrollTo(0, parseInt(sy)); sessionStorage.removeItem('sy'); }

  // ---- paint membership + handles from ST.stories (positions) ----
  function clampStory(i){
    var st=ST.stories[i];
    if(st.e < st.s) st.e = st.s;
    var prev = i>0 ? ST.stories[i-1] : null, next = i<ST.stories.length-1 ? ST.stories[i+1] : null;
    if(prev && st.s <= prev.e) st.s = prev.e + 1;
    if(next && st.e >= next.s) st.e = next.s - 1;
    if(st.e < st.s) st.e = st.s;
  }
  function paint(){
    document.querySelectorAll('.handle, .story-label').forEach(function(x){ x.remove(); });
    ROWS.forEach(function(r){ r.classList.add('row--non'); r.style.removeProperty('--rib'); });
    ST.stories.forEach(function(st,i){
      var col = ST.colors[i % ST.colors.length];
      for(var p=st.s; p<=st.e; p++){ var r=ROWS[p]; if(!r) continue; r.classList.remove('row--non'); r.style.setProperty('--rib', col); }
      addHandle(st.s, 'start', i, col); addHandle(st.e, 'end', i, col);
    });
  }
  function addHandle(p, edge, i, col){
    var r = ROWS[p]; if(!r) return;
    var h = document.createElement('div'); h.className='handle handle--'+edge;
    h.setAttribute('data-story', i); h.setAttribute('data-edge', edge); h.style.setProperty('--rib', col);
    h.title = (edge==='start'?'drag to move where this story starts':'drag to move where this story ends');
    h.innerHTML = '<span class="grip">⠿</span>';
    if(edge==='start'){
      var lab=document.createElement('span'); lab.className='story-label'; lab.style.setProperty('--rib',col);
      lab.textContent='✦ Story '+(i+1)+' · '+(ST.stories[i].title||'untitled');
      r.appendChild(lab);
    }
    r.appendChild(h);
  }

  // ---- drag ----
  var drag = null;
  function posUnder(y){
    // find the row whose vertical band contains the cursor — x-independent, so it
    // works regardless of the centered/padded column width
    for(var i=0;i<ROWS.length;i++){
      var rc = ROWS[i].getBoundingClientRect();
      if(y >= rc.top && y < rc.bottom) return i;
    }
    if(ROWS.length && y < ROWS[0].getBoundingClientRect().top) return 0;
    return ROWS.length ? ROWS.length-1 : null;
  }
  document.addEventListener('mousedown', function(e){
    var h = e.target.closest && e.target.closest('.handle'); if(!h) return;
    e.preventDefault();
    drag = { i: +h.getAttribute('data-story'), edge: h.getAttribute('data-edge') };
    document.body.classList.add('dragging');
  });
  document.addEventListener('mousemove', function(e){
    if(!drag) return;
    var p = posUnder(e.clientY); if(p==null) { autoscroll(e.clientY); return; }
    var st = ST.stories[drag.i];
    if(drag.edge==='start') st.s = p; else st.e = p;
    clampStory(drag.i); paint(); autoscroll(e.clientY);
  });
  document.addEventListener('mouseup', function(){
    if(!drag) return;
    var st = ST.stories[drag.i]; var d = drag; drag = null; document.body.classList.remove('dragging');
    post({session:SID, op:'resize', story:d.i, edge:d.edge,
          segment_id: ST.order[ d.edge==='start' ? st.s : st.e ]}, false);
  });
  function autoscroll(y){
    var m=90; if(y<m) window.scrollBy(0,-14); else if(y>window.innerHeight-m) window.scrollBy(0,14);
  }

  // ---- buttons ----
  var nb=document.getElementById('newstory');
  if(nb) nb.addEventListener('click', function(){ post({session:SID, op:'new_story'}, true); });
  var ub=document.getElementById('undo');
  if(ub) ub.addEventListener('click', function(){ post({session:SID, op:'undo'}, true); });
  var lock=document.getElementById('lock');
  if(lock) lock.addEventListener('click', function(){ post({session:SID, op:'verify', value: lock.className.indexOf('lock--on')<0}, true); });
  document.querySelectorAll('.sc-del').forEach(function(b){
    b.addEventListener('click', function(){
      if(!confirm('Delete this story bar? (Undo can bring it back.)')) return;
      post({session:SID, op:'delete', story:+b.getAttribute('data-story')}, true);
    });
  });
  document.querySelectorAll('.sc-title, .sc-world').forEach(function(inp){
    var t=null;
    inp.addEventListener('input', function(){
      var i=+inp.getAttribute('data-story'), field=inp.getAttribute('data-field');
      if(field==='title' && ST.stories[i]){ ST.stories[i].title=inp.value; paint(); }
      clearTimeout(t); t=setTimeout(function(){ post({session:SID, op:'edit', story:i, field:field, value:inp.value}, false); }, 450);
    });
  });

  // ---- audio ----
  var player=document.getElementById('player'), stopAt=null, cur=null;
  if(player){
    player.addEventListener('timeupdate', function(){ if(stopAt!=null && player.currentTime>=stopAt) player.pause(); });
    function reset(){ stopAt=null; if(cur){ cur.textContent='▶'; cur=null; } }
    player.addEventListener('pause', reset); player.addEventListener('ended', reset);
    document.querySelectorAll('.play[data-start]').forEach(function(btn){
      btn.addEventListener('click', function(e){ e.stopPropagation();
        var s=parseFloat(btn.getAttribute('data-start')), en=parseFloat(btn.getAttribute('data-end'));
        if(cur===btn){ player.pause(); return; }
        if(cur) cur.textContent='▶';
        stopAt = ((en && en-s<14)?en:s+5) + 0.4; player.currentTime=Math.max(0,s-0.4);
        player.play().then(function(){ cur=btn; btn.textContent='❚❚'; }).catch(reset);
      });
    });
  }
  paint();
})();
"""


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Story ground truth · {name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..600&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{ --night:#13111f; --night2:#181530; --panel:#1d1934; --panel2:#221d3d; --ink:#f1e9da; --ink-soft:#bcb0c4;
    --ink-faint:#7a7090; --line:#322b4e; --gold:#e9b873; --gold-soft:#caa15f; --rule:#3a3358; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--night); color:var(--ink); font-family:"Newsreader",Georgia,serif; font-size:16px; line-height:1.5;
    background-image:radial-gradient(120% 70% at 50% -10%, rgba(233,184,115,.10), transparent 60%), radial-gradient(80% 50% at 90% 0%, rgba(127,176,163,.06), transparent 55%);
    background-attachment:fixed; }}
  body.dragging {{ cursor:row-resize; user-select:none; }}
  body::before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:999; opacity:.05;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)'/%3E%3C/svg%3E"); }}
  .wrap {{ max-width:60rem; margin:0 auto; padding:2.2rem clamp(1rem,4vw,3rem) 6rem; }}
  header {{ display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap; margin-bottom:.3rem; }}
  h1 {{ font-family:"Fraunces",serif; font-weight:500; font-size:2.1rem; letter-spacing:-.01em; margin:0; }}
  h1 .moon {{ color:var(--gold); }}
  .tag {{ font-family:"JetBrains Mono",monospace; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-faint); }}
  .lede {{ color:var(--ink-soft); font-size:1rem; font-style:italic; margin:.4rem 0 1.4rem; max-width:47rem; }}
  .lede b {{ color:var(--gold-soft); font-style:normal; font-weight:600; }}
  .pills {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:1.3rem; }}
  .pill {{ text-decoration:none; color:var(--ink-soft); background:var(--panel); border:1px solid var(--line); border-radius:11px;
    padding:.5rem .8rem; display:flex; flex-direction:column; gap:.05rem; min-width:7rem; transition:transform .15s, border-color .15s; }}
  .pill:hover {{ transform:translateY(-2px); border-color:var(--rule); }}
  .pill--on {{ background:var(--panel2); border-color:var(--gold-soft); color:var(--ink); box-shadow:0 0 0 1px rgba(233,184,115,.25), 0 8px 30px -12px rgba(233,184,115,.4); }}
  .pill-name {{ font-family:"Fraunces",serif; font-weight:500; font-size:1.02rem; }}
  .pill-meta {{ font-family:"JetBrains Mono",monospace; font-size:.64rem; color:var(--ink-faint); }}
  .pill--ver .pill-meta {{ color:var(--gold-soft); }}
  .session-head {{ display:flex; align-items:center; gap:1rem; margin:.2rem 0 1rem; flex-wrap:wrap; }}
  .session-head h2 {{ font-family:"Fraunces",serif; font-weight:500; font-size:1.5rem; margin:0; }}
  .sh-meta {{ font-family:"JetBrains Mono",monospace; font-size:.7rem; color:var(--ink-faint); }}
  .sh-actions {{ margin-left:auto; display:flex; gap:.5rem; align-items:center; }}
  .lock, .btn {{ cursor:pointer; font-family:"JetBrains Mono",monospace; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase;
    padding:.45rem .8rem; border-radius:8px; border:1px solid var(--rule); background:transparent; color:var(--ink-soft); transition:.15s; }}
  .lock:hover, .btn:hover {{ border-color:var(--gold-soft); color:var(--ink); }}
  .btn--gold {{ border-color:var(--gold-soft); color:var(--gold); }}
  .lock--on {{ background:linear-gradient(var(--gold),var(--gold-soft)); color:#2a1e0a; border-color:var(--gold); font-weight:600; }}
  .legend {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(15rem,1fr)); gap:.7rem; margin-bottom:1.6rem; }}
  .story-card {{ background:linear-gradient(180deg, var(--panel2), var(--panel)); border:1px solid var(--line); border-left:4px solid var(--rib);
    border-radius:12px; padding:.8rem .9rem; }}
  .sc-top {{ display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem; }}
  .sc-dot {{ width:.6rem; height:.6rem; border-radius:50%; background:var(--rib); box-shadow:0 0 10px -1px var(--rib); }}
  .sc-n {{ font-family:"Fraunces",serif; font-weight:600; font-size:.9rem; }}
  .sc-span {{ margin-left:auto; font-family:"JetBrains Mono",monospace; font-size:.6rem; color:var(--ink-faint); }}
  .sc-del {{ background:transparent; border:none; color:var(--ink-faint); cursor:pointer; font-size:.8rem; padding:0 .2rem; }}
  .sc-del:hover {{ color:#e89; }}
  .sc-title {{ width:100%; background:transparent; border:none; border-bottom:1px dashed var(--rule); color:var(--ink);
    font-family:"Fraunces",serif; font-size:1.05rem; padding:.15rem 0; margin-bottom:.3rem; outline:none; }}
  .sc-title:focus {{ border-bottom-color:var(--gold); }}
  .sc-world {{ width:100%; background:transparent; border:none; color:var(--ink-soft); font-family:"Newsreader",serif; font-style:italic;
    font-size:.82rem; padding:.1rem 0; outline:none; }}
  .sc-world::placeholder, .sc-title::placeholder {{ color:var(--ink-faint); }}
  .empty {{ color:var(--ink-faint); font-style:italic; }}
  .transcript {{ border-top:1px solid var(--line); }}
  .row {{ display:grid; grid-template-columns:1.4rem 2.6rem 2.4rem 1fr; align-items:baseline; gap:.55rem; padding:.2rem .4rem .2rem .9rem;
    position:relative; border-left:3px solid var(--rib, transparent); }}
  .row .rib {{ display:none; }}
  .row:hover {{ background:rgba(255,255,255,.025); }}
  .row--non {{ opacity:.4; border-left-color:transparent; }}
  .row--non:hover {{ opacity:.72; }}
  .seg-id {{ font-family:"JetBrains Mono",monospace; font-size:.62rem; color:var(--ink-faint); text-align:right; }}
  .seg-t {{ font-family:"JetBrains Mono",monospace; font-size:.62rem; color:var(--ink-faint); }}
  .seg-tx {{ font-size:.98rem; }}
  .muted {{ color:var(--ink-faint); }}
  .play {{ background:transparent; border:1px solid var(--rule); color:var(--ink-soft); cursor:pointer; border-radius:50%;
    width:1.3rem; height:1.3rem; font-size:.55rem; line-height:1; padding:0; }}
  .play:hover {{ border-color:var(--gold-soft); color:var(--gold); }}
  .play--off {{ border:none; color:var(--line); text-align:center; cursor:default; }}
  /* drag handles on a story's first/last row */
  .handle {{ position:absolute; left:-2px; width:1.5rem; height:1.05rem; display:flex; align-items:center; justify-content:center;
    cursor:row-resize; color:#1a1626; background:var(--rib); border-radius:4px; z-index:6; box-shadow:0 1px 6px -1px rgba(0,0,0,.5); }}
  .handle--start {{ top:-0.55rem; }}
  .handle--end {{ bottom:-0.55rem; }}
  .handle .grip {{ font-size:.7rem; line-height:1; transform:scaleY(.7); }}
  .handle:hover {{ filter:brightness(1.15); transform:scale(1.12); }}
  .story-label {{ position:absolute; left:1.7rem; top:-0.7rem; font-family:"Fraunces",serif; font-size:.74rem; color:var(--rib);
    background:var(--night); padding:0 .4rem; border-radius:4px; white-space:nowrap; z-index:5; }}
  .statusbar {{ position:fixed; bottom:0; left:0; right:0; background:rgba(19,17,31,.88); backdrop-filter:blur(8px); border-top:1px solid var(--line);
    padding:.5rem clamp(1rem,4vw,3rem); display:flex; align-items:center; gap:1rem; z-index:40; }}
  .sb-hint {{ color:var(--ink-faint); font-size:.78rem; font-style:italic; }}
  .sb-hint b {{ color:var(--gold-soft); font-style:normal; }}
  .stat {{ margin-left:auto; font-family:"JetBrains Mono",monospace; font-size:.68rem; color:var(--ink-faint); }}
  .stat.ok {{ color:#8fc0a8; }} .stat.bad {{ color:#e89; }}
</style></head>
<body data-session="{current}">
  {audio_el}
  <div class="wrap">
    <header><h1>Story Ground <span class="moon">Truth</span></h1>
      <span class="tag">bedtime segmentation · {n_story} stories / {n_seg} segments</span></header>
    <p class="lede">Each story is a <b>bar</b> — drag its <b>top handle</b> to where the story starts, its <b>bottom handle</b>
      to where it ends. Drag to fine-tune (the page scrolls if you reach the edge). Everything outside a bar is non-story and dimmed.
      ▶ to hear a line.</p>
    <nav class="pills">{pills}</nav>
    <div class="session-head"><h2>{name}</h2><span class="sh-meta">{n_seg} segments · {n_story} stories</span>
      <span class="sh-actions"><button id="newstory" class="btn btn--gold">＋ new story</button>
        <button id="undo" class="btn" title="undo the last change">↶ undo</button>
        <button id="lock" class="{ver_cls}">{ver_txt}</button></span></div>
    <div class="legend">{cards}</div>
    <div class="transcript">{rows}</div>
  </div>
  <div class="statusbar"><span class="sb-hint">Edits save to <b>segmentation-truth.json</b>. Drag a handle to the first/last real story line; lock when a session looks right.</span>
    <span id="stat" class="stat"></span></div>
  <script>window.__STATE__ = {state};</script>
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
                self._send(200, render_html(q.get("session", [None])[0]))
            elif u.path == "/audio":
                self._audio(q.get("session", [None])[0])
            elif u.path == "/segmentation-truth.json":
                self._send(200, json.dumps(load_truth(), indent=2, ensure_ascii=False), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}', "application/json"); return
            sid, op = body.get("session"), body.get("op")
            if sid not in SESSIONS:
                self._send(400, '{"error":"bad session"}', "application/json"); return
            data = load_truth(); S = data["sessions"][sid]; stories = S["stories"]
            segs = session_segments(sid); order = _ordered_ids(segs); posmap = {x: i for i, x in enumerate(order)}
            if op in ("resize", "new_story", "delete", "edit"):
                _push_undo(data, sid)
            if op == "resize":
                i = body.get("story"); seg = body.get("segment_id")
                if isinstance(i, int) and 0 <= i < len(stories) and seg in posmap:
                    st = stories[i]
                    if body.get("edge") == "start":
                        st["start"] = seg
                        if posmap[st["start"]] > posmap[st["end"]]:  # collapse if dragged past the end
                            st["end"] = st["start"]
                    else:
                        st["end"] = seg
                        if posmap[st["end"]] < posmap[st["start"]]:
                            st["start"] = st["end"]
                    S["stories"] = _coerce_stories(segs, stories)
            elif op == "new_story":
                covered = set()
                for st in stories:
                    if st["start"] in posmap and st["end"] in posmap:
                        covered.update(range(posmap[st["start"]], posmap[st["end"]] + 1))
                gap = next((p for p in range(len(order)) if p not in covered), len(order) - 1)
                stories.append({"start": order[gap], "end": order[gap], "title": "New story", "world": ""})
                S["stories"] = _coerce_stories(segs, stories)
            elif op == "delete":
                i = body.get("story")
                if isinstance(i, int) and 0 <= i < len(stories):
                    del stories[i]
            elif op == "edit":
                i = body.get("story")
                if isinstance(i, int) and 0 <= i < len(stories) and body.get("field") in ("title", "world"):
                    stories[i][body["field"]] = body.get("value", "")
            elif op == "verify":
                S["verified"] = bool(body.get("value"))
            elif op == "undo":
                h = data.get("_undo", {}).get(sid, [])
                if h:
                    S["stories"] = _coerce_stories(segs, h.pop())
            else:
                self._send(400, '{"error":"bad op"}', "application/json"); return
            save_truth(data)
            self._send(200, '{"ok":true}', "application/json")

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), H)
    print(f"story ground truth ->  http://127.0.0.1:{port}/")
    print(f"sidecar            ->  {SIDECAR.relative_to(ROOT)}   (edits auto-save; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    ap = argparse.ArgumentParser(description="Story-segmentation ground-truth ledger")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=5057)
    args = ap.parse_args()
    if args.serve:
        serve(args.port)
    else:
        HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
        HTML_OUT.write_text(render_html())
        print(f"wrote {HTML_OUT}  (open with --serve for audio + saving)")


if __name__ == "__main__":
    main()
