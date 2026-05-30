#!/usr/bin/env python3
"""EMP Count (step 5): tally hand-coded axial-labels into the failure-mode pivot,
and dump the data needed for the two bookkeeping debts.

axial-labels.json schema: {"labels":[{segmentId, codes:[...], createdAt, updatedAt}]}.
`codes` is a list of mode tags ("M1".."M10","NotA"). Read-only on session data.

The HTML pivot is an interactive working doc: sortable columns, a Human column
(hand-coded), a Mechanical column (what the audio sweeps caught that hands can't),
a Total (Human + Mechanical), and a free-form Notes column. Notes persist to
emp/results/pivot-notes.json on disk -- but only when the page is *served*, because
a browser can't write files on its own.

Run:  python3 emp/src/count.py                              # pivot + debt data to stdout
      python3 emp/src/count.py --html emp/results/pivot.html  # write the static HTML pivot
      python3 emp/src/count.py --serve                        # serve it with live, file-backed notes
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
# Mechanical floors from the committed sweep summaries (NOT plain segment counts).
# These get their own "Mechanical" column; Total = Human + Mechanical.
#   M3 = TMAS-44 gap floor (39, story-scope >=1.0s)
#   M7 = TMAS-45 drift floor (96 words) -- a ROUGH floor, not the real shape (see footnote **)
MECH = {"M3": 39, "M7": 96}

# ---- free-form notes (the interactive part), persisted to disk ----
NOTES_PATH = ROOT / "emp" / "results" / "pivot-notes.json"
NOTES_ABOUT = ("Structured notes for the EMP failure-mode pivot, keyed by failure mode -- a "
               "triage doc for the health of the speech-to-text system. Edited live in "
               "pivot.html when served (python emp/src/count.py --serve), or by hand here. "
               "Safe to commit -- thoughts about modes, no transcript text.")
AXES = {
    "root_cause": "The underlying reason, not the symptom. Several modes share a cause "
                  "(e.g. character/performed voices drives M3, M4 and M5).",
    "fix_locus": "Where a fix would live: ours = post-processing we control | model = needs a "
                 "better model we don't control | unrecoverable = signal/hardware floor | mixed.",
    "detection": "How we'd catch this error automatically (method step 4). code-signal = code "
                 "cross-checking the transcript against the audio/diarization | code-text = code on "
                 "the transcript text or a reference list | llm-judge = an LLM reading the text | "
                 "hybrid = signal features feeding a judge | ears = needs a human listen | none = "
                 "not feasible / unrecoverable. (TMAS is recognition, so most detection is "
                 "code-against-signal; a text-only LLM can't hear the audio.)",
    "subtypes": "Distinct failures bundled under one mode that may need different fixes "
                "(the M2A/M2B and M9-carve pattern).",
    "thoughts": "Free-form thinking; editable live in the served pivot.",
}
# editable dropdown values for the detection field (blank = unset)
DETECTION_OPTIONS = ["", "code-signal", "code-text", "llm-judge", "hybrid", "ears", "none"]
FIELDS_DEFAULT = {"root_cause": "", "fix_locus": "",
                  "detection": "", "subtypes": [], "thoughts": ""}


def load_notes():
    """Return {_about, _axes, [other _* metadata], notes:{mode->{...fields}}} with every
    mode present and coerced to the structured schema (an old plain-string note folds into
    `thoughts`). Any top-level _-prefixed metadata (e.g. _lens) is preserved across saves."""
    raw = json.load(open(NOTES_PATH)) if NOTES_PATH.exists() else {}
    data = {k: v for k, v in raw.items() if k.startswith("_")}
    data.setdefault("_about", NOTES_ABOUT)
    data.setdefault("_axes", AXES)
    src = raw.get("notes", {}) if isinstance(raw.get("notes"), dict) else {}
    notes = {}
    for m in MODES:
        cur = src.get(m, {})
        if isinstance(cur, str):
            cur = {"thoughts": cur}
        merged = {**FIELDS_DEFAULT, **(cur if isinstance(cur, dict) else {})}
        if not isinstance(merged.get("subtypes"), list):
            merged["subtypes"] = []
        notes[m] = merged
    data["notes"] = notes
    return data


def save_notes(data):
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(NOTES_PATH, "w"), indent=2, ensure_ascii=False)


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


_WORD_RE = re.compile(r"\W*([\w\-'’]+)\W*")
def clean_word(w):
    m = _WORD_RE.fullmatch((w or "").strip())
    return (m.group(1) if m else (w or "").strip()).lower().replace("’", "'")


def m9_breakout(m9cases):
    """Name-instance counts per M9 case per session: word occurrences of each case's
    variant tokens inside that session's M9-tagged segments. Counts NAME INSTANCES, not
    segments, so they do not sum to the M9 segment count. Returns (case_keys, {case:{session:n}})."""
    cases = [k for k in m9cases if isinstance(k, str) and k.startswith("M9")]
    vmap = {}
    for c in cases:
        for v in m9cases[c].get("variants", []):
            vmap[v.lower()] = c
    # Private overlay: real family-name variants (M9a) live in a gitignored sidecar so they
    # stay out of the public repo. Used for counting only; never rendered to HTML.
    priv = ROOT / "emp" / "results" / "pivot-notes.private.json"
    if priv.exists():
        for c, vs in json.load(open(priv)).get("m9_variants", {}).items():
            if c in cases:
                for v in vs:
                    vmap[v.lower()] = c
    out = {c: collections.Counter() for c in cases}
    for sid, name in S:
        m9segs = {L["segmentId"] for L in labels(sid)
                  if "M9" in L.get("codes", []) and isinstance(L.get("segmentId"), int)}
        tr = jload(ROOT / "sessions" / sid / "transcript-rich.json")
        segs = tr.get("segments", []) if isinstance(tr, dict) else []
        for seg in segs:
            if not isinstance(seg, dict) or seg.get("id") not in m9segs:
                continue
            for w in seg.get("words", []):
                c = vmap.get(clean_word(w.get("word", "")))
                if c:
                    out[c][name] += 1
    return cases, out


# ============================ HTML rendering ============================

PIVOT_CSS = """
 :root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 body{margin:2.5rem auto;max-width:1100px;color:#1a1a1a;padding:0 1rem}
 h1{font-size:1.4rem;margin:0 0 .2rem}
 .sub{color:#666;font-size:.85rem;margin:0 0 1.4rem;line-height:1.5}
 .sub code{background:#f3f4f6;padding:0 .2rem;border-radius:3px}
 table{border-collapse:collapse;width:100%;font-size:.85rem}
 th,td{padding:.4rem .55rem;text-align:center;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums}
 th[data-col]{cursor:pointer;user-select:none}
 th[data-col]:hover{color:#000;background:#fafafa}
 th.mode{text-align:left;color:#999;font-weight:600;width:3rem}
 td.lbl{text-align:left;white-space:nowrap}
 td.mech{color:#555}
 td.human{color:#222}
 td.tot,th.tot{font-weight:700;border-left:1px solid #ddd}
 sup.fn{color:#be2828;font-weight:700;font-size:.7em;margin-left:1px}
 thead th{border-bottom:2px solid #333;font-size:.8rem;vertical-align:bottom}
 tr.nota td,tr.nota th{color:#999}
 tr.pinned td,tr.pinned th{border-top:1px solid #ccc}
 .arr{color:#be2828;font-size:.8em}
 .notes{font-size:.82rem;color:#444;margin-top:1.4rem;line-height:1.6;border-top:1px solid #eee;padding-top:1rem}
 /* notes column */
 th.notes-h{text-align:left;color:#999;min-width:15rem}
 td.note-cell{text-align:left;vertical-align:top;border-left:1px solid #eee;min-width:15rem}
 .note{min-height:1.2rem;outline:none;padding:.15rem .3rem;border-radius:4px;white-space:pre-wrap;font-size:.82rem;line-height:1.45;cursor:text}
 .note:hover{background:#fafafa}
 .note:focus{background:#fffbe6;box-shadow:inset 0 0 0 1px #f0d98c}
 .note:empty:before{content:attr(data-ph);color:#bbb}
 .stat{display:block;font-size:.68rem;margin-top:.1rem;min-height:.85rem;color:#2a8a3e}
 .stat.warn{color:#be7a28}
 /* structured triage header inside each notes cell */
 .meta{margin-bottom:.4rem;line-height:1.6}
 .chip{display:inline-block;font-size:.66rem;background:#eef1f4;color:#33415c;border-radius:10px;padding:.06rem .5rem;margin:0 .25rem .22rem 0}
 .chip.sev{background:#fbe6e6;color:#9a2a2a;font-weight:600}
 .chip.cause{background:#e9f3ec;color:#2f6b3a}
 .subs{margin:.15rem 0 .1rem 1rem;padding:0;font-size:.72rem;color:#666}
 .subs li{margin:.05rem 0}
 /* detection dropdown (method step 4: how we'd catch this automatically) */
 .detect-wrap{display:inline-block;font-size:.68rem;color:#555;margin:0 .25rem .22rem 0}
 select.detect{font-size:.68rem;padding:.05rem .2rem;border:1px solid #cdd3da;border-radius:6px;background:#f7f9fb;color:#33415c;margin-left:.15rem;cursor:pointer}
 /* M9-by-case derived sub-rows (name-instances, not segments) */
 tr.m9sub td,tr.m9sub th{background:#fbfcfd;border-bottom:1px dotted #e8e8e8}
 tr.m9sub th.mode{font-weight:600;color:#7a86a0}
 tr.m9sub td.lbl{padding-left:1.3rem;font-style:italic;white-space:normal;color:#555}
 td.m9sub-cell{color:#999;font-variant-numeric:tabular-nums}
 td.m9note .m9notetext{font-size:.74rem;color:#555;line-height:1.45;margin:.1rem 0}
 td.m9note .m9vars{font-size:.67rem;color:#aaa;margin-top:.15rem}
"""

PIVOT_JS = """
(function(){
  var table = document.querySelector('table');
  if(!table) return;
  var curCol = -1, curDir = 'desc';
  function sortRows(idx, numeric, dir){
    var tb = table.tBodies[0];
    var all = Array.prototype.slice.call(tb.rows);
    var sub = all.filter(function(r){return r.classList.contains('m9sub');});
    var pinned = all.filter(function(r){return r.classList.contains('pinned');});
    var rows = all.filter(function(r){return !r.classList.contains('pinned') && !r.classList.contains('m9sub');});
    rows.sort(function(a,b){
      var av = a.cells[idx].getAttribute('data-sort'); if(av===null) av=a.cells[idx].textContent;
      var bv = b.cells[idx].getAttribute('data-sort'); if(bv===null) bv=b.cells[idx].textContent;
      if(numeric){ av=parseFloat(av)||0; bv=parseFloat(bv)||0; return dir==='desc'?bv-av:av-bv; }
      av=(''+av).toLowerCase(); bv=(''+bv).toLowerCase();
      return dir==='desc'? bv.localeCompare(av) : av.localeCompare(bv);
    });
    rows.forEach(function(r){tb.appendChild(r);});
    pinned.forEach(function(r){tb.appendChild(r);});   /* clean-segment row stays pinned at the bottom */
    /* keep the M9-by-case sub-rows grouped directly under the M9 row */
    var m9=null;
    rows.forEach(function(r){var th=r.querySelector('th.mode'); if(th && th.textContent.trim()==='M9') m9=r;});
    if(m9){ var ref=m9.nextSibling; sub.forEach(function(s){ tb.insertBefore(s, ref); }); }
  }
  table.querySelectorAll('th[data-col]').forEach(function(th){
    th.addEventListener('click', function(){
      var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
      var numeric = th.classList.contains('num');
      var dir = (curCol===idx && curDir==='desc') ? 'asc' : 'desc';
      sortRows(idx, numeric, dir);
      curCol = idx; curDir = dir;
      table.querySelectorAll('th[data-col] .arr').forEach(function(x){x.remove();});
      var a = document.createElement('span'); a.className='arr';
      a.textContent = dir==='desc' ? ' \\u25BC' : ' \\u25B2';
      th.appendChild(a);
    });
  });
  /* notes: debounced auto-save to pivot-notes.json (only works when served) */
  var served = location.protocol.indexOf('http')===0;
  var timers = {};
  document.querySelectorAll('.note').forEach(function(el){
    el.addEventListener('input', function(){
      var mode = el.getAttribute('data-mode');
      var stat = document.querySelector('.stat[data-mode="'+mode+'"]');
      if(!served){ stat.textContent='\\u26A0 run count.py --serve to save'; stat.className='stat warn'; return; }
      stat.textContent='saving\\u2026'; stat.className='stat';
      clearTimeout(timers[mode]);
      timers[mode] = setTimeout(function(){
        fetch('/notes',{method:'POST',headers:{'Content-Type':'application/json'},
          body: JSON.stringify({mode: mode, text: el.innerText})})
        .then(function(r){ stat.textContent = r.ok?'saved \\u2713':'save failed'; stat.className = r.ok?'stat':'stat warn'; })
        .catch(function(){ stat.textContent='save failed'; stat.className='stat warn'; });
      }, 500);
    });
  });
  /* detection dropdown: save the chosen detector type on change */
  document.querySelectorAll('.detect').forEach(function(sel){
    sel.addEventListener('change', function(){
      var mode = sel.getAttribute('data-mode');
      var stat = document.querySelector('.stat[data-mode="'+mode+'"]');
      if(!served){ if(stat){ stat.textContent='\\u26A0 run count.py --serve to save'; stat.className='stat warn'; } return; }
      if(stat){ stat.textContent='saving\\u2026'; stat.className='stat'; }
      fetch('/notes',{method:'POST',headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode: mode, detection: sel.value})})
      .then(function(r){ if(stat){ stat.textContent = r.ok?'saved \\u2713':'save failed'; stat.className = r.ok?'stat':'stat warn'; } })
      .catch(function(){ if(stat){ stat.textContent='save failed'; stat.className='stat warn'; } });
    });
  });
})();
"""


def render_html(piv, recs):
    nd = load_notes()
    notes = nd["notes"]
    m9cases = nd.get("_m9_cases", {})
    m9_keys, m9_counts = m9_breakout(m9cases) if m9cases else ([], {})
    fail_vals = [piv[m][n] for m in MODES if m != "NotA" for n in NM]
    mx = max(fail_vals) if fail_vals else 1

    body_rows = []
    for m in MODES:
        rc = piv[m]
        human = sum(rc.values())
        mech = MECH.get(m)
        total = human + (mech or 0)

        cells = []
        for n in NM:
            v = rc[n]
            if m == "NotA":
                bg = "#f3f4f6"
            elif v:
                bg = f"rgba(190,40,40,{0.08 + 0.72 * (v / mx):.3f})"
            else:
                bg = "transparent"
            cells.append(f'<td data-sort="{v}" style="background:{bg}">{v}</td>')

        if mech is None:
            mech_html = '<td class="mech" data-sort="0">&mdash;</td>'
        else:
            mark = "*" if m == "M3" else "**"
            mech_html = f'<td class="mech" data-sort="{mech}">{mech}<sup class="fn">{mark}</sup></td>'

        note = notes.get(m, {})
        fix = note.get("fix_locus", "")
        detection = note.get("detection", ""); cause = note.get("root_cause", "")
        subs = note.get("subtypes", []); thoughts = note.get("thoughts", "")
        chips = []
        if cause: chips.append(f'<span class="chip cause">cause: {html.escape(cause)}</span>')
        if fix:   chips.append(f'<span class="chip">fix: {html.escape(fix)}</span>')
        detect_html = ""
        if m != "NotA":
            opt_label = {"": "(set detector)"}
            opts = "".join(
                f'<option value="{o}"{" selected" if o == detection else ""}>'
                f'{html.escape(opt_label.get(o, o))}</option>' for o in DETECTION_OPTIONS)
            detect_html = (f'<label class="detect-wrap">detect: '
                           f'<select class="detect" data-mode="{m}">{opts}</select></label>')
        subs_html = ('<ul class="subs">' + "".join(f'<li>{html.escape(s)}</li>' for s in subs)
                     + '</ul>') if subs else ''
        meta_inner = "".join(chips) + detect_html + subs_html
        meta_html = ('<div class="meta">' + meta_inner + '</div>') if meta_inner else ''
        sort_key = thoughts.lower().replace(chr(10), " ")[:40]
        note_html = (
            f'<td class="note-cell" data-sort="{html.escape(sort_key, quote=True)}">'
            + meta_html +
            f'<div class="note" contenteditable="true" data-mode="{m}" data-ph="add a thought…">'
            f'{html.escape(thoughts)}</div>'
            f'<span class="stat" data-mode="{m}"></span></td>'
        )

        label = html.escape(MODE_LABELS.get(m, m))
        cls = ' class="nota pinned"' if m == "NotA" else ""
        body_rows.append(
            f'<tr{cls}><th class="mode" data-sort="{m}">{m}</th>'
            f'<td class="lbl" data-sort="{label.lower()}">{label}</td>'
            f'{"".join(cells)}'
            f'<td class="human" data-sort="{human}">{human}</td>'
            f'{mech_html}'
            f'<td class="tot" data-sort="{total}">{total}</td>'
            f'{note_html}</tr>'
        )

        # inject the M9-by-case derived rows (name-instances) directly under M9
        if m == "M9" and m9_keys:
            for cc in m9_keys:
                meta = m9cases[cc]
                inst = m9_counts.get(cc, {})
                cc_tot = sum(inst.values())
                scells = "".join(f'<td class="m9sub-cell">{inst.get(n, 0) or "&middot;"}</td>' for n in NM)
                det = meta.get("detection", ""); cnotes = meta.get("notes", ""); cvars = meta.get("variants", [])
                chip = f'<span class="chip">detect: {html.escape(det)}</span>' if det else ''
                vlist = ('<div class="m9vars">' + ", ".join(html.escape(v) for v in cvars) + '</div>') if cvars else ''
                ncell = (f'<td class="note-cell m9note"><div class="meta">{chip}</div>'
                         f'<div class="m9notetext">{html.escape(cnotes)}</div>{vlist}</td>')
                clabel = html.escape(meta.get("label", cc))
                body_rows.append(
                    f'<tr class="m9sub"><th class="mode">{cc}</th>'
                    f'<td class="lbl">&#8627; {clabel}</td>'
                    f'{scells}'
                    f'<td class="human">{cc_tot}</td>'
                    f'<td class="mech">&mdash;</td>'
                    f'<td class="tot">{cc_tot}</td>'
                    f'{ncell}</tr>'
                )

    head_cells = "".join(f'<th class="num" data-col>{html.escape(n)}</th>' for n in NM)
    rows_html = chr(10).join(body_rows)

    footnotes = (
        '<p class="notes">'
        '<b>* Mode 3 &mdash; missed real speech.</b> A coder can only flag a miss when '
        'something shows up to click &mdash; a half-dropped line, or a visible gap. But often '
        'someone speaks and the transcript shows <i>nothing at all</i> for that moment &mdash; '
        'no line, not even an &ldquo;[unintelligible]&rdquo; marker &mdash; so there is nothing '
        'to click and it can&rsquo;t be hand-counted. A separate script catches these by comparing '
        'who was talking (from the speaker detector) against what got written down. It found '
        f'<b>{MECH["M3"]}</b> clear cases (Portal 29, Moon 4, Cruel Baby 4, Rubber Ducky 2, '
        'Pandavas 0); the Total adds those to the 18 a coder could see.<br><br>'
        '<b>** Mode 7 &mdash; word at the wrong time.</b> Whisper <i>guesses</i> each '
        'word&rsquo;s start and end time instead of measuring it, so the times drift &mdash; '
        'usually the first word of a segment, anchored to the silence just before anyone speaks. '
        'It&rsquo;s hard to spot by eye, so the hand count (17) is far too low. Checking every '
        f'word against the actual sound, a script found <b>{MECH["M7"]}</b> words clearly sitting '
        'on silence with their real audio right beside them. Treat that as a <i>rough floor, not '
        'the true size</i>: measured as a spread, about 12% of all words start &ge;0.1s late, '
        '8% &ge;a quarter-second, 3% &ge;half a second &mdash; over a thousand words. The fix is '
        '&ldquo;forced alignment&rdquo; (re-snapping each word to the audio), tracked as TMAS-50.'
        '<br><br><b>M9a / M9b / M9c</b> split Mode 9 (names) by who decides the correct spelling: '
        '<b>real people</b> (a known family roster), <b>improvised</b> names (no canon, only '
        'self-consistency), and <b>sourced canon</b> (an external reference library). Those rows '
        'count <i>name instances</i> &mdash; occurrences of the mis-rendered name inside M9 '
        'segments &mdash; so they don&rsquo;t sum to M9&rsquo;s 134 segments. The name-to-case map '
        'is editable in <code>pivot-notes.json</code> (<code>_m9_cases</code>); re-run to recount.'
        '</p>'
    )

    sub = (
        f'<p class="sub">Hand-coded segments across the five story sessions ({recs} coded). '
        'Each session cell counts segments carrying that mode (one segment can carry more than '
        'one). <b>Human</b> = what coders tagged by hand. <b>Mechanical</b> = what the audio '
        'sweeps caught that hands can&rsquo;t (Modes 3 &amp; 7 &mdash; see the notes below). '
        '<b>Total</b> = Human + Mechanical. <b>Click any column heading to sort</b> (the '
        'clean-segment row stays pinned at the bottom). The <b>Notes</b> column carries a small '
        'triage header per mode (cause / where a fix would live / how it&rsquo;d be detected / '
        'sub-types) above your free-text thoughts; the <b>detect</b> dropdown and your thoughts '
        'save to <code>pivot-notes.json</code> when served '
        '(<code>python emp/src/count.py --serve</code>). Generated by <code>emp/src/count.py</code>.</p>'
    )

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
        '<title>EMP failure-mode pivot</title>\n'
        '<style>' + PIVOT_CSS + '</style></head><body>\n'
        '<h1>EMP &mdash; failure-mode pivot</h1>\n'
        + sub + '\n'
        '<table>\n<thead><tr>'
        '<th class="mode" data-col>&nbsp;</th>'
        '<th class="lbl" data-col>Failure mode</th>'
        + head_cells +
        '<th class="num" data-col>Human</th>'
        '<th class="num" data-col>Mechanical</th>'
        '<th class="num tot" data-col>Total</th>'
        '<th class="notes-h" data-col>Notes</th>'
        '</tr></thead>\n<tbody>\n'
        + rows_html +
        '\n</tbody></table>\n'
        + footnotes +
        '\n<script>' + PIVOT_JS + '</script>\n'
        '</body></html>'
    )


# ============================ serve (the bridge to disk) ============================

def serve(port):
    """Serve the pivot and persist Notes edits to NOTES_PATH. Browser can't write
    files on its own; this tiny server is the bridge."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/pivot.html", "/index.html"):
                piv, _, recs, _ = compute()
                self._send(200, render_html(piv, recs))
            elif path == "/pivot-notes.json":
                self._send(200, json.dumps(load_notes(), indent=2, ensure_ascii=False),
                           "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/notes":
                self._send(404, "not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}', "application/json"); return
            data = load_notes()
            mode = body.get("mode")
            if mode in data["notes"]:
                # live edits from the page carry one field at a time
                if "detection" in body:
                    data["notes"][mode]["detection"] = body.get("detection", "")
                elif "text" in body:
                    data["notes"][mode]["thoughts"] = body.get("text", "")
                else:
                    self._send(400, '{"error":"mode needs text or detection"}', "application/json"); return
            elif isinstance(body.get("notes"), dict):
                for k, v in body["notes"].items():
                    if k not in data["notes"]:
                        continue
                    if isinstance(v, dict):
                        data["notes"][k] = {**FIELDS_DEFAULT, **v}
                    else:
                        data["notes"][k]["thoughts"] = v
            else:
                self._send(400, '{"error":"need {mode,text|detection} or {notes}"}', "application/json"); return
            save_notes(data)
            self._send(200, '{"ok":true}', "application/json")

        def log_message(self, *a):  # keep the console quiet
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"EMP pivot   ->  http://127.0.0.1:{port}/")
    print(f"notes file  ->  {NOTES_PATH.relative_to(ROOT)}   (edits auto-save here; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    ap = argparse.ArgumentParser(description="EMP failure-mode count / pivot")
    ap.add_argument("--html", metavar="PATH", help="write the pivot as a self-contained HTML artifact")
    ap.add_argument("--serve", action="store_true", help="serve the pivot with live, file-backed notes")
    ap.add_argument("--port", type=int, default=5055, help="port for --serve (default 5055)")
    args = ap.parse_args()

    # ensure the notes file exists so it's there to read / edit / discuss
    if not NOTES_PATH.exists():
        save_notes(load_notes())

    if args.serve:
        serve(args.port)
        return

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
        outp.write_text(render_html(piv, recs), encoding="utf-8")
        print(f"\nwrote HTML pivot -> {outp.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
