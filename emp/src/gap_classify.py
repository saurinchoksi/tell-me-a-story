"""TMAS-46 — suggest a code for each `[unintelligible]` gap (read-only).

An `[unintelligible]` gap is one of three things, coded by ear:
  M8   — no real speech there; the gap was wrongly injected (pyannote false-fired).
  NotA — real speech, too quiet for a stranger to decode; the pipeline was right.
  M3   — real speech loud enough to decode; the pipeline missed it.

What the signals can and can't do (measured against all 5 sessions' existing codes):
  * LOUDNESS FAILS. The three classes overlap completely in volume (a loud chuckle
    is M8; a loud mumble is NotA), so a loudness rule scores 29% — worse than always
    guessing NotA (46%). The call is about speech *clarity*, not volume.
  * WHISPER-ON-CLIP works for the tractable half. Running the recognizer on each
    isolated gap clip, its `no_speech_prob` cleanly separates "nothing to transcribe"
    (M8) from real speech: the M8-vs-speech call scores ~82% (M8 flag ~86% precise).
  * M3 vs NotA does NOT mechanize. Whisper decodes both into plausible text (it
    hallucinates confident words even on undecodable audio), so that finer call stays
    an ear judgment — but Whisper's transcription attempt is shown as a decode AID.

So the tool auto-suggests **M8** (nothing to transcribe) and hands the rest to the ear
as **real speech — M3 or NotA**, with Whisper's best-guess transcript on each card.

Read-only on session data. Writes only under emp/results/:
  emp/results/gap-classify/summary.md            (committable; aggregate only)
  emp/results/gap-classify/whisper-cache.json    (feature cache; text of family audio -> gitignored)
  emp/results/visuals/<id>/gap-classify.html      (gitignored; transcript context)
  emp/results/visuals/<id>/gapclip-<n>.mp3        (gitignored; pre-cut audio)

Usage:
  ./venv/bin/python emp/src/gap_classify.py            # classify + score, print table
  ./venv/bin/python emp/src/gap_classify.py --html      # + check-by-ear cards & clips
  ./venv/bin/python emp/src/gap_classify.py --refresh   # recompute Whisper features
"""

import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "emp" / "src"))
import mlx_whisper  # noqa: E402
from mlx_whisper.audio import SAMPLE_RATE, load_audio  # noqa: E402
from api.helpers import get_session_dir  # noqa: E402  (kept: standard sweep boilerplate)
from gap_analysis import (  # noqa: E402
    SESSIONS, load_session, mmss, real_segments, unintelligible_segments,
)
from timestamp_drift_analysis import CHUNK_SEC, load_axial_codes, speaker_coverage  # noqa: E402

NAMES = {s["id"]: s["name"] for s in SESSIONS}
OUTDIR = ROOT / "emp" / "results" / "visuals"
SUMMARY_DIR = ROOT / "emp" / "results" / "gap-classify"
CACHE = SUMMARY_DIR / "whisper-cache.json"
NOTES = SUMMARY_DIR / "heard-notes.json"  # what the human hears per gap (by-ear ground truth)
CORE = ("M8", "NotA", "M3")

MODEL = "mlx-community/whisper-large-v3-mlx"
PAD = 0.3         # seconds of context each side for the clip Whisper reads
NSP_M8 = 0.9      # no_speech_prob at/above this (or zero words) -> "nothing to transcribe" = M8
NOISE_PCT = 20    # percentile of session dB = its noise floor (loudness shown for reference only)
LEAD, TAIL = 1.5, 0.6  # audio-clip padding, seconds

_sess = {}


def session_signals(sdir):
    """(raw 16kHz mono audio, per-20ms-chunk dBFS). Mirrors load_rms_db but keeps
    the raw array too, so Whisper and the loudness readout share one decode."""
    key = str(sdir)
    if key not in _sess:
        a = np.array(load_audio(str(sdir / "audio.m4a"))).astype(np.float32)
        hop = int(CHUNK_SEC * SAMPLE_RATE)
        n = len(a) // hop
        rms = np.sqrt((a[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-10))
        _sess[key] = (a, db)
    return _sess[key]


def whisper_feat(audio, rec, cache):
    """Run Whisper on the isolated gap clip (cached). -> {text, alp, nsp, nw}."""
    key = f"{rec['sid']}|{rec['id']}"
    ent = cache.get(key)
    if ent and ent.get("model") == MODEL and ent.get("pad") == PAD:
        return ent
    i0 = max(0, int((rec["start"] - PAD) * SAMPLE_RATE))
    i1 = min(len(audio), int((rec["end"] + PAD) * SAMPLE_RATE))
    r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                               fp16=True, verbose=False)
    segs = r.get("segments", [])
    text = (r.get("text") or "").strip()
    ent = {
        "text": text,
        "alp": float(np.mean([s["avg_logprob"] for s in segs])) if segs else None,
        "nsp": float(np.max([s["no_speech_prob"] for s in segs])) if segs else 1.0,
        "nw": len(text.split()),
        "model": MODEL, "pad": PAD,
    }
    cache[key] = ent
    return ent


def classify(rec):
    """Whisper-based suggestion: 'M8' (nothing to transcribe) or 'speech' (ear call)."""
    if rec["nsp"] >= NSP_M8 or rec["nw"] == 0:
        return "M8"
    return "speech"


def is_clean(rec):
    """Human code is exactly one of M8/NotA/M3 — the scorable answer key."""
    return len(rec["human"]) == 1 and rec["human"][0] in CORE


def load_notes():
    """{_about, notes: {'<sid>|<gap id>': {heard: str}}} — the human's by-ear transcription
    of each gap (the real answer key for the M3-vs-NotA call the model can't make).
    Preserves any _-prefixed metadata across saves."""
    raw = json.load(open(NOTES)) if NOTES.exists() else {}
    data = {k: v for k, v in raw.items() if k.startswith("_")}
    data.setdefault("_about", "What the human hears in each [unintelligible] gap (TMAS-46 "
                    "by-ear ground truth). Keyed by '<session id>|<gap id>'.")
    src = raw.get("notes", {}) if isinstance(raw.get("notes"), dict) else {}
    data["notes"] = {k: (v if isinstance(v, dict) else {"heard": v}) for k, v in src.items()}
    return data


def save_notes(data):
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(NOTES, "w"), indent=2, ensure_ascii=False)


def gather(refresh=False):
    cache = {} if refresh else (json.load(open(CACHE)) if CACHE.exists() else {})
    records = []
    for s in SESSIONS:
        sid = s["id"]
        sdir, transcript, diar = load_session(sid)
        gaps = unintelligible_segments(transcript)
        if not gaps:
            continue
        audio, db = session_signals(sdir)
        floor = float(np.percentile(db, NOISE_PCT))
        codes = load_axial_codes(sdir)
        reals = sorted(real_segments(transcript), key=lambda x: x["start"])
        for g in sorted(gaps, key=lambda x: x["start"]):
            n = len(db)
            i0 = max(0, min(int(g["start"] / CHUNK_SEC), n - 1))
            i1 = max(i0, min(int(g["end"] / CHUNK_SEC), n - 1))
            peak = float(db[i0:i1 + 1].max())
            _, spk_total = speaker_coverage(g["start"], g["end"], diar["segments"])
            prev_t, next_t = "—", "—"
            for r in reals:
                if r["start"] < g["start"]:
                    prev_t = (r.get("text") or "").strip()
                elif r["start"] > g["start"]:
                    next_t = (r.get("text") or "").strip()
                    break
            rec = {
                "sid": sid, "name": NAMES.get(sid, sid), "sdir": sdir,
                "id": g["id"], "start": g["start"], "end": g["end"],
                "dur": g["end"] - g["start"], "floor": floor, "peak": peak,
                "spk_total": spk_total, "human": codes.get(g["id"], []),
                "prev": prev_t[:90], "next": next_t[:90],
            }
            rec.update(whisper_feat(audio, rec, cache))
            records.append(rec)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    return records


def score(records):
    """M8-vs-speech agreement over the clean-coded gaps (the mechanizable axis)."""
    clean = [r for r in records if is_clean(r)]
    conf = {"M8": {"M8": 0, "speech": 0}, "speech": {"M8": 0, "speech": 0}}
    for r in clean:
        exp = "M8" if r["human"][0] == "M8" else "speech"
        conf[exp][classify(r)] += 1
    agree = conf["M8"]["M8"] + conf["speech"]["speech"]
    return clean, conf, agree


# --- reporting ---------------------------------------------------------------
def print_table(records):
    print(f"\n{'session':12} {'gap id':15} {'dur':>5} {'human':>6} {'nsp':>5} {'alp':>6} "
          f"{'nw':>3} {'sugg':>6}  whisper heard")
    print("-" * 104)
    for r in sorted(records, key=lambda x: (x["name"], x["start"])):
        human = "/".join(r["human"]) or "—"
        alp = f"{r['alp']:6.2f}" if r["alp"] is not None else "   nan"
        print(f"{r['name']:12} {r['id']:15} {r['dur']:5.2f} {human:>6} {r['nsp']:5.2f} "
              f"{alp} {r['nw']:3} {classify(r):>6}  {r['text'][:40]!r}")


def print_score(records):
    clean, conf, agree = score(records)
    m8_pred = conf["M8"]["M8"] + conf["speech"]["M8"]
    print(f"\nM8-vs-speech agreement (the mechanizable axis): {agree}/{len(clean)} "
          f"= {agree / len(clean):.0%}   (loudness baseline 29%, majority-guess 46%)")
    print(f"  M8 flag precision {conf['M8']['M8']}/{m8_pred} · "
          f"recall {conf['M8']['M8']}/{conf['M8']['M8'] + conf['M8']['speech']}")
    print("  confusion (rows=your code bucket, cols=suggested):")
    print(f"    {'':7}{'M8':>7}{'speech':>7}")
    for h in ("M8", "speech"):
        print(f"    {h:7}{conf[h]['M8']:>7}{conf[h]['speech']:>7}")
    print("  (M3 vs NotA is not auto-split — Whisper decodes both; that stays an ear call.)")
    held = [r for r in records if not is_clean(r)]
    if held:
        print(f"\n  held out — another mode owns these ({len(held)}): "
              + ", ".join(f"{r['id']}={'/'.join(r['human'])}" for r in held))


# --- check-by-ear page + clips -----------------------------------------------
def cut_clips(records):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    for i, r in enumerate(records, 1):
        frm = max(0.0, r["start"] - LEAD)
        out = OUTDIR / r["sid"] / f"gapclip-{i}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(r["sdir"] / "audio.m4a"),
               "-t", f"{r['dur'] + LEAD + TAIL:.3f}", "-ac", "1", "-ar", "22050",
               "-c:a", "libmp3lame", "-q:a", "6", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        r["_clip"] = f"gapclip-{i}.mp3"
        if res.returncode != 0 or not out.exists():
            print(f"  clip FAIL {r['id']}: {res.stderr[-160:]}")


def render_cards(recs, notes):
    e = html.escape
    saved = notes.get("notes", {})

    def card(r):
        key = f"{r['sid']}|{r['id']}"
        note = saved.get(key, {}).get("heard", "")  # what the human already typed (empty at first)
        sugg = classify(r)
        human = "/".join(r["human"]) or "—"
        clean = is_clean(r)
        exp = "M8" if (clean and r["human"][0] == "M8") else ("speech" if clean else None)
        bad = clean and exp != sugg
        heard = r["text"] or "(silence — Whisper heard nothing)"
        chip = ('<span class="chip m8">likely M8 — nothing to transcribe</span>'
                if sugg == "M8" else
                '<span class="chip sp">real speech — your call: M3 or NotA</span>')
        hchip = (f'<span class="chip human">your code: {e(human)}'
                 f'{" ✓" if clean and not bad else (" ✗" if bad else "")}</span>')
        return (
            f'<div class="card{" bad" if bad else ""}"><div class="row">'
            f'<audio src="{r["_clip"]}" preload="none"></audio>'
            f'<button class="play" onclick="play(this)">▶</button>'
            f'<div class="body">'
            f'<div class="heard">Whisper heard: “{e(heard)}”</div>'
            f'<div class="chips">{chip} {hchip}</div>'
            f'<div class="ctx"><span class="lbl">before</span>"{e(r["prev"])}" '
            f'<span class="lbl">after</span>"{e(r["next"])}"</div>'
            f'<div class="src">{e(r["name"])} · {r["id"]} · {mmss(r["start"])}–{mmss(r["end"])} '
            f'({r["dur"]:.2f}s) · no-speech {r["nsp"]:.2f} · peak {r["peak"]:.0f}/floor '
            f'{r["floor"]:.0f} dB</div>'
            f'<div class="hear"><textarea class="hearbox" data-key="{e(key)}" rows="2" '
            f'placeholder="type exactly what you hear in this clip…">{e(note)}</textarea>'
            f'<span class="savestat"></span></div>'
            f'</div></div></div>')

    # A gap already coded as some OTHER mode (M2/M7/M4/...) is owned by that mode —
    # show it apart so it doesn't clutter the review. An UNcoded gap (future session,
    # no human code) is not held out — it's grouped by the suggestion like any other.
    held = [r for r in recs if r["human"] and not is_clean(r)]
    rest = [r for r in recs if not (r["human"] and not is_clean(r))]
    m8 = [r for r in rest if classify(r) == "M8"]
    speech = [r for r in rest if classify(r) == "speech"]

    o = ['<!doctype html><html><head><meta charset="utf-8">',
         '<title>[unintelligible] gaps — code check</title><style>',
         "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
         "max-width:760px;margin:32px auto;padding:0 20px;color:#1a1a1a;line-height:1.5}",
         "h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 10px}",
         ".intro{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;font-size:14px}",
         ".card{border:1px solid #e2e8f0;border-radius:11px;padding:11px 13px;margin-bottom:10px}",
         ".card.bad{background:#fef2f2;border-color:#fca5a5}",
         ".row{display:flex;gap:12px;align-items:flex-start}",
         ".play{flex:none;background:#2563eb;color:#fff;border:none;border-radius:8px;"
         "width:40px;height:40px;font-size:15px;cursor:pointer}.play.playing{background:#16a34a}",
         ".body{flex:1}.heard{font-size:16px;margin-bottom:5px}",
         ".chips{margin-bottom:4px}.chip{color:#fff;border-radius:6px;padding:2px 7px;"
         "font-size:12px;font-weight:700;margin-right:5px}",
         ".chip.m8{background:#f59e0b}.chip.sp{background:#2563eb}.chip.human{background:#475569}",
         ".ctx{color:#475569;font-size:13px}.lbl{color:#94a3b8;font-size:11px;text-transform:uppercase;margin-right:3px}",
         ".src{color:#94a3b8;font-size:11px;margin-top:3px}",
         ".hear{margin-top:7px;display:flex;gap:6px;align-items:flex-start}",
         ".hearbox{flex:1;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:7px;"
         "padding:6px 8px;font-size:13px;font-family:inherit;resize:vertical}",
         ".hearbox:focus{outline:none;border-color:#2563eb}",
         ".savestat{font-size:11px;color:#16a34a;white-space:nowrap;padding-top:8px}",
         "</style></head><body>",
         "<h1>[unintelligible] gaps — code check</h1>",
         '<div class="intro">The recognizer was re-run on each isolated clip. Gaps where it '
         "heard <b>nothing to transcribe</b> are almost always <b>M8</b> (the gap was wrongly "
         "injected) — those are grouped first. For the rest there <b>is</b> real speech; Whisper's "
         "best-guess transcript is shown as a decoding aid, but whether a stranger could truly "
         "make it out (<b>M3</b>) or not (<b>NotA</b>) is your ear-call. ▶ plays a pre-cut clip. "
         "Where a code already exists it's shown with ✓/✗ against the M8-vs-speech suggestion. "
         "<b>Type exactly what you hear</b> in the box on each card — it saves to disk (run with "
         "<code>--serve</code>) and becomes the by-ear answer key.</div>"]
    o.append(f"<h2>Likely M8 — nothing to transcribe ({len(m8)})</h2>")
    o += [card(r) for r in sorted(m8, key=lambda x: (x["name"], x["start"]))] or ["<p>none</p>"]
    o.append(f"<h2>Real speech — code M3 or NotA by ear ({len(speech)})</h2>")
    o += [card(r) for r in sorted(speech, key=lambda x: (x["name"], x["start"]))] or ["<p>none</p>"]
    if held:
        o.append(f"<h2>Held out — another mode already owns these ({len(held)})</h2>")
        o += [card(r) for r in sorted(held, key=lambda x: (x["name"], x["start"]))]

    o.append("""<script>
let cur=null,curBtn=null;
function play(btn){const a=btn.closest('.card').querySelector('audio');
 if(cur&&cur!==a)cur.pause();
 if(curBtn&&curBtn!==btn){curBtn.classList.remove('playing');curBtn.textContent='▶';}
 cur=a;curBtn=btn;a.currentTime=0;const p=a.play();if(p&&p.catch)p.catch(()=>{});
 btn.classList.add('playing');btn.textContent='♪';
 a.onended=()=>{btn.classList.remove('playing');btn.textContent='▶';};}
var served=location.protocol.indexOf('http')===0;
document.querySelectorAll('textarea.hearbox').forEach(function(t){
 var s=t.parentNode.querySelector('.savestat'),timer=null;
 t.addEventListener('input',function(){
  if(!served){if(s)s.textContent='⚠ run --serve to save';return;}
  if(s)s.textContent='…';
  if(timer)clearTimeout(timer);
  timer=setTimeout(function(){
   fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:t.dataset.key,heard:t.value})})
    .then(function(r){if(s)s.textContent=r.ok?'saved ✓':'save failed';})
    .catch(function(){if(s)s.textContent='save failed';});
  },500);});});
</script></body></html>""")
    return "\n".join(o)


def write_html(records):
    notes = load_notes()
    by_sid = {}
    for r in records:
        by_sid.setdefault(r["sid"], []).append(r)
    for sid, recs in by_sid.items():
        recs.sort(key=lambda x: x["start"])
        cut_clips(recs)
        page = OUTDIR / sid / "gap-classify.html"
        page.write_text(render_cards(recs, notes))
        print(f"  wrote {page.relative_to(ROOT)} ({len(recs)} cards)")


def serve(port):
    """Serve the per-session cards + clips and persist 'what I hear' edits to NOTES.
    A browser can't write disk on its own; this tiny server is the bridge (see the
    interactive-artifact pattern). Renders fresh on each GET so saved notes show on reload."""
    import http.server

    records = gather()
    by_sid = {}
    for r in records:
        by_sid.setdefault(r["sid"], []).append(r)
    for recs in by_sid.values():
        recs.sort(key=lambda x: x["start"])
        cut_clips(recs)  # ensure the clips exist to serve

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
            if path in ("/", "/index.html"):
                links = "".join(
                    f'<li><a href="/{s["id"]}/gap-classify.html">{html.escape(NAMES.get(s["id"], s["id"]))}</a>'
                    f' — {len(by_sid[s["id"]])} gaps</li>'
                    for s in SESSIONS if s["id"] in by_sid)
                self._send(200, "<!doctype html><meta charset=utf-8>"
                           "<body style='font-family:sans-serif;max-width:600px;margin:40px auto'>"
                           f"<h1>[unintelligible] gaps — code check</h1><ul>{links}</ul>")
                return
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0] in by_sid and parts[1] == "gap-classify.html":
                self._send(200, render_cards(by_sid[parts[0]], load_notes()))
                return
            if len(parts) == 2 and parts[1].endswith(".mp3"):
                clip = OUTDIR / parts[0] / parts[1]
                if clip.exists():
                    self._send(200, clip.read_bytes(), "audio/mpeg")
                    return
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}', "application/json")
                return
            key = body.get("key")
            if not key:
                self._send(400, '{"error":"need key"}', "application/json")
                return
            data = load_notes()
            rec = data["notes"].get(key, {})
            rec["heard"] = body.get("heard", "")  # field-routed: only touch 'heard'
            data["notes"][key] = rec
            save_notes(data)
            self._send(200, '{"ok":true}', "application/json")

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"gap check  ->  http://127.0.0.1:{port}/")
    print(f"notes file ->  {NOTES.relative_to(ROOT)}   (what-you-hear auto-saves here; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def write_summary(records):
    clean, conf, agree = score(records)
    held = [r for r in records if not is_clean(r)]
    m8_pred = conf["M8"]["M8"] + conf["speech"]["M8"]
    L = [
        "# `[unintelligible]` gap classification — summary\n",
        "Read-only. For each injected `[unintelligible]` gap, the recognizer is re-run on the "
        "isolated clip; `no_speech_prob` decides **M8 (nothing to transcribe)** vs **real speech**. "
        "Scored against the existing by-ear codes. Gaps a human coded as another mode are held out.\n",
        "## What the signals can do\n",
        "- **Loudness fails** — the three classes overlap completely in volume (a loud chuckle is "
        "M8, a loud mumble is NotA). A loudness rule scores 29%, worse than always guessing NotA (46%).",
        "- **Whisper-on-clip separates M8 from real speech** — the mechanizable axis.",
        "- **M3 vs NotA does not mechanize** — Whisper decodes both into plausible text, so that finer "
        "call stays an ear judgment; Whisper's transcript is shown as a decode aid, not a verdict.\n",
        f"**M8-vs-speech agreement: {agree}/{len(clean)} = {agree / len(clean):.0%}.** "
        f"M8 flag precision {conf['M8']['M8']}/{m8_pred}, "
        f"recall {conf['M8']['M8']}/{conf['M8']['M8'] + conf['M8']['speech']}.\n",
        "| your bucket \\ suggested | M8 | speech |",
        "|---|---|---|",
        f"| M8 | {conf['M8']['M8']} | {conf['M8']['speech']} |",
        f"| speech (NotA+M3) | {conf['speech']['M8']} | {conf['speech']['speech']} |\n",
        "## Per session (suggested)\n",
        "| session | gaps | M8 | real speech | held out |",
        "|---|---|---|---|---|",
    ]
    for s in SESSIONS:
        rs = [r for r in records if r["sid"] == s["id"]]
        if not rs:
            continue
        L.append(f"| {s['name']} | {len(rs)} | {sum(classify(r) == 'M8' for r in rs)} | "
                 f"{sum(classify(r) == 'speech' for r in rs)} | "
                 f"{sum(not is_clean(r) for r in rs)} |")
    dis = [r for r in clean if ("M8" if r["human"][0] == "M8" else "speech") != classify(r)]
    L.append(f"\n## M8-vs-speech disagreements ({len(dis)}) — check by ear\n")
    for r in sorted(dis, key=lambda x: (x["name"], x["start"])):
        L.append(f"- {r['name']} `{r['id']}` ({mmss(r['start'])}, {r['dur']:.2f}s): you coded "
                 f"**{r['human'][0]}**, suggested **{classify(r)}** "
                 f"(no-speech {r['nsp']:.2f}, {r['nw']} words heard)")
    L.append(f"\n## Held out — another mode owns these ({len(held)})\n")
    for r in sorted(held, key=lambda x: (x["name"], x["start"])):
        L.append(f"- {r['name']} `{r['id']}` ({mmss(r['start'])}) coded {'/'.join(r['human']) or '—'}")
    (SUMMARY_DIR / "summary.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {(SUMMARY_DIR / 'summary.md').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="TMAS-46 gap classifier (read-only)")
    ap.add_argument("--html", action="store_true", help="write check-by-ear cards + clips + summary")
    ap.add_argument("--serve", action="store_true", help="serve the cards with a live 'what I hear' box")
    ap.add_argument("--port", type=int, default=8902, help="port for --serve (default 8902)")
    ap.add_argument("--refresh", action="store_true", help="recompute Whisper features (ignore cache)")
    args = ap.parse_args()

    if args.serve:
        serve(args.port)
        return

    records = gather(refresh=args.refresh)
    print(f"loaded {len(records)} [unintelligible] gaps across "
          f"{len({r['sid'] for r in records})} sessions")
    print_table(records)
    print_score(records)
    if args.html:
        print("\nwriting cards + clips:")
        write_html(records)
        write_summary(records)


if __name__ == "__main__":
    main()
