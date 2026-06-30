"""Curated 'did we drop a real word?' check page (read-only).

A short, hand-picked listening checklist: ~10 segments where the word stream is
missing a small word the sentence still has, deliberately EXCLUDING the
jacked-up segments (Whisper echoes, duplicate loops like "I'm going to take a
few more minutes" x12, single-word loops like "Human Human", non-English
garbage). Each card plays a tiny pre-cut clip (the moment + a lead-in) so a
human can judge by ear: was the struck-through word really spoken (a real loss)
or did Whisper pad the sentence (no loss)?

Why pre-cut clips: the session audio.m4a has its metadata at the end of the
file, which makes browser streaming + seeking flaky. Cutting a ~6s mp3 per card
sidesteps all of that — each clip loads and plays instantly.

Picks are by (session, segment id), verified clean by hand from the
characterization in text_word_check.py. Read-only on session data. Writes:
  emp/results/visuals/word-loss-check.html  + clip-1.mp3 .. clip-N.mp3
(all gitignored — they hold family audio/transcript text).
"""

import difflib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "emp" / "src"))
from api.helpers import get_session_dir  # noqa: E402
from gap_analysis import real_segments, SESSIONS  # noqa: E402

NAMES = {s["id"]: s["name"] for s in SESSIONS}
OUTDIR = ROOT / "emp" / "results" / "visuals"
LEAD, TAIL = 1.5, 0.6  # seconds of audio before / after the segment

# Hand-picked clean candidates: (session_id, segment_id). All normal-duration,
# coherent English, NOT echoes/loops/garbage. Mostly interior function-word
# drops (the clearest test) plus a few edge drops.
PICKS = [
    ("20260304-205354", 32),    # "What did [you] say?"
    ("20260414-213156", 398),   # "But [why] can't it be Soro, Boro?"
    ("20260414-213156", 234),   # "...what do [you] mean?..."
    ("20251207-202105", 291),   # "How do I do [it] that way?"
    ("20260304-205354", 150),   # "...it was going [to be] Casper."
    ("20260211-210718", 171),   # "That's [what] Bheem would say."
    ("20260211-210718", 233),   # "Like, what's [a] war like?"
    ("20251210-203654", 241),   # "I see them bloom for me and [you]."
    ("20260414-213156", 116),   # "Oh, [look], they said."
    ("20260304-205354", 263),   # "[Last] chance for blankets."
]


def rebuilt_text(seg):
    return "".join(w["word"] for w in seg.get("words", []))


def dropped_tokens(text, rebuilt):
    a, b = text.split(), rebuilt.split()
    drops = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op in ("delete", "replace") and i2 > i1:
            drops.extend(a[i1:i2])
    return drops


def diff_html(text, rebuilt):
    a, b = text.split(), rebuilt.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            parts.append(html.escape(" ".join(a[i1:i2])))
        elif i2 > i1:
            parts.append(f'<span class="drop">{html.escape(" ".join(a[i1:i2]))}</span>')
    return " ".join(p for p in parts if p)


_cache = {}


def load(sid):
    if sid not in _cache:
        sdir = get_session_dir(ROOT / "sessions", sid)
        _cache[sid] = (sdir, json.load(open(sdir / "transcript-rich.json")))
    return _cache[sid]


def build():
    cards = []
    for n, (sid, segid) in enumerate(PICKS, 1):
        sdir, t = load(sid)
        seg = next((s for s in t["segments"] if s.get("id") == segid), None)
        if seg is None:
            print(f"WARN: {sid} seg {segid} not found, skipping")
            continue
        reals = sorted(real_segments(t), key=lambda s: s["start"])
        prev_t = next_t = "—"
        for s in reals:
            if s["start"] < seg["start"]:
                prev_t = (s.get("text") or "").strip()
            elif s["start"] > seg["start"] and next_t == "—":
                next_t = (s.get("text") or "").strip()
                break
        rebuilt = rebuilt_text(seg)
        drops = dropped_tokens(seg.get("text") or "", rebuilt)
        cards.append({
            "n": n, "sid": sid, "sdir": sdir, "name": NAMES.get(sid, sid), "id": segid,
            "start": seg.get("start", 0.0), "end": seg.get("end", 0.0),
            "word": " ".join(drops), "sent": diff_html(seg.get("text") or "", rebuilt),
            "prev": prev_t[:80], "next": next_t[:80],
        })
    return cards


def cut_clips(cards):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    for c in cards:
        frm = max(0.0, c["start"] - LEAD)
        dur = (c["end"] - c["start"]) + LEAD + TAIL
        out = OUTDIR / f"clip-{c['n']}.mp3"
        cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(c["sdir"] / "audio.m4a"),
               "-t", f"{dur:.3f}", "-ac", "1", "-ar", "22050",
               "-c:a", "libmp3lame", "-q:a", "6", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0 and out.exists()
        print(f"  clip-{c['n']}.mp3  {'OK' if ok else 'FAIL'}  ({c['name']} seg {c['id']}, "
              f"{frm:.1f}s +{dur:.1f}s)")
        if not ok:
            print(r.stderr[-300:])


def render(cards):
    e = html.escape
    out = ['<!doctype html><html><head><meta charset="utf-8">']
    out.append("<title>Did we drop a real word? — listen check</title><style>")
    out.append("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
               "max-width:720px;margin:36px auto;padding:0 20px;color:#1a1a1a;line-height:1.5}")
    out.append("h1{font-size:23px;margin:0 0 6px}")
    out.append(".intro{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:10px;"
               "padding:16px 18px;font-size:15px;margin:14px 0 10px}")
    out.append("#tally{position:sticky;top:0;background:#fffbe6;border:1px solid #fde68a;"
               "border-radius:8px;padding:8px 14px;font-size:14px;font-weight:600;"
               "margin-bottom:16px;z-index:5}")
    out.append(".card{border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;margin-bottom:14px}")
    out.append(".card.yes{background:#ecfdf5;border-color:#6ee7b7}")
    out.append(".card.no{background:#fef2f2;border-color:#fca5a5}")
    out.append(".row{display:flex;gap:14px;align-items:flex-start}")
    out.append(".play{flex:none;background:#2563eb;color:#fff;border:none;border-radius:9px;"
               "padding:11px 16px;font-size:15px;cursor:pointer}.play:hover{background:#1d4ed8}")
    out.append(".play.playing{background:#16a34a}")
    out.append(".body{flex:1}")
    out.append(".q{font-size:13px;color:#666;margin-bottom:3px}.q b{color:#b91c1c}")
    out.append(".sent{font-size:17px;margin-bottom:6px}")
    out.append(".drop{color:#b91c1c;text-decoration:line-through;font-weight:700}")
    out.append(".ctx{color:#64748b;font-size:13px}")
    out.append(".lbl{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-right:3px}")
    out.append(".mark{flex:none;display:flex;flex-direction:column;gap:6px}")
    out.append(".mark button{border:1px solid #cbd5e1;background:#fff;border-radius:7px;"
               "padding:6px 10px;font-size:13px;cursor:pointer;white-space:nowrap}")
    out.append(".mark button:hover{background:#f1f5f9}")
    out.append(".src{color:#94a3b8;font-size:11px;margin-top:6px}")
    out.append("</style></head><body>")

    out.append("<h1>Did we drop a real word?</h1>")
    out.append('<div class="intro"><b>10 spots to listen to.</b> Each is a normal line where '
               "the timed-word list is missing one small word the sentence still has (shown "
               '<span class="drop">struck through</span>). I left OUT all the Whisper junk — '
               "echoes, repeated-line loops, garbage. <br><br><b>Just listen and decide:</b> "
               "click <b>▶ Play</b> (it starts a moment early so you hear the lead-in). If you "
               "hear the struck-through word actually spoken, it's a <b>real word we dropped</b> "
               "— hit ✓. If it's not really said (Whisper guessed), hit ✗. Then tell me the "
               "count and we'll know how real this is.</div>")
    out.append('<div id="tally">Marked 0 of 10</div>')

    for c in cards:
        word = c["word"] or "the missing word"
        out.append('<div class="card">')
        out.append('<div class="row">')
        out.append(f'<button class="play" onclick="play(this)">▶ Play</button>')
        out.append('<div class="body">')
        out.append(f'<div class="q">#{c["n"]} · did they actually say <b>"{e(word)}"</b>?</div>')
        out.append(f'<div class="sent">{c["sent"]}</div>')
        out.append(f'<div class="ctx"><span class="lbl">before</span>"{e(c["prev"])}" '
                   f'<span class="lbl">after</span>"{e(c["next"])}"</div>')
        out.append(f'<div class="src">{e(c["name"])} · seg {c["id"]} · {c["start"]:.1f}s</div>')
        out.append('</div>')
        out.append('<div class="mark">'
                   '<button onclick="mark(this,1)">✓ real</button>'
                   '<button onclick="mark(this,0)">✗ not said</button></div>')
        out.append('</div>')
        out.append(f'<audio src="clip-{c["n"]}.mp3" preload="none"></audio>')
        out.append('</div>')

    out.append("""<script>
let playing=null, playingBtn=null;
function play(btn){
  const aud=btn.closest('.card').querySelector('audio');
  if(playing && playing!==aud){ playing.pause(); }
  if(playingBtn && playingBtn!==btn){ playingBtn.classList.remove('playing'); playingBtn.textContent='▶ Play'; }
  playing=aud; playingBtn=btn;
  aud.currentTime=0;
  const p=aud.play(); if(p&&p.catch)p.catch(()=>{});
  btn.classList.add('playing'); btn.textContent='♪ Playing';
  aud.onended=()=>{ btn.classList.remove('playing'); btn.textContent='▶ Replay'; };
}
function mark(btn,val){
  const card=btn.closest('.card');
  card.classList.remove('yes','no');
  card.classList.add(val? 'yes':'no');
  const y=document.querySelectorAll('.card.yes').length;
  const n=document.querySelectorAll('.card.no').length;
  document.getElementById('tally').textContent =
    `Marked ${y+n} of 10  —  ${y} real word dropped, ${n} not actually said`;
}
</script>""")
    out.append("</body></html>")
    return "\n".join(out)


def main():
    cards = build()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Cutting clips:")
    cut_clips(cards)
    outpath = OUTDIR / "word-loss-check.html"
    outpath.write_text(render(cards))
    print(f"\nWrote page: {outpath.relative_to(ROOT)} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
