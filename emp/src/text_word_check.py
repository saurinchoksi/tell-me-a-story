"""Segment text vs word-stream consistency sweep (read-only).

Each transcript segment carries two parallel renderings of the same speech: a
`text` string (Whisper's original segment line) and a `words` list (the
word-level tokens, after cleaning + name normalization). They are supposed to
say the same thing. `corrections.py` now heals `text` from `words` at
correction time, but sessions processed before that fix can carry a stale
`text`. This sweep finds every segment where `text` != join(`words`) across the
coded sessions and sorts the divergence into three kinds, so a human can decide
what to do with each:

  - DROPPED (text has words the stream doesn't)  -> needs the ear.
        Almost always a Whisper echo/duplicate: a coherent `text` whose words
        got crushed into a sub-segment with the middle dropped (e.g. text
        "Can you believe that?" but words "Can that?" in 0.14s, right after a
        clean "Can you believe it?"). Healing text->words would just swap one
        ghost spelling for another; dropping the whole echo may be the right
        move. Only an ear can tell a real-but-mistimed phrase from an echo, so
        these get check-by-ear cards.

  - NAME (same word count, a token differs) -> safe to heal, no ear needed.
        The name-normalization case the EMP flagged: `words` were corrected
        (fondos -> Pandavas) but the old `text` kept the mishearing. Here the
        word stream is the MORE correct one, so rebuilding text from words is
        unambiguously right. Listed for completeness, not for review.

  - ADDED (stream has words text doesn't) -> rare; surfaced in the summary.

Cross-signal, per the EMP sweep recipe: transcript text x word stream, plus the
human's existing `axial-labels.json` codes so a segment another failure mode
already owns (a hallucination/echo coded M2, M6, M10...) is split into a
greyed "already coded" section, not re-counted as open work.

READ-ONLY. Never writes session JSON. Outputs:
  - per-session check page emp/results/visuals/<id>/text-word-check.html
    (gitignored — holds transcript text), audio wired to the real sessions/
    audio.m4a via a correct relative path so the Play buttons work as file://
  - a committable aggregate summary emp/results/text-word-check/summary.md
    (counts only, no transcript text), echoed to stdout.

Examples:
    python emp/src/text_word_check.py            # all coded sessions
    python emp/src/text_word_check.py --session 20260117-202237
"""

import argparse
import difflib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from api.helpers import get_session_dir  # noqa: E402
from gap_analysis import mmss, real_segments, SESSIONS  # noqa: E402

NAMES = {s["id"]: s["name"] for s in SESSIONS}

# A kept word stream this dense is physically impossible for real speech
# (~2-4 words/sec is normal) — a strong "this is an echo/ghost" pre-sort hint.
CRAMMED_WPS = 6.0
CRAMMED_DUR = 0.30


# --- helpers -----------------------------------------------------------------
def rebuilt_text(seg):
    """The text as it would read if rebuilt from the word stream (the fix)."""
    return "".join(w["word"] for w in seg.get("words", []))


def classify(seg):
    """None if consistent, else 'dropped' | 'name' | 'added'."""
    text = seg.get("text") or ""
    rebuilt = rebuilt_text(seg)
    if text == rebuilt:
        return None
    nt, nw = len(text.split()), len(rebuilt.split())
    if nt > nw:
        return "dropped"
    if nt == nw:
        return "name"
    return "added"


def load_axial(sdir):
    """segmentId -> [codes]. Keys are stored as both int and str for lookup."""
    p = sdir / "axial-labels.json"
    if not p.exists():
        return {}
    out = {}
    for lab in json.load(open(p)).get("labels", []):
        sid = lab.get("segmentId")
        codes = lab.get("codes", [])
        out[sid] = codes
        out[str(sid)] = codes
    return out


def diff_html(text, rebuilt):
    """Token diff: dropped words struck red, stream-only words green."""
    a, b = text.split(), rebuilt.split()
    sm = difflib.SequenceMatcher(None, a, b)
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(html.escape(" ".join(a[i1:i2])))
        else:
            if i2 > i1:
                parts.append(f'<span class="drop">{html.escape(" ".join(a[i1:i2]))}</span>')
            if j2 > j1:
                parts.append(f'<span class="ins">{html.escape(" ".join(b[j1:j2]))}</span>')
    return " ".join(p for p in parts if p)


def analyze(session_id):
    sdir = get_session_dir(ROOT / "sessions", session_id)
    transcript = json.load(open(sdir / "transcript-rich.json"))
    axial = load_axial(sdir)
    reals = sorted(real_segments(transcript), key=lambda s: s["start"])

    def context(seg):
        prev_t = next_t = "—"
        for s in reals:
            if s["start"] < seg["start"]:
                prev_t = (s.get("text") or "").strip()
            elif s["start"] > seg["start"] and next_t == "—":
                next_t = (s.get("text") or "").strip()
                break
        return prev_t, next_t

    findings = []
    for seg in transcript["segments"]:
        if not seg.get("words"):  # injected [unintelligible] gaps keep their text
            continue
        kind = classify(seg)
        if kind is None:
            continue
        start, end = seg.get("start", 0.0), seg.get("end", 0.0)
        dur = max(0.0, end - start)
        rebuilt = rebuilt_text(seg)
        nwords = len(rebuilt.split())
        wps = nwords / dur if dur > 0 else 999.0
        sid = seg.get("id")
        codes = axial.get(sid) or axial.get(str(sid)) or []
        prev_t, next_t = context(seg)
        findings.append({
            "id": sid, "start": start, "end": end, "dur": dur,
            "kind": kind, "text": seg.get("text") or "", "rebuilt": rebuilt,
            "nwords": nwords, "wps": wps,
            "crammed": dur < CRAMMED_DUR or wps > CRAMMED_WPS,
            "codes": codes, "prev": prev_t, "next": next_t,
        })
    return sdir, findings


# --- rendering ---------------------------------------------------------------
def audio_src(session_id):
    """Relative path from emp/results/visuals/<id>/ back to the real audio."""
    return f"../../../../sessions/{session_id}/audio.m4a"


def render_check_html(session_id, name, findings):
    e = html.escape
    dropped = [f for f in findings if f["kind"] == "dropped"]
    to_check = [f for f in dropped if not f["codes"]]
    coded = [f for f in dropped if f["codes"]]
    names = [f for f in findings if f["kind"] == "name"]
    added = [f for f in findings if f["kind"] == "added"]

    out = ['<!doctype html><html><head><meta charset="utf-8">']
    out.append(f"<title>{e(name)} — text vs words to check</title><style>")
    out.append("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
               "max-width:780px;margin:40px auto;padding:0 20px;color:#1a1a1a;line-height:1.5}")
    out.append("h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 10px}")
    out.append(".sub{color:#666;font-size:14px;margin-bottom:20px}")
    out.append(".explain{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:10px;"
               "padding:16px 18px;font-size:14px;margin-bottom:20px}")
    out.append(".card{border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;"
               "margin-bottom:12px;display:flex;gap:14px;align-items:flex-start}")
    out.append(".play{flex:none;background:#2563eb;color:#fff;border:none;border-radius:8px;"
               "padding:10px 14px;font-size:15px;cursor:pointer}.play:hover{background:#1d4ed8}")
    out.append(".meta{font-weight:600;font-size:14px}")
    out.append(".diff{font-size:15px;margin:6px 0}")
    out.append(".drop{color:#b91c1c;text-decoration:line-through;font-weight:600}")
    out.append(".ins{color:#15803d;font-weight:600}")
    out.append(".ctx{color:#555;font-size:13px;margin-top:6px}")
    out.append(".lbl{color:#99a;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-right:4px}")
    out.append(".tag{display:inline-block;font-size:11px;background:#fde68a;color:#92400e;"
               "border-radius:5px;padding:1px 7px;margin-left:6px}")
    out.append(".coded{opacity:.6;background:#fafafa}")
    out.append(".code{display:inline-block;font-size:11px;background:#e0e7ff;color:#3730a3;"
               "border-radius:5px;padding:1px 7px;margin-left:6px}")
    out.append("audio{width:100%;margin:6px 0 14px}")
    out.append("table{border-collapse:collapse;font-size:13px;margin-top:8px}"
               "td,th{border:1px solid #e2e8f0;padding:5px 9px;text-align:left}")
    out.append(".note{color:#888;font-size:13px;margin-top:24px}")
    out.append("</style></head><body>")

    out.append(f"<h1>{e(name)}: text vs words to check</h1>")
    out.append(f'<div class="sub">Session {e(session_id)}</div>')
    out.append('<div class="explain"><b>What this is.</b> Each segment stores the spoken '
               "line two ways — a <b>text</b> line and the underlying <b>word</b> tokens. "
               "These cards are segments where they disagree because the word stream is "
               "<b>missing words the text still has</b>. Almost always this is a Whisper "
               "<b>echo</b>: a tidy text line whose words got crushed into a fraction of a "
               "second right after the real line. <br><br><b>How to judge one.</b> Click "
               "<b>▶ Play</b> (it starts ~1.5s early, so you hear the line before it too). "
               "If you hear the words said for real, the word stream is wrong (a real bug). "
               "If it's silence, noise, or just an echo of the previous line, this segment "
               "is a ghost and should be dropped, not healed. The "
               '<span class="tag">⚠ crammed</span> tag marks segments far too fast to be '
               "real speech — strong echo suspects.</div>")

    out.append(f'<audio id="aud" controls src="{audio_src(session_id)}" preload="none"></audio>')

    out.append(f"<h2>To check by ear — {len(to_check)} dropped-word segments</h2>")
    if not to_check:
        out.append('<div class="sub">None open.</div>')
    for f in to_check:
        out.append('<div class="card">')
        out.append(f'<button class="play" onclick="play({f["start"]:.2f},{f["end"]:.2f})">▶ Play</button><div>')
        tag = '<span class="tag">⚠ crammed</span>' if f["crammed"] else ""
        out.append(f'<div class="meta">seg {e(str(f["id"]))} · {mmss(f["start"])}–{mmss(f["end"])} · '
                   f'{f["dur"]:.2f}s · {f["nwords"]} words kept{tag}</div>')
        out.append(f'<div class="diff">{diff_html(f["text"], f["rebuilt"])}</div>')
        out.append(f'<div class="ctx"><span class="lbl">before</span>"{e(f["prev"][:90])}" '
                   f'<span class="lbl">after</span>"{e(f["next"][:90])}"</div>')
        out.append("</div></div>")

    if coded:
        out.append(f"<h2>Already coded — skip ({len(coded)})</h2>")
        out.append('<div class="sub">These dropped-word segments are already labelled by a '
                   "failure mode, so they're owned — no action here.</div>")
        for f in coded:
            codes = "".join(f'<span class="code">{e(c)}</span>' for c in f["codes"])
            out.append('<div class="card coded">')
            out.append(f'<button class="play" onclick="play({f["start"]:.2f},{f["end"]:.2f})">▶ Play</button><div>')
            out.append(f'<div class="meta">seg {e(str(f["id"]))} · {mmss(f["start"])}–{mmss(f["end"])} · '
                       f'{f["dur"]:.2f}s{codes}</div>')
            out.append(f'<div class="diff">{diff_html(f["text"], f["rebuilt"])}</div>')
            out.append("</div></div>")

    if names:
        out.append(f"<h2>Name corrections — safe to heal, no ear needed ({len(names)})</h2>")
        out.append('<div class="sub">The word stream is the corrected one; rebuilding text '
                   "from it is unambiguously right.</div>")
        out.append("<table><tr><th>seg</th><th>text (stale)</th><th>words (correct)</th></tr>")
        for f in names:
            out.append(f"<tr><td>{e(str(f['id']))}</td><td>{e(f['text'].strip()[:70])}</td>"
                       f"<td>{e(f['rebuilt'].strip()[:70])}</td></tr>")
        out.append("</table>")

    if added:
        out.append(f"<h2>Stream has extra words ({len(added)})</h2>")
        out.append("<table><tr><th>seg</th><th>text</th><th>words</th></tr>")
        for f in added:
            out.append(f"<tr><td>{e(str(f['id']))}</td><td>{e(f['text'].strip()[:70])}</td>"
                       f"<td>{e(f['rebuilt'].strip()[:70])}</td></tr>")
        out.append("</table>")

    out.append('<div class="note">Read-only view. Audio is the real session file via a '
               "relative path; nothing here modifies the transcript.</div>")
    out.append('<script>const aud=document.getElementById("aud");let timer=null;'
               'function play(s,e){if(timer)clearTimeout(timer);'
               'var from=Math.max(0,s-1.5);aud.currentTime=from;aud.play();'
               'timer=setTimeout(function(){aud.pause();},(e-from+0.8)*1000);}</script>')
    out.append("</body></html>")
    return "\n".join(out)


def build_summary(rows):
    lines = ["# Text vs word-stream consistency sweep", ""]
    lines.append("Segments where a segment's `text` line disagrees with its `words` stream, "
                 "across the coded sessions. **dropped** = text has words the stream lost "
                 "(mostly Whisper echoes — needs the ear); **name** = a token differs at equal "
                 "length (name normalization not propagated to text — safe to heal from words); "
                 "**added** = stream has extra words (rare). `coded` = a dropped segment a human "
                 "already labelled (owned by that mode). Generated by "
                 "`emp/src/text_word_check.py` (read-only). Per-session check pages with audio: "
                 "`emp/results/visuals/<id>/text-word-check.html` (local only — transcript text).")
    lines.append("")
    lines.append("| Session | dropped (to check) | dropped (already coded) | name (safe heal) | added | total stale |")
    lines.append("|---|---|---|---|---|---|")
    tot = [0, 0, 0, 0, 0]
    for r in rows:
        tot = [tot[i] + r[i + 1] for i in range(5)]
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    lines.append(f"| **TOTAL** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{tot[3]}** | **{tot[4]}** |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- The **name** column is safe to heal mechanically (rebuild `text` from `words`); "
                 "the fix already lives in `corrections.py` for new sessions.")
    lines.append("- The **dropped** column is the ear question: real-but-mistimed speech (a word-stream "
                 "bug) vs a Whisper echo (drop the segment). The check pages exist to settle each.")
    lines.append("- Counts are a floor over the coded sessions only; read-only, no transcript text here.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Text vs word-stream consistency sweep (read-only).")
    p.add_argument("--session", help="One session id (default: all coded sessions).")
    args = p.parse_args()

    if args.session:
        ids = [args.session]
    else:
        ids = sorted(d.name for d in (ROOT / "sessions").iterdir()
                     if (d / "axial-labels.json").exists())

    rows = []
    visroot = ROOT / "emp" / "results" / "visuals"
    for sid in ids:
        name = NAMES.get(sid, sid)
        sdir, findings = analyze(sid)
        dropped = [f for f in findings if f["kind"] == "dropped"]
        to_check = sum(1 for f in dropped if not f["codes"])
        coded = sum(1 for f in dropped if f["codes"])
        nname = sum(1 for f in findings if f["kind"] == "name")
        nadded = sum(1 for f in findings if f["kind"] == "added")
        outdir = visroot / sid
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "text-word-check.html"
        outpath.write_text(render_check_html(sid, name, findings))
        rows.append([name, to_check, coded, nname, nadded, len(findings)])
        print(f"{name} ({sid}): {to_check} to-check, {coded} coded, {nname} name, "
              f"{nadded} added · {len(findings)} stale total")
        print(f"   check page: {outpath.relative_to(ROOT)}")

    if not args.session:
        sumpath = ROOT / "emp" / "results" / "text-word-check" / "summary.md"
        sumpath.parent.mkdir(parents=True, exist_ok=True)
        sumpath.write_text(build_summary(rows))
        print(f"\nWrote summary: {sumpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
