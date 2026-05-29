"""Find missed speech that left no trace, by comparing diarization vs transcript.

When someone is talking but Whisper writes nothing, the transcript has no
segment for that stretch — so a coder reviewing segment-by-segment never sees
it. This read-only sweep surfaces those moments: for every diarization segment
(who was talking) it subtracts the transcript-segment spans (what got
transcribed) and reports the leftover slivers as missed-speech gaps.

Coverage is computed at SEGMENT granularity (a whole segment's [start, end]
counts as covered) and includes BOTH transcribed-word segments AND pipeline-
injected "[unintelligible]" placeholders (_source == "diarization_gap"). That
is the exact thing being measured: speech with NO trace at all — no
transcribed line, and not even an [unintelligible] marker. An [unintelligible]
segment is a visible, reviewable trace (a clickable line in the validator), so
a region the pipeline already flagged that way is NOT a miss here; sorting those
segments into real-speech-vs-not is a separate job (TMAS-46).

This reconstructs and generalizes the original one-off Moon Story analysis
(sessions/20251207-195607/gap-analysis.html) — and refines it: the original
EXCLUDED [unintelligible] segments from coverage, so it re-flagged those
already-visible regions as gaps; counting them as coverage scopes the result to
true no-trace missed speech. It deliberately does NOT use
speaker.py:detect_unintelligible_gaps —
that filters out monologue pauses, which would under-count the floor.

For each session it writes emp/results/visuals/<id>/gap-analysis.html (three aligned
strips — transcript / diarization / gaps — plus a per-gap table). It also
prints, and writes to --summary-out, a cross-session table of gap counts and
missed seconds at two thresholds (>=0.3s candidate, >=1.0s high-confidence),
split story-scope vs whole-session.

Read-only: never modifies session JSON. The per-session HTML lives under the
gitignored sessions/ tree (it contains transcript text); the summary holds only
aggregate numbers and is safe to commit.

Examples:
    python emp/src/gap_analysis.py                 # 4 new sessions + validate Moon Story
    python emp/src/gap_analysis.py --session 20260129-204404 --story-end seg:594
    python emp/src/gap_analysis.py --session 20251210-203654 --story-end 9:37.12
"""

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _visdir(session_dir):  # per-session visuals -> emp/results/visuals/<id>/
    d = ROOT / "emp" / "results" / "visuals" / session_dir.name
    d.mkdir(parents=True, exist_ok=True)
    return d
sys.path.insert(0, str(ROOT))

from api.helpers import get_session_dir  # noqa: E402

# --- rendering constants (match the original Moon Story analysis) ------------
X_OFFSET = 110.0          # left margin before t=0, in px
PX_PER_SEC = 2.6          # horizontal scale
GRID_SEC = 30             # gridline spacing
SPEAKER_COLORS = ["#2196f3", "#ff9800", "#9c27b0", "#00bcd4",
                  "#8bc34a", "#e91e63", "#ffc107", "#795548"]
TRANSCRIPT_COLOR = "#4caf50"
UNINTEL_COLOR = "#9e9e9e"   # [unintelligible] placeholders — count as coverage
GAP_COLOR = "#f44336"

DEFAULT_MIN_GAP = 0.3
DEFAULT_HC = 1.0

# The five EMP counting-sample sessions. story_end_seg is the id of the last
# in-scope (story) segment; the wind-down after it is out of scope. is_baseline
# marks Moon Story, which is computed for validation but NOT re-written.
SESSIONS = [
    # Baseline expectations are pinned to the CURRENT transcript-rich.json under
    # the strict missed-speech rule ([unintelligible] counts as coverage). The >=1.0s
    # floor (4 / 9.7s) is the stable anchor — unchanged across the published
    # hand-made file, the broad reconstruction, and strict missed speech (no
    # [unintelligible] overlap lands above 1.0s in Moon Story). At >=0.3s strict
    # gives 65 / 38.4s; the original hand-made HTML showed 65 / 38.8s (broad:
    # it counted one 0.78s [unintelligible] region, but missed a 0.42s sliver
    # that a later seg-51 re-enrichment drift later exposed — the two offsets
    # happen to land on the same gap count).
    {"name": "Moon Story", "id": "20251207-195607", "story_end_seg": 152,
     "is_baseline": True,
     "expected": {"story_gaps_03": 65, "story_sec_03": 38.4,
                  "story_gaps_10": 4, "story_sec_10": 9.7}},
    {"name": "Cruel Baby", "id": "20251207-202105", "story_end_seg": 278},
    {"name": "Rubber Ducky", "id": "20251210-203654", "story_end_seg": 236},
    {"name": "Pandavas", "id": "20260117-202237", "story_end_seg": None,
     "no_boundary": True},
    {"name": "Portal Story", "id": "20260129-204404", "story_end_seg": 594},
]


# --- pure interval helpers ---------------------------------------------------
def merge_intervals(intervals):
    """Merge a list of (start, end) into sorted, non-overlapping spans."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return []
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def subtract_intervals(seg, covered):
    """Return the parts of seg=(start, end) not inside any covered span.

    covered must be a merged, sorted list (output of merge_intervals).
    """
    ds, de = seg
    out = []
    cur = ds
    for cs, ce in covered:
        if ce <= cur:
            continue
        if cs >= de:
            break
        if cs > cur:
            out.append((cur, min(cs, de)))
        cur = max(cur, ce)
        if cur >= de:
            break
    if cur < de:
        out.append((cur, de))
    return out


def mmss(t):
    """Format seconds as m:ss.cc (e.g. 6:34.14)."""
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:05.2f}"


def mmss_grid(t):
    """Format seconds as m:ss for axis labels (e.g. 6:30)."""
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def speaker_label(spk):
    """'SPEAKER_03' -> 'Speaker 3' for the plain view; pass through otherwise."""
    if spk and spk.startswith("SPEAKER_"):
        tail = spk.split("_", 1)[1]
        if tail.isdigit():
            return f"Speaker {int(tail)}"
    return spk or "?"


# --- data loading ------------------------------------------------------------
def load_session(session_id):
    sdir = get_session_dir(ROOT / "sessions", session_id)
    with open(sdir / "transcript-rich.json") as f:
        transcript = json.load(f)
    with open(sdir / "diarization.json") as f:
        diarization = json.load(f)
    return sdir, transcript, diarization


def real_segments(transcript):
    """Segments with actual transcribed words (excludes [unintelligible]).

    Used for the readable before/after context and the green transcript strip —
    NOT for coverage. See covering_segments for the coverage rule.
    """
    return [s for s in transcript["segments"]
            if s.get("_source") != "diarization_gap" and s["end"] > s["start"]]


def unintelligible_segments(transcript):
    """Pipeline-injected [unintelligible] placeholder segments."""
    return [s for s in transcript["segments"]
            if s.get("_source") == "diarization_gap" and s["end"] > s["start"]]


def covering_segments(transcript):
    """Every segment that means 'a trace exists here' — both transcribed-word
    segments AND [unintelligible] placeholders.

    This is the coverage rule. A missed-speech gap is speech with NO
    segment at all: not a transcribed line, and not even an [unintelligible]
    marker. The reason [unintelligible] counts as coverage: it is a *visible,
    reviewable* trace — a clickable line in the validator that says "someone
    spoke here, couldn't decode it." This is specifically the failure that is
    invisible to review, so a region the pipeline already flagged
    [unintelligible] is NOT a miss here. Deciding whether those [unintelligible]
    segments hold real speech (vs noise) is a separate job — TMAS-46.
    """
    return [s for s in transcript["segments"] if s["end"] > s["start"]]


def seg_end_time(transcript, seg_id):
    for s in transcript["segments"]:
        if s.get("id") == seg_id:
            return s["end"]
    raise ValueError(f"segment id {seg_id!r} not found")


# --- gap detection -----------------------------------------------------------
def find_gaps(diarization, transcript, min_gap):
    """Find missed-speech gaps: where the detector heard a voice but NO segment exists.

    Coverage = every segment (transcribed words AND [unintelligible] markers),
    per covering_segments. A gap is an uncovered sliver of a diarization
    segment, >= min_gap seconds, tagged with that segment's speaker. So a gap
    means speech with no trace at all — not a transcribed line, not even an
    [unintelligible] placeholder; that is exactly the no-trace missed speech we want.

    Faithful to the original sweep in every other respect: per-diarization-
    segment subtraction, so a rare same-instant overlap from two speakers would
    be listed twice.
    """
    covered = merge_intervals(
        [(s["start"], s["end"]) for s in covering_segments(transcript)]
    )
    segs_by_start = sorted(real_segments(transcript), key=lambda s: s["start"])

    gaps = []
    for d in diarization.get("segments", []):
        ds, de, spk = d["start"], d["end"], d["speaker"]
        if de - ds <= 0:
            continue
        for gs, ge in subtract_intervals((ds, de), covered):
            if ge - gs >= min_gap:
                gaps.append({"start": gs, "end": ge, "dur": ge - gs,
                             "speaker": spk})
    gaps.sort(key=lambda g: g["start"])

    # Attach nearest preceding / following transcript segment for the table.
    for g in gaps:
        prev_seg = None
        next_seg = None
        for s in segs_by_start:
            if s["start"] < g["start"]:
                prev_seg = s
            elif s["start"] >= g["end"] and next_seg is None:
                next_seg = s
                break
        g["prev"] = prev_seg
        g["next"] = next_seg
    return gaps


def tally(gaps, threshold, story_end):
    """Counts and missed-seconds at a duration threshold.

    Returns dict with story-scope (start < story_end) and whole-session totals.
    If story_end is None, story == whole.
    """
    keep = [g for g in gaps if g["dur"] >= threshold]
    whole_n = len(keep)
    whole_s = sum(g["dur"] for g in keep)
    if story_end is None:
        return {"story_n": whole_n, "story_s": whole_s,
                "whole_n": whole_n, "whole_s": whole_s}
    story = [g for g in keep if g["start"] < story_end]
    return {"story_n": len(story), "story_s": sum(g["dur"] for g in story),
            "whole_n": whole_n, "whole_s": whole_s}


# --- HTML rendering ----------------------------------------------------------
def _x(t):
    return X_OFFSET + t * PX_PER_SEC


def _bar(t_start, t_end):
    """(x, width) for a time span, with a 0.8px minimum width like the original."""
    return _x(t_start), max(0.8, (t_end - t_start) * PX_PER_SEC)


def render_html(session, transcript, diarization, gaps, story_end, min_gap):
    segs = real_segments(transcript)
    unintel = unintelligible_segments(transcript)
    diar = diarization.get("segments", [])
    total_dur = max(
        [s["end"] for s in transcript["segments"]]
        + [d["end"] for d in diar]
    )

    speakers = sorted({d["speaker"] for d in diar})
    color = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
             for i, spk in enumerate(speakers)}

    width = int(_x(total_dur) + 40)
    e = html.escape

    out = []
    out.append('<!doctype html><html><head><meta charset="utf-8">')
    out.append(f"<title>{e(session['name'])} — Gap Analysis</title><style>")
    out.append("body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;padding:24px;margin:0}")
    out.append("h1{font-size:18px;margin:0 0 6px}h2{font-size:14px;margin:24px 0 8px;color:#bbb}")
    out.append(".legend span{display:inline-block;padding:2px 10px;margin-right:6px;font-size:11px;border-radius:3px}")
    out.append(".scroll{overflow-x:auto;border:1px solid #222;background:#0a0a0a;margin:14px 0}")
    out.append("svg{display:block}")
    out.append("table{border-collapse:collapse;font-size:12px}td,th{padding:5px 9px;border:1px solid #2a2a2a;text-align:left}")
    out.append("th{background:#1a1a1a;color:#aaa}.story{background:#2a1010}.tail{color:#777}")
    out.append("rect:hover{stroke:#fff;stroke-width:1}")
    out.append("</style></head><body>")
    out.append(f"<h1>{e(session['name'])} — Tier 1 Gap Analysis (diarization vs transcript)</h1>")

    # header summary line
    counted = [g for g in gaps if story_end is None or g["start"] < story_end]
    counted_s = sum(g["dur"] for g in counted)
    if session.get("no_boundary"):
        out.append(
            f'<div style="color:#999;font-size:12px">Session {e(session["id"])} · '
            f'<b style="color:#fa0">NO wind-down boundary marked</b> — whole-session counts '
            f'(may include a lullaby tail) · min gap {min_gap}s · '
            f'<b style="color:#f55">{len(counted)} gaps, {counted_s:.1f}s missed</b> · '
            f'{len(unintel)} [unintelligible] placeholders (coverage, not missed speech — TMAS-46)</div>'
        )
    else:
        tail = [g for g in gaps if g["start"] >= story_end]
        tail_s = sum(g["dur"] for g in tail)
        out.append(
            f'<div style="color:#999;font-size:12px">Session {e(session["id"])} · '
            f'story 0–{mmss(story_end)} · min gap {min_gap}s · '
            f'<b style="color:#f55">{len(counted)} in-story gaps, {counted_s:.1f}s missed</b> · '
            f'{len(tail)} tail gaps, {tail_s:.1f}s (out of scope) · '
            f'{len(unintel)} [unintelligible] placeholders (coverage, not missed speech — TMAS-46)</div>'
        )

    # logic note — what a gap means, in plain terms
    out.append(
        '<div style="color:#888;font-size:12px;margin-top:8px;max-width:900px">'
        'A <b style="color:#f77">MISSED-SPEECH GAP</b> is where the speaker detector heard a voice but '
        '<b>no segment of any kind exists</b> — no transcribed line, and not even an '
        '[unintelligible] marker. The green (transcript) and gray ([unintelligible]) bars below '
        '<b>both count as coverage</b>; gaps fall only in the bare spots between them. An '
        '[unintelligible] segment is a visible, reviewable line, so it is not a miss here — '
        'classifying those is a separate task (TMAS-46).</div>'
    )

    # legend
    out.append('<div class="legend" style="margin-top:14px">')
    out.append(f'<span style="background:{TRANSCRIPT_COLOR};color:#000">transcript</span>')
    out.append(f'<span style="background:{UNINTEL_COLOR};color:#000">[unintelligible]</span>')
    for spk in speakers:
        out.append(f'<span style="background:{color[spk]};color:#fff">{e(spk)}</span>')
    out.append(f'<span style="background:{GAP_COLOR};color:#fff">missed speech</span>')
    if story_end is not None:
        out.append('<span style="background:#fff;color:#000">story end ↓</span>')
    out.append("</div>")

    # SVG (wrapped in a horizontal-scroll container for long sessions)
    out.append('<div class="scroll">')
    out.append(f'<svg width="{width}" height="200">')
    t = 0
    while t <= total_dur:
        x = _x(t)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="200" stroke="#1d1d1d"/>')
        out.append(f'<text x="{x + 2:.1f}" y="11" fill="#555" font-size="9">{mmss_grid(t)}</text>')
        t += GRID_SEC
    if story_end is not None:
        sx = _x(story_end)
        out.append(f'<line x1="{sx:.3f}" y1="18" x2="{sx:.3f}" y2="200" '
                   f'stroke="#fff" stroke-dasharray="3,3" opacity="0.6"/>')

    # strip 1: transcript coverage — green (transcribed words) + gray ([unintelligible])
    out.append('<text x="6" y="43.0" fill="#bbb" font-size="11">transcript</text>')
    for s in segs:
        x, w = _bar(s["start"], s["end"])
        txt = e((s.get("text") or "")[:60])
        out.append(
            f'<rect x="{x:.1f}" y="24" width="{w:.1f}" height="30" fill="{TRANSCRIPT_COLOR}" '
            f'opacity="0.75"><title>seg {e(str(s.get("id")))}: '
            f'{mmss(s["start"])}-{mmss(s["end"])} · {txt}</title></rect>'
        )
    for s in unintel:
        x, w = _bar(s["start"], s["end"])
        out.append(
            f'<rect x="{x:.1f}" y="24" width="{w:.1f}" height="30" fill="{UNINTEL_COLOR}" '
            f'opacity="0.85"><title>{mmss(s["start"])}-{mmss(s["end"])} · [unintelligible] '
            f'(counts as coverage, not a missed-speech gap)</title></rect>'
        )

    # strip 2: diarization
    out.append('<text x="6" y="83.0" fill="#bbb" font-size="11">diarization</text>')
    for d in diar:
        x, w = _bar(d["start"], d["end"])
        out.append(
            f'<rect x="{x:.1f}" y="64" width="{w:.1f}" height="30" fill="{color[d["speaker"]]}" '
            f'opacity="0.75"><title>{e(d["speaker"])} · {mmss(d["start"])}-{mmss(d["end"])} '
            f'({d["end"] - d["start"]:.2f}s)</title></rect>'
        )

    # strip 3: gaps
    out.append('<text x="6" y="123.0" fill="#bbb" font-size="11">missed speech</text>')
    for g in gaps:
        x, w = _bar(g["start"], g["end"])
        out.append(
            f'<rect x="{x:.1f}" y="104" width="{w:.1f}" height="30" fill="{GAP_COLOR}" '
            f'stroke="#ff0" stroke-width="0.5"><title>GAP {mmss(g["start"])}-{mmss(g["end"])} '
            f'({g["dur"]:.2f}s) · {e(g["speaker"])}</title></rect>'
        )
    out.append("</svg></div>")

    # table
    scope_label = "SESSION" if session.get("no_boundary") else None
    n_story = sum(1 for g in gaps if story_end is None or g["start"] < story_end)
    out.append(f"<h2>Flagged gaps — {len(gaps)} total ({n_story} counted)</h2>")
    out.append("<table><tr><th>#</th><th>start</th><th>end</th><th>dur</th>"
               "<th>speaker</th><th>scope</th><th>prev seg</th><th>next seg</th></tr>")
    for i, g in enumerate(gaps, 1):
        if scope_label:
            scope, cls = "SESSION", "story"
        elif g["start"] < story_end:
            scope, cls = "STORY", "story"
        else:
            scope, cls = "tail", "tail"

        def seg_label(s):
            if not s:
                return ""
            return e(f"[{s.get('id')}] {(s.get('text') or '')[:54]}")

        out.append(
            f'<tr class="{cls}"><td>{i}</td><td>{mmss(g["start"])}</td>'
            f'<td>{mmss(g["end"])}</td><td>{g["dur"]:.2f}s</td>'
            f'<td>{e(g["speaker"])}</td><td>{scope}</td>'
            f'<td>{seg_label(g["prev"])}</td><td>{seg_label(g["next"])}</td></tr>'
        )
    out.append("</table></body></html>")
    return "\n".join(out)


def render_simple_html(session, transcript, gaps, story_end, hc):
    """A plain, human-checkable view of the high-confidence gaps.

    One card per gap, each with a Play button that seeks the audio to a second
    before the gap and plays it — so a reviewer can confirm by ear that a voice
    really is there. Far simpler than the full timeline view.
    """
    e = html.escape

    def lab(t):
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    in_scope = [g for g in gaps
                if g["dur"] >= hc and (story_end is None or g["start"] < story_end)]
    total_s = sum(g["dur"] for g in in_scope)

    out = []
    out.append('<!doctype html><html><head><meta charset="utf-8">')
    out.append(f"<title>{e(session['name'])} — gaps to check</title><style>")
    out.append("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
               "max-width:760px;margin:40px auto;padding:0 20px;color:#1a1a1a;line-height:1.5}")
    out.append("h1{font-size:22px;margin:0 0 4px}.sub{color:#666;font-size:14px;margin-bottom:22px}")
    out.append(".explain{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:10px;"
               "padding:16px 18px;font-size:15px;margin-bottom:22px}")
    out.append(".count{font-size:15px;color:#444;margin:0 0 16px}")
    out.append(".gap{border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin-bottom:12px;"
               "display:flex;gap:14px;align-items:flex-start}")
    out.append(".play{flex:none;background:#2563eb;color:#fff;border:none;border-radius:8px;"
               "padding:10px 14px;font-size:15px;cursor:pointer}.play:hover{background:#1d4ed8}")
    out.append(".meta{font-weight:600;font-size:15px}.ctx{color:#555;font-size:14px;margin-top:6px}")
    out.append(".lbl{color:#99a;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:0 4px}")
    out.append(".blank{color:#c026d3;font-weight:600}")
    out.append("audio{width:100%;margin-bottom:6px}.note{color:#888;font-size:13px;margin-top:24px}")
    out.append("</style></head><body>")

    out.append(f"<h1>{e(session['name'])}: moments to check</h1>")
    scope_txt = ("whole session — no wind-down boundary marked"
                 if story_end is None else f"story portion only (up to {lab(story_end)})")
    out.append(f'<div class="sub">Session {e(session["id"])} · {scope_txt}</div>')

    out.append('<div class="explain"><b>What this is.</b> Sometimes someone talks but the '
               "transcriber writes nothing — the words just vanish, and because there's no line "
               "in the transcript, you'd never catch it by reading. This page finds those moments "
               "by comparing <b>who was talking</b> (the speaker detector) against <b>what got "
               "written down</b>. Each card below is a stretch where a voice was detected but "
               "<b>no segment of any kind exists</b> — no transcribed line, and not even an "
               "[unintelligible] marker. (Spots the pipeline already flagged [unintelligible] are "
               "left out: those are visible, clickable lines in the transcript, so they're a "
               "separate question.)<br><br><b>How to check one.</b> Click <b>▶ Play</b> — it "
               "starts a second early and plays the gap. You should hear a voice, often quiet, "
               "that isn't written anywhere. If you only hear silence or noise, that one's a "
               "false alarm.</div>")

    out.append('<audio id="aud" controls src="audio.m4a" preload="none"></audio>')
    out.append(f'<div class="count">Showing the <b>{len(in_scope)}</b> clearest cases — each is '
               f"≥1 second of detected talking with nothing transcribed, {total_s:.1f}s in all. "
               "(Shorter stretches, often breath-pauses, are left out — see the detailed view.)</div>")

    for g in in_scope:
        prev_t = e((g["prev"].get("text") or "").strip()[:90]) if g["prev"] else "—"
        next_t = e((g["next"].get("text") or "").strip()[:90]) if g["next"] else "—"
        out.append('<div class="gap">')
        out.append(f'<button class="play" onclick="play({g["start"]:.2f},{g["end"]:.2f})">▶ Play</button>')
        out.append('<div>')
        out.append(f'<div class="meta">{lab(g["start"])} → {lab(g["end"])} · '
                   f'{g["dur"]:.1f}s · {e(speaker_label(g["speaker"]))}</div>')
        out.append(f'<div class="ctx"><span class="lbl">before</span>"{prev_t}"'
                   f'<span class="blank"> [nothing transcribed] </span>'
                   f'<span class="lbl">after</span>"{next_t}"</div>')
        out.append("</div></div>")

    if not in_scope:
        out.append('<div class="count">No gaps of 1 second or longer in scope for this session.</div>')

    out.append('<div class="note">Audio is the <code>audio.m4a</code> file beside this page. '
               "The full view — every candidate plus the timeline strips — is "
               "<code>gap-analysis.html</code>.</div>")
    out.append('<script>const aud=document.getElementById("aud");let timer=null;'
               'function play(s,e){if(timer)clearTimeout(timer);'
               'var from=Math.max(0,s-1.0);aud.currentTime=from;aud.play();'
               'timer=setTimeout(function(){aud.pause();},(e-from+0.6)*1000);}</script>')
    out.append("</body></html>")
    return "\n".join(out)


# --- orchestration -----------------------------------------------------------
def process(session, min_gap, hc, write_html, regenerate_baseline):
    sdir, transcript, diarization = load_session(session["id"])
    story_end = (None if session["story_end_seg"] is None
                 else seg_end_time(transcript, session["story_end_seg"]))

    gaps = find_gaps(diarization, transcript, min_gap)
    t03 = tally(gaps, min_gap, story_end)
    t10 = tally(gaps, hc, story_end)

    do_write = write_html and (not session.get("is_baseline") or regenerate_baseline)
    if do_write:
        out_path = _visdir(sdir) / "gap-analysis.html"
        out_path.write_text(
            render_html(session, transcript, diarization, gaps, story_end, min_gap)
        )
        written = str(out_path.relative_to(ROOT))
    else:
        written = "(not written — baseline preserved)" if session.get("is_baseline") else "(skipped)"

    # The plain, checkable view is a new file — no hand-made baseline to keep,
    # so write it for every session including Moon Story.
    simple_path = _visdir(sdir) / "gaps-to-check.html"
    simple_path.write_text(render_simple_html(session, transcript, gaps, story_end, hc))

    return {"session": session, "story_end": story_end,
            "t03": t03, "t10": t10, "written": written,
            "simple": str(simple_path.relative_to(ROOT)),
            "unintel_n": len(unintelligible_segments(transcript))}


def validate_baseline(result):
    exp = result["session"].get("expected")
    if not exp:
        return None
    got = {"story_gaps_03": result["t03"]["story_n"],
           "story_sec_03": round(result["t03"]["story_s"], 1),
           "story_gaps_10": result["t10"]["story_n"],
           "story_sec_10": round(result["t10"]["story_s"], 1)}
    ok = (got["story_gaps_03"] == exp["story_gaps_03"]
          and abs(got["story_sec_03"] - exp["story_sec_03"]) <= 0.1
          and got["story_gaps_10"] == exp["story_gaps_10"]
          and abs(got["story_sec_10"] - exp["story_sec_10"]) <= 0.1)
    return ok, exp, got


def build_summary(results, min_gap, hc):
    lines = []
    lines.append("# Missed-speech sweep — where the transcript has no segment at all")
    lines.append("")
    lines.append(f"Where the speaker detector heard a voice but **no segment of any kind "
                 f"exists** — neither a transcribed line nor an [unintelligible] marker. That is "
                 f"missed speech with no trace at all. [unintelligible] placeholders "
                 f"count as coverage (they're visible, reviewable lines — classifying them is "
                 f"TMAS-46), so they are NOT counted as gaps here. Candidate threshold ≥{min_gap}s, "
                 f"high-confidence ≥{hc}s. Story-scope excludes the end-of-session wind-down. "
                 f"Generated by `emp/src/gap_analysis.py`.")
    lines.append("")
    lines.append("| Session | missed-speech floor — story ≥1.0s | story ≥0.3s | whole ≥1.0s | whole ≥0.3s | [unintelligible] (→TMAS-46) |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        s = r["session"]
        name = s["name"] + (" *(no boundary — whole-session)*" if s.get("no_boundary") else "")
        lines.append(
            f"| {name} "
            f"| **{r['t10']['story_n']} gaps / {r['t10']['story_s']:.1f}s** "
            f"| {r['t03']['story_n']} / {r['t03']['story_s']:.1f}s "
            f"| {r['t10']['whole_n']} / {r['t10']['whole_s']:.1f}s "
            f"| {r['t03']['whole_n']} / {r['t03']['whole_s']:.1f}s "
            f"| {r['unintel_n']} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- **A gap = speech with NO segment at all.** [unintelligible] placeholders "
                 "count as coverage, so a region the pipeline already flagged that way is not a "
                 "a miss here — it is visible/reviewable, and classifying it is TMAS-46. The last "
                 "column counts those placeholders per session (the TMAS-46 universe).")
    lines.append("- The **missed-speech floor** column (story-scope, ≥1.0s) is the unbiased "
                 "high-confidence count for the EMP pivot table.")
    lines.append("- Counts are mechanical and include monologue pauses (a storyteller "
                 "going quiet mid-story counts as missed speech); they are a floor, "
                 "not a hand-verified tally.")
    lines.append("- Pandavas has no wind-down boundary in its notes, so its story and "
                 "whole-session figures are identical and may include a lullaby tail.")
    lines.append("- Per-session visual breakdowns: `emp/results/visuals/<id>/gap-analysis.html` "
                 "(local only — they contain transcript text).")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(
        description="Find missed-speech gaps: diarization vs transcript coverage."
    )
    p.add_argument("--session", help="Process a single session id (ad hoc).")
    p.add_argument("--story-end", help="Story end for --session: 'seg:N' or 'M:SS' "
                                       "(omit for whole-session).")
    p.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP,
                   help=f"Candidate threshold seconds (default {DEFAULT_MIN_GAP}).")
    p.add_argument("--hc-threshold", type=float, default=DEFAULT_HC,
                   help=f"High-confidence threshold seconds (default {DEFAULT_HC}).")
    p.add_argument("--summary-out",
                   default="emp/results/gap-analysis/summary.md",
                   help="Where to write the cross-session summary markdown.")
    p.add_argument("--regenerate-baseline", action="store_true",
                   help="Also overwrite Moon Story's preserved HTML.")
    args = p.parse_args()

    if args.session:
        story_end_seg = None
        sess = {"name": args.session, "id": args.session, "story_end_seg": None}
        if args.story_end:
            if args.story_end.startswith("seg:"):
                sess["story_end_seg"] = int(args.story_end[4:])
            else:
                m, s = args.story_end.split(":")
                # store as a resolved time via a synthetic shim
                sess["_story_end_time"] = int(m) * 60 + float(s)
        sessions = [sess]
    else:
        sessions = SESSIONS

    results = []
    for session in sessions:
        # ad-hoc M:SS override: resolve without a segment lookup
        if "_story_end_time" in session:
            sdir, transcript, diarization = load_session(session["id"])
            story_end = session["_story_end_time"]
            gaps = find_gaps(diarization, transcript, args.min_gap)
            (_visdir(sdir) / "gap-analysis.html").write_text(
                render_html(session, transcript, diarization, gaps, story_end, args.min_gap)
            )
            (_visdir(sdir) / "gaps-to-check.html").write_text(
                render_simple_html(session, transcript, gaps, story_end, args.hc_threshold)
            )
            r = {"session": session, "story_end": story_end,
                 "t03": tally(gaps, args.min_gap, story_end),
                 "t10": tally(gaps, args.hc_threshold, story_end),
                 "written": str((_visdir(sdir) / "gap-analysis.html").relative_to(ROOT)),
                 "simple": str((_visdir(sdir) / "gaps-to-check.html").relative_to(ROOT)),
                 "unintel_n": len(unintelligible_segments(transcript))}
        else:
            r = process(session, args.min_gap, args.hc_threshold,
                        write_html=True, regenerate_baseline=args.regenerate_baseline)
        results.append(r)

        s = r["session"]
        scope = "whole-session" if r["story_end"] is None else f"story 0–{mmss(r['story_end'])}"
        print(f"\n{s['name']} ({s['id']}) · {scope}")
        print(f"  ≥{args.min_gap}s: {r['t03']['story_n']:>3} story / "
              f"{r['t03']['whole_n']:>3} whole gaps · "
              f"{r['t03']['story_s']:.1f}s story / {r['t03']['whole_s']:.1f}s whole")
        print(f"  ≥{args.hc_threshold}s: {r['t10']['story_n']:>3} story / "
              f"{r['t10']['whole_n']:>3} whole gaps · "
              f"{r['t10']['story_s']:.1f}s story / {r['t10']['whole_s']:.1f}s whole")
        print(f"  [unintelligible] placeholders (coverage, not missed speech): {r.get('unintel_n', '?')}")
        print(f"  html:  {r['written']}")
        print(f"  check: {r.get('simple', '—')}")

        v = validate_baseline(r)
        if v is not None:
            ok, exp, got = v
            tag = "PASS" if ok else "FAIL"
            print(f"  >>> Moon Story validation: {tag}")
            if not ok:
                print(f"      expected {exp}")
                print(f"      got      {got}", file=sys.stderr)

    if not args.session:
        summary_path = ROOT / args.summary_out
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(build_summary(results, args.min_gap, args.hc_threshold))
        print(f"\nWrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
