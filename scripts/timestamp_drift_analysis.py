"""Check word timestamps against the audio (the Mode 7 "timestamp drift" floor).

Whisper stamps every word with a start and end time, but those times are
*inferred* from the model's attention rather than measured — so they drift.
A word the transcript places at 5:02 may really be at 5:03. By ear you can only
catch the blatant cases; this read-only sweep checks every word against the
underlying sound so the real count is no longer invisible.

For each word in transcript-rich.json it asks a few plain questions:

  1. Is there loud-enough audio where the word claims to be?  (acoustic check)
     The audio is split into 20ms chunks and turned into loudness (RMS, in dBFS).
     A word that really happened lights up several chunks well above the room's
     quiet floor; a word whose timestamp drifted onto a silent spot does not.
  2. Is *anyone* detected speaking in that window?  (diarization presence)
  3. If the window is quiet and empty, is the word's real audio just *next door*?
     (loud audio within ~0.75s means the stamp drifted off a word that exists.)

A word is sorted into one of:

  * "drifted"        — quiet and empty here, but loud audio sits right beside the
                       window. The word exists; its timestamp slid off it. The
                       clean Mode 7 case.
  * "isolated"       — quiet, empty, AND nothing loud anywhere near. There is no
                       word here at all — more likely a hallucination than a
                       drifted stamp, i.e. a DIFFERENT failure mode, so it is held
                       OUT of the Mode 7 floor and reported on its own.
  * "quiet / maybe real" — quiet, but someone IS detected speaking. AMBIGUOUS:
                       maybe drift, maybe genuine quiet speech (a child's whisper)
                       sitting near the noise floor. Quiet real speech is a
                       *missed-speech* question (#13, see gap_analysis.py /
                       TMAS-44), NOT timestamp drift, so it is kept OUT of the
                       headline floor — by design, so Mode 7 and #13 don't
                       double-count.
  * "ok"             — loud enough to hold a word (it is plainly there; a loud
                       spot the detector didn't label is a diarization gap, not
                       drift).

The headline "Mode 7 floor" is "drifted" — the case provable from the waveform
itself. The isolated and quiet/maybe-real buckets are reported alongside but
held out, so the floor stays to what the audio alone can prove.

A "wrong-speaker" check (flagging a word whose speaker label disagreed with the
rest of its sentence) was tried and removed: spot-checking showed it was not
catching drift at all, but the diarizer splitting ONE storyteller into several
speaker labels — character voices especially read as different speakers. That
is a diarization-granularity issue, a different failure mode, so it does not
belong in a timestamp-drift tool.

Sibling of gap_analysis.py (TMAS-44): same session list, same output trio
(per-session timeline HTML + a plain "drift-to-check" view + a cross-session
summary), and it reuses that tool's pure interval/format helpers rather than
copying them. The one genuinely new ingredient is reading the waveform: it
borrows MLX Whisper's audio loader, exactly as the ticket suggests.

Read-only: never modifies session JSON. Per-session HTML lives under the
gitignored sessions/ tree (it contains transcript text); the summary holds only
aggregate numbers and is safe to commit.

Examples:
    python scripts/timestamp_drift_analysis.py                 # all five sessions
    python scripts/timestamp_drift_analysis.py --session 20260129-204404 --story-end seg:594
    python scripts/timestamp_drift_analysis.py --session 20251210-203654 --story-end 9:37.12
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
from mlx_whisper.audio import SAMPLE_RATE, load_audio

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # so we can import the sibling tool

# Reuse the pure helpers and the shared session list from the missed-speech
# sibling — same EMP sample, same interval math, same render constants.
from gap_analysis import (  # noqa: E402
    GRID_SEC,
    PX_PER_SEC,
    SESSIONS,
    SPEAKER_COLORS,
    X_OFFSET,
    load_session,
    mmss,
    mmss_grid,
    real_segments,
    seg_end_time,
)

# M10 (broken-Whisper "filler-loop") regions are a DIFFERENT failure mode: the
# looped filler word ("Hmm.", "Right.") is a decoder artifact, not a real word
# whose timestamp drifted — so it must not count toward the M7 floor (the same
# no-double-counting rule that holds out isolated/quiet). The canonical region
# definitions live in populate_mode10.py; importing them keeps this tool and the
# M10 fold from ever disagreeing.
from populate_mode10 import STRETCHES  # noqa: E402

# --- tunable thresholds ------------------------------------------------------
# These are deliberately simple and transparent (an honest floor beats a clever
# one). All are exposed on the CLI so they can be swept per session.
CHUNK_SEC = 0.02             # RMS window — 20ms, per the ticket
DEFAULT_NOISE_PCT = 20       # the dBFS at this percentile estimates the quiet floor
DEFAULT_QUIET_MARGIN = 12    # dB above the floor a word window must reach to be "loud"
DEFAULT_EMPTY_FRAC = 0.10    # below this diarization coverage, nobody is talking
DEFAULT_NEIGHBOR_SEC = 0.75  # how far beside the window to look for the word's real audio
QUIET_RANGE_DB = 15.0        # dB past the loud threshold that maps to full confidence

# --- categories --------------------------------------------------------------
# Drift is judged with signals independent of how Whisper set the timestamp: the
# raw loudness in the claimed window and just beside it.
FLOOR = ("drifted",)              # the clean Mode 7 floor: drift on a segment NOT already coded as another failure
ISOLATED = ("isolated",)          # quiet with nothing loud near — likely a DIFFERENT mode (hallucination)
AMBIGUOUS = ("quiet_ambiguous",)  # quiet but a speaker is present — maybe real soft speech (#13)
CODED = ("drifted_coded",)        # drift on a segment a human already coded as another failure — not double-counted

# Axial failure-mode codes other than M7 itself. A drifted word on a segment
# carrying one of these is a downstream symptom of an already-counted failure
# (M2 hallucination, M4 wrong speaker, ...), so it is held OUT of the M7 floor —
# the same no-double-counting rule as M10. "NotA" (not a failure) and "M7" do
# not count as another mode: those segments stay eligible as genuine M7.
def other_failure_codes(codes):
    return [c for c in codes if c not in ("NotA", "M7")]


CAT_COLOR = {
    "ok": "#4caf50",
    "drifted": "#f44336",
    "drifted_coded": "#78909c",
    "isolated": "#fdd835",
    "quiet_ambiguous": "#9c27b0",
}
CAT_LABEL = {
    "ok": "ok",
    "drifted": "drifted",
    "drifted_coded": "already coded (other mode)",
    "isolated": "isolated silence",
    "quiet_ambiguous": "quiet / maybe real",
}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# Backchannels / filler vocalizations. A drifted candidate whose word is one of
# these is *suspect*: more likely a non-speech sound rendered as a word (Mode 2)
# or an unmarked filler loop than a genuinely mistimed real word. Used ONLY to
# float these to the top of the check page for an ear pass — never to change the
# count. The ear decides; a text pattern must not silently drop anything.
FILLER_WORDS = {
    "hmm", "hm", "hmmm", "mm", "mmm", "mhm", "mmhm", "mmhmm", "mhmm",
    "uh", "uhh", "uhhuh", "um", "umm", "er", "err", "oh", "ohh",
    "ah", "ahh", "huh", "yeah", "yep", "yup", "right", "ok", "okay",
}


def is_filler_word(word):
    """True if a word is a backchannel/filler vocalization (review aid only).

    Normalizes by lowercasing and dropping non-letters, so "-hmm." -> "hmm" and
    "Huh?" -> "huh" both match.
    """
    return re.sub(r"[^a-z]", "", (word or "").lower()) in FILLER_WORDS


# --- audio → loudness --------------------------------------------------------
def load_rms_db(audio_path):
    """Load audio and return per-20ms-chunk loudness in dBFS (a 1-D array).

    Uses MLX Whisper's loader (16kHz mono float). load_audio returns an MLX
    array, so it is converted to NumPy before the math.
    """
    audio = np.array(load_audio(str(audio_path))).astype(np.float64)
    hop = int(CHUNK_SEC * SAMPLE_RATE)
    n = len(audio) // hop
    chunks = audio[:n * hop].reshape(n, hop)
    rms = np.sqrt((chunks ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def session_floor(db, noise_pct, margin):
    """(noise_floor_db, loud_threshold_db) from the session's own distribution.

    Each session has its own room tone, so the floor is estimated per session
    rather than hard-coded. The threshold is a fixed margin above that floor.
    """
    noise_floor = float(np.percentile(db, noise_pct))
    return noise_floor, noise_floor + margin


def window_loudness(db, start, end):
    """Loudest 20ms chunk (dBFS) inside [start, end].

    Max, not mean: even a short real word lights up a few chunks, while a
    drifted word sees only floor across the whole window. Clamped to the array.
    """
    n = len(db)
    i0 = _clip(int(start / CHUNK_SEC), 0, n - 1)
    i1 = _clip(int(end / CHUNK_SEC), i0, n - 1)
    return float(db[i0:i1 + 1].max())


# --- diarization overlap (pure) ----------------------------------------------
def speaker_coverage(start, end, diar_segments):
    """Fraction of [start, end] each detected speaker covers, plus the total.

    Returns ({speaker: frac}, total_frac). total_frac is capped at 1.0 (two
    speakers can overlap the same instant). frac is overlap-seconds / window.
    """
    dur = end - start
    if dur <= 0:
        return {}, 0.0
    by_spk = {}
    for d in diar_segments:
        overlap = min(end, d["end"]) - max(start, d["start"])
        if overlap > 0:
            by_spk[d["speaker"]] = by_spk.get(d["speaker"], 0.0) + overlap
    by_spk = {spk: sec / dur for spk, sec in by_spk.items()}
    return by_spk, min(1.0, sum(by_spk.values()))


# --- per-word classification (pure) ------------------------------------------
def classify_word(loud_db, threshold_db, total_frac, loud_nearby,
                  empty_frac=DEFAULT_EMPTY_FRAC, quiet_range=QUIET_RANGE_DB):
    """Sort one word into a drift category with a Mode 7 confidence score.

    Signals (all independent of how Whisper set the timestamp):
      loud_db      — is the claimed window loud enough to hold a word?
      total_frac   — is anyone detected speaking in the window?
      loud_nearby  — is there loud audio just outside the window (the real word)?

    See the module docstring for what each category means. The score is in
    [0, 1]; only "drifted" carries high confidence.
    """
    loud = loud_db >= threshold_db
    nobody = total_frac < empty_frac
    quietness = _clip((threshold_db - loud_db) / quiet_range, 0.0, 1.0)

    if loud:
        # Loud enough to hold a word — it is plainly here. A loud spot the
        # detector didn't label is a diarization gap, not drift.
        cat, score = "ok", 0.0
    elif nobody:
        if loud_nearby:
            # Quiet and empty here, but the word's real audio is right next
            # door — the timestamp drifted off it. The clean Mode 7 case.
            cat, score = "drifted", _clip(0.6 + 0.4 * quietness, 0.6, 1.0)
        else:
            # Quiet, empty, and nothing loud anywhere near — no word here at
            # all. More likely a hallucination than a drifted stamp (a
            # different failure mode), so held out of the Mode 7 floor.
            cat, score = "isolated", _clip(0.4 + 0.2 * quietness, 0.4, 0.6)
    else:
        # Quiet but a speaker is detected — could be drift or genuine quiet
        # speech (#13). Ambiguous on purpose; kept out of the headline floor.
        cat, score = "quiet_ambiguous", _clip(0.3 + 0.2 * quietness, 0.3, 0.5)

    return {
        "category": cat,
        "score": round(score, 3),
        "loud_db": round(loud_db, 1),
        "total_frac": round(total_frac, 3),
    }


# --- session analysis --------------------------------------------------------
def load_axial_codes(session_dir):
    """Read-only: segmentId -> [axial codes] from axial-labels.json, or {} if none.

    The drift tool never writes this file; it only reads the human coding to
    decide which drift candidates are already explained by another failure mode.
    """
    path = session_dir / "axial-labels.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {e["segmentId"]: e.get("codes", []) for e in json.load(f).get("labels", [])}


def m10_filler_ids(transcript, session_id):
    """Segment ids folded into an M10 filler-loop for this session.

    A folded segment is one inside a configured stretch whose stripped text is
    the looped filler word — exactly populate_mode10.py's fold set. Real-content
    lines interleaved in a stretch are NOT folded, so they stay eligible for a
    genuine drift flag.
    """
    stretches = [s for s in STRETCHES if s["session"] == session_id]
    if not stretches:
        return set()
    ids = set()
    for seg in transcript["segments"]:
        sid = seg.get("id")
        # Real segments carry integer ids; pipeline-injected [unintelligible]
        # placeholders use string ids ("gap_1") and are never filler loops.
        if not isinstance(sid, int):
            continue
        text = (seg.get("text") or "").strip()
        if any(s["lo"] <= sid <= s["hi"] and text == s["word"] for s in stretches):
            ids.add(sid)
    return ids


def analyze_session(session, noise_pct, margin, empty_frac, neighbor_sec):
    """Classify every word in a session. Returns everything render/tally need."""
    sdir, transcript, diarization = load_session(session["id"])
    audio_path = next(sdir.glob("audio.*"), None)
    if audio_path is None:
        raise FileNotFoundError(f"no audio file in {sdir}")

    db = load_rms_db(audio_path)
    noise_floor, threshold = session_floor(db, noise_pct, margin)
    diar = diarization.get("segments", [])
    drop_ids = m10_filler_ids(transcript, session["id"])
    axial = load_axial_codes(sdir)

    records = []
    seg_rows = []
    m10_excluded = 0
    for seg in real_segments(transcript):
        if seg.get("id") in drop_ids:
            # Looped filler word (M10), not a drifted real word — skip entirely.
            m10_excluded += sum(1 for w in seg.get("words", []) if w["end"] > w["start"])
            continue
        seg_other = other_failure_codes(axial.get(seg.get("id"), []))
        words = seg.get("words", [])
        seg_records = []
        for wi, word in enumerate(words):
            start, end = word["start"], word["end"]
            if end <= start:
                continue
            _, total = speaker_coverage(start, end, diar)
            loud_db = window_loudness(db, start, end)
            loud_nearby = window_loudness(db, start - neighbor_sec, end + neighbor_sec) >= threshold
            verdict = classify_word(loud_db, threshold, total, loud_nearby, empty_frac)
            # A drift candidate on a segment a human already coded as another
            # failure is a downstream symptom of that mode — held out of the M7
            # floor (it still shows on the check page, tagged with its code).
            if verdict["category"] == "drifted" and seg_other:
                verdict["category"] = "drifted_coded"
            verdict["axial"] = seg_other
            rec = {"seg_id": seg.get("id"), "wi": wi, "start": start, "end": end,
                   "word": (word.get("word") or "").strip(), **verdict}
            records.append(rec)
            seg_records.append(rec)

        n_floor = sum(1 for r in seg_records if r["category"] in FLOOR)
        n_other = sum(1 for r in seg_records if r["category"] not in ("ok",) + FLOOR)
        if n_floor or n_other:
            seg_rows.append({"seg": seg, "records": seg_records, "n_floor": n_floor,
                             "n_other": n_other, "n_words": len(seg_records)})
    seg_rows.sort(key=lambda r: (r["n_floor"], r["n_other"]), reverse=True)

    return {"sdir": sdir, "transcript": transcript, "diarization": diarization,
            "db": db, "noise_floor": noise_floor, "threshold": threshold,
            "records": records, "seg_rows": seg_rows, "m10_excluded": m10_excluded}


def tally(records, story_end):
    """Per-category word counts, split story-scope vs whole-session."""
    def bucket(rs):
        return {
            "floor": sum(1 for r in rs if r["category"] in FLOOR),
            "coded": sum(1 for r in rs if r["category"] in CODED),
            "isolated": sum(1 for r in rs if r["category"] in ISOLATED),
            "ambiguous": sum(1 for r in rs if r["category"] in AMBIGUOUS),
            "total": len(rs),
        }
    whole = bucket(records)
    if story_end is None:
        return {"story": whole, "whole": whole}
    return {"story": bucket([r for r in records if r["start"] < story_end]),
            "whole": whole}


def severity_distribution(seg_rows):
    """How many segments carry 1, 2, 3, or 4+ drifted (floor) words."""
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for r in seg_rows:
        n = r["n_floor"]
        if n <= 0:
            continue
        buckets["4+" if n >= 4 else str(n)] += 1
    return buckets


# --- HTML rendering ----------------------------------------------------------
def _x(t):
    return X_OFFSET + t * PX_PER_SEC


def _bar(t_start, t_end):
    return _x(t_start), max(0.8, (t_end - t_start) * PX_PER_SEC)


def render_html(session, analysis, story_end):
    """Full timeline view: loudness profile, words tinted by drift category,
    diarization — plus a per-segment table of the flagged words."""
    e = html.escape
    db = analysis["db"]
    threshold = analysis["threshold"]
    noise_floor = analysis["noise_floor"]
    diar = analysis["diarization"].get("segments", [])
    records = analysis["records"]
    seg_rows = analysis["seg_rows"]

    total_dur = max([analysis["transcript"]["segments"][-1]["end"]]
                    + [d["end"] for d in diar] + [len(db) * CHUNK_SEC])
    speakers = sorted({d["speaker"] for d in diar})
    color = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
             for i, spk in enumerate(speakers)}
    width = int(_x(total_dur) + 40)

    # loudness profile: 0.5s columns, height above the noise floor
    lo, hi = noise_floor - 6.0, float(db.max())
    span = max(1e-6, hi - lo)
    LOUD_TOP, LOUD_BOT = 24.0, 74.0

    def loud_y(value_db):
        return LOUD_BOT - _clip((value_db - lo) / span, 0.0, 1.0) * (LOUD_BOT - LOUD_TOP)

    counted = [r for r in records if story_end is None or r["start"] < story_end]
    n_floor = sum(1 for r in counted if r["category"] in FLOOR)
    n_coded = sum(1 for r in counted if r["category"] in CODED)
    n_iso = sum(1 for r in counted if r["category"] in ISOLATED)
    n_amb = sum(1 for r in counted if r["category"] in AMBIGUOUS)
    n_m10 = analysis.get("m10_excluded", 0)

    out = []
    out.append('<!doctype html><html><head><meta charset="utf-8">')
    out.append(f"<title>{e(session['name'])} — Timestamp Drift</title><style>")
    out.append("body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;padding:24px;margin:0}")
    out.append("h1{font-size:18px;margin:0 0 6px}h2{font-size:14px;margin:24px 0 8px;color:#bbb}")
    out.append(".legend span{display:inline-block;padding:2px 10px;margin-right:6px;font-size:11px;border-radius:3px}")
    out.append(".scroll{overflow-x:auto;border:1px solid #222;background:#0a0a0a;margin:14px 0}")
    out.append("svg{display:block}")
    out.append("table{border-collapse:collapse;font-size:12px}td,th{padding:5px 9px;border:1px solid #2a2a2a;text-align:left;vertical-align:top}")
    out.append("th{background:#1a1a1a;color:#aaa}.story{background:#2a1010}.tail{color:#777}")
    out.append(".w{padding:1px 4px;border-radius:3px;margin:1px;display:inline-block}")
    out.append("rect:hover{stroke:#fff;stroke-width:1}")
    out.append("</style></head><body>")
    out.append(f"<h1>{e(session['name'])} — Timestamp Drift (Mode 7) check</h1>")

    scope_txt = ("whole session — no wind-down boundary marked"
                 if story_end is None else f"story 0–{mmss(story_end)}")
    out.append(
        f'<div style="color:#999;font-size:12px">Session {e(session["id"])} · {scope_txt} · '
        f'noise floor {noise_floor:.1f} dBFS, loud ≥ {threshold:.1f} dBFS · '
        f'<b style="color:#f55">{n_floor} drifted (Mode 7 floor)</b> · '
        f'<b style="color:#90a4ae">{n_coded} already coded (other mode, excluded)</b> · '
        f'<b style="color:#fd5">{n_iso} isolated (≈hallucination)</b> · '
        f'<b style="color:#c79">{n_amb} quiet / maybe-real (→ #13)</b>'
        + (f' · <span style="color:#777">{n_m10} M10 filler-loop words excluded</span>'
           if n_m10 else '')
        + '</div>'
    )
    out.append(
        '<div style="color:#888;font-size:12px;margin-top:8px;max-width:920px">'
        'Each word carries a start/end time from Whisper. This checks those times against '
        'the sound. The top strip is <b>loudness</b> (taller = louder; the dashed line is the '
        '"a word should be at least this loud" threshold). The middle strip is every word, '
        '<b>tinted by what we found</b>: <span style="color:#f55">red</span> = near-silent here '
        'but the real word is loud right beside it (the time <b>drifted</b> off it); '
        '<span style="color:#90a4ae">slate</span> = drifted, but a human already coded that '
        'segment as another failure (M2, M4, …), so it is counted there, not as M7; '
        '<span style="color:#fd5">yellow</span> = quiet with nothing loud anywhere near (no word '
        'there at all — likely a hallucination, a different problem, so not counted); '
        '<span style="color:#c8f">purple</span> = quiet but someone IS detected — maybe drift, '
        'maybe genuine quiet speech, set aside as a missed-speech question (#13). '
        'The bottom strip is who the detector heard.</div>'
    )

    out.append('<div class="legend" style="margin-top:14px">')
    for cat in ("drifted", "drifted_coded", "isolated", "quiet_ambiguous", "ok"):
        out.append(f'<span style="background:{CAT_COLOR[cat]};color:#000">{CAT_LABEL[cat]}</span>')
    if story_end is not None:
        out.append('<span style="background:#fff;color:#000">story end ↓</span>')
    out.append("</div>")

    out.append('<div class="scroll">')
    out.append(f'<svg width="{width}" height="210">')
    t = 0
    while t <= total_dur:
        x = _x(t)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="210" stroke="#1d1d1d"/>')
        out.append(f'<text x="{x + 2:.1f}" y="11" fill="#555" font-size="9">{mmss_grid(t)}</text>')
        t += GRID_SEC
    if story_end is not None:
        sx = _x(story_end)
        out.append(f'<line x1="{sx:.3f}" y1="18" x2="{sx:.3f}" y2="210" '
                   f'stroke="#fff" stroke-dasharray="3,3" opacity="0.6"/>')

    # strip 1: loudness profile + threshold line
    out.append('<text x="6" y="50.0" fill="#bbb" font-size="11">loudness</text>')
    step = 0.5
    chunks_per = int(step / CHUNK_SEC)
    i = 0
    while i < len(db):
        seg_db = float(db[i:i + chunks_per].max())
        x = _x(i * CHUNK_SEC)
        y = loud_y(seg_db)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.8, step * PX_PER_SEC):.1f}" '
                   f'height="{LOUD_BOT - y:.1f}" fill="#3a6ea5" opacity="0.8"/>')
        i += chunks_per
    ty = loud_y(threshold)
    out.append(f'<line x1="{_x(0):.1f}" y1="{ty:.1f}" x2="{_x(total_dur):.1f}" y2="{ty:.1f}" '
               f'stroke="#f7c948" stroke-dasharray="4,3" opacity="0.8"/>')

    # strip 2: words tinted by category
    out.append('<text x="6" y="103.0" fill="#bbb" font-size="11">words</text>')
    for r in records:
        x, w = _bar(r["start"], r["end"])
        opacity = "0.9" if r["category"] != "ok" else "0.30"
        title = (f'{mmss(r["start"])}-{mmss(r["end"])} · "{r["word"]}" · '
                 f'{CAT_LABEL[r["category"]]} (score {r["score"]}) · '
                 f'loud {r["loud_db"]}dB, voice {int(r["total_frac"] * 100)}%')
        out.append(f'<rect x="{x:.1f}" y="88" width="{w:.1f}" height="30" '
                   f'fill="{CAT_COLOR[r["category"]]}" opacity="{opacity}">'
                   f'<title>{e(title)}</title></rect>')

    # strip 3: diarization
    out.append('<text x="6" y="147.0" fill="#bbb" font-size="11">diarization</text>')
    for d in diar:
        x, w = _bar(d["start"], d["end"])
        out.append(f'<rect x="{x:.1f}" y="132" width="{w:.1f}" height="30" '
                   f'fill="{color[d["speaker"]]}" opacity="0.75">'
                   f'<title>{e(d["speaker"])} · {mmss(d["start"])}-{mmss(d["end"])}</title></rect>')
    out.append("</svg></div>")

    # per-segment table
    out.append(f"<h2>Segments with flagged words — {len(seg_rows)} segments</h2>")
    out.append("<table><tr><th>#</th><th>seg</th><th>start</th><th>severity</th>"
               "<th>flagged words</th><th>segment text</th></tr>")
    for i, row in enumerate(seg_rows, 1):
        seg = row["seg"]
        cls = "tail" if (story_end is not None and seg["start"] >= story_end) else "story"
        flagged = []
        for r in row["records"]:
            if r["category"] == "ok":
                continue
            flagged.append(
                f'<span class="w" style="background:{CAT_COLOR[r["category"]]};color:#000" '
                f'title="{e(CAT_LABEL[r["category"]])} · score {r["score"]} · '
                f'{mmss(r["start"])}">{e(r["word"]) or "·"}</span>'
            )
        out.append(
            f'<tr class="{cls}"><td>{i}</td><td>{e(str(seg.get("id")))}</td>'
            f'<td>{mmss(seg["start"])}</td>'
            f'<td>{row["n_floor"]}✕ / {row["n_other"]}? of {row["n_words"]}</td>'
            f'<td>{"".join(flagged)}</td>'
            f'<td>{e((seg.get("text") or "").strip()[:80])}</td></tr>'
        )
    out.append("</table></body></html>")
    return "\n".join(out)


def render_simple_html(session, analysis, story_end, cap=100):
    """A plain, human-checkable view of the highest-confidence drift words.

    One card per word, each with a Play button that seeks a second early and
    plays through — so a reviewer can confirm by ear that the word is NOT where
    its timestamp claims. The clean drift (the Mode 7 floor) is the to-check
    list; drift on segments already coded as another failure goes in a separate
    grey box, tagged with its code, to skip.
    """
    e = html.escape
    by_seg = {s.get("id"): s for s in analysis["transcript"]["segments"]}

    def lab(t):
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    def mmss_ms(t):
        m, s = divmod(float(t), 60)
        return f"{int(m)}:{s:06.3f}"

    def in_scope(r):
        return story_end is None or r["start"] < story_end

    # Two groups: clean drift (the M7 floor, to check by ear) and drift on
    # segments a human already coded as another failure (shown to skip).
    clean = [r for r in analysis["records"] if r["category"] in FLOOR and in_scope(r)]
    clean.sort(key=lambda r: (is_filler_word(r["word"]), r["score"]), reverse=True)
    coded = [r for r in analysis["records"] if r["category"] in CODED and in_scope(r)]
    coded.sort(key=lambda r: r["score"], reverse=True)
    total_clean = len(clean)
    shown_clean = clean[:cap]

    out = []
    out.append('<!doctype html><html><head><meta charset="utf-8">')
    out.append(f"<title>{e(session['name'])} — drift to check</title><style>")
    out.append("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
               "max-width:760px;margin:40px auto;padding:0 20px;color:#1a1a1a;line-height:1.5}")
    out.append("h1{font-size:22px;margin:0 0 4px}.sub{color:#666;font-size:14px;margin-bottom:22px}")
    out.append(".explain{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:10px;"
               "padding:16px 18px;font-size:15px;margin-bottom:22px}")
    out.append(".count{font-size:15px;color:#444;margin:0 0 16px}")
    out.append(".card{border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin-bottom:12px;"
               "background:#fff;display:flex;gap:14px;align-items:flex-start}")
    out.append(".coded{background:#f1f5f9;border:1px solid #cbd5e1;border-radius:12px;"
               "padding:16px 18px;margin-top:28px}")
    out.append(".chead{font-size:16px;font-weight:700;color:#475569;margin:0 0 6px}")
    out.append(".cnote{font-size:14px;color:#475569;margin-bottom:14px}")
    out.append(".codetag{background:#475569;color:#fff;font-size:11px;font-weight:700;"
               "padding:2px 7px;border-radius:4px;margin-left:4px}")
    out.append(".filler{color:#c2410c;font-weight:600}")
    out.append(".play{flex:none;background:#2563eb;color:#fff;border:none;border-radius:8px;"
               "padding:10px 14px;font-size:15px;cursor:pointer}.play:hover{background:#1d4ed8}")
    out.append(".meta{font-weight:600;font-size:15px}.ctx{color:#555;font-size:14px;margin-top:6px}")
    out.append(".reason{font-size:14px;margin-top:4px}.tag{font-size:11px;text-transform:uppercase;"
               "letter-spacing:.04em;padding:2px 7px;border-radius:4px;color:#fff}")
    out.append(".here{background:#fde68a;font-weight:600;padding:0 3px;border-radius:3px}")
    out.append(".player{position:sticky;top:0;background:#fff;padding:12px 0 10px;margin-bottom:14px;"
               "z-index:10;border-bottom:1px solid #e2e8f0}")
    out.append("audio{width:100%;margin-bottom:4px;display:block}")
    out.append(".now{font-variant-numeric:tabular-nums;font-size:15px;color:#2563eb;font-weight:700}")
    out.append(".note{color:#888;font-size:13px;margin-top:24px}")
    out.append("</style></head><body>")

    out.append(f"<h1>{e(session['name'])}: timestamps to check</h1>")
    scope_txt = ("whole session — no wind-down boundary marked"
                 if story_end is None else f"story portion only (up to {lab(story_end)})")
    out.append(f'<div class="sub">Session {e(session["id"])} · {scope_txt}</div>')

    out.append('<div class="explain"><b>What this is.</b> Each word in the transcript carries a '
               "start and end time, but Whisper guesses those times rather than measuring them, so "
               "they drift. This page lists the words whose time looks <b>clearly wrong</b>: the spot "
               "is near-silent while the word's real sound is loud right beside it. "
               "The list below is what to <b>check by ear</b>. Words on segments you already coded as "
               "another failure (a hallucination, wrong speaker, …) are pulled into a separate "
               "grey box at the bottom — those are counted under that other mode, so you can skip "
               "them here. "
               "<br><br><b>How to check one.</b> Click <b>▶ Play</b> — it starts a second early and "
               "plays through the word's claimed time. If you do NOT hear that word right there, the "
               "timestamp drifted. If you DO hear it, it's a false alarm.</div>")

    out.append('<div class="player"><audio id="aud" controls src="audio.m4a" preload="none"></audio>'
               '<div class="now">▶ playing at <span id="nowt">0:00.000</span></div></div>')

    cap_note = "" if total_clean <= cap else f" (of {total_clean} — showing the {cap} highest-confidence)"

    def card(r):
        seg = by_seg.get(r["seg_id"])
        text = (seg.get("text") or "").strip() if seg else ""
        word = r["word"]
        ctx = e(text[:120])
        if word and word in text:
            ctx = e(text[:120]).replace(e(word), f'<span class="here">{e(word)}</span>', 1)
        codes = r.get("axial") or []
        if codes:
            tag = f'<span class="codetag">already: {e(", ".join(codes))}</span>'
            reason = ("This segment is already coded as " + e(", ".join(codes)) + " — that failure "
                      "is counted there, so it is not added to the M7 count. Shown only to confirm.")
        else:
            tag = (f'<span class="tag" style="background:{CAT_COLOR[r["category"]]}">'
                   f'{CAT_LABEL[r["category"]]}</span>')
            reason = ("The audio at this timestamp is near-silent, but the word's real sound is "
                      "loud just beside it — the timestamp drifted off the actual word.")
            if is_filler_word(word):
                reason += (' <span class="filler">Filler-shaped — confirm it isn\'t a '
                           'non-speech sound (a different problem) before counting it.</span>')
        rows = ['<div class="card">']
        rows.append(f'<button class="play" onclick="play({r["start"]:.2f},{r["end"]:.2f})">▶ Play</button>')
        rows.append('<div>')
        rows.append(f'<div class="meta">"{e(word) or "·"}" — claims {mmss_ms(r["start"])} → '
                    f'{mmss_ms(r["end"])} {tag}</div>')
        rows.append(f'<div class="reason">{reason}</div>')
        rows.append(f'<div class="ctx">in: "{ctx}"</div>')
        rows.append("</div></div>")
        return "\n".join(rows)

    out.append(f'<div class="count"><b>{len(shown_clean)}</b> Mode 7 candidates to check by ear'
               f"{cap_note}"
               + (f" · <b>{len(coded)}</b> more sit on already-coded segments (in the grey box below "
                  "— skip)." if coded else ".")
               + "</div>")

    # Section A — the M7 floor: clean drift to confirm by ear.
    for r in shown_clean:
        out.append(card(r))
    if not shown_clean:
        out.append('<div class="count">No clean Mode 7 candidates in scope for this session.</div>')

    # Section B — drift on segments already coded as another failure: skip.
    if coded:
        out.append('<div class="coded">')
        out.append(f'<div class="chead">Already coded as another failure — {len(coded)} '
                   "(skip for M7)</div>")
        out.append('<div class="cnote">These words also tripped the drift detector, but you already '
                   "coded their segment as a different failure (a hallucination, wrong speaker, name, "
                   "…). That failure is counted under its own mode, so these are <b>not</b> added to "
                   "the Mode 7 count — listed here only so you can confirm the existing code.</div>")
        for r in coded:
            out.append(card(r))
        out.append("</div>")

    out.append('<div class="note">Audio is the <code>audio.m4a</code> beside this page. The full '
               "view — every word plus the timeline strips — is <code>timestamp-drift.html</code>.</div>")
    out.append('<script>'
               'const aud=document.getElementById("aud"),nowt=document.getElementById("nowt");'
               'let timer=null,raf=null;'
               'function fmt(t){var m=Math.floor(t/60),s=t-m*60;return m+":"+(s<10?"0":"")+s.toFixed(3);}'
               'function tick(){nowt.textContent=fmt(aud.currentTime);raf=requestAnimationFrame(tick);}'
               'aud.addEventListener("play",function(){if(!raf)tick();});'
               'aud.addEventListener("pause",function(){if(raf){cancelAnimationFrame(raf);raf=null;}'
               'nowt.textContent=fmt(aud.currentTime);});'
               'aud.addEventListener("seeked",function(){nowt.textContent=fmt(aud.currentTime);});'
               'function play(s,e){if(timer)clearTimeout(timer);'
               'var from=Math.max(0,s-1.0);aud.currentTime=from;aud.play();'
               'timer=setTimeout(function(){aud.pause();},(e-from+0.6)*1000);}'
               '</script>')
    out.append("</body></html>")
    return "\n".join(out)


# --- cross-session summary ---------------------------------------------------
def build_summary(results, noise_pct, margin):
    lines = []
    lines.append("# Tier 1 timestamp-drift sweep (Mode 7 floor)")
    lines.append("")
    lines.append("Whisper guesses each word's start/end time instead of measuring it, so the times "
                 "drift. This sweep checks every word against the sound. The **Mode 7 floor** is the "
                 "case provable from the waveform alone — **drifted**: the word's window is "
                 "near-silent, but its real audio is loud **right beside it**, so the timestamp slid "
                 "off a word that exists. Several signals are reported but held OUT of the floor, on "
                 "purpose. **already-coded** drift sits on a segment a human already coded as another "
                 "failure (M2 hallucination, M4 wrong speaker, …) — that failure is counted under its "
                 "own mode, so counting it here too would double-count; it is excluded (same rule as "
                 "M10). **isolated** (near-silent with nothing loud anywhere near) is more likely a "
                 "hallucination than drift, a *different* mode. **quiet / maybe-real** (quiet but a "
                 "speaker is detected) is possibly genuine soft speech, a missed-speech / #13 question "
                 "(see the gap sweep). Holding these out keeps the floor to drift that is the segment's "
                 "*only* flagged problem. "
                 f"Loud threshold = the p{noise_pct} noise floor + {margin} dB, per session. "
                 "Story-scope excludes the end-of-session wind-down. Generated by "
                 "`scripts/timestamp_drift_analysis.py`.")
    lines.append("")
    lines.append("| Session | Mode 7 floor — story (drifted) | already-coded (excluded) "
                 "| isolated (≈halluc.) | quiet/maybe-real (→#13) | words checked | whole-session floor |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        s = r["session"]
        name = s["name"] + (" *(no boundary — whole-session)*" if s.get("no_boundary") else "")
        st = r["tally"]["story"]
        wh = r["tally"]["whole"]
        pct = (100.0 * st["floor"] / st["total"]) if st["total"] else 0.0
        lines.append(
            f"| {name} | **{st['floor']} words ({pct:.1f}%)** | {st['coded']} "
            f"| {st['isolated']} | {st['ambiguous']} "
            f"| {st['total']} | {wh['floor']} |"
        )
    lines.append("")
    lines.append("Severity — segments carrying multiple drifted (floor) words (story-scope):")
    lines.append("")
    lines.append("| Session | 1 word | 2 words | 3 words | 4+ words |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        d = r["severity"]
        lines.append(f"| {r['session']['name']} | {d['1']} | {d['2']} | {d['3']} | {d['4+']} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- The **Mode 7 floor** is the **drifted** count alone — silence where the word "
                 "claims to be, with its real audio loud right beside it. It rests on the waveform, "
                 "not on the speaker labels. Mechanical, not a hand-verified tally — spot-check with "
                 "each session's `drift-to-check.html`.")
    lines.append("- **drifted** is almost always a segment's *first* word: Whisper anchors a word to "
                 "the segment start, which often falls in the pause before anyone speaks, so the "
                 "stamp sits ~0.5s before the real onset.")
    lines.append("- **already-coded** drift is held out of the floor: when a drift candidate sits on a "
                 "segment a human already coded as another failure (M2/M4/M9/M1/…), that failure is "
                 "counted under its own mode — so the drift is a downstream symptom, not an "
                 "independent M7. Cross-referenced read-only against each session's "
                 "`axial-labels.json`. This is what drops the raw mechanical count (e.g. Moon 17, "
                 "Portal 60) to the floor below.")
    lines.append("- **isolated** and **quiet / maybe-real** are held out too: the first is likely a "
                 "hallucination (a different mode), the second likely genuine soft speech (#13). "
                 "Keeping all of these out stops Mode 7 from absorbing other modes.")
    m10 = [(r["session"]["name"], r["m10_excluded"]) for r in results if r.get("m10_excluded")]
    if m10:
        detail = ", ".join(f"{name} {n}" for name, n in m10)
        lines.append(f"- **M10 filler-loop words excluded from the floor** ({detail}). Inside a "
                     "broken-Whisper stretch the looped filler (\"Hmm.\", \"Right.\") is a decoder "
                     "artifact, not a real word whose time drifted — so it is a Mode 10 event, not "
                     "Mode 7, and is dropped here. Regions per `scripts/populate_mode10.py`.")
    lines.append("- A **wrong-speaker** check (flagging a word whose speaker label disagreed with its "
                 "sentence) was tried and removed: ear spot-checks showed it was catching the "
                 "diarizer splitting one storyteller into several speakers — character voices "
                 "especially — not timestamp drift. That diarization-granularity issue is a separate "
                 "failure mode and doesn't belong in this tool.")
    lines.append("- Thresholds are per-session adaptive (each room has its own quiet floor) and are "
                 "tunable on the CLI; the floor moves if they change.")
    lines.append("- Per-session visual breakdowns: `sessions/<id>/timestamp-drift.html` and "
                 "`sessions/<id>/drift-to-check.html` (local only — they contain transcript text).")
    return "\n".join(lines) + "\n"


# --- orchestration -----------------------------------------------------------
def process(session, args):
    story_end = None
    if "_story_end_time" in session:
        story_end = session["_story_end_time"]

    analysis = analyze_session(session, args.noise_pct, args.quiet_margin,
                               args.empty_frac, args.neighbor_sec)
    if story_end is None and session.get("story_end_seg") is not None:
        story_end = seg_end_time(analysis["transcript"], session["story_end_seg"])

    sdir = analysis["sdir"]
    html_path = sdir / "timestamp-drift.html"
    html_path.write_text(render_html(session, analysis, story_end))
    check_path = sdir / "drift-to-check.html"
    check_path.write_text(render_simple_html(session, analysis, story_end))

    counted = [r for r in analysis["seg_rows"]
               if story_end is None or r["seg"]["start"] < story_end]
    return {
        "session": session,
        "story_end": story_end,
        "tally": tally(analysis["records"], story_end),
        "severity": severity_distribution(counted),
        "noise_floor": analysis["noise_floor"],
        "threshold": analysis["threshold"],
        "m10_excluded": analysis["m10_excluded"],
        "html": str(html_path.relative_to(ROOT)),
        "check": str(check_path.relative_to(ROOT)),
    }


def main():
    p = argparse.ArgumentParser(
        description="Check word timestamps against the audio (Mode 7 drift floor)."
    )
    p.add_argument("--session", help="Process a single session id (ad hoc).")
    p.add_argument("--story-end", help="Story end for --session: 'seg:N' or 'M:SS' "
                                       "(omit for whole-session).")
    p.add_argument("--noise-pct", type=int, default=DEFAULT_NOISE_PCT,
                   help=f"Percentile estimating the noise floor (default {DEFAULT_NOISE_PCT}).")
    p.add_argument("--quiet-margin", type=float, default=DEFAULT_QUIET_MARGIN,
                   help=f"dB above the floor a word must reach to be 'loud' (default {DEFAULT_QUIET_MARGIN}).")
    p.add_argument("--empty-frac", type=float, default=DEFAULT_EMPTY_FRAC,
                   help=f"Diarization coverage below which nobody is talking (default {DEFAULT_EMPTY_FRAC}).")
    p.add_argument("--neighbor-sec", type=float, default=DEFAULT_NEIGHBOR_SEC,
                   help=f"How far beside the window to look for the word's real audio, separating "
                        f"drift from hallucination (default {DEFAULT_NEIGHBOR_SEC}).")
    p.add_argument("--summary-out",
                   default="experiments/results/tier1-timestamp-drift/summary.md",
                   help="Where to write the cross-session summary markdown.")
    args = p.parse_args()

    if args.session:
        sess = {"name": args.session, "id": args.session, "story_end_seg": None}
        if args.story_end:
            if args.story_end.startswith("seg:"):
                sess["story_end_seg"] = int(args.story_end[4:])
            else:
                m, s = args.story_end.split(":")
                sess["_story_end_time"] = int(m) * 60 + float(s)
        sessions = [sess]
    else:
        sessions = SESSIONS

    results = []
    for session in sessions:
        r = process(session, args)
        results.append(r)

        s = r["session"]
        scope = "whole-session" if r["story_end"] is None else f"story 0–{mmss(r['story_end'])}"
        st = r["tally"]["story"]
        print(f"\n{s['name']} ({s['id']}) · {scope}")
        print(f"  floor {r['noise_floor']:.1f} dBFS · loud ≥ {r['threshold']:.1f} dBFS")
        print(f"  Mode 7 floor (story): {st['floor']:>4} drifted words of {st['total']} checked")
        print(f"  already coded as another mode (excluded): {st['coded']}")
        print(f"  isolated (≈hallucination): {st['isolated']}  ·  "
              f"quiet / maybe-real (→ #13): {st['ambiguous']}")
        if r["m10_excluded"]:
            print(f"  M10 filler-loop words excluded: {r['m10_excluded']}")
        print(f"  html:  {r['html']}")
        print(f"  check: {r['check']}")

    if not args.session:
        summary_path = ROOT / args.summary_out
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(build_summary(results, args.noise_pct, args.quiet_margin))
        print(f"\nWrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
