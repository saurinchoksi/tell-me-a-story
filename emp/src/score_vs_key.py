#!/usr/bin/env python3
"""Score a transcript against the 36-item by-ear answer key — joined by AUDIO TIME.

The measuring stick for the "Whisper-with-context" investigation: how close does a
transcription get to Choksi's by-ear name key ON ITS OWN, with no human correction?

Why time-join, not position-join: re-transcribing (or any re-enrich) changes segment ids
and word indices, so we can't join by (segmentId, wordIndex). But every key note carries
`wordStart` (== the original word's audio `start`, present on all 36), and the audio is the
same file — so we join each answer to whatever word the transcript places at that TIME.

For each key note we look at every transcript word whose start falls within ±TOL of the
note's wordStart (drift tolerance), plus the single nearest word, and count a HIT if any of
them matches the answer (`text`) by exact cleaned spelling OR shared Double-Metaphone code
(sound-alike — the honest bar, since "Bhima"/"Bheem" is a correct recovery). We also record
the nearest word verbatim, so a human can eyeball whether the prompt FORCED a wrong name
(introduced hallucination) rather than recovering the real one.

Read-only. Usage:
    python emp/src/score_vs_key.py <session_id> <transcript.json> [--tol 1.0] [--label NAME]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from detectors.phonetics import clean, codes  # noqa: E402

UNKNOWABLE = {"???", "?", ""}  # answers a transcriber cannot be expected to produce (e.g. the Cyrillic hallucination)


def load_key(session_dir: Path) -> list[dict]:
    """The by-ear key: one entry per note carrying an answer + the audio time to join on."""
    notes = json.loads((session_dir / "validation-notes.json").read_text()).get("notes", [])
    key = []
    for n in notes:
        ans = (n.get("text") or "").strip()
        ws = n.get("wordStart")
        if not ans or ws is None:
            continue
        key.append({"answer": ans, "word_start": float(ws),
                    "on_screen": (n.get("wordText") or "").strip(),
                    "segment_id": n.get("segmentId"), "word_index": n.get("wordIndex")})
    return key


def flatten_words(transcript: dict) -> list[dict]:
    """All words across segments as {word, start, end}, sorted by start. Works for both
    transcript-rich.json and a raw mlx_whisper.transcribe result (same segment/word shape)."""
    out = []
    for s in transcript.get("segments", []):
        for w in s.get("words", []):
            if w.get("start") is None:
                continue
            out.append({"word": (w.get("word") or "").strip(),
                        "start": float(w["start"]), "end": float(w.get("end", w["start"]))})
    out.sort(key=lambda w: w["start"])
    return out


def _matches(token: str, answer: str) -> str | None:
    """"exact" if cleaned spellings equal, "dm" if they share a Double-Metaphone code, else None."""
    ct, ca = clean(token), clean(answer)
    if not ct or not ca:
        return None
    if ct == ca:
        return "exact"
    if codes(ct) & codes(ca):
        return "dm"
    return None


def score(key: list[dict], words: list[dict], tol: float = 1.0) -> dict:
    """Per-note scoring by time-join. Returns rows + tallies. A hit = some word within ±tol
    of the note's wordStart matches the answer (exact or sound-alike). `nearest` is the closest
    word by time, for eyeballing what the transcript actually put there."""
    rows = []
    tally = {"exact": 0, "dm": 0, "miss": 0, "unknowable": 0}
    for k in key:
        ws, ans = k["word_start"], k["answer"]
        near = [w for w in words if abs(w["start"] - ws) <= tol]
        nearest = min(words, key=lambda w: abs(w["start"] - ws)) if words else None
        if ans in UNKNOWABLE:
            tally["unknowable"] += 1
            match, matched_word = "unknowable", ""
        else:
            best = None
            for w in near or ([nearest] if nearest else []):
                m = _matches(w["word"], ans)
                if m == "exact":
                    best, matched_word = "exact", w["word"]; break
                if m == "dm" and best is None:
                    best, matched_word = "dm", w["word"]
            match = best or "miss"
            matched_word = matched_word if best else ""
            tally[match] += 1
        rows.append({"answer": ans, "on_screen": k["on_screen"], "word_start": round(ws, 2),
                     "nearest": nearest["word"] if nearest else "", "matched_word": matched_word,
                     "match": match})
    scoreable = len(key) - tally["unknowable"]
    hits = tally["exact"] + tally["dm"]
    return {"rows": rows, "tally": tally, "n_key": len(key), "scoreable": scoreable,
            "hits": hits, "recall": round(hits / scoreable, 3) if scoreable else 0.0}


def print_report(res: dict, label: str) -> None:
    print(f"\n{'='*90}\n{label}\n{'='*90}")
    print(f"  {'TIME':>7}  {'ANSWER':16} {'NEAREST WORD':18} {'MATCHED':14} RESULT")
    for r in sorted(res["rows"], key=lambda x: x["word_start"]):
        flag = {"exact": "OK exact", "dm": "OK ~sound", "miss": "XX miss",
                "unknowable": "-- n/a"}[r["match"]]
        print(f"  {r['word_start']:7.1f}  {r['answer']:16.16} {r['nearest']:18.18} "
              f"{r['matched_word']:14.14} {flag}")
    t = res["tally"]
    print(f"\n  HITS {res['hits']}/{res['scoreable']} (recall {res['recall']}) "
          f"| exact {t['exact']}  sound-alike {t['dm']}  miss {t['miss']}  "
          f"unknowable {t['unknowable']} (excluded)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("transcript", help="path to a transcript JSON (rich or a re-transcribe result)")
    ap.add_argument("--tol", type=float, default=1.0, help="time tolerance in seconds")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    session_dir = get_session_dir(ROOT / "sessions", a.session_id)
    key = load_key(session_dir)
    transcript = json.loads(Path(a.transcript).read_text())
    res = score(key, flatten_words(transcript), tol=a.tol)
    print_report(res, a.label or f"{a.session_id}  <-  {a.transcript}")
