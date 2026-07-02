#!/usr/bin/env python3
"""Census of non-name transcript garbage — the scoping sweep for the next quality arc.

Choksi's observation (2026-07-01, while marking the by-ear name key): beyond the names,
the transcripts carry other junk — a Cyrillic word that was never said, stuck repeats,
words Whisper itself doesn't believe. This sweep COUNTS that junk mechanically across all
sessions so the next arc starts from a denominator instead of anecdotes.

Classes counted (each precise, each with its held-out exclusions per tmas-eval-sweep):
  script    — tokens containing non-Latin letters (какие). English sessions only, so any
              non-Latin script is Whisper drifting language, not speech.
  repeat    — a run of >= 3 identical cleaned words inside one segment's words ("Patient
              Patient Patient"). Runs inside the known M10 broken-Whisper stretches are
              EXCLUDED (that mode owns them); 2x repeats are NOT counted (often real
              speech: "favorite favorite").
  zeroprob  — words failing filters.near_zero_probability (prob < 0.01) — Whisper's own
              disbelief. Reported per-cluster (adjacent zero-prob words merge).
  symbol    — tokens that are music notation / bracketed artifacts / punctuation-only.

Segments a human already coded in axial-labels.json (any mode) are counted separately in
the report ("already coded") — this sweep's number is the UNCODED residue, so nothing is
double-counted against the existing failure-mode pivot.

READ-ONLY. Committable summary (counts only, no transcript text) to
emp/results/garbage-census/summary.md; per-session detail + check-by-ear cards (pre-cut
clips — transcript text, so gitignored) to emp/results/visuals/<sid>/garbage-*.

Usage: python emp/src/garbage_census.py [--cards N]
"""
import argparse
import base64
import html
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from filters import near_zero_probability  # noqa: E402
from populate_mode10 import STRETCHES  # noqa: E402

NON_LATIN = re.compile(r"[^\x00-\x7f]")
LATIN_LETTER = re.compile(r"[a-zA-Z]")
SYMBOLY = re.compile(r"^[\W♪♫\[\]()\-–—.…]+$")


def clean_token(raw: str) -> str:
    return re.sub(r"[^a-z]", "", raw.strip().lower())


def coded_segment_ids(session_dir: Path) -> set:
    p = session_dir / "axial-labels.json"
    if not p.exists():
        return set()
    return {l["segmentId"] for l in json.loads(p.read_text()).get("labels", [])}


def m10_positions(sid: str) -> set:
    out = set()
    for s in STRETCHES:
        if s["session"] == sid:
            out |= set(range(s["lo"], s["hi"] + 1))
    return out


def census_session(session_dir: Path):
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    sid = session_dir.name
    coded = coded_segment_ids(session_dir)
    pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}
    m10 = m10_positions(sid)

    findings = []  # {cls, segment_id, start, end, token(s), coded}
    for seg in rich["segments"]:
        words = seg.get("words") or []
        seg_coded = seg["id"] in coded
        in_m10 = pos_of.get(seg["id"], -1) in m10

        # script + symbol per word
        for w in words:
            raw = (w.get("word") or "").strip()
            if not raw:
                continue
            if NON_LATIN.search(raw) and not LATIN_LETTER.search(raw):
                findings.append({"cls": "script", "segment_id": seg["id"],
                                 "start": w.get("start"), "end": w.get("end"),
                                 "token": raw, "coded": seg_coded})
            elif SYMBOLY.match(raw):
                findings.append({"cls": "symbol", "segment_id": seg["id"],
                                 "start": w.get("start"), "end": w.get("end"),
                                 "token": raw, "coded": seg_coded})

        # repeat runs (>=3 identical cleaned words), skipping M10-owned segments
        if not in_m10:
            run_start = 0
            cl = [clean_token(w.get("word") or "") for w in words]
            i = 0
            while i < len(cl):
                j = i
                while j + 1 < len(cl) and cl[j + 1] == cl[i] and cl[i]:
                    j += 1
                if j - i + 1 >= 3:
                    findings.append({"cls": "repeat", "segment_id": seg["id"],
                                     "start": words[i].get("start"), "end": words[j].get("end"),
                                     "token": f"{words[i]['word'].strip()} x{j - i + 1}",
                                     "coded": seg_coded})
                i = j + 1

        # zero-prob clusters (adjacent zero-prob words merge)
        i = 0
        while i < len(words):
            if near_zero_probability(words[i]):
                j = i
                while j + 1 < len(words) and near_zero_probability(words[j + 1]):
                    j += 1
                findings.append({"cls": "zeroprob", "segment_id": seg["id"],
                                 "start": words[i].get("start"), "end": words[j].get("end"),
                                 "token": " ".join(w["word"].strip() for w in words[i:j + 1])[:40],
                                 "coded": seg_coded})
                i = j + 1
            else:
                i += 1
    return findings


def cut_clip(session_dir: Path, start: float, end: float, out: Path):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    frm = max(0.0, (start or 0) - 1.5)
    dur = ((end or start or 0) - (start or 0)) + 1.5 + 1.0
    cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(session_dir / "audio.m4a"),
           "-t", f"{dur:.3f}", "-ac", "1", "-ar", "22050", "-c:a", "libmp3lame",
           "-q:a", "6", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, default=15, help="check-by-ear cards for the worst N")
    a = ap.parse_args()

    all_findings = {}
    for d in sorted((ROOT / "sessions").iterdir()):
        if not d.is_dir() or not (d / "transcript-rich.json").exists():
            continue
        if d.name == "00000000-000000":
            continue  # fixture
        all_findings[d.name] = census_session(d)

    # ---- committable summary (counts only) ----
    lines = ["# Non-name garbage census", "",
             "Counts of mechanically-detectable non-name junk per session. `uncoded` is the",
             "residue no human has coded yet (the next arc's denominator); `coded` overlaps",
             "existing failure-mode codes and is excluded from the headline. Repeat runs inside",
             "the known M10 stretches are excluded entirely (that mode owns them).", ""]
    classes = ["script", "repeat", "zeroprob", "symbol"]
    lines.append("| session | " + " | ".join(f"{c} (uncoded/coded)" for c in classes) + " |")
    lines.append("|---|" + "---|" * len(classes))
    totals = defaultdict(lambda: [0, 0])
    for sid, fs in all_findings.items():
        row = [sid]
        for c in classes:
            un = sum(1 for f in fs if f["cls"] == c and not f["coded"])
            co = sum(1 for f in fs if f["cls"] == c and f["coded"])
            totals[c][0] += un
            totals[c][1] += co
            row.append(f"{un}/{co}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **total** | " + " | ".join(f"**{totals[c][0]}/{totals[c][1]}**" for c in classes) + " |")
    outdir = ROOT / "emp/results/garbage-census"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- check-by-ear cards for the worst N uncoded (script + repeat + zeroprob clusters, by duration) ----
    cands = []
    for sid, fs in all_findings.items():
        for f in fs:
            if f["coded"] or f["start"] is None:
                continue
            f["session"] = sid
            f["span"] = (f["end"] or f["start"]) - f["start"]
            cands.append(f)
    order = {"script": 0, "repeat": 1, "zeroprob": 2, "symbol": 3}
    cands.sort(key=lambda f: (order[f["cls"]], -f["span"]))
    picks = cands[:a.cards]

    cards_html = ""
    clipdir = ROOT / "emp/results/visuals/garbage-clips"
    clipdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(picks):
        sdir = ROOT / "sessions" / f["session"]
        clip = clipdir / f"g{i}.mp3"
        ok = cut_clip(sdir, f["start"], f["end"], clip)
        audio_tag = ""
        if ok:
            b = base64.b64encode(clip.read_bytes()).decode()
            audio_tag = f'<audio controls preload="none" src="data:audio/mpeg;base64,{b}"></audio>'
        cards_html += (f'<div class="card"><div class="h">{i+1}. [{f["cls"]}] '
                       f'{esc(f["session"])} seg {esc(f["segment_id"])} @ {f["start"]:.1f}s</div>'
                       f'<div class="tok">{esc(f["token"])}</div>{audio_tag}</div>')

    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Garbage census — check by ear</title><style>"
            "body{font-family:-apple-system,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#fafafa}"
            ".card{background:#fff;border-radius:12px;padding:12px 16px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}"
            ".h{font-size:.78rem;color:#888}.tok{font-family:ui-monospace,monospace;font-weight:600;margin:.3em 0}"
            "audio{width:100%}</style></head><body>"
            "<h2>Non-name garbage: the worst uncoded finds</h2>"
            "<p style='color:#666;font-size:.88rem'>Question per card: was anything really said here, "
            "and is the transcript's token junk?</p>" + cards_html + "</body></html>")
    out = ROOT / "emp/results/visuals/garbage-census-earcheck.html"
    out.write_text(page)
    print(f"\ncards: {out}")


def esc(x):
    return html.escape(str(x))


if __name__ == "__main__":
    main()
