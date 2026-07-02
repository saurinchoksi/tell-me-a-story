#!/usr/bin/env python3
"""Missed-speech recovery probe — can a context-primed re-decode rescue no-trace speech?

The lead (2026-07-01, namefix validation): a clip re-decode with the world's cast in
Whisper's ear recovered "when the Pandavas and Kauravas" that blind Whisper had dropped
ENTIRELY, and that Choksi's own ear barely caught. Missed speech (a voice on the diarizer,
no segment at all) has resisted every recovery attempt so far (the quiet-speech experiments,
8 approaches, no recovery). This probe measures whether the context-primed re-decode is
different.

Method: for each missed-speech gap (gap_analysis.find_gaps — no transcribed line, not even
an [unintelligible] marker) in the Mahabharata-world sessions, re-decode [start-1.0,
end+1.0] with the world+cast prompt, keep only re-decoded words whose midpoint falls INSIDE
the gap, and report per gap: recovered words or nothing. Ear-check cards (embedded clips)
for every gap that yielded words — a recovered phrase is only real once a human confirms it
(the prompt can force text into noise; that is exactly what the cards test).

READ-ONLY. Committable memo (counts only) -> emp/results/missed-speech-probe/summary.md;
cards (transcript text + audio) -> emp/results/visuals/missed-speech-probe.html (gitignored).

Usage: python emp/src/missed_speech_probe.py [session ...]   (default: the 2 Mahabharata sessions)
"""
import base64
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gap_analysis import find_gaps, load_session  # noqa: E402
import worldcast  # noqa: E402

MODEL = "mlx-community/whisper-large-v3-mlx"
MIN_GAP = 1.0     # same floor as the missed-speech sweep's default findings
PAD = 1.0         # context each side of the gap

DEFAULT_SESSIONS = ["20260117-202237", "20260211-210718"]  # both Mahabharata world


def mahabharata_prompt() -> str:
    split = json.loads((ROOT / "data/worlds-cache/mahabharata.json").read_text())
    cast = worldcast.correction_cast(split)
    import namefix
    return namefix.build_prompt("Mahabharata", cast)


def cut_clip(sdir: Path, start: float, end: float, out: Path) -> bool:
    ff = shutil.which("ffmpeg") or "ffmpeg"
    frm = max(0.0, start - 1.5)
    cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(sdir / "audio.m4a"),
           "-t", f"{end - frm + 1.0:.3f}", "-ac", "1", "-ar", "22050",
           "-c:a", "libmp3lame", "-q:a", "6", str(out)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and out.exists()


def main(session_ids):
    import mlx_whisper
    from mlx_whisper.audio import SAMPLE_RATE, load_audio

    prompt = mahabharata_prompt()
    print(f"[probe] prompt: {prompt[:100]}...", file=sys.stderr)

    rows = []
    for sid in session_ids:
        sdir, transcript, diarization = load_session(sid)
        gaps = find_gaps(diarization, transcript, MIN_GAP)
        print(f"[probe] {sid}: {len(gaps)} missed-speech gaps >= {MIN_GAP}s", file=sys.stderr)
        if not gaps:
            continue
        audio = np.array(load_audio(str(sdir / "audio.m4a"))).astype(np.float32)
        for g in gaps:
            gs, ge = g["start"], g["end"]
            i0 = max(0, int((gs - PAD) * SAMPLE_RATE))
            i1 = int((ge + PAD) * SAMPLE_RATE)
            r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                                       word_timestamps=True, initial_prompt=prompt,
                                       condition_on_previous_text=True, verbose=False)
            clip_t0 = i0 / SAMPLE_RATE
            inside = []
            for s in r.get("segments", []):
                for w in s.get("words", []):
                    if w.get("start") is None:
                        continue
                    mid = clip_t0 + (w["start"] + w.get("end", w["start"])) / 2
                    if gs <= mid <= ge:
                        inside.append((w.get("word") or "").strip())
            rows.append({"session": sid, "start": gs, "end": ge, "dur": round(ge - gs, 2),
                         "speaker": g.get("speaker"), "recovered": " ".join(inside),
                         "sdir": sdir})
            print(f"    {gs:7.1f}-{ge:7.1f} ({ge-gs:4.1f}s) -> "
                  f"{' '.join(inside)[:60] or '(nothing)'}", file=sys.stderr)

    yielded = [r for r in rows if r["recovered"]]
    # ---- committable memo (counts only) ----
    outdir = ROOT / "emp/results/missed-speech-probe"
    outdir.mkdir(parents=True, exist_ok=True)
    memo = ["# Missed-speech recovery probe", "",
            f"Gaps probed (voice on the diarizer, no transcript trace, >= {MIN_GAP}s): "
            f"{len(rows)} across {len(session_ids)} Mahabharata-world sessions.",
            f"Gaps where a world+cast-primed re-decode produced words INSIDE the gap: "
            f"{len(yielded)} of {len(rows)}.", "",
            "A produced word is a CANDIDATE recovery, not a confirmed one — the prompt can",
            "force text into noise. Every yielding gap has an ear-check card (gitignored",
            "visuals). The decision this feeds: whether context-primed re-decoding earns a",
            "place as a missed-speech treatment, where 8 prior approaches recovered nothing.", ""]
    per = {}
    for r in rows:
        per.setdefault(r["session"], [0, 0])
        per[r["session"]][0] += 1
        per[r["session"]][1] += bool(r["recovered"])
    memo.append("| session | gaps | yielded words |")
    memo.append("|---|---|---|")
    for sid, (n, y) in per.items():
        memo.append(f"| {sid} | {n} | {y} |")
    (outdir / "summary.md").write_text("\n".join(memo) + "\n")
    print("\n".join(memo))

    # ---- ear-check cards for every yielding gap ----
    clipdir = ROOT / "emp/results/visuals/missed-speech-clips"
    clipdir.mkdir(parents=True, exist_ok=True)
    cards = ""
    for i, r in enumerate(yielded):
        clip = clipdir / f"ms{i}.mp3"
        tag = ""
        if cut_clip(r["sdir"], r["start"], r["end"], clip):
            b = base64.b64encode(clip.read_bytes()).decode()
            tag = f'<audio controls preload="none" src="data:audio/mpeg;base64,{b}"></audio>'
        cards += (f'<div class="card"><div class="h">{i+1}. {html.escape(r["session"])} '
                  f'{r["start"]:.1f}-{r["end"]:.1f}s ({r["dur"]}s, {html.escape(str(r["speaker"]))})</div>'
                  f'<div class="tok">re-decode heard: &ldquo;{html.escape(r["recovered"])}&rdquo;</div>'
                  f'{tag}<div class="q">Is that really said in the clip?</div></div>')
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Missed-speech probe — ear check</title><style>"
            "body{font-family:-apple-system,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#fafafa}"
            ".card{background:#fff;border-radius:12px;padding:12px 16px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}"
            ".h{font-size:.78rem;color:#888}.tok{font-family:ui-monospace,monospace;font-weight:600;margin:.3em 0}"
            ".q{font-size:.84rem;color:#a67c00;margin-top:.4em}audio{width:100%}"
            "</style></head><body><h2>Missed-speech probe: candidate recoveries</h2>" + cards + "</body></html>")
    (ROOT / "emp/results/visuals/missed-speech-probe.html").write_text(page)
    print(f"\ncards: emp/results/visuals/missed-speech-probe.html ({len(yielded)} yielding gaps)")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SESSIONS)
