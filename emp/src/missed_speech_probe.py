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

READ-ONLY on session data. Committable memo (counts only) ->
emp/results/missed-speech-probe/summary.md; the ear-check page (earcheck lib: audio +
verdict + "what I hear", saving to a co-edited sidecar) -> gitignored
emp/results/visuals/missed-speech-probe.html. Probe rows persist to a JSON so the page
regenerates without the GPU.

Usage:
    python emp/src/missed_speech_probe.py [session ...]   # run the model (default: the 2 Mahabharata sessions)
    python emp/src/missed_speech_probe.py --page-only     # rebuild the page from saved rows (no GPU)
    python emp/src/missed_speech_probe.py --serve         # serve the page with live saving
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gap_analysis import find_gaps, load_session  # noqa: E402
import worldcast  # noqa: E402
import earcheck  # noqa: E402

MODEL = "mlx-community/whisper-large-v3-mlx"
MIN_GAP = 1.0     # same floor as the missed-speech sweep's default findings
PAD = 1.0         # context each side of the gap

DEFAULT_SESSIONS = ["20260117-202237", "20260211-210718"]  # both Mahabharata world
ROWS_PATH = ROOT / "emp/results/visuals/missed-speech-rows.json"
SIDECAR = ROOT / "emp/results/visuals/missed-speech-hearings.json"
PAGE = ROOT / "emp/results/visuals/missed-speech-probe.html"
CLIPDIR = ROOT / "emp/results/visuals/missed-speech-clips"
SERVE_CMD = "python emp/src/missed_speech_probe.py --serve"
PORT = 8769


def mahabharata_prompt() -> str:
    split = json.loads((ROOT / "data/worlds-cache/mahabharata.json").read_text())
    cast = worldcast.correction_cast(split)
    import namefix
    return namefix.build_prompt("Mahabharata", cast)


def render_page() -> str:
    """The ear-check page from the saved rows (no GPU). Every yielding gap = one card with
    the candidate text, the clip, verdicts, and the "what I hear" box (the standing rule)."""
    rows = json.loads(ROWS_PATH.read_text())
    cards = []
    for i, r in enumerate(y for y in rows if y["recovered"]):
        cid = f"{r['session']}-{r['start']:.1f}"
        clip = earcheck.cut_clip_b64(ROOT / "sessions" / r["session"] / "audio.m4a",
                                     r["start"], r["end"], CLIPDIR / f"ms{i}.mp3")
        cards.append({
            "id": cid,
            "header": f"{i+1}. {earcheck.esc(r['session'])} · {r['start']:.1f}–{r['end']:.1f}s "
                      f"({r['dur']}s, {earcheck.esc(str(r['speaker']))})",
            "body_html": f"re-decode heard: <span class='tok'>&ldquo;{earcheck.esc(r['recovered'])}&rdquo;</span>",
            "clip": clip,
            "verdicts": ["really said", "partly right", "not said", "can't tell"],
        })
    return earcheck.build_page(
        "Missed-speech probe: candidate recoveries",
        "Each stretch had a voice on the speaker-detector but NO transcript at all. The line "
        "shown is what a context-primed re-listen produced. Play the clip; type what YOU hear; "
        "pick a verdict.",
        cards, SIDECAR, SERVE_CMD)


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
                         "speaker": g.get("speaker"), "recovered": " ".join(inside)})
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

    # ---- persist rows; build the ear-check page (earcheck lib: hear-box + verdicts) ----
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    PAGE.write_text(render_page())
    print(f"\npage: {PAGE} ({len(yielded)} yielding gaps; serve: {SERVE_CMD})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--serve" in sys.argv:
        PAGE.write_text(render_page())
        earcheck.serve(render_page, SIDECAR, PORT)
    elif "--page-only" in sys.argv:
        PAGE.write_text(render_page())
        print(f"rebuilt {PAGE} from saved rows (no model run)")
    else:
        main(args or DEFAULT_SESSIONS)
