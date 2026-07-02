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
emp/results/garbage-census/summary.md; the ear-check page (earcheck lib: audio + verdict +
"what I hear" saving to a co-edited sidecar) -> gitignored emp/results/visuals/.

Usage:
    python emp/src/garbage_census.py [--cards N]
    python emp/src/garbage_census.py --serve      # serve the card page with live saving
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from filters import near_zero_probability  # noqa: E402
from populate_mode10 import STRETCHES  # noqa: E402
import earcheck  # noqa: E402

SIDECAR = ROOT / "emp/results/visuals/garbage-hearings.json"
PAGE = ROOT / "emp/results/visuals/garbage-census-earcheck.html"
PICKS_PATH = ROOT / "emp/results/visuals/garbage-picks.json"
CLIPDIR = ROOT / "emp/results/visuals/garbage-clips"
SERVE_CMD = "python emp/src/garbage_census.py --serve"
PORT = 8770

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


def render_page() -> str:
    """Ear-check page from the saved picks (earcheck lib: hear-box + verdicts — the
    standing rule: every listening artifact takes 'what I hear')."""
    picks = json.loads(PICKS_PATH.read_text())
    cards = []
    for i, f in enumerate(picks):
        cid = f"{f['session']}-{f['segment_id']}-{f['start']:.1f}"
        clip = earcheck.cut_clip_b64(ROOT / "sessions" / f["session"] / "audio.m4a",
                                     f["start"], f.get("end") or f["start"],
                                     CLIPDIR / f"g{i}.mp3")
        cards.append({
            "id": cid,
            "header": f"{i+1}. [{earcheck.esc(f['cls'])}] {earcheck.esc(f['session'])} "
                      f"seg {earcheck.esc(f['segment_id'])} @ {f['start']:.1f}s",
            "body_html": f"transcript token: <span class='tok'>{earcheck.esc(f['token'])}</span>",
            "clip": clip,
            "verdicts": ["junk (nothing said)", "real speech, wrong token",
                         "token is right", "can't tell"],
        })
    return earcheck.build_page(
        "Garbage census: the worst uncoded finds",
        "Mechanically-flagged non-name junk. Play each clip; type what you actually hear; "
        "pick a verdict.",
        cards, SIDECAR, SERVE_CMD)


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

    # ---- pick the worst N uncoded (script + repeat + zeroprob, by duration), persist, page ----
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
    PICKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(json.dumps(picks, indent=2, ensure_ascii=False))
    PAGE.write_text(render_page())
    print(f"\npage: {PAGE} ({len(picks)} cards; serve: {SERVE_CMD})")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        PAGE.write_text(render_page())
        earcheck.serve(render_page, SIDECAR, PORT)
    else:
        main()
