#!/usr/bin/env python3
"""Validate ONLY the new piece: Qwen3.5 pass-2 world naming in the PRODUCTION segmenter
(src/story_segment_qwen35.py). Boundaries are a verbatim port of the validated sweep, so
this isolates world recognition: feed each hand-marked truth story-span's full lines to
pass2_name and compare the world it returns to the human truth label. Cheap (~7 calls).

Key checks: Pandavas -> a Mahabharata world (non-empty); Portal's engine story -> Thomas;
the made-up stories -> "" (empty = canon off). World strings can't be exact-matched fairly
(free text), so this prints pred-vs-truth to eyeball.

    ./venv/bin/python emp/src/validate_pass2_worlds.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from segment import SESSIONS, load_truth                                  # noqa: E402
from story_segment import load_segments, full_region_lines               # noqa: E402
from story_segment_qwen35 import make_reader, pass2_name                 # noqa: E402


def main():
    truth = load_truth()
    gen = make_reader()
    print(f"{'session':14} {'truth world':34} -> pred world", flush=True)
    print("-" * 72)
    for sid, name in SESSIONS.items():
        segs = load_segments(str(ROOT / "sessions" / sid))
        pos_of = {s["id"]: i for i, s in enumerate(segs)}
        for st in truth[sid]:
            sp, ep = pos_of.get(st["start"]), pos_of.get(st["end"])
            if sp is None or ep is None:
                print(f"  {name:12} (truth ids {st['start']}-{st['end']} absent — skipped)")
                continue
            region = {"start_pos": min(sp, ep), "end_pos": max(sp, ep)}
            title, world = pass2_name(gen, full_region_lines(segs, region), raw_log=[])
            tw = st.get("world", "") or "(original)"
            print(f"  {name:12} {tw[:34]:34} -> {world!r}   title={title!r}", flush=True)


if __name__ == "__main__":
    main()
