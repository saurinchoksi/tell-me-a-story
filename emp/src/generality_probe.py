#!/usr/bin/env python3
"""Generality probe: on a session whose world is NOT recognizable, the chain must do NOTHING.

The no-false-positive property has to hold on stories that aren't from a known world — an
invented bedtime story has no canon to correct toward, and a world the model can't place
must not trigger guesses. The chain's safety valve is step 2 (recognize the world from the
name list; abstains -> stop). This probe runs exactly that step on the given sessions and
reports the decision. Expected on the KPop held-out and invented-story sessions: world="" ->
ZERO corrective actions.

One Qwen load for all sessions. Read-only.
Usage: python emp/src/generality_probe.py <session_id> [<session_id> ...]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from detectors.story_names._worker import build_regions  # noqa: E402
from detectors.story_names._names import story_segments  # noqa: E402
from detectors.story_names._audit import story_name_cards  # noqa: E402
from detectors.story_names import _qwen35  # noqa: E402
from qwen35 import make_reader  # noqa: E402

OUT = ROOT / "emp/results/visuals/whisper-context/generality_probe.json"

if __name__ == "__main__":
    sids = sys.argv[1:]
    if not sids:
        raise SystemExit("usage: generality_probe.py <session_id> [...]")
    t0 = time.time()
    gen = make_reader()
    print(f"qwen loaded in {time.time()-t0:.0f}s", file=sys.stderr)

    results = []
    for sid in sids:
        sdir = get_session_dir(ROOT / "sessions", sid)
        rich = json.loads((sdir / "transcript-rich.json").read_text())
        pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}
        stories = rich.get("_stories") or []
        for r in build_regions(stories, pos_of):
            segs = story_segments(rich, r, pos_of)
            cards = story_name_cards(segs, recover=True)
            names = sorted({s for c in cards for s in c["surface"]})
            world = _qwen35.recognize_world(gen, names)
            action = "CHAIN PROCEEDS (world recognized)" if world else "NO ACTION (abstained — correct for invented/unknown)"
            results.append({"session": sid, "story": r["idx"], "title": r.get("title", ""),
                            "n_names": len(names), "recognized_world": world, "verdict": action})
            print(f"[{time.time()-t0:4.0f}s] {sid} story {r['idx']} ({len(names)} names) "
                  f"-> world={world!r} => {action}", file=sys.stderr)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("WROTE", OUT, file=sys.stderr)
