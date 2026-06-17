#!/usr/bin/env python3
"""Cold test: run a chosen world-classifier prompt on the HELD-OUT real sessions.

These recordings were never used to tune the prompt, so this measures generalization
— and directly re-runs the Feb-11 case (the Mahabharata session the production prompt
read as "made-up", catching 0 canon). The prompt sees only the transcript; the answer
key (coldkey.json, gitignored) is read only afterward, to score. We feed the WHOLE
transcript (the decision the benchmark justified); a session that mixes several stories
is a coarse single-label test, which is fine for these two (one is Mahabharata
throughout, the other improvised throughout).

LOCAL-ONLY: Gemma via MLX. Predictions echo transcript fragments, so cold-pred.json is
gitignored alongside the key.

    ./venv/bin/python emp/src/cold_worlds.py --prompt general_reasoned
    ./venv/bin/python emp/src/cold_worlds.py --prompt general_reasoned 20260211-210718
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from segment import make_reader, extract_json, MODEL_ID  # noqa: E402
from bench_worlds import PROMPTS  # noqa: E402
import score_worlds as sw  # noqa: E402

BENCH = ROOT / "emp" / "results" / "worlds-bench"
KEY = BENCH / "coldkey.json"


def render_transcript(session_id):
    """Whole transcript as [seg_id] "text" lines (production's rendering, unsampled)."""
    rich = json.loads((ROOT / "sessions" / session_id / "transcript-rich.json").read_text())
    lines = []
    for seg in rich["segments"]:
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(f'[{seg["id"]}] "{t}"')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="general_reasoned", choices=list(PROMPTS),
                    help="which candidate prompt to cold-test (default: general_reasoned)")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("sessions", nargs="*", help="session ids (default: all in coldkey.json)")
    args = ap.parse_args()

    key = json.loads(KEY.read_text())["sessions"]
    sessions = args.sessions or list(key)
    for s in sessions:
        if s not in key:
            ap.error(f"{s} not in coldkey.json")

    print(f"loading {MODEL_ID} …", file=sys.stderr)
    gen = make_reader()
    template = PROMPTS[args.prompt]

    rows, preds = [], []
    for sid in sessions:
        transcript = render_transcript(sid)                      # model sees ONLY this
        raw = gen(template.replace("{story}", transcript), max_tokens=args.max_tokens)
        obj = extract_json(raw)
        world = str(obj.get("world", "")).strip() if isinstance(obj, dict) else ""
        preds.append({"session": sid, "world_pred": world, "raw": raw})

        k = key[sid]                                             # key read AFTER prediction
        story = {"bucket": "made_up" if k["made_up"] else "canon",
                 "world_gold": k["true_world"], "world_aliases": k["aliases"]}
        outcome = sw.classify(story, world)
        rows.append((sid, k["true_world"] or "(made-up)", world, outcome))

    (BENCH / "cold-pred.json").write_text(json.dumps(
        {"prompt": args.prompt, "model": MODEL_ID, "results": preds}, indent=2))

    print("\n" + "=" * 70)
    print(f"COLD TEST — prompt {args.prompt!r}")
    print("=" * 70)
    for sid, true_w, pred, outcome in rows:
        flag = "PASS" if outcome == "correct" else "FAIL"
        print(f"  [{flag}] {sid}  expected={true_w!r}  predicted={pred!r}  ({outcome})")
    n_pass = sum(1 for *_, o in rows if o == "correct")
    print(f"\n  {n_pass}/{len(rows)} correct")


if __name__ == "__main__":
    main()
