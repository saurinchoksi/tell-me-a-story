#!/usr/bin/env python3
"""Measure the world-grounded normalizer on a session — the before/after diff (the Sierra deliverable).

Runs `worldnorm.world_normalize` (cold, empty world dicts) on a session's HONEST transcript and,
when a by-ear name key exists in that session's validation-notes.json, joins the normalizer's
decision at each keyed position to the human's answer. For each name it prints:

    heard-now | old on-screen | recognized world | action (auto->X / queued->X / unchanged) | answer | ✓/✗

so you can see exactly what the cold, dictionary-empty system does: recognize the world from the
name list, auto-apply only genuine sound-alikes, and QUEUE (keep honest) the non-sound-alike
garbles the old world-blind normalizer used to confidently mis-substitute.

Read-only on session data. Usage:  venv/bin/python src/worldnorm_measure.py <session-id> [<id> ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
import worldnorm  # noqa: E402


def load_key(session_dir: Path) -> dict:
    """The by-ear name key from validation-notes.json, keyed (segment_id, word_index) -> answer.
    A note is a name-correction when its `text` (correct name) differs from `wordText` (on-screen)."""
    path = session_dir / "validation-notes.json"
    if not path.exists():
        return {}
    notes = json.loads(path.read_text()).get("notes", [])
    key = {}
    for n in notes:
        wt, tx = (n.get("wordText") or "").strip(), (n.get("text") or "").strip()
        sid, wi = n.get("segmentId"), n.get("wordIndex")
        if tx and wt and tx.lower() != wt.lower() and sid is not None and wi is not None:
            key[(sid, wi)] = {"answer": tx, "on_screen": wt}
    return key


def decision_index(result: dict) -> dict:
    """(segment_id, word_index) -> the normalizer's decision, from auto + pending occurrences."""
    idx = {}
    for action, groups in (("auto", result["auto"]), ("pending", result["pending"])):
        for g in groups:
            for o in g["occurrences"]:
                idx[(o["segment_id"], o["word_index"])] = {
                    "action": action, "canonical": g["canonical"], "method": g["method"],
                    "heard": g["heard"], "confident": g["suggestion_confident"],
                    "vote_count": g.get("vote_count"),
                }
    return idx


def current_words(transcript: dict) -> dict:
    """(segment_id, word_index) -> the current (honest) surface word the normalizer actually sees."""
    out = {}
    for s in transcript["segments"]:
        for wi, w in enumerate(s.get("words", [])):
            out[(s["id"], wi)] = w["word"].strip()
    return out


def measure(session_id: str, use_cached: bool = False) -> None:
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    transcript = json.loads((session_dir / "transcript-rich.json").read_text())
    key = load_key(session_dir)

    # Cache the model result (gitignored, transcript-derived) so scoring can be re-run without
    # reloading Qwen. --cached reuses the last run; otherwise run the model and refresh the cache.
    cache = ROOT / "emp" / "results" / "visuals" / session_id / "worldnorm-result.json"
    print(f"\n{'='*100}\nSESSION {session_id}\n{'='*100}")
    if use_cached and cache.exists():
        print(f"Using cached result: {cache}\n")
        result = json.loads(cache.read_text())
    else:
        print("Running world_normalize (cold: empty world dicts, first contact)... this loads Qwen3.5-4B.\n")
        result = worldnorm.world_normalize(transcript, world_dicts={})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    for w in result["worlds"]:
        print(f"  Story {w['story_id']} \"{w['title']}\": saved world={w['saved_world']!r} "
              f"-> RECOGNIZED {w['recognized_world']!r}  ({w['n_names']} candidate names)")
    print(f"\n  auto-applied: {len(result['auto'])} names   queued for review: {len(result['pending'])} names")

    # -------- everything the normalizer decided (whole session, not just keyed) --------
    print(f"\n  {'-'*40} ALL DECISIONS {'-'*40}")
    print(f"  {'HEARD':16} {'ACTION':8} {'-> SUGGESTION':22} {'METHOD':12} {'VOTES':6} OCC")
    for g in sorted(result["auto"], key=lambda x: -len(x["occurrences"])):
        print(f"  {g['heard']:16} {'AUTO':8} -> {g['canonical']:20} {g['method']:12} "
              f"{str(g.get('vote_count') or ''):6} {len(g['occurrences'])}")
    for g in sorted(result["pending"], key=lambda x: -len(x["occurrences"])):
        print(f"  {g['heard']:16} {'QUEUE':8} -> {g['canonical']:20} {g['method']:12} "
              f"{str(g.get('vote_count') or ''):6} {len(g['occurrences'])}")

    if not key:
        print("\n  (no by-ear key for this session — decisions above are unscored)\n")
        return

    # -------- scored against the by-ear key --------
    idx = decision_index(result)
    now = current_words(transcript)
    print(f"\n  {'-'*36} SCORED vs BY-EAR KEY ({len(key)} items) {'-'*36}")
    print(f"  {'SEG/WI':9} {'HEARD-NOW':14} {'ON-SCREEN':12} {'ACTION':7} {'-> RESULT':20} {'ANSWER':14} MATCH")
    tally = {"auto_right": 0, "auto_wrong": 0, "pending_right": 0, "pending_wrong": 0, "unchanged": 0}
    for (sid, wi), k in sorted(key.items()):
        heard_now = now.get((sid, wi), "?")
        d = idx.get((sid, wi))
        answer = k["answer"]
        if d is None:
            action, result_txt, match = "—", "(unchanged/honest)", ""
            tally["unchanged"] += 1
        else:
            action = "AUTO" if d["action"] == "auto" else "QUEUE"
            result_txt = f"{d['method']}: {d['canonical']}"
            hit = d["canonical"].lower() == answer.lower()
            match = "OK" if hit else "XX"
            tally[f"{d['action']}_{'right' if hit else 'wrong'}"] += 1
        print(f"  {str(sid)+'/'+str(wi):9} {heard_now:14.14} {k['on_screen']:12.12} "
              f"{action:7} {result_txt:20.20} {answer:14.14} {match}")

    print(f"\n  SUMMARY (vs {len(key)}-item by-ear key):")
    print(f"    auto-applied & correct : {tally['auto_right']}")
    print(f"    auto-applied & WRONG   : {tally['auto_wrong']}   <- the number to watch (false substitutions)")
    print(f"    queued & would-be-right: {tally['pending_right']}   (kept honest; a human bless makes it right)")
    print(f"    queued & would-be-wrong: {tally['pending_wrong']}   (queued, so a human rejects it -> no harm)")
    print(f"    left unchanged (honest): {tally['unchanged']}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_cached = "--cached" in args
    ids = [a for a in args if not a.startswith("-")] or ["20260211-210718"]
    for sid in ids:
        measure(sid, use_cached=use_cached)
