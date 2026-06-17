#!/usr/bin/env python3
"""Assemble the per-cell part files into the frozen benchmark `stories.json`.

Reads emp/results/worlds-bench/parts/*.json (one per world/flavor cell), validates
them, assigns stable ids, renders each story's `full_text` the way the model sees it
in production ([i] "line"), and writes stories.json. Validation is the orchestrator
review the plan calls for: counts, label rules, and two leakage checks —
  - a canon/hybrid medium|subtle story must NOT name its world aloud (would be a
    free giveaway), and
  - a made_up story must not contain a real franchise name (a mislabel).
ERRORS block the write; WARNINGS are printed for a human to judge.

    ./venv/bin/python emp/src/assemble_worlds_bench.py            # validate + write
    ./venv/bin/python emp/src/assemble_worlds_bench.py --check    # validate only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "emp" / "results" / "worlds-bench"
PARTS = BENCH / "parts"
OUT = BENCH / "stories.json"

# Fixed order -> stable ids, grouped canon / hybrid / made_up.
CELL_ORDER = [
    "canon_star_wars", "canon_he_man", "canon_harry_potter", "canon_paw_patrol",
    "canon_octonauts", "canon_lesser_known",
    "hybrid_star_wars", "hybrid_heman_hp", "hybrid_kids",
    "madeup_obvious", "madeup_fantasy", "madeup_scifi", "madeup_animals", "madeup_traps",
]
EXPECT = {"canon": 35, "hybrid": 15, "made_up": 50}
REQUIRED = {"bucket", "difficulty", "world_gold", "world_aliases", "lines", "tell_note"}
BUCKETS = {"canon", "hybrid", "made_up"}
DIFFS = {"obvious", "medium", "subtle"}

# Real franchise tokens that must NOT appear verbatim in a made_up story (a mislabel /
# slip). The traps are meant to EVOKE these without naming them.
DENY = [
    "star wars", "jedi", "skywalker", "darth", "death star", "yoda", "chewbacca",
    "millennium falcon", "hogwarts", "harry potter", "hermione", "dumbledore",
    "voldemort", "quidditch", "he-man", "skeletor", "grayskull", "eternia",
    "paw patrol", "octonauts", "barnacles", "kwazii", "frozen", "elsa", "arendelle",
    "batman", "gotham", "wolverine", "x-men", "professor x", "hobbit", "frodo",
    "gandalf", "mordor", "narnia", "aslan", "mowgli", "tarzan", "snow white",
    "redwall", "moomin", "tintin", "hilda",
]


def render_full(lines):
    return "\n".join(f'[{i}] "{ln}"' for i, ln in enumerate(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    args = ap.parse_args()

    errors, warnings, stories = [], [], []
    counts = {b: 0 for b in BUCKETS}

    for cell in CELL_ORDER:
        p = PARTS / f"{cell}.json"
        if not p.exists():
            errors.append(f"MISSING part file: {p.name}")
            continue
        try:
            batch = json.loads(p.read_text())["stories"]
        except (json.JSONDecodeError, KeyError) as e:
            errors.append(f"{cell}: unreadable ({e})")
            continue
        for k, st in enumerate(batch):
            tag = f"{cell}[{k}]"
            missing = REQUIRED - set(st)
            if missing:
                errors.append(f"{tag}: missing keys {missing}")
                continue
            if st["bucket"] not in BUCKETS:
                errors.append(f"{tag}: bad bucket {st['bucket']!r}")
            if st["difficulty"] not in DIFFS:
                errors.append(f"{tag}: bad difficulty {st['difficulty']!r}")
            if len(st["lines"]) < 6:
                errors.append(f"{tag}: only {len(st['lines'])} lines (<6)")

            lines_lc = " \n ".join(st["lines"]).lower()
            if st["bucket"] == "made_up":
                if st["world_gold"] or st["world_aliases"]:
                    errors.append(f"{tag}: made_up must have empty world_gold/aliases")
                hits = [d for d in DENY if d in lines_lc]
                if hits:
                    warnings.append(f"{tag}: made_up text contains franchise token(s) {hits}")
            else:  # canon / hybrid
                if not st["world_gold"]:
                    errors.append(f"{tag}: {st['bucket']} must name a world_gold")
                if st["difficulty"] in ("medium", "subtle"):
                    wg = st["world_gold"].lower()
                    # whole-name leak is the real concern (e.g. "star wars", "harry potter");
                    # partial-token matching would false-alarm on words like "star".
                    if wg in lines_lc:
                        warnings.append(f"{tag}: {st['difficulty']} story names its world "
                                        f"{st['world_gold']!r} aloud (giveaway)")

            counts[st["bucket"]] = counts.get(st["bucket"], 0) + 1
            sid = f"s{len(stories) + 1:03d}"
            stories.append({
                "id": sid, "cell": cell, "bucket": st["bucket"],
                "difficulty": st["difficulty"], "world_gold": st["world_gold"],
                "world_aliases": st["world_aliases"], "lines": st["lines"],
                "full_text": render_full(st["lines"]), "tell_note": st["tell_note"],
            })

    # count check
    for b, want in EXPECT.items():
        if counts.get(b, 0) != want:
            errors.append(f"bucket {b}: have {counts.get(b, 0)}, expected {want}")

    print(f"assembled {len(stories)} stories  "
          f"(canon={counts.get('canon', 0)} hybrid={counts.get('hybrid', 0)} "
          f"made_up={counts.get('made_up', 0)})")
    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  ! {w}")
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  X {e}")
        sys.exit("\nNOT writing stories.json — fix errors first.")

    if args.check:
        print("\n--check: validation passed; not writing.")
        return
    OUT.write_text(json.dumps({
        "_about": ("100 fabricated bedtime stories with known world labels — the "
                   "world-classifier benchmark. Synthetic (no family content), so committable. "
                   "Gold = world_gold/world_aliases/bucket/difficulty; the runner "
                   "(bench_worlds.py) reads only id+lines."),
        "n": len(stories), "counts": counts, "stories": stories,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
