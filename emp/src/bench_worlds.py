#!/usr/bin/env python3
"""World-classifier benchmark RUNNER — offline, local-only, FIREWALLED.

Runs a candidate world-naming prompt over the 100 fabricated stories in
`emp/results/worlds-bench/stories.json` and writes the model's predictions to
`emp/results/worlds-bench/pred/<candidate>[.sampled].json`. The companion
`score_worlds.py` is the ONLY side that reads the gold labels — this runner reads
each story's `id` + `lines` and NOTHING ELSE (it never looks at `world_gold`,
`bucket`, or `difficulty`). That firewall is what keeps the eval honest, mirroring
the audit/score split in `audit_names.py` / `score_names.py`.

We tune ONE thing — the prompt — so the input is held constant at the full story
(the decision the benchmark exists to justify). `--sampled` is an ablation that
re-imposes production's old 12-line head/middle/tail thinning, to measure how much
the sampling alone was costing us.

LOCAL-ONLY: Gemma-4 E4B via MLX (needs the venv with mlx-vlm). No cloud.

    ./venv-mlx-vlm/bin/python emp/src/bench_worlds.py                  # all candidates, full story
    ./venv-mlx-vlm/bin/python emp/src/bench_worlds.py --candidate current
    ./venv-mlx-vlm/bin/python emp/src/bench_worlds.py --candidate current --sampled
    ./venv-mlx-vlm/bin/python emp/src/bench_worlds.py --limit 4        # quick smoke
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))            # emp/src — for `segment`
sys.path.insert(0, str(ROOT / "src"))    # src — segment also adds this

from segment import make_reader, extract_json, MODEL_ID  # noqa: E402

BENCH = ROOT / "emp" / "results" / "worlds-bench"
STORIES = BENCH / "stories.json"
PRED_DIR = BENCH / "pred"

# --- candidate prompts --------------------------------------------------------
# Each template carries a literal "{story}" token (filled by .replace, so the JSON
# braces below need no escaping). Every prompt must elicit a JSON object with a
# "world" field — empty string means "original / made up, no known world".
PROMPTS = {
    # The current PRODUCTION prompt, verbatim — the "before" floor. Note its
    # dataset-specific example worlds and the "Do not use any outside list" line.
    "current": (
        'You are reading ONE complete bedtime story told by a parent to a young '
        'child (some lines are sampled, in order). Give it a short descriptive '
        'title, and name the WORLD it is set in — inferred ONLY from what you read. '
        'Examples of a world: "Thomas & Friends (plus invented engines)", '
        '"Mahabharata", "original / made-up". Do not use any outside list; name '
        'what you actually see.\n\n'
        'Return JSON only, no other text:\n'
        '{"title": "<short title>", "world": "<the world / canon>"}\n\n'
        'Story lines:\n{story}\n'
    ),
    # General + abstain: no dataset-specific examples, and the explicit invitation
    # to USE world knowledge (the deliberate reversal of "do not use any list").
    "general_abstain": (
        'You are reading the transcript of ONE bedtime story a parent told a child, '
        'out loud — so it rambles and proper names may be mis-heard.\n\n'
        'Decide what fictional world this story is set in:\n'
        '- If it clearly takes place in a REAL, widely-known story world (a book, '
        'film, show, myth, or franchise many people would recognize), give that '
        "world's common name.\n"
        '- If it is an original, made-up story that does not come from any known '
        'world, return an empty string.\n\n'
        'Use your own knowledge of well-known stories to recognize the world even '
        'when names are garbled or the title is never spoken. If you are not '
        'confident it is a real, known world, return empty.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<the world\'s name, or empty if made up>"}\n\n'
        'Transcript:\n{story}\n'
    ),
    # General + two-step reasoning before the verdict.
    "general_reasoned": (
        'You are reading the transcript of ONE bedtime story a parent told a child, '
        'out loud (rambling; proper names may be mis-transcribed).\n\n'
        'Think in two quick steps, then answer:\n'
        '1) Note the distinctive character names, place names, and signature words.\n'
        '2) Do those match a REAL, widely-known story world (book, film, show, myth, '
        'or franchise)? Names may be garbled — judge by what they sound like.\n\n'
        "If yes, give that world's common name. If it is an original/made-up story "
        'from no known world, return an empty string. When unsure, return empty.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<world name, or empty if made up>"}\n\n'
        'Transcript:\n{story}\n'
    ),
    # Round 2: reasoned + "one garbled signature name is enough; lesser-known
    # worlds count" — aimed at the subtle/less-famous MISSED-CANON cases (the model
    # returned empty rather than guessing wrong).
    "general_reasoned_v2": (
        'You are reading the transcript of ONE bedtime story a parent told a child, '
        'out loud — it rambles and proper names are often mis-transcribed (a name may '
        'be misspelled or split into pieces).\n\n'
        'Work in two quick steps, then answer:\n'
        '1) List the distinctive character names, place names, and signature details '
        '(a unique creature, vehicle, catchphrase, or power). Names may be garbled — '
        'say what each one SOUNDS like.\n'
        '2) Do those point to a REAL, known story world — a book, film, TV show '
        "(children's shows included), myth, comic, or game? Even ONE clearly "
        'recognizable character or place is enough to name the world, and lesser-known '
        'shows/books count as much as famous ones. Recognize it even if the title is '
        'never spoken and the names are garbled.\n\n'
        'If they clearly point to a real world, give its common name. If it is an '
        'original, made-up story from no known world, return an empty string — but only '
        'when you genuinely cannot place it.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<world name, or empty if made up>"}\n\n'
        'Transcript:\n{story}\n'
    ),
    # Round 2: commit-leaning — counter the over-abstention that drove missed-canon.
    "general_commit": (
        'You are reading a transcript of ONE bedtime story told aloud to a child. It '
        'rambles and the proper names are often mis-heard or split into pieces.\n\n'
        'Your job: name the real, well-known story world it comes from — a book, film, '
        "TV show (kids' shows included), myth, comic, or game — OR say it is made up.\n\n"
        'Recognize the world even when the title is never spoken and the names are '
        'garbled: judge characters and places by what they SOUND like, and a single '
        'distinctive one (a unique character, creature, vehicle, or power) is enough. '
        'Lesser-known worlds count as much as famous ones.\n\n'
        'Return an empty string ONLY for a story that is genuinely original and matches '
        'no known world. When a detail clearly echoes a real world, name it rather than '
        'abstaining.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<world name, or empty if made up>"}\n\n'
        'Transcript:\n{story}\n'
    ),
    # General + generic EXAMPLE worlds, chosen to be disjoint from the benchmark
    # AND from the held-out real worlds (so they prime the task, not the answers).
    "general_examples": (
        'You are reading the transcript of ONE bedtime story a parent told a child, '
        'out loud (rambling; names may be mis-heard).\n\n'
        'Name the real, well-known story world it is set in — for example '
        '"Winnie the Pooh", "Greek mythology", "Frozen", or "Peter Rabbit". Use your '
        'own knowledge to recognize it even when names are garbled or the title is '
        'never said.\n\n'
        'If the story is original and made-up — not from any known world — return an '
        'empty string. When unsure, return empty.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<world name, or empty if made up>"}\n\n'
        'Transcript:\n{story}\n'
    ),
    # Round 3 (research-driven): few-shot. 6 balanced worked examples placed FIRST
    # (small models swing ~20pts on demo position), spanning famous/less-famous/made-up
    # incl. a wizard-school TRAP that resolves to empty. Example worlds are DISJOINT from
    # the benchmark and held-out worlds, so they teach the pattern, not the answers.
    "general_fewshot": (
        'Here are examples of reading a bedtime-story transcript and naming its world. '
        'An empty world means an original, made-up story from no known source.\n\n'
        'Example 1\nStory: the chubby yellow bear who loves honey gets his head stuck in '
        'the rabbit hole, and his gloomy grey donkey friend just sighs\n'
        'Answer: {"world": "Winnie the Pooh"}\n\n'
        'Example 2\nStory: the two princess sisters, the older one makes ice and snow '
        'with her hands and runs away to build a palace up the freezing mountain\n'
        'Answer: {"world": "Frozen"}\n\n'
        'Example 3\nStory: the naughty little bunny in the blue coat sneaks under the '
        "fence into Mister Mac-greggor's vegetable garden and loses his jacket\n"
        'Answer: {"world": "Peter Rabbit"}\n\n'
        'Example 4\nStory: the strong hero in the lion skin has to do his twelve labors, '
        'and first he fights the big snake with the many heads that grow back\n'
        'Answer: {"world": "Greek mythology"}\n\n'
        'Example 5\nStory: there was a little sleepy cloud named Pip who was too tired to '
        'rain, so he floated down and napped on the tallest tree while the birds tucked '
        'him in\nAnswer: {"world": ""}\n\n'
        'Example 6\nStory: a boy gets a letter saying he is magic and goes to a foggy '
        'castle school called Thornhallow to learn potions and flying\n'
        'Answer: {"world": ""}\n\n'
        'Now do the same for the story below. It rambles and proper names may be '
        'mis-transcribed — judge garbled names by what they sound like, and use your own '
        'knowledge of well-known stories. If it is an original/made-up story from no '
        'known world, return empty; when unsure, return empty.\n\n'
        'Return JSON only, no other text:\n'
        '{"world": "<world name, or empty if made up>"}\n\n'
        'Story:\n{story}\n'
    ),
}


# --- input rendering (firewall: id + lines only) ------------------------------
def render_full(lines):
    """Every line, in order, as the model sees a story in production: [i] "text"."""
    return "\n".join(f'[{i}] "{ln}"' for i, ln in enumerate(lines))


def render_sampled(lines, head=6, mid=3, tail=3):
    """Production's OLD thinning (sample_region_lines): first `head` + `mid` evenly
    spaced middle + last `tail` lines. The ablation arm."""
    body = list(range(len(lines)))
    if len(body) <= head + mid + tail:
        picks = body
    else:
        mids = [body[len(body) * (k + 1) // (mid + 1)] for k in range(mid)]
        picks = sorted(set(body[:head] + mids + body[-tail:]))
    return "\n".join(f'[{p}] "{lines[p]}"' for p in picks)


def load_inputs():
    """Stories as the runner is ALLOWED to see them: id + lines, nothing else.
    The gold fields (world_gold/bucket/difficulty) sit in the same file but are
    deliberately dropped here — only score_worlds.py may read them."""
    data = json.loads(STORIES.read_text())
    return [{"id": s["id"], "lines": s["lines"]} for s in data["stories"]]


def run_candidate(gen, name, inputs, sampled, max_tokens):
    template = PROMPTS[name]
    render = render_sampled if sampled else render_full
    results = []
    for i, item in enumerate(inputs):
        prompt = template.replace("{story}", render(item["lines"]))
        raw = gen(prompt, max_tokens=max_tokens)
        obj = extract_json(raw)
        world = ""
        if isinstance(obj, dict):
            world = str(obj.get("world", "")).strip()
        results.append({"id": item["id"], "world_pred": world, "raw": raw})
        print(f"  [{i + 1}/{len(inputs)}] {item['id']}: {world!r}", file=sys.stderr)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", nargs="*", default=list(PROMPTS),
                    help="which prompt(s) to run (default: all)")
    ap.add_argument("--sampled", action="store_true",
                    help="ablation: feed the old 12-line head/mid/tail sample, not the full story")
    ap.add_argument("--limit", type=int, default=0, help="only the first N stories (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    for name in args.candidate:
        if name not in PROMPTS:
            ap.error(f"unknown candidate {name!r}; choices: {', '.join(PROMPTS)}")

    inputs = load_inputs()
    if args.limit:
        inputs = inputs[:args.limit]
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} …", file=sys.stderr)
    gen = make_reader()
    mode = "sampled" if args.sampled else "full"

    for name in args.candidate:
        print(f"\n=== candidate {name!r} ({mode}, {len(inputs)} stories) ===", file=sys.stderr)
        results = run_candidate(gen, name, inputs, args.sampled, args.max_tokens)
        out = {
            "_about": "world-classifier benchmark predictions (firewalled runner)",
            "model": MODEL_ID, "candidate": name, "input_mode": mode,
            "n": len(results), "results": results,
        }
        suffix = ".sampled" if args.sampled else ""
        path = PRED_DIR / f"{name}{suffix}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
