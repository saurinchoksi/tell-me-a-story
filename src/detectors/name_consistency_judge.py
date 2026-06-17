"""Offline LLM judge for the M9b detector — recovers dictionary-word names.

The code-only detector drops clusters where every spelling is an ordinary word,
which loses improvised names that are also dictionary words (Bibi, Bacchus). This
hands exactly those ambiguous clusters to Gemma-4 E4B (~4B, the build the M9b
eval found wins at 1.00/1.00 with a few-shot prompt — see emp/src/judge_m9b.py
and emp/writeup/name-consistency-eval.html). The judge reads a few example lines
per cluster and decides "character name, or ordinary words?".

This module is BOTH the caller and the worker:
  - `make_judge()` returns judge(candidates) -> set[cluster_id]. It runs the model in
    a fresh subprocess of this venv — a clean process so a pyannote MPS allocation
    can't block the Gemma load (finish-and-free for GPU memory).
  - run as `--worker`, it loads the model and judges clusters read as JSON on stdin,
    emitting kept cluster_ids as JSON on stdout.

It is NEVER called from the live API (a ~30s model load can't sit in a page-view
request). Only `detect.py --judge` supplies it; the result survives normal Monitor
viewing and is reverted to code-only if the transcript changes (re-run to restore).
"""
import json
import subprocess
import sys
from pathlib import Path

VLM_PYTHON = Path(sys.executable)
MODEL = "mlx-community/gemma-4-e4b-it-4bit"
RUNS = 3

# The winning prompt from the M9b sweep (v3: character-context framing + two
# worked examples). Kept verbatim — the 1.00/1.00 result is tied to this phrasing.
PROMPT = (
    "You are checking a transcript of a parent telling a young child a bedtime "
    "story. A group of similar-sounding capitalized words was flagged. Using the "
    "example lines, decide whether they name a CHARACTER in the story (a person, "
    "creature, or toy that is spoken to, greeted, or that does something), or are "
    "just ordinary language.\n\n"
    "Answer WORD if they are everyday words, interjections (Oh, Yeah, Whoa), "
    "question words (What, There), or family address terms (Daddy, Mommy) used in "
    "ordinary speech — even if one odd spelling is mixed in.\n"
    "Answer NAME if at least one example line uses the word as a character's name "
    "(the word is greeted, spoken to, or performs an action).\n\n"
    "Examples:\n"
    'Flagged spellings: Yeah, Whoa\n'
    'Example lines:\n- "Yeah, they all said."\n- "Whoa, look at that!"\n'
    'Answer: WORD\n\n'
    'Flagged spellings: Zogg, Zog\n'
    'Example lines:\n- "And Zogg flew up high."\n- "Then Zog landed near the pond."\n'
    'Answer: NAME\n\n'
    "Now classify this one.\n"
    "Flagged spellings: {spellings}\n"
    "Example lines:\n{sentences}\n\n"
    "Answer with exactly one word: NAME or WORD."
)


def make_judge():
    """Return judge(candidates) -> set[cluster_id]. Each candidate:
    {cluster_id, spellings: [...], examples: [...]}."""
    def judge(candidates):
        proc = subprocess.run(
            [str(VLM_PYTHON), str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(candidates), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"judge worker failed:\n{proc.stderr[-2000:]}")
        return set(json.loads(proc.stdout)["kept"])

    return judge


def _worker():
    """Runs in a fresh subprocess. Reads candidates on stdin, emits kept ids."""
    from collections import Counter
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    candidates = json.loads(sys.stdin.read())
    model, processor = load(MODEL)
    kept = []
    for c in candidates:
        prompt = PROMPT.format(spellings=", ".join(c["spellings"]),
                               sentences="\n".join(f"- {s}" for s in c["examples"]))
        verdicts = []
        for _ in range(RUNS):
            p = apply_chat_template(processor, model.config, prompt)
            res = generate(model, processor, p, max_tokens=8, temperature=0.0, verbose=False)
            up = (getattr(res, "text", res) or "").strip().upper()
            verdicts.append("NAME" if ("NAME" in up and "WORD" not in up) else "WORD")
        if Counter(verdicts).most_common(1)[0][0] == "NAME":
            kept.append(c["cluster_id"])
    print(json.dumps({"kept": kept}))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
