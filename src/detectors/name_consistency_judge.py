"""Offline LLM judge for the M9b detector — recovers dictionary-word names.

The code-only detector drops clusters where every spelling is an ordinary word,
which loses improvised names that are also dictionary words (Bibi, Bacchus). This
hands exactly those ambiguous clusters to Gemma-4 E4B (~4B, the build the M9b
eval found wins at 1.00/1.00 with a few-shot prompt — see emp/src/judge_m9b.py
and emp/writeup/name-consistency-eval.html). The judge reads a few example lines
per cluster and decides "character name, or ordinary words?".

`make_judge()` returns judge(candidates) -> set[cluster_id]. The model runs via the
shared model_runner — a fresh subprocess of this venv, a clean process so a pyannote
MPS allocation can't block the Gemma load, freeing GPU memory on exit.

It is NEVER called from the live API (a ~30s model load can't sit in a page-view
request). Only `detect.py --judge` supplies it; the result survives normal Monitor
viewing and is reverted to code-only if the transcript changes (re-run to restore).
"""
from collections import Counter

from model_runner import run_model

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


def _judge_clusters(candidates):
    """Module-level (picklable for the spawned subprocess). Loads Gemma-4 E4B once and
    returns the cluster_ids that majority-vote to NAME. Each candidate:
    {cluster_id, spellings: [...], examples: [...]}."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

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
    return kept


def make_judge():
    """Return judge(candidates) -> set[cluster_id]. Runs Gemma in a fresh subprocess
    (via model_runner) so the ~30s model load gets a clean GPU process and frees it on
    exit. Each candidate: {cluster_id, spellings: [...], examples: [...]}."""
    def judge(candidates):
        return set(run_model(_judge_clusters, candidates, timeout=600))

    return judge
