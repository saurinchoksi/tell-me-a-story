#!/usr/bin/env python3
"""LLM-judge for M9b: recover the dictionary-word improvised names.  EXPERIMENT.

The code-only consistency detector clusters capitalized tokens phonetically and
flags any name spelled >1 way. Its common-word filter (drop clusters that are all
ordinary words) buys precision but loses the improvised names that are *also*
dictionary words (Bibi, Bacchus) — a lexical filter can't tell "Bibi the
character" from "bibi the word". That call is contextual, so we hand exactly that
residual to a local LLM:

  - cluster with the dictionary filter OFF  -> recovers every inconsistent name
  - clusters that are NOT all-common (Jiraki/Jameis) -> keep by code, no LLM
  - clusters that ARE all-common (Bibi/…, Yeah/Whoa) -> the JUDGE decides, from
    example lines, whether they are one character's NAME or ordinary words.

Scored two ways: the judge's keep/drop agreement vs the variant-map truth, and the
end-to-end detector (code-kept + judge-kept) precision/recall vs the human M9
coding (reusing score_m9b's ground truth). Per-cluster verdicts (which echo name
variants and private context lines) go to a gitignored dump; the console prints
only aggregates and improvised-name cluster spellings — never the context lines.

Backends (run under the matching venv):
  ./venv/bin/python          emp/src/judge_m9b.py --backend mlx-lm  --model mlx-community/Qwen3-8B-8bit
  ./venv/bin/python          emp/src/judge_m9b.py --backend mlx-lm  --model mlx-community/gemma-3-12b-it-4bit
  ./venv-mlx-vlm/bin/python  emp/src/judge_m9b.py --backend mlx-vlm --model mlx-community/gemma-4-e2b-it-4bit
  ./venv-mlx-vlm/bin/python  emp/src/judge_m9b.py --backend mlx-vlm --model mlx-community/gemma-4-e4b-it-4bit
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.phonetics import clean, codes, is_capitalized  # noqa: E402
from detectors.name_consistency import NameConsistencyDetector, MIN_NAME_LEN  # noqa: E402
from score_m9b import variant_to_case, clean_word, segment_cases  # noqa: E402

SESSIONS = ["20251207-202105", "20260129-204404"]  # the two M9b-bearing sessions
EXAMPLES_PER_CLUSTER = 3
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# --- Prompt variants (the iteration loop) -------------------------------------
_V2 = (
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
    "Flagged spellings: {spellings}\n"
    "Example lines:\n{sentences}\n\n"
    "Answer with exactly one word: NAME or WORD."
)

_V3_SHOTS = (
    "Examples:\n"
    'Flagged spellings: Yeah, Whoa\n'
    'Example lines:\n- "Yeah, they all said."\n- "Whoa, look at that!"\n'
    'Answer: WORD\n\n'
    'Flagged spellings: Zogg, Zog\n'
    'Example lines:\n- "And Zogg flew up high."\n- "Then Zog landed near the pond."\n'
    'Answer: NAME\n\n'
)

PROMPTS = {
    "v1": (
        "You are checking a transcript of a parent telling a young child a bedtime "
        "story. A group of capitalized words was flagged because they sound alike and "
        "might be one made-up character's NAME that the transcriber spelled "
        "inconsistently. From the example lines, decide whether they are really one "
        "character's NAME, or just ordinary words or interjections (like 'Oh', "
        "'Yeah', 'What', 'There') that happen to sound alike.\n\n"
        "Flagged spellings: {spellings}\n"
        "Example lines:\n{sentences}\n\n"
        "Answer with exactly one word: NAME or WORD."
    ),
    "v2": _V2,
    # v3 = v2 with two worked examples (one WORD, one NAME) prepended.
    "v3": _V2.replace("Flagged spellings: {spellings}",
                      _V3_SHOTS + "Now classify this one.\nFlagged spellings: {spellings}"),
}

# "prod" drives the EXACT shipped M9b judge string (Qwen 3.5 4B), so validation
# scores the production prompt, never a paraphrase of it.
try:
    from detectors.name_consistency_judge import PROMPT as _PROD_PROMPT
    PROMPTS["prod"] = _PROD_PROMPT
except Exception:  # pragma: no cover - import guard for partial checkouts
    pass


# --- Clustering (filter OFF; enumerate candidates) ----------------------------
class _UF:
    def __init__(self): self.p = {}
    def add(self, x): self.p.setdefault(x, x)
    def find(self, x):
        r = x
        while self.p[r] != r: r = self.p[r]
        while self.p[x] != r: self.p[x], x = r, self.p[x]
        return r
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def cluster_session(sid, is_common):
    """Every inconsistent (>1 spelling), min-length cluster in the session, with
    its all_common flag and example sentences. Mirrors name_consistency.run()'s
    clustering with the common-word filter held off."""
    base = ROOT / "sessions" / sid
    rich = json.loads((base / "transcript-rich.json").read_text())
    seg_text = {str(s["id"]): (s.get("text") or "").strip() for s in rich["segments"]}

    occ = []          # (seg_id, raw, cleaned)
    form_codes = {}
    for seg in rich["segments"]:
        for w in seg.get("words", []):
            raw = w["word"].strip()
            c = clean(raw)
            if not c or not is_capitalized(raw) or len(c) < MIN_NAME_LEN:
                continue
            occ.append((str(seg["id"]), raw, c))
            form_codes.setdefault(c, codes(c))

    uf = _UF()
    for c in form_codes: uf.add(c)
    code_to_forms = defaultdict(list)
    for form, fcs in form_codes.items():
        for code in fcs: code_to_forms[code].append(form)
    for forms in code_to_forms.values():
        for other in forms[1:]: uf.union(forms[0], other)

    cl_forms, cl_surface, cl_segs, cl_raws = (defaultdict(set), defaultdict(set),
                                              defaultdict(set), defaultdict(set))
    for sid_, raw, c in occ:
        r = uf.find(c)
        cl_forms[r].add(c); cl_surface[r].add(raw); cl_segs[r].add(sid_); cl_raws[r].add(raw)

    clusters = []
    for r, forms in cl_forms.items():
        if len(forms) < 2:
            continue
        all_common = all(is_common(f) for f in forms)
        segs = sorted(cl_segs[r], key=lambda s: int(s) if s.isdigit() else 1e9)
        clusters.append({
            "spellings": sorted(cl_surface[r]),
            "cleaned": sorted(forms),
            "raws": sorted(cl_raws[r]),
            "segments": segs,
            "all_common": all_common,
            "examples": [seg_text[s] for s in segs[:EXAMPLES_PER_CLUSTER] if seg_text.get(s)],
        })
    return clusters, seg_text


# --- Backends -----------------------------------------------------------------
def make_judge(backend, model_id):
    """Return classify(prompt_text) -> raw string. Loads the model once."""
    if backend == "mlx-vlm":
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor = load(model_id)

        def classify(prompt_text):
            p = apply_chat_template(processor, model.config, prompt_text)
            res = generate(model, processor, p, max_tokens=8, temperature=0.0, verbose=False)
            return (getattr(res, "text", res) or "").strip()
        return classify

    if backend == "qwen35":
        # Qwen3.5-4B via the production recipe (mlx_vlm loader + enable_thinking=False,
        # plain text). The mlx-vlm/mlx-lm paths above mis-drive it; reuse src/qwen35.py
        # so the bench runs the EXACT runtime the M9c detector ships.
        from qwen35 import make_reader
        gen = make_reader(model_id)

        def classify(prompt_text):
            return gen(prompt_text, max_tokens=32)
        return classify

    if backend == "mlx-lm":
        import mlx_lm
        model, tok = mlx_lm.load(model_id)
        no_think = "qwen3" in model_id.lower()  # Qwen3 emits <think>…</think> otherwise

        def classify(prompt_text):
            content = ("/no_think\n" + prompt_text) if no_think else prompt_text
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True)
            resp = mlx_lm.generate(model, tok, prompt=formatted, max_tokens=64, verbose=False)
            return _THINK_RE.sub("", resp).strip()
        return classify

    raise ValueError(f"unknown backend {backend!r}")


def verdict_of(raw):
    up = (raw or "").upper()
    has_name, has_word = "NAME" in up, "WORD" in up
    if has_name and not has_word: return "NAME"
    if has_word and not has_name: return "WORD"
    return "?"


# --- Main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mlx-lm", "mlx-vlm", "qwen35"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="v1", choices=list(PROMPTS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("sessions", nargs="*", default=None)
    args = ap.parse_args()
    sids = args.sessions or SESSIONS
    template = PROMPTS[args.prompt]

    vmap = variant_to_case()
    det = NameConsistencyDetector()  # for its _is_common (system dictionary)
    short = args.model.split("/")[-1]
    print(f"=== judge {short}  prompt={args.prompt}  runs={args.runs} ===")
    print(f"loading {args.model} ({args.backend}) ...")
    classify = make_judge(args.backend, args.model)
    print("loaded.\n")

    agg = Counter()
    pooled_truth = pooled_flagseg = pooled_tp = pooled_namehit = 0
    pooled_m9b = pooled_caught = 0
    for sid in sids:
        clusters, seg_text = cluster_session(sid, det._is_common)
        base = ROOT / "sessions" / sid
        human = {str(l["segmentId"]): l["codes"]
                 for l in json.loads((base / "axial-labels.json").read_text())["labels"]
                 if l.get("codes")}
        rich = json.loads((base / "transcript-rich.json").read_text())
        seg_by = {str(s["id"]): s for s in rich["segments"]}
        human_m9 = {s for s, c in human.items() if any(x.startswith("M9") for x in c)}
        m9b_truth = {s for s in human_m9 if "M9b" in segment_cases(seg_by.get(s, {}), vmap)}

        auto_keep = [c for c in clusters if not c["all_common"]]
        candidates = [c for c in clusters if c["all_common"]]
        flagged_segs = set()
        for c in auto_keep:
            flagged_segs |= set(c["segments"])

        dump = []
        print(f"--- {sid}: {len(clusters)} clusters "
              f"({len(auto_keep)} code-kept, {len(candidates)} judged) ---")
        for c in candidates:
            # ground truth: is this cluster a real name? (any spelling in the variant map)
            cases = {vmap[clean_word(t)] for t in c["raws"] if clean_word(t) in vmap}
            is_name = bool(cases)
            prompt = template.format(spellings=", ".join(c["spellings"]),
                                     sentences="\n".join(f"- {s}" for s in c["examples"]))
            raws = []
            for _ in range(args.runs):
                try:
                    raws.append(classify(prompt))
                except Exception as e:
                    raws.append("CRASH: " + repr(e)[:120])
            verdicts = [verdict_of(r) for r in raws]
            majority = Counter(verdicts).most_common(1)[0][0]
            stable = len(set(verdicts)) == 1
            keep = majority == "NAME"
            correct = keep == is_name
            if keep:
                flagged_segs |= set(c["segments"])
            agg["names" if is_name else "nonnames"] += 1
            if is_name and keep: agg["names_kept"] += 1
            if is_name and not keep: agg["names_dropped"] += 1
            if not is_name and not keep: agg["nonnames_dropped"] += 1
            if not is_name and keep: agg["nonnames_kept"] += 1
            agg["stable"] += stable
            if any("CRASH" in v for v in raws): agg["crashes"] += 1
            dump.append({"spellings": c["spellings"], "segments": c["segments"],
                         "is_name_truth": is_name, "cases": sorted(cases),
                         "verdicts": verdicts, "majority": majority, "stable": stable,
                         "correct": correct, "examples": c["examples"], "raw": raws})
            tag = "NAME" if is_name else "word"
            mark = "ok" if correct else "**MISS**"
            print(f"    {str(c['spellings']):<40} truth={tag:<5} "
                  f"{str(verdicts):<20} -> {majority:<5} {mark}")

        # end-to-end detector score vs the human M9b coding
        tp = len(flagged_segs & m9b_truth)
        namehit = sum(1 for s in flagged_segs if segment_cases(seg_by.get(s, {}), vmap))
        caught = len(m9b_truth & flagged_segs)
        pooled_truth += len(m9b_truth); pooled_flagseg += len(flagged_segs)
        pooled_tp += tp; pooled_namehit += namehit
        pooled_m9b += len(m9b_truth); pooled_caught += caught
        outdir = ROOT / "emp" / "results" / "visuals" / sid
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"m9b-judge-{short}-{args.prompt}.json").write_text(json.dumps(
            {"_about": "M9b LLM-judge verdicts over the all-common ambiguous clusters; "
                       "NAME=keep. Gitignored (name variants + context lines).",
             "model": args.model, "prompt": args.prompt, "session": sid,
             "runs": args.runs, "clusters": dump}, indent=2))

    # --- scorecard ---
    names, nonnames = agg["names"], agg["nonnames"]
    judged = names + nonnames
    j_prec = agg["names_kept"] / (agg["names_kept"] + agg["nonnames_kept"]) if (agg["names_kept"] + agg["nonnames_kept"]) else 0
    j_rec = agg["names_kept"] / names if names else 0
    print("\n" + "=" * 64)
    print(f"JUDGE SCORECARD  {short} / {args.prompt}  (over {judged} ambiguous clusters)")
    print(f"  names kept:     {agg['names_kept']}/{names}   (dropped {agg['names_dropped']} -> recall damage)")
    print(f"  non-names dropped: {agg['nonnames_dropped']}/{nonnames}   (kept {agg['nonnames_kept']} -> precision leak)")
    print(f"  stable/{args.runs}runs: {agg['stable']}/{judged}    crashes: {agg['crashes']}")
    print(f"  judge keep-decision: precision {j_prec:.3f}  recall {j_rec:.3f}")
    print("  --- end-to-end detector (code-kept + judge-kept) vs human M9b ---")
    if pooled_flagseg:
        print(f"  M9b precision  = {pooled_tp}/{pooled_flagseg} = {pooled_tp/pooled_flagseg:.3f}")
        print(f"  name precision = {pooled_namehit}/{pooled_flagseg} = {pooled_namehit/pooled_flagseg:.3f}")
    if pooled_m9b:
        print(f"  M9b recall     = {pooled_caught}/{pooled_m9b} = {pooled_caught/pooled_m9b:.3f}")


if __name__ == "__main__":
    main()
