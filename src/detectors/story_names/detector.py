"""Offline canon-name auditor (M9c) — the main-venv caller.

run() invokes the worker via model_runner.run_model: a fresh subprocess of this venv
that segments + audits with Gemma and returns {n_word_tokens, flags}. The subprocess
exits when done, freeing its GPU memory; the caller stays model-free.

The worker does the full converged per-story analysis — improvised-name inconsistency
(M9b) *and* sourced-canon recognition (M9c) together — because the canon shield needs
the M9b clustering to protect correctly-spelled real names (e.g. "James") from being
flagged. This detector surfaces only the M9c slice of that analysis (case == "M9c");
M9b is covered by the separate m9b-name-consistency detector, so emitting it here too
would just double a Monitor line. Filtering costs nothing — the converged work has
already happened in the one model pass.

offline_only=True keeps it out of every web request and every non-offline scan; only
`detect.py --story-names` and process_inbox (run_offline=True) trigger it. NEVER call
it from a live API GET/POST — a multi-minute segment+audit can't sit in a web request.
"""
from pathlib import Path

from detectors.base import Detector

# Generous cap: a long session segments + audits across several model calls.
WORKER_TIMEOUT = 1800


class CanonNameDetector(Detector):
    id = "m9c-canon"
    label = "Canon-name mistranscription (Thomas / Mahabharata / known source)"
    failure_mode = "M9c"
    # 0.5.0: emit ALL canon catches (confident AND not), each carrying suggestion_confident + vote_count,
    #   so the VIEW layer (api.helpers.canon_tier) sorts them into confidence TIERS — confident /
    #   best-guess (judge vote >= 4) / low — instead of the detector hard-dropping the non-sound-alike
    #   ones. The scored sweep (emp/src/tune_surfacing_policy.py) showed the best-guess tier lifts
    #   real-canon recall (held-out 0.56 -> 0.78, synthetic spread 0.75 -> 0.85) at a trivial precision
    #   cost the human verdict button absorbs. The vote threshold lives at view time, so it's tunable
    #   without a re-scan. (0.4.0 was the order-robust judge surfacing CONFIDENT catches only.)
    # 0.4.0: ORDER-ROBUST judge. The single judge call is order-sensitive (a borderline name is
    #   caught in one ordering of the name list, missed in another), so the judge now votes across
    #   several deterministic shuffles and keeps names a majority agrees on — reliable on borderline
    #   catches, drops one-off order-noise. (0.3.0 was the single-shot Qwen3.5 cast+judge+phonetic
    #   union; 0.2.0 the Gemma worksheet, kept as the baseline _worker.run.) mlx_vlm wobbles a hair
    #   on Metal even at temp 0; voting averages that out too. Qwen3.5 judges the names directly and
    #   is sound-matched against a Qwen3.5 cast; the two catches are unioned, then dictionary-gated.
    version = "0.5.0-experimental"
    accepts_judge = False
    offline_only = True  # never runs in a web request or a non-offline scan

    def config_fingerprint(self) -> str | None:
        """Hash the voting config, the three judge/world/cast prompts, and the surfacing policy so a
        change to K, the cutoff, the seed, a prompt, or what we emit forces a re-scan (an unrelated
        edit does not). Inputs only — never anything model-derived, which would thrash the staleness
        cache (Metal non-determinism)."""
        import hashlib
        import json
        from detectors.story_names import _qwen35
        payload = json.dumps({
            "rounds": _qwen35.JUDGE_ROUNDS,
            "threshold": _qwen35.JUDGE_THRESHOLD,
            "seed_base": _qwen35.JUDGE_SEED_BASE,
            "judge_prompt": _qwen35.JUDGE_PROMPT,
            "recognize_prompt": _qwen35.RECOGNIZE_PROMPT,
            "cast_prompt": _qwen35.CAST_PROMPT,
            "surface": "all-tiers",   # all canon catches emitted; the view layer tiers them (see run())
        }, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def run(self, session_dir: Path) -> dict:
        from model_runner import run_model
        from detectors.story_names import _worker
        result = run_model(_worker.run_qwen35, str(session_dir), timeout=WORKER_TIMEOUT)
        # Surface the canon slice (the worker computed M9b clusters internally to shield real names),
        # carrying ALL catches — confident and not. Each flag already holds suggestion_confident +
        # vote_count (computed in _worker.expand_combined), so the view layer (api.helpers.canon_tier)
        # sorts them into confidence tiers and decides what shows by default; the detector no longer
        # hard-drops the non-sound-alike "best guess" catches.
        return {
            "n_word_tokens": result["n_word_tokens"],
            "flags": [f for f in result["flags"] if f.get("case") == "M9c"],
        }
