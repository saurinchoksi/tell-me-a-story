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
    # 0.4.0: ORDER-ROBUST judge. The single judge call is order-sensitive (a borderline name is
    #   caught in one ordering of the name list, missed in another), so the judge now votes across
    #   several deterministic shuffles and keeps names a majority agrees on — reliable on borderline
    #   catches, drops one-off order-noise. (0.3.0 was the single-shot Qwen3.5 cast+judge+phonetic
    #   union; 0.2.0 the Gemma worksheet, kept as the baseline _worker.run.) mlx_vlm wobbles a hair
    #   on Metal even at temp 0; voting averages that out too. Qwen3.5 judges the names directly and
    #   is sound-matched against a Qwen3.5 cast; the two catches are unioned, then dictionary-gated.
    #   Surfaces only CONFIDENT catches (the suggested spelling sounds like the heard token): a
    #   not-sound-alike "best guess" is the judge over-reaching — it force-maps an invented name onto
    #   canon when told a world (e.g. Pataki->Paddy) — so those stay out of the Monitor.
    version = "0.4.0-experimental"
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
            "surface": "confident-only",   # only sound-alike catches are emitted (see run())
        }, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def run(self, session_dir: Path) -> dict:
        from model_runner import run_model
        from detectors.story_names import _worker
        result = run_model(_worker.run_qwen35, str(session_dir), timeout=WORKER_TIMEOUT)
        # Surface only the canon slice (the worker computed M9b clusters internally to shield real
        # names) AND only CONFIDENT catches — those whose suggested spelling sounds like the heard
        # token. A not-sound-alike "best guess" is the judge over-reaching (force-mapping an invented
        # name onto canon when told a world), so it stays out: a flag the reviewer can trust beats a
        # screen of maybes. The worker still computes the full set; this is the surfacing policy.
        return {
            "n_word_tokens": result["n_word_tokens"],
            "flags": [f for f in result["flags"]
                      if f.get("case") == "M9c" and f.get("suggestion_confident")],
        }
