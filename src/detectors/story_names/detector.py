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
    # 0.3.0: Qwen3.5 cast+judge+phonetic design (real-data recall 1/11 -> 8/11). Qwen3.5
    #   judges the names directly and is sound-matched against a Qwen3.5-generated cast; the
    #   two catches are unioned, then dictionary-gated. World comes from the Qwen3.5
    #   segmentation. (0.2.0 was the Gemma worksheet + dictionary gate, kept as the baseline
    #   _worker.run.)
    version = "0.3.0-experimental"
    accepts_judge = False
    offline_only = True  # never runs in a web request or a non-offline scan

    def run(self, session_dir: Path) -> dict:
        from model_runner import run_model
        from detectors.story_names import _worker
        result = run_model(_worker.run_qwen35, str(session_dir), timeout=WORKER_TIMEOUT)
        # Surface only the canon slice; the worker still computed (and shielded with)
        # the M9b clusters internally — that is what keeps the M9c flags accurate.
        return {
            "n_word_tokens": result["n_word_tokens"],
            "flags": [f for f in result["flags"] if f.get("case") == "M9c"],
        }
