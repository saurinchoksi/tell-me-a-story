"""Offline per-story name auditor (M9b/M9c) — the main-venv caller.

run() invokes the worker via model_runner.run_model: a fresh subprocess of this venv
that segments + audits with Gemma and returns {n_word_tokens, flags}. The subprocess
exits when done, freeing its GPU memory; the caller stays model-free.

offline_only=True keeps it out of every web request and every non-offline scan; only
`detect.py --story-names` and process_inbox (run_offline=True) trigger it. NEVER call
it from a live API GET/POST — a multi-minute segment+audit can't sit in a web request.
"""
from pathlib import Path

from detectors.base import Detector

# Generous cap: a long session segments + audits across several model calls.
WORKER_TIMEOUT = 1800


class StoryNameDetector(Detector):
    id = "m9bc-story-names"
    label = "Story-name mistranscription (improvised + canon)"
    failure_mode = "M9b/M9c"
    version = "0.1.0-experimental"
    accepts_judge = False
    offline_only = True  # never runs in a web request or a non-offline scan

    def run(self, session_dir: Path) -> dict:
        from model_runner import run_model
        from detectors.story_names import _worker
        return run_model(_worker.run, str(session_dir), timeout=WORKER_TIMEOUT)
