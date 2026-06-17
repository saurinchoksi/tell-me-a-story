"""Offline per-story name auditor (M9b/M9c) — the main-venv caller.

Mirrors name_consistency_judge.make_judge's subprocess pattern: run() spawns the
worker in a fresh subprocess of this venv (a clean process for the Gemma load), passing the session
dir on argv and reading {n_word_tokens, flags} back as JSON on stdout. The caller
imports no model code and no ported audit logic — all of that lives in the worker.

offline_only=True keeps it out of every web request and every non-offline scan; only
`detect.py --story-names` and process_inbox (run_offline=True) trigger it. NEVER call
it from a live API GET/POST — a multi-minute segment+audit can't sit in a web request.
"""
import json
import subprocess
import sys
from pathlib import Path

from detectors.base import Detector

# One merged venv: spawn the worker as a fresh subprocess of THIS interpreter so a
# pyannote MPS allocation can't block the Gemma load (finish-and-free for GPU memory).
VLM_PYTHON = Path(sys.executable)
WORKER = Path(__file__).resolve().parent / "_worker.py"


class StoryNameDetector(Detector):
    id = "m9bc-story-names"
    label = "Story-name mistranscription (improvised + canon)"
    failure_mode = "M9b/M9c"
    version = "0.1.0-experimental"
    accepts_judge = False
    offline_only = True  # never runs in a web request or a non-offline scan

    def run(self, session_dir: Path) -> dict:
        proc = subprocess.run(
            [str(VLM_PYTHON), str(WORKER), str(session_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"story-names worker failed:\n{proc.stderr[-2000:]}")
        result = json.loads(proc.stdout)
        return {"n_word_tokens": result["n_word_tokens"], "flags": result["flags"]}
