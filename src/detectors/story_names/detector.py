"""Offline per-story name auditor (M9b/M9c) — the main-venv caller.

Mirrors name_consistency_judge.make_judge's subprocess pattern: run() spawns the
worker under venv-mlx-vlm (the Gemma-4 build only loads there), passing the session
dir on argv and reading {n_word_tokens, flags} back as JSON on stdout. The caller
imports no model code and no ported audit logic — all of that lives in the worker.

offline_only=True keeps it out of every web request and every non-offline scan; only
`detect.py --story-names` and process_inbox (run_offline=True) trigger it. NEVER call
it from a live API GET/POST — a multi-minute segment+audit can't sit in a web request.
"""
import json
import subprocess
from pathlib import Path

from detectors.base import Detector

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VLM_PYTHON = PROJECT_ROOT / "venv-mlx-vlm" / "bin" / "python"
WORKER = Path(__file__).resolve().parent / "_worker.py"


class StoryNameDetector(Detector):
    id = "m9bc-story-names"
    label = "Story-name mistranscription (improvised + canon)"
    failure_mode = "M9b/M9c"
    version = "0.1.0-experimental"
    accepts_judge = False
    offline_only = True  # never runs in a web request or a non-offline scan

    def run(self, session_dir: Path) -> dict:
        if not VLM_PYTHON.exists():
            raise FileNotFoundError(
                f"mlx-vlm venv not found at {VLM_PYTHON}. Create it with: "
                "python -m venv venv-mlx-vlm && ./venv-mlx-vlm/bin/pip install "
                "'mlx-vlm==0.5.0' metaphone"
            )
        proc = subprocess.run(
            [str(VLM_PYTHON), str(WORKER), str(session_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"story-names worker failed:\n{proc.stderr[-2000:]}")
        result = json.loads(proc.stdout)
        return {"n_word_tokens": result["n_word_tokens"], "flags": result["flags"]}
