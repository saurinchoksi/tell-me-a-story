"""Run a heavy model in a fresh, short-lived subprocess — the finish-and-free pattern.

A model on Apple Silicon (Whisper, pyannote, Qwen via mlx-lm, Gemma via mlx-vlm) holds
the Mac's Metal/MPS memory in a caching allocator that is NOT reliably released when the
model object is deleted — and one framework's held memory (pyannote/torch) can block
another (mlx) from loading. The only sure way to free it is to END the process: on exit
the OS reclaims all of it. So each heavy step runs its work in a spawned subprocess that
loads the model, does the work, returns the result, and exits.

One venv now: the subprocess is the same interpreter (spawn), so every model stack is
available. The worker function must be module-level (picklable by reference); its args
and return value must be picklable. This is the single primitive behind the word-fixer
(normalize.py), the M9b name judge, and the story-name auditor.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError


def run_model(fn, /, *args, timeout: int = 600, **kwargs):
    """Run fn(*args, **kwargs) in a fresh spawned subprocess and return its result.

    The subprocess exits when fn returns, releasing all of its GPU memory before the
    next model loads. fn must be a module-level callable; its args/kwargs and return
    value must be picklable. Raises TimeoutError if fn does not finish within `timeout`
    seconds; any exception fn raises propagates to the caller.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"model subprocess timed out after {timeout}s")
