"""LLM-based normalization of phonetic name mishearings."""

import json
import re
from datetime import datetime, timezone

from model_cache import cached_or_run, fingerprint
from model_runner import run_model


MODEL = "mlx-community/Qwen3-8B-8bit"
MAX_TOKENS = 512

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

DEFAULT_PROMPT = """This text is a transcription of a spoken conversation. Some proper nouns may be phonetically distorted or misheared by the transcription model.

Look at the text and identify any words that might be phonetic mishearings of proper nouns (names of people, places, or groups).

ONLY return corrections for words that actually appear in the text.
If no mishearings appear, return an empty list.

Return JSON only:
{{"corrections": [{{"transcribed": "...", "correct": "..."}}]}}

Text:
{text}"""


def _mlx_worker(prompt_text: str, model: str, max_tokens: int) -> str:
    """Run MLX-LM inference in an isolated subprocess. Module-level for spawn pickling."""
    import gc

    import mlx.core as mx
    import mlx_lm

    model_obj, tokenizer = mlx_lm.load(model)
    messages = [{"role": "user", "content": "/no_think\n" + prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response = mlx_lm.generate(
        model_obj,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=False,
    )
    del model_obj, tokenizer
    gc.collect()
    mx.clear_cache()
    return response


def _call_mlx(prompt_text: str, model: str, timeout: int) -> str:
    """Call MLX-LM in a fresh subprocess (clean GPU slate after pyannote) and strip the
    model's <think> block. The finish-and-free isolation lives in model_runner.run_model;
    pyannote's MPS allocations would otherwise block Qwen3-8B-8bit from loading."""
    response = run_model(_mlx_worker, prompt_text, model, MAX_TOKENS, timeout=timeout)
    return _THINK_RE.sub("", response).strip()


def _parse_llm_corrections(response: str) -> list[dict]:
    """Parse LLM response into corrections list with 3-tier fallback.

    Tier 1: Regex search for {"corrections": [...]} in response
    Tier 2: Extract from ```json or ``` code blocks
    Tier 3: Direct json.loads on stripped response

    Args:
        response: Raw LLM output string

    Returns:
        List of correction dicts with 'transcribed' and 'correct' keys.
        Returns [] if model returned empty object or empty corrections.

    Raises:
        ValueError: If response cannot be parsed as JSON by any tier
    """
    stripped = response.strip()
    parsed = None

    # Tier 1: Regex for corrections JSON
    match = re.search(r'\{"corrections":\s*\[.*\]\s*\}', stripped, re.DOTALL)
    if match:
        parsed = json.loads(match.group())

    # Tier 2: Code block extraction
    if parsed is None:
        if "```json" in stripped:
            json_str = stripped.split("```json")[1].split("```")[0]
            parsed = json.loads(json_str.strip())
        elif "```" in stripped:
            json_str = stripped.split("```")[1].split("```")[0]
            parsed = json.loads(json_str.strip())

    # Tier 3: Direct parse
    if parsed is None:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            tail = stripped[-200:] if len(stripped) > 200 else stripped
            raise ValueError(f"Could not parse LLM response: ...{tail}")

    corrections = parsed.get("corrections", [])
    if not isinstance(corrections, list):
        return []
    return [c for c in corrections if isinstance(c, dict)
            and isinstance(c.get("transcribed"), str)
            and isinstance(c.get("correct"), str)]


def llm_normalize(
    text: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = MODEL,
    timeout: int = 300,
    cache_dir=None,
) -> tuple[list[dict], dict]:
    """Identify phonetic mishearings in text using a local LLM.

    Sends the full text to a local MLX-LM model which returns corrections
    for words that appear to be phonetic mishearings of known names.

    Args:
        text: Transcript text to analyze
        prompt: Prompt template with {text} placeholder
        model: MLX-community model identifier
        timeout: Request timeout in seconds (passed through to backend)
        cache_dir: If given, corrections are cached there keyed on (text + model +
            prompt). A later call with the same input and config reuses them and skips
            the model entirely — so a re-enrich that didn't change the transcript text
            or this stage's config doesn't reload Qwen.

    Returns:
        Tuple of (corrections, processing_entry).
        corrections: List of dicts with 'transcribed' and 'correct' keys.
        processing_entry: Dict with stage metadata, incl. 'from_cache' (no
            corrections_applied — pipeline adds that after apply_corrections).

    Raises:
        TimeoutError: If inference exceeds timeout
        RuntimeError: If model call fails
        ValueError: If LLM response cannot be parsed
    """
    formatted = prompt.format(text=text)

    def compute():
        return _parse_llm_corrections(_call_mlx(formatted, model, timeout))

    if cache_dir is not None:
        corrections, was_cached = cached_or_run(
            cache_dir, "llm_normalization",
            fingerprint(text), fingerprint(model + "\n" + prompt), compute)
    else:
        corrections, was_cached = compute(), False

    entry = {
        "stage": "llm_normalization",
        "model": model,
        "status": "success",
        "from_cache": was_cached,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return corrections, entry
