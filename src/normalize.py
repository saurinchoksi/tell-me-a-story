"""LLM-based normalization of phonetic name mishearings."""

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone


MODEL = "qwen3:8b"

DEFAULT_PROMPT = """This text is from a conversation about the Mahabharata epic. A child is pronouncing Sanskrit names phonetically, often with significant distortion.

Look at the text and identify any words that might be phonetic mishearings of Mahabharata character or group names (like Pandavas, Kauravas, Yudhishthira, Duryodhana, Dhritarashtra, Pandu, etc.)

ONLY return corrections for words that actually appear in the text.
If no mishearings appear, return an empty list.

Return JSON only:
{{"corrections": [{{"transcribed": "...", "correct": "..."}}]}}

Text:
{text}"""


def _call_ollama(prompt: str, model: str, timeout: int) -> str:
    """Call Ollama REST API and return the response text.

    Args:
        prompt: Formatted prompt string
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        Response text from Ollama

    Raises:
        TimeoutError: If the request exceeds timeout
        RuntimeError: If the API call fails
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise TimeoutError(f"Ollama request timed out after {timeout}s") from e
        raise RuntimeError(f"Ollama call failed: {e}") from e

    return body["response"]


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
) -> tuple[list[dict], dict]:
    """Identify phonetic mishearings in text using a local LLM.

    Sends the full text to a local Ollama model which returns corrections
    for words that appear to be phonetic mishearings of known names.

    Args:
        text: Transcript text to analyze
        prompt: Prompt template with {text} placeholder
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        Tuple of (corrections, processing_entry).
        corrections: List of dicts with 'transcribed' and 'correct' keys.
        processing_entry: Dict with stage metadata (no corrections_applied —
            pipeline adds that after apply_corrections).

    Raises:
        TimeoutError: If ollama exceeds timeout
        RuntimeError: If ollama call fails
        ValueError: If LLM response cannot be parsed
    """
    formatted = prompt.format(text=text)
    response = _call_ollama(formatted, model, timeout)
    corrections = _parse_llm_corrections(response)
    entry = {
        "stage": "llm_normalization",
        "model": model,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return corrections, entry
