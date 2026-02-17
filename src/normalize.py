"""LLM-based normalization of phonetic name mishearings."""

import json
import re
import subprocess


DEFAULT_PROMPT = """This text is from a conversation about the Mahabharata epic. A child is pronouncing Sanskrit names phonetically, often with significant distortion.

Look at the text and identify any words that might be phonetic mishearings of Mahabharata character or group names (like Pandavas, Kauravas, Yudhishthira, Duryodhana, Dhritarashtra, Pandu, etc.)

ONLY return corrections for words that actually appear in the text.
If no mishearings appear, return an empty list.

Return JSON only:
{{"corrections": [{{"transcribed": "...", "correct": "..."}}]}}

Text:
{text}"""


def _call_ollama(prompt: str, model: str, timeout: int) -> str:
    """Call Ollama CLI and return stdout.

    Args:
        prompt: Formatted prompt string
        model: Ollama model name
        timeout: Subprocess timeout in seconds

    Returns:
        Raw stdout string from ollama

    Raises:
        subprocess.TimeoutExpired: If ollama exceeds timeout
        RuntimeError: If subprocess fails
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama returned exit code {result.returncode}: {result.stderr[:500]}"
        )

    return result.stdout


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

    # Tier 1: Regex for corrections JSON
    match = re.search(r'\{"corrections":\s*\[.*?\]\}', stripped, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        return parsed.get("corrections", [])

    # Tier 2: Code block extraction
    if "```json" in stripped:
        json_str = stripped.split("```json")[1].split("```")[0]
        parsed = json.loads(json_str.strip())
        return parsed.get("corrections", [])
    elif "```" in stripped:
        json_str = stripped.split("```")[1].split("```")[0]
        parsed = json.loads(json_str.strip())
        return parsed.get("corrections", [])

    # Tier 3: Direct parse
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        tail = stripped[-200:] if len(stripped) > 200 else stripped
        raise ValueError(f"Could not parse LLM response: ...{tail}")

    return parsed.get("corrections", [])


def llm_normalize(
    text: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = "qwen3:8b",
    timeout: int = 300,
) -> list[dict]:
    """Identify phonetic mishearings in text using a local LLM.

    Sends the full text to a local Ollama model which returns corrections
    for words that appear to be phonetic mishearings of known names.

    Args:
        text: Transcript text to analyze
        prompt: Prompt template with {text} placeholder
        model: Ollama model name
        timeout: Subprocess timeout in seconds

    Returns:
        List of correction dicts, each with 'transcribed' and 'correct' keys.
        Returns [] if no mishearings found.

    Raises:
        subprocess.TimeoutExpired: If ollama exceeds timeout
        RuntimeError: If ollama subprocess fails
        ValueError: If LLM response cannot be parsed
    """
    formatted = prompt.format(text=text)
    response = _call_ollama(formatted, model, timeout)
    return _parse_llm_corrections(response)
