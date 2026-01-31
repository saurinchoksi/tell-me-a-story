"""Compare local LLMs on speaker correction task.

Loads a processed transcript with speaker labels, sends to multiple LLMs
for correction, and generates a comparison report.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# Model configurations
MODELS = [
    {"name": "Qwen3 8B", "ollama_name": "qwen3:8b", "temperature": None},
    {"name": "DeepSeek R1 8B", "ollama_name": "deepseek-r1:8b", "temperature": 0.6},
    {"name": "Gemma 3 12B", "ollama_name": "gemma3:12b-it-qat", "temperature": None},
]

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT_SECONDS = 300  # 5 minutes

# Input/output paths
INPUT_PATH = Path("sessions/processed/New Recording 63_short.json")
OUTPUT_DIR = Path("test_outputs")


def format_for_llm(utterances: list[dict]) -> str:
    """Convert utterances to LLM-friendly format with speaker tags.

    Maps speaker labels to compact tags:
    - SPEAKER_00 -> <spk:0>
    - SPEAKER_01 -> <spk:1>
    - None -> <spk:?>

    Args:
        utterances: List of utterance dicts with 'speaker' and 'text' keys

    Returns:
        String with speaker tags followed by utterance text.
    """
    parts = []
    for utt in utterances:
        speaker = utt.get("speaker")
        text = utt.get("text", "").strip()

        if speaker == "SPEAKER_00":
            tag = "<spk:0>"
        elif speaker == "SPEAKER_01":
            tag = "<spk:1>"
        else:
            tag = "<spk:?>"

        parts.append(f"{tag} {text}")

    return " ".join(parts)


def parse_llm_output(raw_output: str) -> tuple[str, str]:
    """Split LLM output into reasoning and corrected transcript.

    Looks for "CORRECTED:" delimiter (case-insensitive) to separate
    the model's reasoning from its corrected output. If no delimiter,
    treats entire output as the corrected transcript.

    Args:
        raw_output: Raw text output from the LLM

    Returns:
        Tuple of (reasoning, corrected_transcript).
    """
    # Case-insensitive search for CORRECTED: line
    pattern = re.compile(r"^CORRECTED:\s*", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(raw_output)

    if match:
        reasoning = raw_output[:match.start()].strip()
        corrected = raw_output[match.end():].strip()
        return reasoning, corrected
    else:
        # No delimiter found - treat entire output as corrected transcript
        return "", raw_output.strip()


def run_ollama(model: str, prompt: str, temperature: Optional[float] = None) -> str:
    """Call Ollama API with the given model and prompt.

    Args:
        model: Ollama model name (e.g., "qwen3:8b")
        prompt: The full prompt to send
        temperature: Optional temperature override

    Returns:
        The model's response text.

    Raises:
        requests.RequestException: On API errors
        requests.Timeout: If request exceeds timeout
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    result = response.json()
    return result.get("response", "")


def count_markers(transcript: str) -> dict:
    """Count speaker markers in a transcript.

    Args:
        transcript: Text containing speaker markers

    Returns:
        Dict with counts for each marker type:
        - spk_0: confident speaker 0
        - spk_1: confident speaker 1
        - spk_0_uncertain: uncertain speaker 0
        - spk_1_uncertain: uncertain speaker 1
        - spk_unknown: ambiguous/unknown
    """
    return {
        "spk_0": len(re.findall(r"<spk:0>", transcript)),
        "spk_1": len(re.findall(r"<spk:1>", transcript)),
        "spk_0_uncertain": len(re.findall(r"<spk:0\?>", transcript)),
        "spk_1_uncertain": len(re.findall(r"<spk:1\?>", transcript)),
        "spk_unknown": len(re.findall(r"<spk:\?>", transcript)),
    }


def compare_transcripts(original: str, corrected: str) -> dict:
    """Compare original and corrected transcripts.

    Uses positional word matching to identify changes.

    Args:
        original: Original transcript with speaker markers
        corrected: Corrected transcript from LLM

    Returns:
        Dict with change counts:
        - resolved_confident: <spk:?> -> <spk:0> or <spk:1>
        - resolved_uncertain: <spk:?> -> <spk:0?> or <spk:1?>
        - left_ambiguous: <spk:?> unchanged
        - other_changes: other speaker reassignments
    """
    # Extract marker sequences from both transcripts
    # Valid markers: <spk:0>, <spk:1>, <spk:0?>, <spk:1?>, <spk:?>
    marker_pattern = r"<spk:(?:0\??|1\??|\?)>"

    original_markers = re.findall(marker_pattern, original)
    corrected_markers = re.findall(marker_pattern, corrected)

    stats = {
        "resolved_confident": 0,
        "resolved_uncertain": 0,
        "left_ambiguous": 0,
        "other_changes": 0,
    }

    # Compare markers positionally
    for i, orig in enumerate(original_markers):
        if i >= len(corrected_markers):
            # Corrected has fewer markers - skip
            continue

        corr = corrected_markers[i]

        if orig == "<spk:?>":
            # Originally unknown
            if corr in ("<spk:0>", "<spk:1>"):
                stats["resolved_confident"] += 1
            elif corr in ("<spk:0?>", "<spk:1?>"):
                stats["resolved_uncertain"] += 1
            elif corr == "<spk:?>":
                stats["left_ambiguous"] += 1
            else:
                stats["other_changes"] += 1
        elif orig != corr:
            # Speaker changed (was known, now different)
            stats["other_changes"] += 1

    return stats


def build_prompt(transcript: str) -> str:
    """Build the full prompt for speaker correction.

    Args:
        transcript: Formatted transcript with speaker markers

    Returns:
        Complete prompt string for the LLM.
    """
    return f"""This is a conversation between a parent and young child. Some words are marked <spk:?> because the speaker is unknown.

Resolve the <spk:?> markers. Do NOT change existing <spk:0> or <spk:1> assignments.

Output ONLY the corrected transcript with all <spk:?> resolved. No explanation, no formatting, just the transcript.

Transcript:
{transcript}"""


def generate_markdown_report(results: list[dict], original: str) -> str:
    """Generate a markdown comparison report.

    Args:
        results: List of result dicts from each model run
        original: Original formatted transcript

    Returns:
        Markdown string for the report.
    """
    lines = []

    # Header
    lines.append("# LLM Speaker Correction Comparison")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Original transcript info
    original_counts = count_markers(original)
    lines.append("## Original Transcript")
    lines.append("")
    lines.append(f"- Total markers: {sum(original_counts.values())}")
    lines.append(f"- Speaker 0: {original_counts['spk_0']}")
    lines.append(f"- Speaker 1: {original_counts['spk_1']}")
    lines.append(f"- Unknown: {original_counts['spk_unknown']}")
    lines.append("")
    lines.append("```")
    lines.append(original)
    lines.append("```")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Time | Resolved (confident) | Resolved (uncertain) | Left ambiguous | Other changes | Reasoning quality |")
    lines.append("|-------|------|---------------------|---------------------|----------------|---------------|-------------------|")

    for r in results:
        stats = r.get("stats", {})
        time_str = f"{r.get('time_seconds', 0):.1f}s"
        lines.append(
            f"| {r['model_name']} | {time_str} | "
            f"{stats.get('resolved_confident', '-')} | "
            f"{stats.get('resolved_uncertain', '-')} | "
            f"{stats.get('left_ambiguous', '-')} | "
            f"{stats.get('other_changes', '-')} | "
            f"_TODO_ |"
        )

    lines.append("")

    # Per-model sections
    for r in results:
        lines.append(f"## {r['model_name']}")
        lines.append("")

        if r.get("error"):
            lines.append(f"**Error:** {r['error']}")
            lines.append("")
            # Still show raw output if available (helps debug format issues)
            if r.get("raw_output"):
                lines.append("### Raw Output")
                lines.append("")
                lines.append("```")
                lines.append(r["raw_output"])
                lines.append("```")
                lines.append("")
            continue

        lines.append(f"**Time:** {r.get('time_seconds', 0):.1f}s")
        lines.append("")

        # Reasoning
        lines.append("### Reasoning")
        lines.append("")
        lines.append("```")
        lines.append(r.get("reasoning", "(none)"))
        lines.append("```")
        lines.append("")

        # Corrected transcript
        lines.append("### Corrected Transcript")
        lines.append("")
        lines.append("```")
        lines.append(r.get("corrected", "(none)"))
        lines.append("```")
        lines.append("")

        # Marker counts
        if r.get("corrected"):
            counts = count_markers(r["corrected"])
            lines.append("### Marker Counts")
            lines.append("")
            lines.append(f"- Speaker 0: {counts['spk_0']}")
            lines.append(f"- Speaker 1: {counts['spk_1']}")
            lines.append(f"- Speaker 0 (uncertain): {counts['spk_0_uncertain']}")
            lines.append(f"- Speaker 1 (uncertain): {counts['spk_1_uncertain']}")
            lines.append(f"- Unknown: {counts['spk_unknown']}")
            lines.append("")

    return "\n".join(lines)


def main():
    """Run comparison across all models and generate report."""
    # Load input JSON
    input_path = Path(__file__).parent.parent / INPUT_PATH
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        session = json.load(f)

    # Validate JSON structure
    if "stories" not in session or not session["stories"]:
        print(f"Error: No stories found in {input_path}")
        return

    story = session["stories"][0]
    if "utterances" not in story:
        print(f"Error: No utterances in first story")
        return

    # Extract utterances from first story
    utterances = story["utterances"]
    print(f"Loaded {len(utterances)} utterances from {INPUT_PATH}")

    # Format for LLM
    original = format_for_llm(utterances)
    print(f"Formatted transcript: {len(original)} chars")
    print()

    # Build prompt
    prompt = build_prompt(original)

    # Run each model
    results = []
    for model_config in MODELS:
        model_name = model_config["name"]
        ollama_name = model_config["ollama_name"]
        temperature = model_config["temperature"]

        print(f"Running {model_name}...")

        result = {
            "model_name": model_name,
            "ollama_name": ollama_name,
        }

        try:
            start_time = time.time()
            raw_output = run_ollama(ollama_name, prompt, temperature)
            elapsed = time.time() - start_time

            reasoning, corrected = parse_llm_output(raw_output)

            result["time_seconds"] = elapsed
            result["raw_output"] = raw_output
            result["reasoning"] = reasoning
            result["corrected"] = corrected

            # Compare if we got corrected output
            if corrected:
                result["stats"] = compare_transcripts(original, corrected)
            else:
                result["stats"] = {}
                result["error"] = "No CORRECTED: delimiter found in output"

            print(f"  Completed in {elapsed:.1f}s")

        except requests.Timeout:
            result["error"] = f"Timeout after {TIMEOUT_SECONDS}s"
            result["time_seconds"] = TIMEOUT_SECONDS
            print(f"  Timeout!")

        except requests.ConnectionError:
            result["error"] = "Could not connect to Ollama at localhost:11434. Is it running?"
            result["time_seconds"] = 0
            print(f"  Connection failed - is Ollama running?")

        except requests.RequestException as e:
            result["error"] = str(e)
            result["time_seconds"] = 0
            print(f"  Error: {e}")

        results.append(result)
        print()

    # Generate report
    report = generate_markdown_report(results, original)

    # Save report
    output_dir = Path(__file__).parent.parent / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    output_filename = f"llm_comparison_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
