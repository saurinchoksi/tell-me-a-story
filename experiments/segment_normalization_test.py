#!/usr/bin/env python3
"""
Test segment-by-segment LLM normalization vs full-transcript.

Usage:
    python experiments/segment_normalization_test.py
"""

import json
import re
import subprocess
from pathlib import Path

TRANSCRIPT_PATH = Path(__file__).parent / "results" / "run_0.json"
MODEL = "qwen3:8b"

PROMPT_TEMPLATE = """This text is from a conversation about the Mahabharata epic. A child is pronouncing Sanskrit names phonetically, often with significant distortion.

Look at the text and identify any words that might be phonetic mishearings of Mahabharata character or group names (like Pandavas, Kauravas, Yudhishthira, Duryodhana, Dhritarashtra, Pandu, etc.)

ONLY return corrections for words that actually appear in the text.
If no mishearings appear, return an empty list.

Return JSON only:
{{"corrections": [{{"transcribed": "...", "correct": "..."}}]}}

Text:
{text}"""


def call_ollama(prompt: str, debug: bool = False) -> dict | None:
    """Call Ollama and parse JSON response."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        response = result.stdout.strip()
        
        if debug:
            print(f"\n[RAW RESPONSE ({len(response)} chars):]")
            print(response[:1000])
            print("[END RAW]\n")
        
        # Find JSON object in response (handles thinking mode)
        match = re.search(r'\{"corrections":\s*\[.*?\]\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        
        # Fallback: try code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        return json.loads(response.strip())
    except Exception as e:
        print(f" [ERROR: {e}]")
        print(f" [Response length: {len(result.stdout) if 'result' in dir() else 'N/A'}]")
        print(f" [Response tail: ...{result.stdout[-500:] if 'result' in dir() else 'N/A'}]")
        return None


def test_segments(data: dict) -> dict:
    """Test each segment individually."""
    segments = data["segments"]
    all_corrections = {}
    
    valid_segments = [
        s for s in segments 
        if s.get("text", "").strip()
    ]
    
    print(f"Testing {len(valid_segments)} segments...\n")
    
    for i, seg in enumerate(valid_segments):
        text = seg["text"].strip()
        
        print(f"[{i+1}/{len(valid_segments)}] \"{text[:60]}{'...' if len(text) > 60 else ''}\"", end="", flush=True)
        
        prompt = PROMPT_TEMPLATE.format(text=text)
        result = call_ollama(prompt)
        
        if result and result.get("corrections"):
            for c in result["corrections"]:
                # Skip template placeholders
                if c["transcribed"] == "..." or c["correct"] == "...":
                    continue
                key = c["transcribed"].lower()
                if key not in all_corrections:
                    all_corrections[key] = c["correct"]
                    print(f" → {c['transcribed']} → {c['correct']}")
                else:
                    print()
        else:
            if result:
                print(f" (empty: {result})")
            else:
                print(" (no result)")
    
    return all_corrections


def main():
    print(f"Model: {MODEL}\n")
    
    with open(TRANSCRIPT_PATH) as f:
        data = json.load(f)
    
    corrections = test_segments(data)
    
    # Ground truth
    ground_truth = {"fondos", "fondo", "goros", "yudister", "dhrashtra"}
    found = set(corrections.keys())
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Found: {sorted(found)}")
    print(f"Ground truth: {sorted(ground_truth)}")
    print(f"Recall: {len(found & ground_truth)}/{len(ground_truth)}")
    print(f"Missed: {sorted(ground_truth - found)}")


if __name__ == "__main__":
    main()
