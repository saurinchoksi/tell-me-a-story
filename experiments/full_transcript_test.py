#!/usr/bin/env python3
"""
Test full-transcript LLM normalization (one shot).

Usage:
    python experiments/full_transcript_test.py
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


def call_ollama(prompt: str) -> tuple[dict | None, str]:
    """Call Ollama and parse JSON response. Returns (result, raw_response)."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=300  # 5 min for full transcript
        )
        response = result.stdout.strip()
        
        # Find JSON object in response (handles thinking mode)
        match = re.search(r'\{"corrections":\s*\[.*?\]\}', response, re.DOTALL)
        if match:
            return json.loads(match.group()), response
        
        # Fallback: try code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip()), response
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
            return json.loads(json_str.strip()), response
        
        return json.loads(response.strip()), response
    except Exception as e:
        print(f"ERROR: {e}")
        if 'result' in dir():
            print(f"Response length: {len(result.stdout)}")
            print(f"Response tail: ...{result.stdout[-500:]}")
        return None, ""


def main():
    print(f"Model: {MODEL}")
    print(f"Transcript: {TRANSCRIPT_PATH}\n")
    
    with open(TRANSCRIPT_PATH) as f:
        data = json.load(f)
    
    # Concatenate all segment text
    full_text = data["text"]
    print(f"Full transcript length: {len(full_text)} chars\n")
    
    prompt = PROMPT_TEMPLATE.format(text=full_text)
    
    print("Calling Ollama (full transcript)...")
    result, raw_response = call_ollama(prompt)
    
    print(f"\n{'='*60}")
    print("RAW RESPONSE (last 1000 chars)")
    print('='*60)
    print(f"...{raw_response[-1000:]}")
    
    print(f"\n{'='*60}")
    print("PARSED RESULT")
    print('='*60)
    if result:
        for c in result.get("corrections", []):
            print(f"  {c['transcribed']} → {c['correct']}")
    else:
        print("  (no result)")
    
    # Ground truth comparison
    ground_truth = {"fondos", "fondo", "goros", "yudister", "dhrashtra"}
    if result:
        found = {c["transcribed"].lower() for c in result.get("corrections", [])}
        # Filter template placeholders
        found = {f for f in found if f != "..."}
    else:
        found = set()
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print('='*60)
    print(f"Found: {sorted(found)}")
    print(f"Ground truth: {sorted(ground_truth)}")
    print(f"Recall: {len(found & ground_truth)}/{len(ground_truth)}")
    print(f"Hits: {sorted(found & ground_truth)}")
    print(f"Missed: {sorted(ground_truth - found)}")
    print(f"Extra: {sorted(found - ground_truth)}")
    
    # Save results
    output = {
        "model": MODEL,
        "approach": "full_transcript",
        "transcript_length": len(full_text),
        "result": result,
        "ground_truth": list(ground_truth),
        "found": list(found),
        "recall": len(found & ground_truth) / len(ground_truth),
        "hits": list(found & ground_truth),
        "missed": list(ground_truth - found),
        "extra": list(found - ground_truth)
    }
    
    output_path = Path(__file__).parent / "results" / "full_transcript_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
