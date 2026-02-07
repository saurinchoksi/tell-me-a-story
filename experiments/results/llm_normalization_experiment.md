# LLM Normalization Experiment: Full Transcript vs Segment-by-Segment

**Date:** 2026-02-06  
**Model:** qwen3:8b (Ollama, local)  
**Test File:** run_0.json (baseline transcript, no initial_prompt)

---

## Hypothesis

Segment-by-segment LLM processing might catch context-dependent mishearings that full-transcript processing misses. Alternatively, it might lose context and perform worse.

---

## Method

### Full Transcript (Single Shot)
- Concatenate all segment text into one prompt
- Single LLM call
- ~5 seconds

### Segment-by-Segment
- Each segment processed independently  
- 89 LLM calls
- ~15 minutes total

### Prompt Used (Both Approaches)
```
This text is from a conversation about the Mahabharata epic. A child is pronouncing Sanskrit names phonetically, often with significant distortion.

Look at the text and identify any words that might be phonetic mishearings of Mahabharata character or group names (like Pandavas, Kauravas, Yudhishthira, Duryodhana, Dhritarashtra, Pandu, etc.)

ONLY return corrections for words that actually appear in the text.
If no mishearings appear, return an empty list.

Return JSON only:
{"corrections": [{"transcribed": "...", "correct": "..."}]}
```

### Ground Truth
Manually identified phonetic mishearings in transcript:
| Transcribed | Correct | Notes |
|-------------|---------|-------|
| fondos | Pandavas | Group name |
| fondo | Pandu | Individual (father) |
| goros | Kauravas | Group name |
| yudister | Yudhishthira | Individual |
| dhrashtra | Dhritarashtra | Individual |

---

## Results

### Recall Comparison

| Approach | Recall | Caught | Missed | False Positives |
|----------|--------|--------|--------|----------------|
| Full transcript | 5/5 (100%) | fondos, fondo, goros, yudister, dhrashtra | — | 0 |
| Segment-by-segment | 4/5 (80%) | fondos, goros, yudister, dhrashtra | fondo* | 3 |

*fondo missed due to 120s timeout on segment 37, not model failure

**Key finding:** Full transcript achieves perfect recall with zero false positives.

### Precision Analysis

#### Full Transcript
- **True Positives:** 5
- **False Positives:** 0
- **Precision:** 100%
- **Recall:** 100%

#### Segment-by-Segment
- **True Positives:** 4 (ground truth) + 1 bonus (pondovas)
- **False Positives:** 3 (English words mapped to Sanskrit)
- **Precision:** ~63%

### Detailed Segment-by-Segment Output

#### True Positives (Ground Truth)
| Segment | Transcribed | Corrected To |
|---------|-------------|--------------|
| 2 | fondos | Pandavas |
| 2 | goros | Kauravas |
| 11 | Yudister | Yudhishthira |
| 45 | Dhrashtra | Dhritarashtra |

#### Bonus Catches (Valid, Not in Ground Truth)
| Segment | Transcribed | Corrected To | Notes |
|---------|-------------|--------------|-------|
| 62 | Pondovas | Pandavas | Alternative pronunciation |

#### Hindi Variants Caught (Pass 2 Territory)
| Segment | Transcribed | Corrected To | Notes |
|---------|-------------|--------------|-------|
| 8 | Duryodhan | Duryodhana | Hindi shortening |
| 44 | Yudhisthir | Yudhishthira | Hindi shortening |

These are valid corrections but belong to Pass 2 (dictionary-based standardization), not Pass 1 (phonetic mishearing detection).

#### False Positives (Hallucinations)
| Segment | Transcribed | Corrected To | Why Wrong |
|---------|-------------|--------------|-----------|
| 26 | best | Bhishma | "best" is English |
| 29 | father | Dhritarashtra | "father" is English |
| 58 | dad | Pandu | "dad" is English |

The model over-triggered on English words that contextually relate to Mahabharata characters.

---

## Technical Observations

### Response Format Issues
- Model sometimes returns `{}` instead of `{"corrections": []}`
- Regex `r'\{"corrections":\s*\[.*?\]\}'` fails on empty object
- Need fallback parsing

### Timeout Behavior
- 120s timeout hit on some segments
- Segment 37 ("older than Fondo") timed out — missed a ground truth item
- Longer segments or complex reasoning can exceed timeout

### Non-Determinism
- Same segment produced different results on different runs
- Early test: fondos→Kauravas, goros→Pandavas (swapped!)
- Later runs: correct mapping
- LLM reasoning is probabilistic

### Thinking Mode
- qwen3:8b outputs reasoning before JSON
- Must extract JSON from end of response
- Reasoning is helpful for debugging but adds latency

---

## Conclusions

### Full Transcript: Clear Winner
1. **Perfect recall** — 5/5 ground truth items caught
2. **Perfect precision** — no false positives
3. **Faster** — single LLM call (~5s vs ~15min)
4. **More consistent** — full context reduces ambiguity
5. **Simpler** — no aggregation/deduplication needed

### Segment-by-Segment: Not Recommended
1. **Lower recall** — 4/5 (missed fondo due to timeout)
2. **Lower precision** — 3 false positives (English words)
3. **Slower** — 89 calls vs 1
4. **Non-deterministic** — results vary between runs
5. **Timeout risk** — long segments may fail

The only advantage was catching "pondovas" as a bonus, but this doesn't justify the tradeoffs.

---

## Recommendation

**Use full transcript for Pass 1.** Perfect recall and precision in testing.

Pass 2 (dictionary) still needed for:
1. **Hindi variants** — Duryodhan → Duryodhana, Yudhisthir → Yudhishthira
2. **Validation filter** — only accept corrections where `transcribed` appears in text (safety check)

---

## Architecture Implication

```
┌─────────────────────────────────────────────────────────┐
│                    NORMALIZATION                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Pass 1: LLM (Full Transcript)                          │
│    Input: Complete transcript text                       │
│    Output: {transcribed → canonical} for mishearings    │
│    Example: "fondos" → "Pandavas"                       │
│             "dhrashtra" → "Dhritarashtra"               │
│                                                          │
│  Pass 2: Dictionary (Hindi Variants)                    │
│    Input: All remaining Sanskrit-looking words          │
│    Output: Standardized canonical forms                 │
│    Example: "Duryodhan" → "Duryodhana"                  │
│             "Yudhisthir" → "Yudhishthira"               │
│                                                          │
│  Validation: Filter impossible corrections              │
│    Reject if transcribed not in source text             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Files

- `run_0.json` — baseline transcript (test input)
- `full_transcript_result.json` — full transcript test results (100% recall)
- `../segment_normalization_test.py` — segment test script
- `../full_transcript_test.py` — full transcript test script
