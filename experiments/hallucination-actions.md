# Hallucination: Actionable Items

Based on `experiments/hallucination_analysis_report_1.md` (2026-02-03), mapped against current pipeline state (2026-02-07).

---

## What exists now

- `clean_transcript()` removes zero-duration words and empty segments — true garbage only
- `filters.py` has a `min_probability()` factory function — available but **never called** in the pipeline
- `query.py` accepts an optional `word_filter` param — but `pipeline.py` doesn't pass one
- No hallucination marking, no duplicate detection, no segment-level flagging

**Philosophy decided:** Mark inline using the same enrichment pattern as normalization. `_hallucination` metadata on words/segments. "Raw artifacts, filtered queries." Not yet built.

---

## Item 1: Low-probability fabrications

**The problem:** Whisper fabricates short utterances ("Well.", "Right.", "Okay.") in silence gaps. 3 of 4 hallucinated segments in the report have `min_word_prob < 0.5`.

**Current handling:** Nothing. These pass through untouched.

**Report recommended:** Add `min_word_prob < 0.5` as segment-level hallucination flag.

**Open questions:**
- Mark at word level, segment level, or both?
- What metadata shape? `_hallucination: {reason: "low_probability", min_prob: 0.133}`?
- Does this become a pipeline stage (like normalization), or part of `clean_transcript()`?
- Threshold: 0.5 catches 3/4. 0.2 catches 2/4. Two-tier (delete vs mark)?
- The report suggested deleting `< 0.2` and marking `0.2–0.5` as `[uncertain]`. Does that conflict with "never destroy data"?

---

## Item 2: Consecutive duplicate segments

**The problem:** Whisper seek windows overlap at boundaries, producing duplicate transcriptions of the same audio. Segment 4 & 5 both transcribe "Well." from ~25.5s. The duplicate has dramatically lower probability (0.133 vs 0.993).

**Current handling:** Nothing.

**Report recommended:** If `current.text == previous.text` and `min_prob(current) < 0.5`, mark as duplicate.

**Open questions:**
- Text comparison: exact match, or fuzzy (stripped/lowered)?
- Only consecutive segments, or within a time window?
- Mark the lower-prob one as duplicate, keep the higher-prob one?
- Same inline enrichment pattern? `_hallucination: {reason: "duplicate", duplicate_of_segment: 4}`?

---

## Item 3: High-confidence hallucination (the hard one)

**The problem:** Segment 23 repeats segment 22's content with prob 0.745 and compression 2.39 (just under 2.5 threshold). Passes all proposed filters.

**Current handling:** Nothing. And no easy automated fix.

**Report recommended:** Either lower compression threshold to 2.4, or add consecutive-repeat detection.

**Open questions:**
- Lowering compression to 2.4 — what's the false positive risk on legitimate speech?
- Consecutive-repeat detection: how similar is "similar enough"? Exact text match? Fuzzy?
- Is this rare enough to punt to the human correction stage?
- How many of these exist across a full corpus (we only have 1 session analyzed)?

---

## Item 4: Missing quiet speech

**The problem:** Arti's quiet questions aren't captured. Three instances in the report. Timestamps show gaps where speech occurred but wasn't transcribed.

**Current handling:** Nothing possible in software — audio simply doesn't contain the speech.

**Report recommended:** Hardware solutions (mic placement, gain). Software: mark gaps > 1s as potential missing speech for human review.

**Open questions:**
- Is gap-marking worth building now, or does the ESP32 device solve this at the source?
- If we mark gaps, where does the marker live? It's not a word or segment — it's an absence.
- Could voice activity detection on the raw audio identify "there was sound here but Whisper produced nothing"?
- Priority: this feels like a capture-layer problem, not a pipeline-layer problem.

---

## Item 5: Pipeline stage architecture

**The problem:** Where does hallucination marking fit in the pipeline sequence?

**Current pipeline:**
```
audio → transcribe → clean → LLM normalize → dict normalize → diarize
```

**Options:**
a) After clean, before normalization (don't normalize hallucinated words)
b) After normalization, before diarization (normalize everything, then flag)
c) As part of clean_transcript() (expand what "clean" means)

**Open questions:**
- If a hallucinated segment contains a real word that got normalized, do we care?
- Should normalization skip words already flagged as hallucinated?
- Is this one stage or two (probability flagging + duplicate detection)?
- Does this need its own `_processing` entry like normalization stages do?

---

## Item 6: Query-time filtering vs pipeline-time marking

**The problem:** `filters.py` and `query.py` already support filtering at query time. The report recommendations mix deletion (pipeline-time) with marking (query-time).

**Current philosophy:** "Raw artifacts, filtered queries." Mark in pipeline, filter in query.

**Open questions:**
- Is the philosophy settled enough to commit? No deletion at pipeline time, only marking?
- If so, the report's "delete < 0.2" recommendation gets overridden — mark instead, let query decide.
- `filters.py` currently only has `min_probability()`. Need additional predicates: `not_hallucinated()`, `not_duplicate()`?
- How does the formatted transcript (`format_transcript()`) handle marked-but-not-filtered words? Show them with a marker? Hide by default?
