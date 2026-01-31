# Whisper Transcription Notes

Technical learnings about Whisper behavior discovered during pipeline development.

---

## Audio Length Affects Transcription Quality Globally

**Discovered:** 2026-01-30

### The Finding

Whisper's transcription quality depends on total audio length in unexpected ways. A ~15-second difference in total audio length (180s vs 195s) can determine whether borderline speech segments in the *first 30 seconds* get transcribed — even though both clips process that same 30-second window identically.

### Test Case

Using "New Recording 63" (a bedtime story recording):

| Clip Duration | Parent's Repeated Question (at 19-22s) | Notes |
|---------------|----------------------------------------|-------|
| 338s (full)   | ✓ Captured | Complete recording |
| 195s (3.25 min) | ✓ Captured | |
| 180s (3 min)  | ✗ Missing | 15 seconds shorter |
| 90s (1.5 min) | ✗ Missing | |
| 60s (1 min)   | ✗ Missing | |
| 38s           | ✗ Missing | |

The missing segment is a parent repeating a child's question — softer speech, slightly overlapping with an "uh-huh" that follows.

### Why This Is Surprising

Whisper processes audio in 30-second windows. The parent's repeated question at 19-22 seconds falls solidly within the *first* window. Context passing between chunks shouldn't matter here.

Yet total audio length changes whether this segment gets transcribed.

### Hypotheses

1. **VAD threshold calibration** — Whisper may use total audio characteristics to set thresholds for what counts as "speech worth transcribing"

2. **Confidence calibration** — Longer recordings might cause Whisper to be more generous with borderline segments throughout

3. **Global planning pass** — The long-form algorithm might do a quick scan of the whole file before detailed transcription, affecting decisions even in early windows

### Whisper Architecture Context

From research on Whisper's design:

- All audio is padded/truncated to exactly 30 seconds before mel spectrogram conversion
- Model trained without attention mask — learns to ignore padding from spectrogram directly  
- Long-form transcription uses sequential chunking with context propagation
- Previous transcription text (up to half max tokens) passed to next chunk via `<|startofprev|>` token
- Timestamp tokens signal when speech overflows 30s window, triggering overlap

The context-passing mechanism explains why *later* chunks benefit from *earlier* content. But it doesn't fully explain why total length affects the *first* chunk's behavior.

### Practical Implications

**For this pipeline:**
- Always process complete recordings (bedtime stories are naturally 5-10+ minutes)
- Short extracted samples are unreliable for validation testing
- Don't use short clips to test transcription quality — results won't match production behavior

**For testing:**
- Use recordings >3 minutes for reliable validation
- When creating test fixtures, include enough audio to cross the threshold
- Ground truth validation should use full recordings, not extracted segments

### Related: Model Size Affects Different Failures

Separate finding: different Whisper model sizes have complementary failure modes. The small model sometimes captures segments the large model misses, and vice versa. This suggests ensemble approaches might improve coverage, but adds complexity.

For now, sticking with `large-v3-turbo` as the primary model since it performs best on the target use case (soft-spoken children's voices).

---

## Future Research

- [ ] Test exact threshold more precisely (somewhere between 180-195s for this recording)
- [ ] Check if threshold varies by audio characteristics (silence ratio, speaker count, etc.)
- [ ] Investigate whether `initial_prompt` parameter helps with short audio
- [ ] Research WhisperX and other wrappers that handle chunking differently

---

*Last updated: 2026-01-30*
