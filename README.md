# Tell Me a Story

A local system for capturing bedtime stories. Audio goes in, speaker-labeled transcripts come out.

Every night, my daughter Arti and I tell stories together. Characters recur, plot threads tangle across weeks of bedtime improv, and she asks questions I never saw coming.

I wanted to capture those conversations — to eventually "see" the stories we're building together. So I built my own pipeline.

## The Mahabharata

Our test recordings are of Arti asking questions about the Mahabharata — the ancient Indian epic I grew up with, now being passed to my daughter. "Why did Duryodhana want to be king?" "What happened to the Pandavas?" These are real questions from a four-year-old, and in some ways, the soul of the project.

The Mahabharata is the first content domain, not the only one. The pipeline is built to handle any storytelling session — improvised adventures, fairy tale retellings, interdimensional portal hopping tales, whatever. Where the Mahabharata shapes things (like name correction), it's pluggable: swap the reference library, swap the prompt, and the same pipeline works for different stories..

## What It Does

Five-stage pipeline that turns a raw audio recording into a speaker-labeled, corrected transcript:

1. **Transcription** — Converts audio to text with word-level timestamps. Long-form models work best for capturing quiet child speech; short clips lose too much context.

2. **Diarization** — Identifies *who* is speaking and *when*, producing time-stamped speaker segments independent of the transcript. A father and daughter sound quite different, which helps — but toddler interjections are still the hardest case.

3. **LLM Normalization** — A local language model reads the full transcript and corrects words that sound like names but got mangled by the transcriber. For our Mahabharata sessions, "Durian" becomes "Duryodhana" and "fondos" becomes "Pandavas." For a different storytelling domain, you'd swap the prompt.

4. **Dictionary Normalization** — A deterministic pass catches spelling variants the LLM misses, using a reference library of known names and their common mishearings. Our Mahabharata library has 56 entries. You'd build a different library for different content — or skip this stage entirely if your stories don't have specialized vocabulary.

5. **Speaker Enrichment** — Merges the diarization segments onto the transcript at word level, so every word knows who said it and how confident that attribution is. Words that fall in silence gaps or between speakers get flagged rather than guessed at.

Each stage adds information without destroying what came before. Original transcriptions are preserved alongside corrections, and every change is tracked with an audit trail. The raw transcript is saved separately and never modified — it's the honest record of what the transcriber heard.

## Hallucination Handling

Speech transcription models hallucinate — they generate confident-sounding text during silence, breathing, or background noise. With a toddler's soft voice and long pauses, this happens a lot.

Three detection filters flag suspect content: silence gaps where no speaker was detected, near-zero probability words where the transcriber itself wasn't confident, and duplicate segments that repeat at processing boundaries. These are review aids, not automated deletion — the filters surface candidates for human judgment.

## Why Local

For recordings of my kid, I wanted everything to runs on local hardware with no network dependency. No API keys, no data leaving the house.

The constraint forced me to understand the full pipeline — model behavior, hallucination patterns, speaker alignment — rather than outsourcing it to a service.

It also let me work with technology I was excited about.

## Capture Device

On the horizon: an ESP32-based capture device so recording disappears into the background. Tap to start, tap to stop — not always listening, no screens, no attention-grabbing LEDs, operable in the dark. A bedside storytelling tool, not a surveillance device.

Audio records to an SD card initially, with WiFi sync to the processing machine planned for later.

## Tech Stack

- **MLX Whisper** — Transcription, optimized for Apple Silicon
- **Pyannote** — Speaker diarization
- **Ollama** — Local LLM for name correction
- **Python** — Pipeline, automated tests (fast/slow split), CLI
- **Flask** — Validation player with waveform visualization and word-level highlighting

## Requirements

- Apple Silicon Mac (for MLX Whisper)
- Hugging Face token (for Pyannote model download)
- Ollama installed locally (for LLM normalization)
- FFmpeg (for audio format conversion)

## Running It

```bash
# Clone and set up
git clone https://github.com/saurinchoksi/tell-me-a-story.git
cd tell-me-a-story
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set Hugging Face token (required for Pyannote)
export HF_TOKEN=your_token_here

# Run full pipeline
python src/pipeline.py sessions/<session-id>/audio.m4a

# Re-enrich an existing session (skip transcription and diarization)
python src/pipeline.py sessions/<session-id>/ --re-enrich
```

Models download on first run.

## Build Log

Development decisions and discoveries are tracked in `changelog.md`. A narrative build log lives on [the project page](https://saurinchoksi.com/portfolio/tell-me-a-story-log.html).

## Background

I'm a Sesame Workshop Writers Room fellow. I wrote for Mo Willems. I earned an Emmy nomination for Daniel Tiger's Neighborhood. Most recently, I built tools for the content team at Kibeam Learning, a company that makes interactive products for kids.
