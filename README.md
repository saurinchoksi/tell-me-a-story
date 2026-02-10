# Tell Me a Story

A local system for capturing bedtime stories. Audio goes in, speaker-labeled transcripts come out.

Every night, my daughter Arti and I tell stories together. Characters recur, plot threads tangle across weeks of bedtime improv, and she asks questions I never saw coming.

I wanted to capture those stories and conversations to eventually "see" it all. So I built my own pipeline.

## What It Does

Four-stage enrichment pipeline, entirely local:

1. **Transcription** — MLX Whisper (large model) produces word-level timestamps
2. **LLM Normalization** — Ollama (qwen3:8b) corrects Sanskrit names that Whisper mishears ("Pandava" from "Pandava's", "Duryodhana" from "Durian")
3. **Dictionary Normalization** — Deterministic pass catches spelling variants the LLM misses, using a 56-entry Mahabharata reference library
4. **Diarization Enrichment** — Pyannote speaker labels merged at the word level, with coverage scores for each attribution

Three hallucination filters run during transcription: silence gap detection, near-zero probability detection, and duplicate segment identification. Whisper hallucinates when it can't decode soft voices — these filters catch the worst of it.

Each stage adds information without destroying what came before. `_original` fields preserve raw Whisper output. `_corrections` chains track every change. Schema-versioned output (currently 1.2.0) with full audit trails.

## Why Local

For recordings of my kid, cloud wasn't an option. The constraint forced me to understand the full pipeline — model behavior, hallucination patterns, speaker alignment — rather than calling an endpoint.

The hard problems live in the gaps between models. Whisper needs 3+ minutes of audio context for reliable transcription. Diarization coverage turns out to be a stronger hallucination signal than word probability. Full-transcript LLM processing beats segment-by-segment for name correction. None of this is obvious from documentation. You learn it by building.

## Tech Stack

- **MLX Whisper** — Transcription (Apple Silicon optimized)
- **Pyannote** — Speaker diarization
- **Ollama** (qwen3:8b) — LLM normalization
- Python pipeline with 93+ automated tests (fast unit / slow integration split)
- Flask-based validation player with waveform visualization and word-level highlighting

## Status

Working pipeline. Both test sessions fully enriched at schema 1.2.0. What happens after transcription — what gets extracted, how stories become searchable, what the archive actually becomes — is still emerging.

On the horizon: story element extraction, a mobile browser interface for session playback with synced captions, and ESP32 capture devices so recording disappears into the background.

## Background

Five years writing children's animation (Emmy-nominated for Daniel Tiger's Neighborhood, plus PBS, Netflix, Nickelodeon, Sesame Workshop). One year as Creative Technologist at Kibeam Learning, building tools and ML pipelines for interactive children's products.

This project sits at the intersection: audio ML applied to a problem I actually have, built with constraints I actually care about.

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
```

Requires Apple Silicon for MLX Whisper. Pyannote and Ollama models download on first run.

## Build Log

Development decisions and discoveries are tracked in `changelog.md`. A narrative build log with visual components lives on [the project page](https://saurinchoksi.com/portfolio/tell-me-a-story-log.html).
