# Tell Me a Story

My daughter Arti is four. Every night, we make up stories together. Characters recur. Worlds layer. Plot threads tangle across weeks of bedtime improv.

I wanted to capture it. Not the polished retellings, the live mess: her interruptions, my detours, the penguin who inexplicably showed up in every story for a month.

Most transcription tools are cloud-based. For family recordings, I wanted something local. So I built my own.

## What It Does

Audio in, speaker-labeled transcript out. Entirely local.

1. **Transcription** — MLX Whisper (large model) produces word-level timestamps
2. **Diarization** — Pyannote separates speakers (parent vs. child voice)
3. **Alignment** — Custom layer merges transcript to speaker segments
4. **Filtering** — Hallucination detection for quiet/unclear speech

The hard part is child speech. Whisper hallucinates when it can't decode soft voices. Pyannote struggles with speaker changes in overlapping conversation. The alignment layer compensates with filtering heuristics I've tuned through iteration.

## Why Local

I could have used cloud APIs. But I wanted to understand the full pipeline, not just call an endpoint. And for recordings of my kid, local felt right.

The constraint forced me to solve problems that managed services abstract away: model selection, hallucination filtering, speaker alignment. That's where the learning lives.

## Tech Stack

- **MLX Whisper** for transcription (Apple Silicon optimized)
- **Pyannote** for speaker diarization
- Python pipeline architecture
- 43 tests, including slow integration tests against real audio

## Status

Working pipeline. Transcribes, diarizes, aligns, saves JSON with word-level timestamps. Still building: story element extraction, searchable archive, the thing that makes captured stories useful months later.

## Background

I spent five years writing children's animation (Daniel Tiger's Neighborhood, Wonder Pets, work with Mo Willems). Then I spent a year as a Creative Technologist at Kibeam Learning, building tools and ML pipelines for interactive children's products.

This project sits at the intersection: audio ML applied to a problem I actually have, built with constraints I actually care about.

## Running It

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/tell-me-a-story.git
cd tell-me-a-story
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set Hugging Face token (required for Pyannote)
export HF_TOKEN=your_token_here

# Run full pipeline
python src/pipeline.py sessions/<session-id>/audio.m4a
```

Requires Apple Silicon for MLX Whisper. Pyannote model downloads on first run (~1GB).

## Build Journal

I've been keeping a daily log in `journal/`. It tracks decisions, experiments, dead ends, and what I'm learning about audio ML along the way.
