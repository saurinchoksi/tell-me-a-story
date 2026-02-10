# Build Principles

How I want to approach building this project. These will evolve.

---

## Simple Over Clever

Prefer simple, easy-to-understand code over "cleaner" architecture that adds complexity. Redundant-but-readable beats elegant-but-tricky. Only accept complexity when there's a clear tradeoff in efficiency, speed, or correctness.

---

## Fail Loud

No silent fallbacks. Use `utt["words"]` not `utt.get("words", [])`. If assumptions break, we want to know immediately — not have bugs hide behind default values.

---

## Capture Generously, Build Features Sparingly

Store data even if you're not sure you'll need it. Cheap to store, impossible to recover. But don't build UI or features for that data until you actually want them.

Data capture principles:
- **Replay-ability** — if you can't recreate it later, store it now
- **Cost of retrieval** — if regenerating is expensive, cache the result
- **Regret heuristic** — bias toward capture when in doubt

---

## Patterns Over Tools

Document the *why* separately from the *what*. When I choose a tool, 
name the underlying need it solves. The tool might change; the pattern 
stays.

Example: "I need speaker diarization" is the pattern. "I'm using 
pyannote.audio" is the current tool choice.

---

## Principles Over Rules

Write guidance that scales to situations I haven't anticipated. Prefer 
"don't swallow errors" over "always log to this specific file." This 
applies to prompts, agent instructions, and my own coding style.

---

## Build With the Agent

Involve Claude in construction, not just execution. Preserve conversation 
context and artifacts (SYNC.md, handoff docs). This way the agent can 
maintain and extend the system later without the switching cost of 
recovering context.

---

## Leave Infrastructure Doors Open

Build for myself first. But don't foreclose on the possibility that 
pieces of this could become shareable—an API, a protocol, a pattern 
others could use. Architecture decisions now can leave doors open.

---

## On Stakes

This project serves two purposes that aren't in conflict: preserving something precious, and demonstrating capability that leads to financial security. The depth is what makes it compelling for both. The personal meaning isn't separate from the career positioning—it's what makes the work real.

---

## Project-Specific Principles

From the project pillars:

- **Zero Latency** (Bret Victor): Immediate connection between intent and 
  result. The capture should disappear.

- **Hard Fun** (Papert): Building this is the learning. Complex 
  engineering as creative material.

- **Calm Tech** (Weiser): Technology that enables rather than extracts. 
  No cloud dependency.

- **The Arti Test**: Would I actually use this with my daughter? If no, 
  don't build it.

- **Honest Transcripts**: Mark unclear speech as `[unintelligible]` rather 
  than deleting or inventing. Preserves truth that something was said; can 
  be filled in later when memory surfaces. Garbled is better than missing. 
  Lies are worse than gaps.

---

## Learning Principles

- Start small. One audio file → one transcript.
- Understand tools well enough to list them as real skills.
- Write tests as I go.
- Document decisions and discoveries in the changelog.

---

## Human-First, Tools-Assisted

**Know the code, then scale with tools.**

See [session-flow.md](session-flow.md) for detailed guidance on:
- Session flow for learning (concept → discuss → logic → code → review)
- Understanding vs. typing — where time is valuable
- When to type vs. when to let Claude generate
- The progression from writing code to orchestrating agents

---

## Documentation

**Changelog** — Structured record of what changed, what was decided, and what was learned. Single file, newest entries at top. Lives in `changelog.md`. Source material for public build log entries. Replaces the per-session journal files (archived in `journal/`).

**Docstrings** — Reference material embedded in code. Document functions as they're written:

```python
def transcribe(audio_path: str) -> dict:
    """Transcribe audio file using MLX Whisper.
    
    Args:
        audio_path: Path to the audio file (.m4a, .wav, etc.)
    
    Returns:
        Dict with 'text', 'language', and 'segments' keys.
    """
```

Docstrings explain *what* and *how*. Inline comments (sparingly) explain *why* for non-obvious reasoning.

**README** — Keep updated with: what the project does, how to install/run, basic usage.

---

## Technical Reference Documents

Detailed findings and tool-specific learnings live in separate docs:

- **[whisper-notes.md](whisper-notes.md)** — Whisper transcription behavior, audio length effects, model quirks
- **[session-flow.md](session-flow.md)** — Claude collaboration patterns, learning flow, when to type vs generate

---

## Future Research

When ready to scale to "Claude Code builds features" phase, research:

- **Claude Code review workflows** — one session writes, another reviews before merge
- **Sub-agents and orchestration** — Claude Code spawning focused helpers
- **Skills and custom instructions** — teaching Claude Code project-specific patterns
- **CLAUDE.md best practices** — project context that persists across sessions

Sources: Anthropic docs/Discord, Claude Code GitHub discussions, Twitter workflows, early adopter blog posts.

---

*Last updated: 2026-01-30*
