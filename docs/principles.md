# Build Principles

How I want to approach building this project. These will evolve.

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

---

## Learning Principles

- Start small. One audio file → one transcript.
- Understand tools well enough to list them as real skills.
- Write tests as I go.
- Document the journey in the build journal.

---

## Human-First, Tools-Assisted

**Know the code, then scale with tools.**

Early on, write code myself. Understand the libraries, the data flow, the 
decisions. No black boxes. This builds the foundation to make confident 
changes later.

As the project grows and patterns stabilize, bring in Claude Code for 
larger implementations, refactors, and multi-file changes. The principle 
from "4 Patterns That Work" applies: if the agent builds it, the agent 
can maintain it.

The goal: never feel blind or scared about what's in the codebase. Know 
it well enough to direct the tools effectively.

**The progression:**

1. **Now:** Write code with Claude Desktop guiding. Learn the pieces.

2. **Soon:** Hand well-defined tasks to Claude Code. Stay in the loop.

3. **Later:** Let Claude Code build features while I focus on architecture 
   and direction.

4. **Eventually:** Orchestrate multiple Claude Code sessions working in 
   parallel—one writing code, one reviewing, one building tests, several 
   working on different features simultaneously.

The arc of this project: from writing a few lines of code myself to 
orchestrating a team of AI agents. That's the builder skill to develop.

Being a builder in 2026 means using these tools confidently—but confidence 
comes from understanding, not from outsourcing understanding.

---

## Documentation

**Journal** — Narrative of what happened each session. Decisions, learnings, what's next. Lives in `journal/`.

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

## Future Research

When ready to scale to "Claude Code builds features" phase, research:

- **Claude Code review workflows** — one session writes, another reviews before merge
- **Sub-agents and orchestration** — Claude Code spawning focused helpers
- **Skills and custom instructions** — teaching Claude Code project-specific patterns
- **CLAUDE.md best practices** — project context that persists across sessions

Sources: Anthropic docs/Discord, Claude Code GitHub discussions, Twitter workflows, early adopter blog posts.

---

*Last updated: 2026-01-21*
