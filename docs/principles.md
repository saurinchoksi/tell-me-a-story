# Build Principles

How I want to approach building this project. These will evolve.

---

## Simple Over Clever

Prefer simple, easy-to-understand code over "cleaner" architecture that adds complexity. Redundant-but-readable beats elegant-but-tricky. Only accept complexity when there's a clear tradeoff in efficiency, speed, or correctness.

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

### Session Flow for Learning

Before writing any code, follow this cadence:

1. **Conceptual overview** — Claude explains the problem, the data structures involved, and the general approach. No code yet.

2. **Discussion** — Choksi asks questions, explores edge cases, challenges assumptions. This is where real understanding happens.

3. **Logic walkthrough** — Claude explains the algorithm step-by-step, often in pseudo-code or plain English. This is where Choksi builds the mental model.

4. **Claude writes the code** — Choksi generally won't be typing code. Claude generates it, then walks through what's interesting vs. mechanical.

5. **Code review** — Discuss the implementation. Choksi asks questions until it feels solid.

5. **Repeat for tests** — Same flow: concept → discuss → decide who writes → review.

The goal is never to feel rushed past understanding. If Claude moves too fast, say so.

### Understanding vs. Typing

Typing code doesn't equal understanding. You can type something and not understand it. You can read something carefully and understand it deeply.

The understanding comes from the mental model — the conceptual discussion, the "wait, why are we doing it this way?" questions, catching confusion before it becomes bugs.

**Where Choksi's time is valuable:**
- Asking clarifying questions
- Challenging assumptions
- Deciding what to build and why
- Reading code and verifying it matches the mental model

**Where Choksi's time isn't valuable:**
- Typing `labeled_words.append({`

Copy-paste the code. The job is to *read it and make sure it matches the mental model*. If something looks wrong or confusing, stop and ask.

The skill being built is: *directing and understanding*, not *typing*. That's the 2026 builder skill.

### When to Type vs. When to Generate

**Type it yourself when:**
- The code introduces new patterns you haven't seen before
- You need to build muscle memory (new library, new syntax)
- The code is short and conceptually dense

**Let Claude generate when:**
- The code follows patterns you've already internalized
- It's mostly boilerplate or API calls
- The learning is in *understanding* the code, not *writing* it

Claude should always call out what's worth your attention in generated code — the conceptually interesting parts vs. the mechanical parts.

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
