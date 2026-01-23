# Tell Me A Story — Content Source

**Purpose:** Raw material for website, LinkedIn, videos, job applications, and other outputs. This is the reference document—pull from here when creating content.

**Last updated:** January 22, 2026

---

## The Hook (2-3 sentences)

A local-first system for capturing bedtime stories. Audio in, speaker-labeled transcripts out. Built for my daughter Arti—and for me to understand my own storytelling voice.

---

## Origin

### The itch

I tell Arti three stories every night. As she gets older, she's steering more—adding characters, driving plot, asking questions that surprise me. These moments are alive and then they're gone.

Part of me is okay with ephemeral. But part of me wants to hold onto them. For me, for her, for curiosity about what patterns might emerge.

I'd already done a project analyzing my old standup comedy (the Voice Bible) and found patterns I didn't realize were there. What would I find in these stories?

### The constraint that shaped everything

No phone in that space. I want it dark, calm. Just a tap or a switch, and it's captured. The capture disappears. This isn't about documenting—it's about not breaking the moment.

### The Mahabharata thread

Around the same time I started building this, Arti got curious about the Mahabharata. My favorite stories—the ones I grew up with. Complicated, morally gray, not easy to tell to a three-year-old. But she's asking.

Figuring out how to tell the Mahabharata to her, over years, watching how those stories develop between us—that's the thing I most want to capture. The fact that my test file (the one I'm debugging diarization with) is me and Arti talking about Yudhishthira and Duryodhana... that's not incidental. That's the soul of the project.

### The curiosity that grew

As I got deeper into it, things kept fitting. Edge computing, local ML, CUDA, Jetson—stuff I wanted to learn anyway. The technical stack felt right for the problem AND for where I want to go professionally.

And it connects to bigger questions I care about: How does AI work with kids? What if the parent is always the interface—not a meat puppet, but a thoughtful human in the loop? What if the user interface for AI, for a child, is me?

I don't know where it goes. But I can see further as I build. That's the point.

---

## Principles

### Bret Victor: Immediate Connection

Creators need an immediate connection to what they're creating. The capture should disappear. The interface should surface what you need without friction.

For this project: No compile-and-run loop between recording and seeing what you captured. No five-step process to find a story. The system should feel like an extension of the storytelling practice, not an interruption to it.

### Seymour Papert: Hard Fun

The best learning happens when you're building something you actually want. The struggle is part of the point. "Hard fun."

For this project: Building this is the learning. I could use an existing transcription service. I'm choosing to understand the pipeline—diarization, alignment, extraction—because that understanding is valuable. The project is a vehicle for skills I want to develop.

### Mark Weiser: Calm Technology

Technology should inform without demanding attention. It should move between the periphery and the center of our awareness, but default to the periphery.

For this project: The capture device (when built) will have no screen, no attention-grabbing LEDs. It lives in the dark, does its job, disappears. The output surfaces when I want it, not when it wants me.

### The Arti Test

Would I actually use this with my daughter? If the answer is no—if it's too intrusive, too extractive, too demanding—don't build it.

---

## The System

### What it does

**Input:** Audio recordings (phone voice memos now, ESP32 device later)

**Processing:**
- Speaker diarization (who is speaking when)
- Transcription with timestamps (what was said)
- Alignment (speaker labels matched to words)

**Output:** Speaker-labeled transcripts. What happens next—extraction, patterns, story elements, a browsable interface—is still emerging.

### The three layers

**Pipeline (core):** The audio processing. Diarization (pyannote.audio), transcription (MLX Whisper), alignment. This is what I'm building now.

**Capture (input):** How audio gets into the system. Phone voice memos for now. Eventually an ESP32 device—screenless, dark-operable, calm.

**Interface (output):** How I access what's been captured. Not yet designed. Will emerge from actual use.

### Local-first

Everything runs locally. No cloud, no OAuth, no API keys for family content.

This isn't just privacy preference. It's about:
- Owning my data (if data is valuable, I want mine)
- Not depending on services that could change or disappear
- Learning edge computing / local ML as a skill
- Refusing to participate in the surveillance economy for something this intimate

---

## Where I'm Coming From

Emmy-nominated children's animation writer (Daniel Tiger's Neighborhood, Mo Willems). Former standup comedian. Now building technical skills in edge computing, ML pipelines, and local-first systems.

I'm targeting creative technologist roles at companies building physical products and screenless experiences for children and families: Yoto, Sesame Workshop, children's museums, Tonies. Places where storytelling and technology meet, and where someone who understands both sides is valuable.

This project is both a meaningful tool for my family AND relevant R&D for the space I want to work in.

---

## The Ethical Edge

### What I'm not building

Once you have transcribed stories, there are obvious generative AI applications:
- Auto-generate new stories in "your style"
- Create illustrations from story descriptions
- Build an AI companion that knows your family's stories

I'm skeptical of most of these directions.

### Why

The point isn't to outsource storytelling to machines. It's to understand and develop your own voice as a storyteller. The system should be a mirror, not a replacement.

There's something important about the parent being the interface for AI when it comes to kids. I'm not a "meat puppet" passing through AI-generated content. I'm using my judgment, my knowledge of my daughter, my sense of what she needs. The AI helps me reflect on my storytelling practice. It doesn't do the storytelling.

### Local-first as ethical stance

Keeping data out of cloud services isn't just about privacy settings. It's refusing to participate in an economy where intimate family moments become training data or advertising targets.

I want to build tools that enable rather than extract. This project is a small example of what that looks like.

---

## Technical Progress (Summary)

Current state as of January 2026:
- Diarization working (pyannote.audio)
- Transcription working (MLX Whisper, large model for child speech)
- Alignment working (speaker labels matched to transcript)
- 20 tests passing

What's next:
- Filter hallucinations (repeated phrases at end of transcripts)
- Save transcripts to file
- Begin thinking about what happens after transcripts

See `/journal/` for detailed build log.

---

## Key Phrases and Framings

**On the origin:**
- "Three stories a night. She's steering now."
- "These moments are alive and then they're gone."
- "No phone in that space. I want it dark, calm."

**On the Mahabharata:**
- "My favorite stories—the ones I grew up with. Complicated, morally gray, not easy to tell to a three-year-old. But she's asking."
- "That's the thing I most want to capture."

**On the technical approach:**
- "The capture disappears."
- "I could use an existing service. I'm choosing to understand the pipeline."

**On ethics:**
- "The point isn't to outsource storytelling to machines."
- "The user interface for AI, for a child, is me."
- "Tools that enable rather than extract."

**On career fit:**
- "Both a meaningful tool for my family AND relevant R&D for the space I want to work in."

---

## Outputs Checklist

- [ ] Website page (`/portfolio/tell-me-a-story.html`)
- [ ] LinkedIn post (first announcement)
- [ ] Weekly build log entries
- [ ] LinkedIn posts (ongoing, less frequent than website)
- [ ] Job application talking points

---

## Notes

- Downstream analysis layer (what happens after transcripts) is TBD. Don't pretend there's a plan yet.
- The Mahabharata thread is the heart, but don't lead with it—it emerges from the origin story.
- Voice Bible at `/Users/choksi/dev/voice-bible/Choksi_Voice_Bible.md` for tone/style when drafting.
