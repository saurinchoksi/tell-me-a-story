# Changelog

Structured record of what changed, what was decided, and what was learned. Newest entries at top. Source material for public build log entries.

Format: **What** (what changed), **Result** (concrete outcome with numbers when available), **Decided** (decisions made and why), **Learned** (insights, principles, surprises). Not all fields required every entry.

## 2026-02-20 — README, test coverage, and removing friction

**What:** README got sample output (before/after showing raw Whisper → enriched transcript), Python version, MIT license, portfolio link. Entry points (`init_session.py`, `process_inbox.py`) got 28 tests. Validator note saves went from sluggish to instant via event delegation. `inbox/` moved to project root (separate from `sessions/` output). Changelog page converted to dynamic shell fetching from GitHub. SYNC workflow simplified to intent/context/result format.
**Result:** 150 fast tests passing. Validator re-renders no longer reattach handlers. Changelog page: 626→270 lines, single source of truth. README now shows the pipeline's value in six lines of before/after.
**Decided:** No framework for the validator — event delegation solves the perf problem without adding a build toolchain. Python version states what we know (3.14) rather than claiming untested compatibility. SYNC tasks communicate what and why; Code decides how.
**Learned:** The most compelling README content isn't architecture descriptions — it's showing "fondos" becoming "Pandavas" with speaker labels. Infrastructure for problems you don't have yet (React, file watchers) costs more than the friction it would remove.

## 2026-02-19 — Second real session and what it revealed

**What:** Processed session 20260218-185123 — an original moon story (not Mahabharata), ~10 min, multiple people in the room. First real-world session through the full pipeline.
**Result:** 190 segments, 4 speakers detected (should be 2), 37 NONE segments, 0 LLM/dictionary corrections (no Sanskrit content). Hallucinated text found over clean audio — a new category the current filters don't catch.
**Decided:** Benchmark against cloud services (AssemblyAI, Deepgram, OpenAI Whisper API) — two files, three services, one experiment. Kill the build log; replace with a visual portfolio page that shows the system working, not describes the journey. Changelog stays as the "go deeper" link.
**Learned:** Controlled test sessions don't predict real-world behavior. Household audio with overlapping voices, interruptions, and multiple speakers is a fundamentally harder problem. The pipeline captured a family moment faithfully — messy parts are solvable engineering, the captured moment isn't reproducible.

## 2026-02-18 — Gaps, inbox, and the pipeline running end to end

**What:** Unintelligible speech gap detection finds moments where diarization sees a speaker but Whisper produces nothing. Validator renders them as `[unintelligible]` cards. `process_inbox.py` is the single entry point — drop audio, run one command. Hit GPU OOM (pyannote + MLX-LM competing for Metal memory), fixed by spawning MLX-LM in a subprocess.
**Result:** 3 gaps detected in session 000 at correct timestamps. Session 20260207-172315 processed via inbox — identical output to manual run (zero text differences, same corrections, same gaps). 122 tests passing, 9 src files.
**Decided:** (1) Hallucination marking goes in `transcript-rich.json` as enrichment, not in validation-notes.json. (2) Pipeline needs Mahabharata layer separated from core — dictionary normalization and LLM prompt are content-specific, pipeline should be content-agnostic. (3) Process more diverse sessions before building editing mechanisms.
**Learned:** Speaker-transition filter for gaps was 6/6 accurate — gaps matching neighbors are pauses, gaps differing from neighbors are interjections. VAD is redundant (diarization runs it internally). Pyannote's MPS allocations survive `gc.collect()` + `torch.mps.empty_cache()` in the same process — subprocess isolation is the real fix. Pipeline is fully deterministic given same hardware: greedy decoding at every stage means identical input → identical output.

## 2026-02-18 — Diarization becomes visible: speaker indicators in validator

**What:** Added speaker visualization to transcript validator (Code delivered core, Desktop fixed bugs + added always-visible filter badges and badge layout). Also decoupled filter badge display from filter toggle state.
**Result:** Segment cards show speaker-colored left borders (blue/berry), speaker badges, wavy underlines on word-level mismatches, and extended tooltips with speaker + coverage. Filter badges (silence gap, near-zero, duplicate) now always visible — toggles only control dimming.
**Decided:** (1) Full-opacity speaker colors on default border state — 15% opacity invisible on 3px. (2) Always-visible filter badges — you should see what a segment *is* before deciding to filter it. (3) Two-row badge layout — speaker always rightmost top, filter badges right-aligned underneath.
**Learned:** CSS cascade from parent card class drives wavy underline color with zero per-word JS — elegant and extensible. Code's `getSpeakerClass()` returning `speaker-00` meant badge class `badge-speaker-${speakerClass}` doubled the prefix. Visual verification caught 3 bugs that unit tests wouldn't — border opacity, class naming, badge visibility are all rendering concerns.

## 2026-02-18 — Pipeline becomes fully self-contained: Ollama out, MLX-LM in

**What:** Replaced Ollama (external inference server) with mlx-lm (Python library) for LLM normalization. Evaluated 4 model variants × 2 sessions in `experiments/mlx_lm_eval/`, then swapped production code.
**Result:** `mlx-community/Qwen3-8B-8bit` with `/no_think` — same 5 corrections per session as Ollama, 15-24s inference, 8300MB load → 0MB unload. All 95 tests pass. Every pipeline stage is now a Python library: import, call, done.
**Decided:** (1) 8-bit over 4-bit — comparable quality but slightly better on ambiguous Sanskrit phonetics, memory fits fine running sequentially on M1 16GB. (2) No-think over thinking mode — thinking caused a 186s infinite loop on session 000003 ("Wait, the user says 'Dhrashtra'... Wait..."), zero output. Chain-of-thought adds nothing to mechanical name correction. (3) Eval before swap — experiments folder, no production code touched until data confirmed the decision.
**Learned:** Ollama's client-server architecture was solving a problem the pipeline doesn't have (multi-user inference). The mismatch showed up as a silent failure mode — the kind of thing calm technology can't tolerate. MLX-LM treats the model the same way the pipeline treats Whisper and pyannote: load, use, unload. Also: Qwen3 emits empty `<think></think>` tags even when told not to think. And raw prompt injection made the model continue the bedtime story instead of analyzing it — chat template is mandatory.

## 2026-02-17 — 12 files become 8, and modules own their identity

**What:** Wave 4 completed: file consolidation (12 → 8 src files) + targeted fixes. README rewritten.
**Result:** `enrichment.py` folded into `diarize.py`, `enrich.py` into `pipeline.py`, `inspect_audio.py` and `query.py` inlined. Each module now owns its model constant and returns its own processing entry. Ollama switched from subprocess to REST API. `_schema_version` killed. LLM response parsing hardened. Session ID validation added. 95 tests.
**Decided:** (1) Module that calls the model owns the identifier — no model strings cross boundaries. (2) Kill `_schema_version` entirely — `_processing` already describes what's in the file, no backwards compatibility contract exists. Code initially kept it; pushed back. (3) README separates Mahabharata as content domain from pipeline mechanics.
**Learned:** "Where should this constant live?" → follow it to its only real consumer. If the consumer has to rename your export to understand it, the name isn't working. If an orchestrator is importing model strings just to stamp metadata, the responsibility is in the wrong place.

## 2026-02-17 — Three artifacts, not five: session folder redesign

**What:** Completed Waves 2-3 of codebase cleanup (renames + small refactors), then worked through all four design questions from the architecture walkthrough. Three decided, one deferred.
**Result:** Session folder simplified from 5 files to 3 artifacts + audio. Tests: 116 → 85 across all waves. `transcript-raw.json` (immutable Whisper output), `diarization.json` (unchanged), `transcript-rich.json` (enriched with corrections, speakers, audio info). `manifest.json`, `audio-info.json` eliminated. `strip_enrichments()`, `create_manifest()`, enrich.py CLI all deleted. `--re-enrich` flag added to pipeline.py. SYNC fully processed.
**Decided:** (1) Save raw transcript separately — never destroy the honest Whisper record. (2) Kill manifest — fold audio hash into `_processing`, one provenance location. (3) Kill enrich.py CLI — one entry point via pipeline.py. (4) Content separation deferred — Mahabharata defaults are overridable params, extract when second domain appears.
**Learned:** "Rich transcript" is actual speech processing terminology (NIST Rich Transcription evaluations) — exactly describes combining ASR + diarization + metadata into one artifact.

## 2026-02-10 — Sub-agent removal and codebase hygiene

**What:** Full codebase review flagged sub-agent architecture (code-reviewer, coder-agent, go.md orchestrator) as solving a problem the project doesn't have. Deleted agent files and orchestration commands. Removed "Use subagents liberally" from SYNC.md. Also removed dead code from `inspect_audio.py`, updated CLAUDE.md to reflect current architecture, rebuilt contaminated venv.
**Result:** 116 tests passing on clean venv. CLAUDE.md now accurately documents enrichment pipeline, query layer, filters. `requirements-lock.txt` created.
**Decided:** Sub-agents don't fit a 14-file project — code review in the same session that wrote the code has full context. Experiments get their own venv to prevent dependency contamination. Before adopting any new practice, ask "does my project actually have this problem?"
**Learned:** Experiment dependencies (clearvoice, opencv) silently upgraded torchaudio to 2.9.1, breaking speechbrain's `list_audio_backends()`. Unpinned `requirements.txt` allowed it. Perfectionism tendency: if a practice sounds smart and aligns with mental model, default is to implement at full fidelity regardless of fit.

---

## 2026-02-09 — Workflow Revision for Opus 4.6

**What:** Audited entire workflow against new Opus 4.6 capabilities (adaptive thinking, compaction, project memory, past chats search). Migrated 29 journal files to single changelog. Rewrote project instructions — identity sections aligned with website, operational sections stripped to essentials.
**Result:** 103KB journals → 19KB changelog. 14 reference files → 9. Project instructions ~40% shorter. Triggers removed except !CLOSE.
**Decided:** Journals are dead — changelog is the single historical record. Project instructions are the contract for guaranteed behavior; memory handles accumulated knowledge; CURRENT.md tracks live state. Voice/Visual Bibles dropped from project scope (they're personal references, not project-specific).
**Learned:** Three-layer model for Claude reliability: instructions (guaranteed, loaded every time) → memory (synthesized, approximate) → files on disk (must be explicitly read). Encoding specific behaviors in memory is unreliable; encode them in instructions.

---

## 2026-02-08c — Autonomous Code Exploration for Quiet Speech

**What:** Three Desktop-driven experiments failed to recover quiet child speech (gain boost, spectral subtraction, Whisper parameter sweep). Pivoted to giving Claude Code full autonomy for open-ended experimentation.
**Result:** Baseline SNR is 30.3dB — the voice is genuinely quiet, not noise-masked. Spectral subtraction actually decreased SNR (Δ-0.6dB). Whisper parameter sweep was completely flat — identical results across every variant. The model isn't on a decision boundary; it confidently detects no speech regardless of decode parameters.
**Decided:** New SYNC task template for autonomous work: tight context + loose approach + "when to stop" = idea exhaustion, not step counts. Branch isolation mandatory. Opus for creative tasks, Sonnet for well-specified execution.
**Learned:** Whisper has a hard floor, not a soft boundary. The quality of the handoff determines the quality of the autonomy — tight context + loose approach = good autonomous work.

## 2026-02-08b — Quiet Speech Recovery: Gain + initial_prompt Trap

**What:** Built experiment to recover Arti's speech at 241.68–242.83s where diarization detected her but Whisper produced nothing. Tested 7 audio variants × 2 prompt strategies.
**Result:** None recovered the speech. Gain/filtering can't create information that isn't there. But discovered that narrative text in `initial_prompt` causes Whisper to skip matching audio — a different failure mode than the earlier spelling issue.
**Decided:** Vocabulary-only for initial_prompt, never narrative text. Accept unrecoverable speech — mark honestly rather than pretend it doesn't exist.
**Learned:** Diarization detection ≠ transcription recovery. Pyannote's task (detect voice activity) is fundamentally easier than Whisper's (decode words). Mic placement matters more than post-processing.

## 2026-02-08 — Re-enrich Script + Architecture Refactor

**What:** Built `src/enrich.py` to run enrichment stages on existing sessions without re-transcribing from audio. Refactored enrichment logic into shared function used by both pipeline.py (new sessions) and enrich.py (existing sessions).
**Result:** Session 000001 enriched: 18 LLM corrections, 16 dictionary corrections, speaker data on every word, schema 1.2.0. 116 tests passing.
**Decided:** Don't redo expensive compute (transcription, diarization) when artifacts already exist. Shared function, not duplicated code. Caller assembles metadata — different callers have different needs for `_processing`.

## 2026-02-07f — Hallucination Filters: Three Patterns, Zero False Positives

**What:** Analyzed all low-probability words across both sessions with diarization coverage data. Built three targeted filters replacing the blunt min_probability approach.
**Result:** Silence gap filter (null speaker + zero coverage + single-word segment): 3 caught, 0 false positives. Near-zero probability (prob < 0.01 + single-word segment): catches edge cases silence gap misses. Duplicate segments: review aid for seek-boundary artifacts.
**Decided:** Single-word segment is the critical structural discriminator, not probability thresholds. Silence gap and near-zero are eventual production defaults. Duplicates for human review only.
**Learned:** The insight came from noticing seg 2 "Why" (null speaker, 0.0 coverage) differs from hallucinations because it lives in an 11-word segment where 10/11 are above 0.99.

## 2026-02-07d — Hallucination Ground Truth + Diarization Enrichment

**What:** Walked through every word below probability 0.5 in session 000001. 24 words flagged. Choksi provided ground truth for each. Then merged diarization into the transcript as an enrichment stage.
**Result:** 4 confirmed hallucinations, 20 real words with low probability. No probability threshold cleanly separates them. But 3/4 hallucinations fall in diarization silence gaps — two independent systems disagreeing is the real signal.
**Decided:** Diarization coverage is the primary hallucination signal, not word probability alone. Word-level speaker enrichment, not segment-level — a single Whisper segment can contain two speakers. diarization.json stays as raw artifact; transcript becomes the single read point.
**Learned:** Two consecutive "Well."s — real (prob 0.993) then hallucinated (prob 0.133) — confirm coverage alone can't catch everything either. Need low prob + coverage context combined.

## 2026-02-07b — Full Pipeline Validation Run

**What:** First full end-to-end pipeline run with all normalization stages on real audio. Created session 000001 to preserve pre-normalization session for comparison.
**Result:** LLM: 18 corrections. Dictionary: 15 corrections (up from 12 — possessives fix working). Zero false positives. Schema 1.1.0 with `_processing` metadata.

## 2026-02-06d — Normalization on Real Audio: Punctuation Bug

**What:** Ran LLM + dictionary normalization against the actual Mahabharata recording for the first time. Built step-through validation script.
**Result:** LLM: 18 corrections (fondos→Pandavas ×7, goros→Kauravas ×4, Yudister→Yudhishthira ×3, Fondo→Pandu ×3, Dhrashtra→Dhritarashtra ×1). Dictionary: 12 corrections (Duryodhan→Duryodhana ×8, Yudhisthir→Yudhishthira ×4). Two passes split perfectly with zero overlap. Zero false positives.
**Decided:** Fix punctuation stripping before continuing validation — the fix was straightforward and seeing real correction counts was more valuable than noting it for later.
**Learned:** Whisper attaches trailing punctuation to words (`"Dhrashtra,"`) which blocks dictionary lookup. `_corrections` chain tracks bare names, not punctuated forms.

## 2026-02-06c — Inline Corrections Architecture

**What:** Designed full normalization pipeline architecture through iterative dialogue. Started with separate normalization.json artifact, ended with inline corrections on the transcript itself.
**Result:** Built Mahabharata reference library (56 entries), dictionary module, pipeline integration with inline `_corrections` chain and `_processing` metadata. 86 tests passing.
**Decided:** Transcript is a living document enriched through pipeline stages. Each stage adds information, never destroys. `_original` = Whisper output, always. Both LLM and dictionary corrections applied inline. Key distinction: variants (misspellings → correct) vs aliases (legitimate alternate names → keep).
**Learned:** The architecture conversation was the most valuable part of the session. Each "what if?" question revealed something the original design didn't handle. "What if I want human corrections?" → same pattern, different `_corrected_by`. The final architecture is simpler and more extensible than what we started with.

## 2026-02-06 — Full Transcript LLM Beats Segment-by-Segment

**What:** Compared segment-by-segment vs full-transcript LLM normalization for Sanskrit name correction.
**Result:** Full transcript: 5/5 recall, 0 false positives, single LLM call. Segment-by-segment: 4/5 recall, 3 false positives (hallucinated English→Sanskrit mappings like "dad"→"Pandu", "best"→"Bhishma").
**Decided:** Full transcript wins definitively. No hybrid approach needed. Two-pass architecture: LLM for phonetic mishearings (context-dependent), dictionary for variant spellings (universal).
**Learned:** Third category of Whisper output beyond "fabricated" and "unintelligible": "partially decoded but wrong" — real speech where Whisper produced something but got the word wrong. This is what normalization is for.

## 2026-02-05 — Segment-Level Filtering + Sanskrit Name Experiment

**What:** Wired probability filter into validator. First attempt used word-level filtering which stripped valid low-confidence words from good segments. Fixed to segment-level inclusion. Also ran initial_prompt experiment for Sanskrit names.
**Result:** The unit of filtering matters. Hallucinations are segment-shaped problems, not word-shaped problems. Initial_prompt vocab list gave phonetic accuracy ("Pandus"), natural sentence biased toward canonical forms ("Pandava"). Both "lie" about what was spoken.
**Decided:** Post-processing normalization, not prompt engineering. Preserve honest transcripts, add canonical field. Aligns with project principle: honest transcripts matter more than clean ones.

## 2026-02-04 — Hallucination Filters Are Useless, Then Reverted

**What:** Cross-referenced validation notes against segment metadata. Temperature and compression_ratio catch 0/5 hallucinations. Word probability catches 3/5. Built `_min_word_prob` enrichment, then reverted it same day.
**Result:** temperature filter: 0% catch rate. compression_ratio: 0%. min_word_prob < 0.5: 60%. But the existing word-level min_probability() filter in filters.py already handles the same cases at query time.
**Decided:** Reverted `_min_word_prob` enrichment. When word-level filtering already solves the problem, segment-level enrichment is unnecessary. Also removed temp/compression analysis from deferred tasks — dead end.
**Learned:** Check if existing tools solve the problem before building new infrastructure.

## 2026-02-03 — Architecture Clarification: Artifacts vs Queries

**What:** Deep dive with "seasoned pipeline architect" framing. Stripped pipeline to three artifacts (audio-info.json, transcript.json, diarization.json). Removed intermediate files.
**Result:** Removed 04-words.json, 05-words-labeled.json, 06-utterances.json. All derivable at query time. `extract_words()` was unnecessary — reshaping data to fit code instead of writing code that traverses data as-is. File naming simplified to `audio.m4a`, `transcript.json`, `diarization.json`.
**Decided:** Filters are query-time predicates, not pipeline stages. Exception: zero-duration words (provably useless, bake into transcribe). Expert advice: "You're asking the right question at the wrong time. Validate transcription first."
**Learned:** "Raw artifacts, filtered queries" — preserve complete data at pipeline stages, apply filters at query time. Enables experimentation without losing source material.

## 2026-02-01 — Course Correction: Wrong Whisper Model

**What:** Previous journal entries contained conclusions based on testing with wrong Whisper model. The 189.21s threshold finding is invalid.
**Result:** With correct large model: no speech dropping at any audio length. Hallucination quality varies with clip length, but the hard threshold doesn't exist.
**Learned:** Always verify which model is running. Conclusions from wrong model testing are worse than no conclusions — they create false confidence.

## 2026-01-31 — Whisper Audio Length Threshold (Later Invalidated)

**What:** Binary searched exact audio length threshold where Whisper's transcription changes. Found 189.21s (3:09.21), validated with 12 trials at 100% determinism.
**Result:** ⚠️ Partially invalidated (Feb 1 — wrong model). Core insight survived: Whisper needs ~3+ minutes of context not to avoid dropping speech, but to get words right for quiet speakers. Short clips produce worse quality than full recordings.
**Learned:** The investigation methodology was sound (binary search, automated script, validation trials). The model configuration was not.

## 2026-01-29 — Heuristics Dead End, Pivot to LLM

**What:** Built gap-filling heuristics for speaker assignment. Raised threshold from 0.5s to 1.25s to catch more gaps. Smelled wrong. Reverted everything.
**Result:** Time thresholds can't distinguish "uh-huh" (listener backchannel → different speaker) from "and the Pandavas" (sentence continuation → same speaker). Only semantic context can.
**Decided:** Revert all gap-filling code. Pivot to LLM post-processing. Reference: DiarizationLM (Google, 2024) achieved 55% WDER reduction using semantic context.
**Learned:** "Use intelligence, not heuristics." The temptation with ML pipelines is to stack heuristics. Each one feels like progress. But they're brittle, arbitrary, and don't generalize.

## 2026-01-27 — Validation Player Spec + Orchestration Research

**What:** Designed validation player tool. Researched Claude Code orchestration patterns (sub-agents, Ralph Wiggum pattern, custom agents).
**Result:** Full spec with Flask server, wavesurfer.js waveform, word-level highlighting, keyboard shortcuts, notes persistence. Recommendation: minimal orchestration for this tool, save custom sub-agents for bigger systems.

## 2026-01-26 — Pipeline Testing on Real Recordings

**What:** Tested pipeline on two new real recordings (5.6 min + 4.8 min bedtime story conversations).
**Result:** Speaker separation worked well. Arti's quiet voice captured in many places. [unintelligible] markers appeared where speech faded. Sanskrit name problems confirmed across multiple recordings: Pandavas→"Fondos"/"Bondos"/"Pondos", Kauravas→"Goros"/"Koros", Dhritarashtra→"The Trasht".

## 2026-01-25 — JSON Output Complete: Audio In → Structured JSON Out

**What:** Built `save_session()` with schema v0.1.0. Session now persists to structured JSON with word-level timestamps.
**Result:** 43 tests passing. Pipeline complete: audio file in → structured JSON out with speaker-labeled utterances and word-level data.
**Decided:** `stories` array (future multi-story splitting), `moments` placeholder (non-story fragments), `processing` placeholder (stats). "Capture generously, build features sparingly." Replay-ability, cost of retrieval, regret heuristic.
**Learned:** Schema design principles — core (stories, utterances, words), forensics (processing stats, versions), future (extracted elements, speaker names). Word-level timestamps enable future caption sync (audio plays, words highlight like Apple Music lyrics).

## 2026-01-24 — Hallucination Filtering: Two Layers, Different Purposes

**What:** Deep dive into hallucination detection. Word-level signals insufficient (some hallucinations have probability 0.85+). Moved to segment-level signals.
**Result:** Two-layer system. Segment filter: temperature == 1.0 OR compression_ratio > 2.5 → replace with [unintelligible]. Word filters: zero-duration, low-probability → remove entirely.
**Decided:** [unintelligible] = real speech we can't decode → preserve. Filtered words = fabricated content → delete. "Honest transcripts, not clean transcripts." Post-filtering over VAD prevention (Silero VAD only 61% accurate, risks losing Arti's quiet moments).
**Learned:** Build vs buy reflection — wrote custom rather than using WhisperX/faster-whisper. First time: build yourself, learn the domain. Second time: use the library. The value isn't the code, it's the judgment.

## 2026-01-23 — Leading Fragments + Model Clarity

**What:** Investigated 7 remaining UNKNOWNs. Found pattern: 5 are turn-starts where diarization missed the boundary — fragments right before the next speaker's utterance.
**Result:** `assign_leading_fragments` reduces UNKNOWNs from 75 → 1 (down from 75 → 7 after merge). Time-based threshold (≤0.5s gap to next utterance). Last remaining UNKNOWN is a contextual hallucination — Whisper heard quiet speech and invented plausible words.
**Decided:** Two types of hallucinations: random repetition ("He knows. He knows.") and contextual (quiet speech → model guesses). The second is sneakier because it sounds right.
**Learned:** Pyannote = speaker diarization (voice characteristics). Whisper = speech recognition (words). Embeddings = voice as a point in mathematical space, similar voices cluster together.

## 2026-01-22 — Alignment Pipeline: First Speaker-Labeled Transcript

**What:** Built alignment pipeline combining transcription with diarization. Word midpoint matching for speaker assignment. Handled diarization gaps and utterance fragmentation.
**Result:** From 75 UNKNOWNs to 7. From 159 fragmented utterances to 23 consolidated ones. Output reads as actual conversation.
**Learned:** "Arti remembering Duryodhana and Yudhishthira. This is what the project is for." Also: the 2026 builder skill is knowing when to go deep and when to let the machine run. Understanding trade-offs matters more than typing speed.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Diarization Working + Model Size Matters for Child Speech

**What:** Got pyannote.audio speaker diarization running. Compared Whisper tiny vs large model on the same audio window (5-25 seconds).
**Result:** Tiny model: absolute silence where Arti speaks (0 words). Large model: "Dad, why do the Fondos and the Goros want to be king?" (12 words). Ten seconds of a child's voice, gone with the wrong model. Large model also captured Arti answering questions throughout: "Durioden", "Yudhishthira", "The which father?", "I forgot."
**Decided:** Use large model (whisper-large-v3-turbo) as baseline. Explore smaller models + preprocessing later.
**Learned:** The whole point of this project is capturing stories told to Arti. If the transcription can't hear her voice, we lose half the conversation — the half that matters most.

## 2026-01-20 — Day 1: Project Started

**What:** Created project structure, first script (inspect_audio.py), first transcription (MLX Whisper on 5.6 min Mahabharata bedtime story). Established build principles, testing approach, progression plan for Claude collaboration.
**Result:** First transcript saved. 7 tests passing. Whisper phonetically guesses Sanskrit names: "Yudhishthira" → "you this there", "Duryodhana" → "D'You're older", "Pandavas" → "fondos/bondos". The story bible has begun.
**Decided:** Backend first (if can't process audio, nothing else matters). Small steps. Write code with Claude Desktop guiding — understanding before automation. The arc: from writing code myself to orchestrating a team of AI agents.
**Learned:** Separate concerns early. A function that returns data is testable. A function that prints is not.
