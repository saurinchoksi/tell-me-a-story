# Changelog

Newest at top. Only the changes that actually shaped the project, written plainly: what I built and what it does, with a real example where it helps. The full step-by-step record lives in git history.

*Claude Code drafts these from each session's work; Saurin Choksi reviews and approves.*

## 2026-06-26 — Moved the last detector onto Qwen, so the project runs on one model

The inconsistency judge was the last thing still running on Gemma. I tested whether Qwen could take it over, and once I wrote the prompt the way Qwen reads instructions, it matched Gemma exactly, even on the one case they used to disagree on: a character named Bacchus that the transcriber sometimes wrote as the ordinary word "because." So the whole pipeline now runs on one local model. Gemma was never really better at this job. It just had a prompt that fit it.

## 2026-06-24 — Overrule wrong name flags

Sometimes a detector flags a name when it shouldn't. For example, the detector marks "Jammus" (a train engine my daughter and I made up) as a misspelling of "James" (a real character from Thomas & Friends), because the two sound almost identical. They aren't the same character; we just invented one that sounds like a real one. I can now correct that false flag right on the monitor page, and it stays corrected through every future scan.

## 2026-06-19 — Moved the story tools onto Qwen 3.5 4B

I wanted the story work to run on one small model instead of juggling more than one. So I tested whether Qwen 3.5 4B could take over what Gemma 4 E4B was doing, and it won: it splits a recording into its stories more cleanly and catches more misspelled names. The splitter, the world-recognition step, and the canon name-checker all moved onto Qwen. One job stayed on Gemma for now, the check that spots a made-up name spelled several ways.

## 2026-06-18 — Built the per-story canon name-checker

For each story it lists the characters of whatever world the story is set in, then flags any the transcriber spelled wrong. In a Mahabharata story it knows the cast includes Bhishma, so when the transcript says "Bishma" it catches the misspelling.

## 2026-06-17 — Got world-recognition working

Built a "is this a known story-world, or just made up?" classifier. For each story it makes one of three calls. It might be from a world the model knows (Thomas & Friends, the Mahabharata, Steven Universe). It might be wholly improvised. Or it might be a mix: a known world with new characters made up on the spot. Knowing the world is what lets the name-checker tell a misspelled real character from an invented one.

## 2026-06-15 — Split each recording into its separate stories

A bedtime recording often holds several stories, and the name checks work best per story, not across a whole session. So I split each recording first. Simple cues help determine where the breaks are: a long silence (three seconds or more), or telltale lines in the transcript like "once upon a time," "the end," or "start the story." Then a small model reads through in order and confirms the real breaks. A last pass fixes over-splitting: if a long pause in the middle of one story (someone got up for milk) made it look like two, it stitches the halves back together when they share the same characters and plot.

## 2026-06-12 — Added a second name detector

It catches a made-up character whose name gets written several different ways in one story, so I can still follow that character through the telling. Plain text-matching catches most. The ones it misses are invented names that are also ordinary words (my daughter has a character named "Bibi"), so those go to a small local model that reads a few lines and decides whether the word is being used as a name.

## 2026-06-11 — Took the name Monitor live

The first detector now runs over every recording and shows its catches on a new Monitor screen. It flags the family's own names when Whisper mangles them, like my daughter's name heard as an ordinary word. The screen lists every flag across all sessions, each with a play button to hear the exact moment. It only points at the transcript, never changes it.

## 2026-06-01 — Tested the family-name detector on unseen recordings

Built the family-name detector and checked it on two recordings it had never seen, against answers I marked by ear. It caught every mis-transcribed name a fresh listen turned up, including spellings of my daughter's name it had never been shown. It raised one false alarm, on an ordinary word that happens to sound like her name.

## 2026-05-28 — Built sweeps to count the failures you miss by hand

Three read-only passes that hunt for failures you'd never see reading the page: speech the transcriber dropped entirely, words stamped at the wrong moment, and stretches where it got stuck repeating one little word ("Right. Right. Right."). Each one surfaced far more cases than counting only what made it onto the page.

## 2026-03-03 — Built cross-session speaker ID

Taught the pipeline to recognize who's speaking and carry that across recordings. From a single voice sample of each of us, it correctly picked out me and my daughter on a session it had never heard. When it's unsure (a silly voice, someone far from the mic) it says "suggested" rather than force a confident match.

## 2026-02-18 — Ran the pipeline end to end from one command

Drop a recording in a folder, run one script, and get back a transcript split by who's speaking, with a timestamp on every word.

## 2026-02-07 — Got name correction working on the real recording

Ran the whole pipeline on the real Mahabharata recording for the first time, name correction included, and it fixed the mangled Sanskrit names cleanly. What made it work was handing the model the whole transcript at once instead of one line at a time. Line by line, with no surrounding context, it had invented wrong fixes like turning "dad" into a Sanskrit name.

## 2026-02-07 — Caught hallucinations where two systems disagree

The tell for a hallucinated word isn't the model's confidence score, it's the shape: it usually sits alone, a one-word segment by itself. The strongest sign is two systems disagreeing, a word the transcriber wrote in a spot where the speaker detector heard only silence. Three simple filters built on those signs catch the made-up words and leave the real quiet ones alone.

## 2026-02-03 — Save the raw transcript, compute the rest on demand

The pipeline keeps three files: the raw transcript exactly as Whisper produced it, the speaker data, and an enriched transcript with corrections layered on top. The raw one is never overwritten, so if a correction later turns out wrong, I can always see what Whisper actually heard, like the original "fondos" before it was fixed to "Pandavas."

## 2026-01-27 — Built the review tool

A player that listens back to a recording with the transcript running alongside. I can mark where it went wrong segment by segment, and hovering a word lights up its audio, so I can hear whether "Yudhishthira" was really said there.

## 2026-01-24 — Settled how to handle Whisper's hallucinations

If someone really spoke but Whisper couldn't make out the words, we keep the spot and mark it unintelligible, like my daughter's too-quiet sentence. If Whisper invented words out of pure silence, we delete them. So the transcript never throws away something that was actually said.

## 2026-01-22 — The first transcript that reads as a conversation

Combined transcription with speaker detection and got the first transcript that reads like an actual back-and-forth, scattered fragments pulled together into real turns. Seeing my daughter remember Duryodhana and Yudhishthira, in print, is the whole point of this.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Child speech needs the large Whisper model

Compared Whisper's small and large models on the same stretch of audio. The small one produced silence where my daughter speaks; the large one caught her full sentence. With the wrong model, ten seconds of her voice simply vanish.

## 2026-01-20 — Day one

First transcription: a 5.6-minute Mahabharata bedtime story through Whisper. It worked, but mangled every Sanskrit name ("Yudhishthira" became "you this there," "Pandavas" became "fondos"). Fixing that became the project's first real problem.
