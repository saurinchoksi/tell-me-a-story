# Changelog

Newest at top. Only the changes that actually shaped the project, written plainly: what I built and what it does, with a real example where it helps. The full step-by-step record lives in git history.

*Claude Code drafts these from each session's work; Saurin Choksi reviews and approves.*

## 2026-07-02 — The new name-fixer listens instead of guessing

Built the replacement for the name-fixer I removed yesterday, around the opposite idea: instead of guessing what a garbled name should be from its spelling, give Whisper another listen. The pipeline now works out which story world it's in from the names alone, asks a small model for that world's cast, including the group names like "Pandavas" that a kid says constantly, and replays a few seconds of audio around each suspect name with that cast in Whisper's ear. On the same Mahabharata night the old fixer ruined, it turned `Bushma` into "Bhishma" and `fondos` into "Pandavas" right through the war, and my by-ear check caught none of its fixes wrong. The few it isn't sure about wait on a new review screen, like `Bheem`, which means Bhima in one line and Arjuna in another and only I can say which. One tap applies or rejects each, and a name I bless is remembered for every night after. On a made-up story it does nothing at all, since an invented name has no right spelling to fix toward.

## 2026-07-01 — Removed the name-fixer that was renaming the heroes as the enemy

The pipeline had a step that tried to fix mis-heard names, but it ran with no idea which story it was in, so it reached for whatever famous name sounded closest. On a Mahabharata night it heard "Pandavas" as `fondos` and then confidently rewrote that as `Bhishma`, a character on the enemy's side, right through the war. I listened back with my daughter's telling in my ears and it was plainly wrong, so I took the step out, and the transcript now keeps the words Whisper actually heard instead of a confident wrong guess. Dropping it also removed the one heavy model the pipeline still loaded, so it now runs on a single small model that fits an 8GB machine.

## 2026-07-01 — Hold Option to hear just one word

When I click a word in the reviewer it plays from there and keeps going, which is what I want when I'm listening through a stretch. But often I just want to hear one word by itself, to check whether Whisper got it right. So now if I hold Option while I click, it plays only that word and stops clean, even a fast little word like "to" or "it" that used to get clipped off at the end. A plain click still plays on from where I clicked, same as before.

## 2026-06-30 — Re-timed every word so I can click one and hear it

Whisper's word timings were off in a consistent way: it marked a word as starting back in the silence, before anyone actually spoke. The opening "Okay," was tagged at 1.70 seconds, but my daughter doesn't say it until 2.06. So I lined every word back up against the audio, matching the words I already trust to where each one really lands. The same pass brought back real words an earlier cleanup step had been quietly deleting, like the "you believe" missing from "Can you believe that?". Now clicking a word in the reviewer lands right on it, instead of a beat early.

## 2026-06-29 — Checked the canon name-checker on recordings it had never seen

The canon name-checker had only ever been graded on the recordings I built it from, so I ran it on two it had never seen. On a Mahabharata story it caught the misspelled names, like `Bishma` for "Bhishma". On a story set in a world it didn't recognize, the movie K-pop Demon Hunters, it stayed quiet instead of inventing a name. I took the experimental label off it.

## 2026-06-26 — Moved the last detector onto Qwen, so the project runs on one model

The inconsistency judge was the last thing still running on Gemma. I tested whether Qwen could take it over, and once I wrote the prompt the way Qwen reads instructions, it matched Gemma exactly, even on the one case they used to disagree on: a character named Bacchus that the transcriber sometimes wrote as the ordinary word `because`. So the whole pipeline now runs on one local model. Gemma was never really better at this job. It just had a prompt that fit it.

## 2026-06-24 — Overrule wrong name flags

Sometimes a detector flags a name it shouldn't, so I added a way to set it straight. It marked "Jammus" (a train engine my daughter and I made up) as a misspelling of "James" (a real Thomas & Friends character), because the two sound almost identical. They aren't the same character. We invented one that just sounds like a real one. Now I can correct that false flag right on the monitor page, and it stays corrected through every future scan.

## 2026-06-19 — Moved the story tools onto Qwen 3.5 4B

I wanted the story work to run on one small model instead of juggling more than one. So I tested whether Qwen 3.5 4B could take over what Gemma 4 E4B was doing, and it won: it splits a recording into its stories more cleanly and catches more misspelled names. The splitter, the world-recognition step, and the canon name-checker all moved onto Qwen. One job stayed on Gemma for now, the check that spots a made-up name spelled several ways.

## 2026-06-18 — Built the per-story canon name-checker

For each story, it lists the characters of whatever world the story is set in, then flags any the transcriber spelled wrong. In a Mahabharata story it knows the cast includes Bhishma, so when the transcript says `Bishma`, it catches the misspelling.

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

## 2026-05-28 — Built sweeps to find the failures I'd miss by hand

Three read-only passes that hunt for failures I'd never catch just reading the page: speech the transcriber dropped entirely, words stamped at the wrong moment, and stretches where it got stuck repeating one little word (`Right. Right. Right.`). Each one turned up far more cases than I'd find by counting only what made it onto the page.

## 2026-03-03 — Built cross-session speaker ID

Taught the pipeline to recognize who's speaking and carry that across recordings. From a single voice sample of each of us, it correctly picked out me and my daughter on a session it had never heard. When it's unsure (a silly voice, someone far from the mic) it says "suggested" rather than force a confident match.

## 2026-02-18 — Ran the pipeline end to end from one command

Now I can drop a recording in a folder, run one script, and get back a transcript split by who's speaking, with a timestamp on every word.

## 2026-02-07 — Got name correction working on the real recording

Ran the whole pipeline on the real Mahabharata recording for the first time, name correction included, and it fixed the mangled Sanskrit names cleanly. What made it work was handing the model the whole transcript at once instead of one line at a time. Line by line, with no surrounding context, it had invented wrong fixes like turning "dad" into a Sanskrit name.

## 2026-02-07 — Flag Whisper's made-up words for review

Whisper sometimes writes words that were never said. The validation tool flags two kinds for me to review: a word Whisper itself barely believes, where its confidence on that word is near zero, and a word sitting where the speaker detector heard only silence. These just point me at the suspect spots; they don't change the transcript.


## 2026-02-03 — Save the raw transcript, compute the rest on demand

The pipeline keeps three files: the raw transcript with the words as Whisper first heard them, the speaker data, and an enriched transcript with corrections layered on top. The raw one is never overwritten, so if a correction later turns out wrong, I can always see what Whisper actually heard, like the original `fondos` before it was fixed to "Pandavas."

## 2026-01-27 — Built the validation tool

Used Claude Code to whip up a "validation tool" where I can listen to a recording and view the transcript running alongside. I mark where the transcript has errors segment by segment.

## 2026-01-24 — Settled how to handle Whisper's hallucinations

If someone really spoke but Whisper couldn't make out the words, we keep the spot and mark it unintelligible, like my daughter's too-quiet sentence. If Whisper invented words out of pure silence, we delete them.

[show some type of example here to make it concrete and clear]

## 2026-01-22 — The first transcript that reads as a conversation

Combined Whisper's output with speaker detection and got the artifact that feels like a real transcript.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Child speech needs the large Whisper model

Compared Whisper's small, medium, and large models on the same stretch of audio. For catching a quiet, young child's speech I need Whisper large. The other models frequently output nothing even though I can hear her talking.

[show a lil example here tomato it concrete and clear]

## 2026-01-20 — Day one!

I hooked up Whisper and ran my first transcription: a ~5 min bedtime convo with my daughter about the Mahabharata. It mostly worked. Looking at the transcript felt a lil magical, but I noticed mangled Sanskrit names throughout. Like "Yudhishthira" became `you this there`. "Pandavas" became `fondos`. 🤔
