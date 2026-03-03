/**
 * WordSpan — a single word within a segment card.
 *
 * Rendered as a <span> with probability coloring, speaker mismatch indicator,
 * click-to-seek, and right-click context menu support. The `.active` class is
 * toggled imperatively by ValidatorPage for 60fps audio sync.
 */

import type { ValidatorWord } from '../types';
import { getWordProbabilityClass, getSpeakerClass } from '../utils/filters';

interface WordSpanProps {
  word: ValidatorWord;
  segmentIndex: number;
  wordIndex: number;
  dominantSpeaker: string | null;
  onSeek: (time: number) => void;
  onContextMenu: (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => void;
}

export default function WordSpan({
  word,
  segmentIndex,
  wordIndex,
  dominantSpeaker,
  onSeek,
  onContextMenu,
}: WordSpanProps) {
  const probClass = getWordProbabilityClass(word.probability);
  const speakerLabel = word._speaker?.label ?? null;
  const mismatch = dominantSpeaker !== null && speakerLabel !== null && speakerLabel !== dominantSpeaker;
  const speakerClass = getSpeakerClass(speakerLabel);
  const hasCorrection = word._corrections && word._corrections.length > 0;

  const tooltip = [
    `prob: ${word.probability.toFixed(3)}`,
    `${word.start.toFixed(2)}s → ${word.end.toFixed(2)}s`,
    `dur: ${(word.end - word.start).toFixed(2)}s`,
    speakerLabel ? `${speakerLabel} (${(word._speaker?.coverage ?? 0).toFixed(2)})` : 'no speaker',
    hasCorrection ? `was: "${word._original}"` : null,
  ].filter(Boolean).join(' | ');

  return (
    <span
      className={[
        'word-span',
        probClass,
        speakerClass,
        mismatch ? 'speaker-mismatch' : '',
        hasCorrection ? 'word-corrected' : '',
      ].filter(Boolean).join(' ')}
      data-segment={segmentIndex}
      data-word={wordIndex}
      data-start={word.start}
      data-end={word.end}
      data-tooltip={tooltip}
      onClick={() => onSeek(word.start)}
      onContextMenu={(e) => onContextMenu(e, segmentIndex, wordIndex)}
    >
      {word.word}
    </span>
  );
}
