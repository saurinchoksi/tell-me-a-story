/**
 * SegmentCard — renders a single transcript segment with badges and word spans.
 *
 * Gap segments (_source === 'diarization_gap') get distinct rendering.
 * React.memo with custom comparator keeps this frozen during audio playback —
 * the .active class is applied imperatively by ValidatorPage for 60fps sync.
 */

import { memo, type RefCallback } from 'react';
import type { ValidatorSegment } from '../types';
import { getDominantSpeaker, getSpeakerClass } from '../utils/filters';
import WordSpan from './WordSpan';
import './SegmentCard.css';

interface SegmentCardProps {
  segment: ValidatorSegment;
  index: number;
  isFiltered: boolean;
  filterReasons: string[];
  allFilterReasons: string[];
  hasNotes: boolean;
  speakerNames: Map<string, string>;
  speakerColorMap: Map<string, string>;
  onSeek: (time: number) => void;
  onContextMenu: (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => void;
  cardRef: RefCallback<HTMLDivElement>;
}

function SegmentCardInner({
  segment,
  index,
  isFiltered,
  filterReasons,
  allFilterReasons,
  hasNotes,
  speakerNames,
  speakerColorMap,
  onSeek,
  onContextMenu,
  cardRef,
}: SegmentCardProps) {
  const isGap = segment._source === 'diarization_gap';
  const dominant = getDominantSpeaker(segment);
  const displayName = dominant ? (speakerNames.get(dominant) ?? dominant) : null;
  const isIdentified = dominant !== null && speakerNames.has(dominant);
  const speakerClass = (dominant && speakerColorMap.get(dominant)) ?? getSpeakerClass(dominant);
  const duration = (segment.end - segment.start).toFixed(2);

  return (
    <div
      ref={cardRef}
      className={[
        'segment-card',
        isGap ? 'segment-gap' : '',
        isFiltered ? 'filtered' : '',
        speakerClass,
      ].filter(Boolean).join(' ')}
      data-segment-index={index}
      data-segment-id={segment.id}
    >
      {/* Speaker name — prominent top-left label */}
      {displayName && (
        <div
          className={`segment-speaker-name ${speakerClass} ${isIdentified ? 'speaker-identified' : 'speaker-raw'}`}
          title={isIdentified && dominant ? dominant : undefined}
        >
          {displayName}
        </div>
      )}

      {/* Header row — clickable to seek */}
      <div
        className="segment-header"
        onClick={() => onSeek(segment.start)}
        onContextMenu={(e) => onContextMenu(e, index, -1)}
      >
        <span className="segment-id">
          {isGap ? `Gap ${segment.id}` : `Segment ${segment.id}`}
        </span>
        <span className="segment-time">
          {segment.start.toFixed(2)}s — {segment.end.toFixed(2)}s ({duration}s)
        </span>
      </div>

      {/* Badge row */}
      <div className="segment-badges">
        {!isGap && (
          <>
            <span className="badge badge-temp" title="Temperature">
              T: {segment.temperature.toFixed(2)}
            </span>
            <span className="badge badge-cr" title="Compression Ratio">
              CR: {segment.compression_ratio.toFixed(2)}
            </span>
          </>
        )}
        {hasNotes && <span className="badge badge-note" title="Has notes">Note</span>}
        {allFilterReasons.map((r) => (
          <span
            key={r}
            className={`badge badge-filter ${filterReasons.includes(r) ? 'badge-filter-active' : ''}`}
            title={r}
          >
            {r}
          </span>
        ))}
      </div>

      {/* Content */}
      {isGap ? (
        <div className="segment-gap-body">
          <span className="gap-label">[unintelligible]</span>
          <span className="gap-duration">{duration}s</span>
        </div>
      ) : (
        <div className="segment-words">
          {segment.words.map((w, wi) => (
            <WordSpan
              key={wi}
              word={w}
              segmentIndex={index}
              wordIndex={wi}
              dominantSpeaker={dominant}
              onSeek={onSeek}
              onContextMenu={onContextMenu}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function areEqual(prev: SegmentCardProps, next: SegmentCardProps): boolean {
  return (
    prev.segment.id === next.segment.id &&
    prev.isFiltered === next.isFiltered &&
    prev.hasNotes === next.hasNotes &&
    prev.filterReasons.length === next.filterReasons.length &&
    prev.filterReasons.every((r, i) => r === next.filterReasons[i]) &&
    prev.allFilterReasons.length === next.allFilterReasons.length &&
    prev.allFilterReasons.every((r, i) => r === next.allFilterReasons[i]) &&
    prev.speakerNames === next.speakerNames &&
    prev.speakerColorMap === next.speakerColorMap
  );
}

const SegmentCard = memo(SegmentCardInner, areEqual);
export default SegmentCard;
