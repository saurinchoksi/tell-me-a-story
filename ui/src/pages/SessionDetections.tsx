import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getSessionDetections, scanSession, audioURL } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import type { DetectionFlag, SessionDetectionsData } from '../types';
import './SessionDetections.css';

// Clip window padding (seconds). We play the flagged token's whole containing
// segment so the audio matches the sentence shown on the card and survives
// word-timestamp drift; the pads just give a running start and tail.
const PREROLL = 0.5;
const TAIL = 0.4;
// If a segment is pathologically long (e.g. a broken-Whisper filler run), fall
// back to a tight window around the word so one click can't play 30 seconds.
const MAX_SEGMENT_CLIP = 12;

/** The [start, end] audio window to play for a flag, or null if untimed. */
function clipWindow(flag: DetectionFlag): [number, number] | null {
  const segStart = flag.segment_start;
  const segEnd = flag.segment_end;
  if (segStart != null && segEnd != null && segEnd - segStart <= MAX_SEGMENT_CLIP) {
    return [segStart, segEnd];
  }
  if (flag.start != null && flag.end != null) {
    return [flag.start, flag.end];
  }
  // Long segment but we have a word start: a tight word window beats nothing.
  if (segStart != null && segEnd != null && flag.start != null) {
    return [flag.start, Math.min(flag.start + 4, segEnd)];
  }
  return null;
}

interface FlagCardProps {
  flag: DetectionFlag;
  isPlaying: boolean;
  canPlay: boolean;
  onToggle: () => void;
}

function FlagCard({ flag, isPlaying, canPlay, onToggle }: FlagCardProps) {
  return (
    <div className="detection-flag">
      <div className="detection-flag-top">
        <button
          className="detection-flag-play"
          onClick={onToggle}
          disabled={!canPlay}
          title={canPlay ? 'Play the audio around this flag' : 'No audio for this session'}
          aria-label={isPlaying ? 'Pause' : 'Play the audio around this flag'}
        >
          {isPlaying ? '❚❚' : '▶'}
        </button>
        <span className="detection-flag-token">{flag.token}</span>
        {'cluster_spellings' in flag ? (
          <>
            <span className="detection-flag-type detection-flag-type--inconsistent">
              inconsistent
            </span>
            <span className="detection-flag-cluster">
              spelled: {flag.cluster_spellings.join(' · ')}
            </span>
          </>
        ) : (
          <>
            <span className="detection-flag-arrow" aria-hidden="true">→</span>
            <span className="detection-flag-canonical">{flag.matched_canonicals.join(', ')}</span>
            <span className={`detection-flag-type detection-flag-type--${flag.match_type}`}>
              {flag.match_type}
            </span>
          </>
        )}
        <span className="detection-flag-meta">
          {flag.start != null ? formatTime(flag.start) : '—'}
          {flag.segment_speaker ? ` · ${flag.segment_speaker}` : ''}
        </span>
      </div>
      {flag.segment_text != null ? (
        <p className="detection-flag-text">{flag.segment_text}</p>
      ) : (
        <p className="detection-flag-text detection-flag-text--missing">
          (segment {String(flag.segment_id)} not found in transcript)
        </p>
      )}
      <div className="detection-flag-codes">
        Double Metaphone: {flag.dm_codes.join(' · ')}
      </div>
    </div>
  );
}

// Keyed by session id so navigating A→B remounts the view — a fresh <audio>
// element (the old one unmounts, stopping playback) and clean refs/state,
// instead of leaking the prior session's audio and flags into the next.
export default function SessionDetections() {
  const { id } = useParams<{ id: string }>();
  return <SessionDetectionsView key={id} id={id} />;
}

function SessionDetectionsView({ id }: { id: string | undefined }) {
  const [data, setData] = useState<SessionDetectionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);

  // Shared clip player — one <audio> element drives every flag's play button.
  const audioRef = useRef<HTMLAudioElement>(null);
  const stopAtRef = useRef<number | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    getSessionDetections(id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  function playClip(key: string, start: number, end: number) {
    const audio = audioRef.current;
    if (!audio) return;
    if (playingKey === key) {
      audio.pause(); // toggle off (onPause clears playingKey)
      return;
    }
    stopAtRef.current = end + TAIL;
    audio.currentTime = Math.max(0, start - PREROLL);
    audio.play().then(() => setPlayingKey(key)).catch(() => setPlayingKey(null));
  }

  function handleTimeUpdate() {
    const audio = audioRef.current;
    if (!audio || stopAtRef.current == null) return;
    if (audio.currentTime >= stopAtRef.current) {
      audio.pause(); // auto-stop at the clip's end
    }
  }

  function handleStopped() {
    stopAtRef.current = null;
    setPlayingKey(null);
  }

  async function handleRescan() {
    if (!id) return;
    setScanning(true);
    setError(null);
    try {
      setData(await scanSession(id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setScanning(false);
    }
  }

  if (loading) return <p>Loading detections...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!data || !id) return null;

  const { date, time } = formatSessionDate(id);
  const sections = Object.entries(data.detectors);

  return (
    <div className="session-detections-page">
      <Link to="/monitor" className="session-detections-back">← Monitor</Link>

      {data.has_audio && (
        <audio
          ref={audioRef}
          src={audioURL(id)}
          preload="metadata"
          onTimeUpdate={handleTimeUpdate}
          onPause={handleStopped}
          onEnded={handleStopped}
        />
      )}

      <div className="session-detections-header">
        <div className="session-detections-header-row">
          <h1>{date}</h1>
          <button
            className="session-detections-rescan"
            onClick={handleRescan}
            disabled={scanning}
            title="Re-scan this session (code detectors + the M9b LLM judge)"
          >
            {scanning ? 'Scanning…' : 'Full re-scan'}
          </button>
        </div>
        <p className="session-detections-subtitle">{time} · session {id}</p>
        {data.warning && <p className="session-detections-warning">⚠ {data.warning}</p>}
      </div>

      {sections.length === 0 ? (
        <div className="session-detections-empty">
          <h2>No detections</h2>
          <p>This session hasn't been scanned (or has no transcript). Use Full re-scan.</p>
        </div>
      ) : (
        sections.map(([detId, result]) => (
          <section key={detId} className="detection-section">
            <div className="detection-section-header">
              <span className="detection-section-mode">{result.failure_mode}</span>
              <h2 className="detection-section-label">{result.label}</h2>
              <span className="detection-section-meta">
                {result.n_flags} flag{result.n_flags !== 1 ? 's' : ''} /{' '}
                {result.n_word_tokens} tokens · v{result.detector_version}
                {result.judge_applied ? ' +judge' : ''} · scanned{' '}
                {new Date(result.run_at).toLocaleString()}
                {result.stale && (
                  <span className="detection-stale" title="Transcript changed since this scan — re-scan">
                    {' '}⟳ stale
                  </span>
                )}
              </span>
            </div>
            {result.n_flags === 0 ? (
              <p className="detection-section-clean">No flags — clean scan.</p>
            ) : (
              <div className="detection-flags">
                {result.flags.map((flag, i) => {
                  const key = `${String(flag.segment_id)}-${flag.word_index}-${i}`;
                  const clip = clipWindow(flag);
                  return (
                    <FlagCard
                      key={key}
                      flag={flag}
                      isPlaying={playingKey === key}
                      canPlay={data.has_audio && clip !== null}
                      onToggle={() => clip && playClip(key, clip[0], clip[1])}
                    />
                  );
                })}
              </div>
            )}
          </section>
        ))
      )}
    </div>
  );
}
