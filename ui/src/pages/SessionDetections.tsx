import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getSessionDetections } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import type { DetectionFlag, SessionDetectionsData } from '../types';
import './SessionDetections.css';

function FlagCard({ flag }: { flag: DetectionFlag }) {
  return (
    <div className="detection-flag">
      <div className="detection-flag-top">
        <span className="detection-flag-token">{flag.token}</span>
        <span className="detection-flag-arrow" aria-hidden="true">→</span>
        <span className="detection-flag-canonical">{flag.matched_canonicals.join(', ')}</span>
        <span className={`detection-flag-type detection-flag-type--${flag.match_type}`}>
          {flag.match_type}
        </span>
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

export default function SessionDetections() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<SessionDetectionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    getSessionDetections(id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <p>Loading detections...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!data || !id) return null;

  const { date, time } = formatSessionDate(id);
  const sections = Object.entries(data.detectors);

  return (
    <div className="session-detections-page">
      <Link to="/monitor" className="session-detections-back">← Monitor</Link>

      <div className="session-detections-header">
        <h1>{date}</h1>
        <p className="session-detections-subtitle">{time} · session {id}</p>
      </div>

      {sections.length === 0 ? (
        <div className="session-detections-empty">
          <h2>No detections</h2>
          <p>This session has no transcript yet — there is nothing to scan.</p>
        </div>
      ) : (
        sections.map(([detId, result]) => (
          <section key={detId} className="detection-section">
            <div className="detection-section-header">
              <span className="detection-section-mode">{result.failure_mode}</span>
              <h2 className="detection-section-label">{result.label}</h2>
              <span className="detection-section-meta">
                {result.n_flags} flag{result.n_flags !== 1 ? 's' : ''} /{' '}
                {result.n_word_tokens} tokens · v{result.detector_version} · scanned{' '}
                {new Date(result.run_at).toLocaleString()}
              </span>
            </div>
            {result.n_flags === 0 ? (
              <p className="detection-section-clean">No flags — clean scan.</p>
            ) : (
              <div className="detection-flags">
                {result.flags.map((flag, i) => (
                  <FlagCard key={`${String(flag.segment_id)}-${flag.word_index}-${i}`} flag={flag} />
                ))}
              </div>
            )}
          </section>
        ))
      )}
    </div>
  );
}
