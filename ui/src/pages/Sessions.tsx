import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listSessions } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import type { SessionSummary } from '../types';
import SessionNote from '../components/SessionNote';
import ValidationStatus from '../components/ValidationStatus';
import './Sessions.css';

const PIPELINE_STAGES = [
  { key: 'has_audio' as const, label: 'Recorded' },
  { key: 'has_transcript' as const, label: 'Transcribed' },
  { key: 'has_diarization' as const, label: 'Diarized' },
  { key: 'has_embeddings' as const, label: 'Embeddings' },
  { key: 'has_identifications' as const, label: 'Identified' },
];

export default function Sessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading sessions...</p>;
  if (error) return <p className="error">Error: {error}</p>;

  if (sessions.length === 0) {
    return (
      <div className="sessions-page">
        <div className="sessions-empty">
          <h2>No stories yet</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="sessions-page">
      <div className="sessions-header">
        <h1>Story Sessions</h1>
        <p className="sessions-subtitle">
          {sessions.length} session{sessions.length !== 1 ? 's' : ''} recorded
        </p>
      </div>

      <div className="session-list-header">
        <span>Date</span>
        <span>Time</span>
        <span>Length</span>
        <span>Pipeline</span>
        <span>Validation</span>
        <span>Notes</span>
        <span className="session-list-header-actions">Actions</span>
      </div>

      <div className="sessions-list">
        {sessions.map((s) => {
          const { date, time } = formatSessionDate(s.id);

          return (
            <div key={s.id} className="session-row">
              <span className="session-row-day">{date}</span>
              <span className="session-row-time">{time}</span>
              <span className="session-row-length">
                {s.duration_seconds != null ? formatTime(s.duration_seconds) : '—'}
              </span>

              <div className="session-row-pipeline">
                {PIPELINE_STAGES.map((stage) => (
                  <div
                    key={stage.key}
                    className={`pipeline-dot ${s[stage.key] ? 'pipeline-dot--done' : 'pipeline-dot--pending'}`}
                    data-label={stage.label}
                    role="img"
                    aria-label={stage.label}
                  />
                ))}
              </div>

              <ValidationStatus
                sessionId={s.id}
                initialStatus={s.validation_status}
              />

              <span
                className="session-row-notes"
                title={`${s.note_count} validation note${s.note_count === 1 ? '' : 's'}`}
              >
                {s.note_count}
              </span>

              <div className="session-row-actions">
                <Link
                  to={`/sessions/${s.id}/speakers`}
                  className="session-action-primary"
                >
                  Speakers
                </Link>
                {s.has_transcript && (
                  <Link
                    to={`/sessions/${s.id}/validate`}
                    className="session-action-secondary"
                  >
                    Validate
                  </Link>
                )}
              </div>

              <SessionNote sessionId={s.id} initialNote={s.note} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
