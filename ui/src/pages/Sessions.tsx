import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listSessions } from '../api/client';
import type { SessionSummary } from '../types';

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
  if (sessions.length === 0) return <p>No sessions found.</p>;

  return (
    <div>
      <h1>Sessions</h1>
      <table className="sessions-table">
        <thead>
          <tr>
            <th>Session</th>
            <th>Audio</th>
            <th>Transcript</th>
            <th>Diarization</th>
            <th>Embeddings</th>
            <th>Identified</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.id}>
              <td>
                <Link to={`/sessions/${s.id}/speakers`}>{s.id}</Link>
              </td>
              <td>{s.has_audio ? 'Y' : '-'}</td>
              <td>{s.has_transcript ? 'Y' : '-'}</td>
              <td>{s.has_diarization ? 'Y' : '-'}</td>
              <td>{s.has_embeddings ? 'Y' : '-'}</td>
              <td>{s.has_identifications ? 'Y' : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
