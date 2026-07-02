import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getNameCorrections,
  blessNameCorrection,
  rejectNameCorrection,
} from '../api/client';
import { formatTime } from '../utils/time';
import type { NameCorrectionsRollup, NameCorrectionItem } from '../types';
import './NameReview.css';

/**
 * The name-review queue — the human half of the namefix stage.
 *
 * The pipeline auto-applies only bulletproof audio-verified fixes; everything it wasn't
 * sure about lands here, grouped by world and then by NAME across the whole batch (one
 * decision covers every session that heard the same thing). Bless = apply everywhere +
 * remember in the world's dictionary forever; Reject = drop it, transcript untouched.
 */
export default function NameReview() {
  const [rollup, setRollup] = useState<NameCorrectionsRollup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null); // heard_cleaned being acted on
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  function refresh() {
    getNameCorrections()
      .then(setRollup)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }
  useEffect(refresh, []);

  async function actOnAll(
    item: NameCorrectionItem,
    act: (sessionId: string, heard: string, canonical?: string) => Promise<unknown>,
    canonical?: string,
  ) {
    setBusy(item.heard_cleaned);
    setError(null);
    try {
      // one human decision covers every session holding this name
      for (const s of item.sessions) {
        await act(s.session_id, item.heard_cleaned, canonical);
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  if (loading) return <p>Loading name queue...</p>;
  if (error) return <p className="name-review-error">{error}</p>;
  if (!rollup || rollup.n_pending_groups === 0) {
    return (
      <div className="name-review">
        <h1>Name review</h1>
        <p className="name-review-empty">
          Nothing waiting — every suspected name is either auto-fixed or resolved.
        </p>
      </div>
    );
  }

  return (
    <div className="name-review">
      <h1>Name review</h1>
      <p className="name-review-sub">
        The pipeline wasn&apos;t sure about these. Bless applies the fix everywhere and the
        world&apos;s dictionary remembers it; Reject leaves the transcript as heard.
      </p>
      {rollup.worlds.map((w) => (
        <section key={w.world} className="name-review-world">
          <h2>{w.world}</h2>
          {w.names.map((item) => {
            const key = item.heard_cleaned;
            const nOcc = item.sessions.reduce((n, s) => n + s.occurrences.length, 0);
            return (
              <div key={key} className="name-card">
                <div className="name-card-head">
                  <span className="name-heard">{item.heard}</span>
                  <span className="name-arrow">→</span>
                  <input
                    className="name-canonical"
                    value={overrides[key] ?? item.canonical}
                    onChange={(e) =>
                      setOverrides({ ...overrides, [key]: e.target.value })
                    }
                    aria-label="Correct spelling"
                  />
                  <span className="name-count">
                    {nOcc} spot{nOcc === 1 ? '' : 's'} · {item.sessions.length} session
                    {item.sessions.length === 1 ? '' : 's'}
                  </span>
                  <div className="name-actions">
                    <button
                      className="name-bless"
                      disabled={busy === key}
                      onClick={() =>
                        actOnAll(item, blessNameCorrection, overrides[key] ?? item.canonical)
                      }
                    >
                      ✓ Bless
                    </button>
                    <button
                      className="name-reject"
                      disabled={busy === key}
                      onClick={() => actOnAll(item, rejectNameCorrection)}
                    >
                      ✕ Reject
                    </button>
                  </div>
                </div>
                <ul className="name-occurrences">
                  {item.sessions.map((s) =>
                    s.occurrences.map((o, i) => (
                      <li key={`${s.session_id}-${o.segment_id}-${o.word_index}-${i}`}>
                        <Link to={`/sessions/${s.session_id}/detections`}>
                          {s.session_id}
                        </Link>{' '}
                        · seg {o.segment_id} · {formatTime(o.start)} ·{' '}
                        <span className="name-token">“{o.token}”</span>
                      </li>
                    )),
                  )}
                </ul>
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}
