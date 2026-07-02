import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useRef } from 'react';
import {
  getNameCorrections,
  blessNameCorrection,
  rejectNameCorrection,
  audioURL,
} from '../api/client';
import type { NameCorrectionOccurrence } from '../types';
import { formatTime } from '../utils/time';
import type { NameCorrectionsRollup, NameCorrectionItem } from '../types';
import './NameReview.css';

/** The occurrence's sentence with the judged word highlighted — several names can fly
 *  by in one span, and this is what tells the reviewer which word the verdict is about. */
function OccurrenceSentence({ o }: { o: NameCorrectionOccurrence }) {
  if (!o.segment_text || o.word_offset == null || o.word_len == null) return null;
  const before = o.segment_text.slice(0, o.word_offset);
  const word = o.segment_text.slice(o.word_offset, o.word_offset + o.word_len);
  const after = o.segment_text.slice(o.word_offset + o.word_len);
  return (
    <div className="name-occ-sentence">
      “{before}
      <mark className="name-occ-target">{word}</mark>
      {after}”
    </div>
  );
}

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
  // occurrence keys the user un-checked (same-sound-two-referents: bless only some spots)
  const [excluded, setExcluded] = useState<Record<string, boolean>>({});

  const occKey = (sid: string, o: NameCorrectionOccurrence) =>
    `${sid}|${o.segment_id}|${o.word_index}`;

  // Shared clip player (the SessionDetections pattern): one <audio> element; the source
  // swaps to whichever session the clicked occurrence belongs to, seeks just before the
  // word, and auto-stops a few seconds later.
  const PREROLL = 1.5;
  const WINDOW = 2.5;
  const audioRef = useRef<HTMLAudioElement>(null);
  const srcSessionRef = useRef<string | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  function playClip(sessionId: string, key: string, start: number) {
    const audio = audioRef.current;
    if (!audio) return;
    if (playingKey === key) {
      audio.pause();
      return;
    }
    if (srcSessionRef.current !== sessionId) {
      audio.src = audioURL(sessionId);
      srcSessionRef.current = sessionId;
    }
    stopAtRef.current = start + WINDOW;
    audio.currentTime = Math.max(0, start - PREROLL);
    audio.play().then(() => setPlayingKey(key)).catch(() => setPlayingKey(null));
  }

  function handleTimeUpdate() {
    const audio = audioRef.current;
    if (!audio || stopAtRef.current == null) return;
    if (audio.currentTime >= stopAtRef.current) audio.pause();
  }

  function handleStopped() {
    stopAtRef.current = null;
    setPlayingKey(null);
  }

  function refresh() {
    getNameCorrections()
      .then(setRollup)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }
  useEffect(refresh, []);

  async function actOnAll(
    item: NameCorrectionItem,
    act: (
      sessionId: string,
      heard: string,
      canonical?: string,
      occurrences?: { segment_id: number | string; word_index: number }[],
    ) => Promise<unknown>,
    canonical?: string,
  ) {
    setBusy(item.heard_cleaned);
    setError(null);
    try {
      // one human decision covers every session holding this name; un-checked spots
      // stay pending (partial bless — the API re-queues the remainder)
      for (const s of item.sessions) {
        const kept = s.occurrences.filter((o) => !excluded[occKey(s.session_id, o)]);
        if (kept.length === 0) continue;
        const partial = kept.length < s.occurrences.length;
        await act(s.session_id, item.heard_cleaned, canonical,
          partial ? kept.map((o) => ({ segment_id: o.segment_id, word_index: o.word_index })) : undefined);
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
      <audio
        ref={audioRef}
        preload="none"
        onTimeUpdate={handleTimeUpdate}
        onPause={handleStopped}
        onEnded={handleStopped}
      />
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
                    s.occurrences.map((o, i) => {
                      const k = occKey(s.session_id, o);
                      return (
                        <li key={`${k}-${i}`}>
                          <button
                            className="name-occ-play"
                            onClick={() => playClip(s.session_id, k, o.start)}
                            title="Play the audio around this word"
                            aria-label={playingKey === k ? 'Pause' : 'Play'}
                          >
                            {playingKey === k ? '❚❚' : '▶'}
                          </button>{' '}
                          <label className="name-occ-check">
                            <input
                              type="checkbox"
                              checked={!excluded[k]}
                              onChange={(e) =>
                                setExcluded({ ...excluded, [k]: !e.target.checked })
                              }
                            />
                          </label>{' '}
                          <Link to={`/sessions/${s.session_id}/detections`}>
                            {s.session_id}
                          </Link>{' '}
                          · seg {o.segment_id} · {formatTime(o.start)} ·{' '}
                          <span className="name-token">“{o.token}”</span>
                          <OccurrenceSentence o={o} />
                        </li>
                      );
                    }),
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
