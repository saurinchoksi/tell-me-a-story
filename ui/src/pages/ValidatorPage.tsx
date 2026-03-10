/**
 * ValidatorPage — transcript validation tool embedded in the React app.
 *
 * Renders outside <Layout> for full viewport width. Loads session transcript +
 * notes, renders a WaveSurfer waveform and scrollable segment cards. Active
 * segment/word highlighting runs at 60fps via refs + direct DOM manipulation
 * (binary search for current segment, class toggle on the card element).
 */

import { useEffect, useRef, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { getSession, getNotes, saveNotes, audioURL, listSessions } from '../api/client';
import type { ValidatorSegment, Note, ContextTarget } from '../types';
import { formatTime } from '../utils/time';
import WaveformPlayer, { type WaveformPlayerHandle } from '../components/WaveformPlayer';
import SegmentCard from '../components/SegmentCard';
import ContextMenu from '../components/ContextMenu';
import NoteModal from '../components/NoteModal';
import NotesDrawer from '../components/NotesDrawer';
import { useValidatorState } from './validator/useValidatorState';
import { useAutoScroll } from './validator/useAutoScroll';
import { useKeyboardShortcuts } from './validator/useKeyboardShortcuts';
import './ValidatorPage.css';

/** Binary search: find the segment containing `time`, or -1. */
function findSegmentIndex(segments: ValidatorSegment[], time: number): number {
  let lo = 0;
  let hi = segments.length - 1;
  let result = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (segments[mid].start <= time) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  // Verify the found segment actually contains the time
  if (result >= 0 && time <= segments[result].end) {
    return result;
  }
  return -1;
}

/** Find the active word within a segment at a given time. */
function findWordIndex(segment: ValidatorSegment, time: number): number {
  for (let i = 0; i < segment.words.length; i++) {
    if (time >= segment.words[i].start && time <= segment.words[i].end) {
      return i;
    }
  }
  return -1;
}

export default function ValidatorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { state, dispatch, derived } = useValidatorState();
  const waveformRef = useRef<WaveformPlayerHandle>(null);

  // Map of segment ID → card DOM element for imperative .active toggling
  const segmentRefsMap = useRef(new Map<number | string, HTMLDivElement>());
  // Track current active indices to clear previous .active classes
  const activeSegRef = useRef(-1);
  const activeWordRef = useRef(-1);
  const segmentsContainerRef = useRef<HTMLDivElement>(null);
  // Ref to segments for 60fps callback (avoids closing over state)
  const segmentsRef = useRef<ValidatorSegment[]>([]);

  // Throttle SET_TIME dispatches (~4fps for display only)
  const lastDispatchRef = useRef(0);

  // Auto-scroll: scrolls active segment into view, pauses on user scroll
  const { scrollToSegment } = useAutoScroll(segmentsContainerRef);

  // Keep segmentsRef in sync with state
  useEffect(() => { segmentsRef.current = state.segments; }, [state.segments]);

  // Load session data
  useEffect(() => {
    if (!id) return;

    Promise.all([getSession(id), getNotes(id)])
      .then(([session, notes]) => {
        const segments = (session.transcript?.segments ?? []) as ValidatorSegment[];
        const speakerNames = new Map<string, string>();
        if (session.identifications?.identifications) {
          for (const ident of session.identifications.identifications) {
            if (ident.profile_name) {
              speakerNames.set(ident.speaker_key, ident.profile_name);
            }
          }
        }
        dispatch({ type: 'LOAD_SESSION', segments, notes, speakerNames });
      })
      .catch((e) => dispatch({ type: 'SET_ERROR', error: e.message }));
  }, [id, dispatch]);

  // Load sessions list for dropdown
  useEffect(() => {
    listSessions()
      .then((sessions) => dispatch({ type: 'SET_SESSIONS', sessions }))
      .catch(() => {}); // non-critical — dropdown just stays empty
  }, [dispatch]);

  // 60fps time update handler — imperative DOM manipulation via refs only.
  // Uses segmentsRef (not state) so callback identity is stable.
  const handleTimeUpdate = useCallback(
    (time: number) => {
      const segments = segmentsRef.current;
      if (segments.length === 0) return;

      // Binary search for active segment
      const segIdx = findSegmentIndex(segments, time);
      const seg = segIdx >= 0 ? segments[segIdx] : null;
      const wordIdx = seg ? findWordIndex(seg, time) : -1;

      // Clear previous active segment
      if (activeSegRef.current !== segIdx) {
        if (activeSegRef.current >= 0 && activeSegRef.current < segments.length) {
          const prevCard = segmentRefsMap.current.get(segments[activeSegRef.current].id);
          if (prevCard) {
            prevCard.classList.remove('active');
            // Clear all active words in previous card
            const prevWords = prevCard.querySelectorAll('.word-span.active');
            prevWords.forEach((el) => el.classList.remove('active'));
          }
        }

        // Activate new segment + auto-scroll
        if (seg) {
          const newCard = segmentRefsMap.current.get(seg.id);
          if (newCard) {
            newCard.classList.add('active');
            scrollToSegment(newCard);
          }
        }
        activeSegRef.current = segIdx;
        activeWordRef.current = -1; // Reset word on segment change
      }

      // Update active word within segment
      if (seg && wordIdx !== activeWordRef.current) {
        const card = segmentRefsMap.current.get(seg.id);
        if (card) {
          // Clear previous active word
          if (activeWordRef.current >= 0) {
            const prevWord = card.querySelector(
              `.word-span[data-word="${activeWordRef.current}"]`,
            );
            prevWord?.classList.remove('active');
          }
          // Activate new word
          if (wordIdx >= 0) {
            const newWord = card.querySelector(
              `.word-span[data-word="${wordIdx}"]`,
            );
            newWord?.classList.add('active');
          }
        }
        activeWordRef.current = wordIdx;
      }

      // Throttled state dispatch for time display (~4fps)
      const now = performance.now();
      if (now - lastDispatchRef.current > 250) {
        dispatch({ type: 'SET_TIME', time });
        dispatch({ type: 'SET_ACTIVE_SEGMENT', index: activeSegRef.current });
        lastDispatchRef.current = now;
      }
    },
    [dispatch, scrollToSegment],
  );

  const handleSeek = useCallback(
    (time: number) => {
      waveformRef.current?.setTime(time);
      waveformRef.current?.play();
    },
    [],
  );

  // Keyboard shortcuts (modal-aware) — placed after handleSeek is defined
  useKeyboardShortcuts({
    waveformRef,
    segments: state.segments,
    activeSegmentIndex: state.activeSegmentIndex,
    filters: state.filters,
    duplicateIds: state.duplicateIds,
    modalOpen: state.noteModal.visible,
    dispatch,
    onSeek: handleSeek,
  });

  const handleContextMenu = useCallback(
    (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => {
      e.preventDefault();
      const seg = state.segments[segmentIndex];
      if (!seg) return;

      if (wordIndex >= 0 && wordIndex < seg.words.length) {
        const w = seg.words[wordIndex];
        dispatch({
          type: 'SHOW_CONTEXT_MENU',
          menu: {
            visible: true,
            x: e.clientX,
            y: e.clientY,
            target: {
              type: 'word',
              segmentId: seg.id,
              segmentIndex,
              wordIndex,
              wordText: w.word.trim(),
              wordStart: w.start,
              timestamp: w.start,
            },
          },
        });
      } else {
        dispatch({
          type: 'SHOW_CONTEXT_MENU',
          menu: {
            visible: true,
            x: e.clientX,
            y: e.clientY,
            target: {
              type: 'segment',
              segmentId: seg.id,
              segmentIndex,
              timestamp: seg.start,
            },
          },
        });
      }
    },
    [state.segments, dispatch],
  );

  // Create stable callback ref factory for segment cards
  const getCardRef = useCallback(
    (segId: number | string) => (el: HTMLDivElement | null) => {
      if (el) {
        segmentRefsMap.current.set(segId, el);
      } else {
        segmentRefsMap.current.delete(segId);
      }
    },
    [],
  );

  // --- Note CRUD ---

  const persistNotes = useCallback(
    (notes: Note[]) => {
      dispatch({ type: 'SET_NOTES', notes });
      if (id) saveNotes(id, notes).catch(console.error);
    },
    [id, dispatch],
  );

  const handleAddNoteFromMenu = useCallback(() => {
    dispatch({
      type: 'SHOW_NOTE_MODAL',
      modal: {
        visible: true,
        target: state.contextMenu.target,
        existingNote: null,
      },
    });
  }, [state.contextMenu.target, dispatch]);

  const handleSaveNote = useCallback(
    (text: string, target: ContextTarget, existingNote: Note | null) => {
      const now = new Date().toISOString();
      let updated: Note[];

      if (existingNote) {
        // Edit existing
        updated = state.notes.map((n) =>
          n.id === existingNote.id ? { ...n, text } : n,
        );
      } else {
        // Create new
        const newNote: Note = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          segmentId: target.segmentId ?? null,
          wordIndex: target.wordIndex ?? null,
          wordText: target.wordText ?? null,
          wordStart: target.wordStart ?? null,
          timestamp: target.timestamp ?? 0,
          text,
          createdAt: now,
        };
        updated = [...state.notes, newNote];
      }

      persistNotes(updated);
      dispatch({ type: 'HIDE_NOTE_MODAL' });
    },
    [state.notes, persistNotes, dispatch],
  );

  const handleDeleteNote = useCallback(
    (noteId: string) => {
      const updated = state.notes.filter((n) => n.id !== noteId);
      persistNotes(updated);
      dispatch({ type: 'HIDE_NOTE_MODAL' });
    },
    [state.notes, persistNotes, dispatch],
  );

  const handleDeleteAllNotes = useCallback(() => {
    persistNotes([]);
  }, [persistNotes]);

  const handleEditNote = useCallback(
    (note: Note) => {
      const target: ContextTarget = note.wordText
        ? { type: 'word', segmentId: note.segmentId ?? undefined, wordText: note.wordText, wordStart: note.wordStart ?? undefined, timestamp: note.timestamp }
        : note.segmentId !== null
          ? { type: 'segment', segmentId: note.segmentId, timestamp: note.timestamp }
          : { type: 'timestamp', timestamp: note.timestamp };

      dispatch({
        type: 'SHOW_NOTE_MODAL',
        modal: { visible: true, target, existingNote: note },
      });
    },
    [dispatch],
  );

  // Stable dismiss/cancel callbacks for ContextMenu + NoteModal portals
  const handleDismissMenu = useCallback(() => dispatch({ type: 'HIDE_CONTEXT_MENU' }), [dispatch]);
  const handleCancelModal = useCallback(() => dispatch({ type: 'HIDE_NOTE_MODAL' }), [dispatch]);

  if (state.loading) {
    return (
      <div className="validator-page">
        <div className="validator-loading">Loading session...</div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="validator-page">
        <div className="validator-error">
          <p>Error: {state.error}</p>
          <Link to="/sessions">Back to sessions</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="validator-page">
      {/* Header */}
      <header className="validator-header">
        <Link to="/sessions" className="validator-back">Sessions</Link>
        {state.sessions.length > 0 ? (
          <select
            className="validator-session-select"
            value={id}
            onChange={(e) => navigate(`/sessions/${e.target.value}/validate`)}
          >
            {state.sessions.map((s) => (
              <option key={s.id} value={s.id}>{s.id}</option>
            ))}
          </select>
        ) : (
          <span className="validator-session-id">{id}</span>
        )}
        <span className="validator-time">
          {formatTime(state.currentTime)} / {formatTime(state.duration)}
        </span>
      </header>

      {/* Waveform */}
      {id && (
        <WaveformPlayer
          ref={waveformRef}
          audioUrl={audioURL(id)}
          onTimeUpdate={handleTimeUpdate}
          onReady={(dur) => dispatch({ type: 'SET_DURATION', duration: dur })}
          onPlayPause={(p) => dispatch({ type: 'SET_PLAYING', playing: p })}
          playbackRate={state.playbackRate}
        />
      )}

      {/* Controls bar */}
      <div className="validator-controls">
        <div className="controls-playback">
          <button
            className="btn btn--small"
            onClick={() => waveformRef.current?.playPause()}
          >
            {state.playing ? 'Pause' : 'Play'}
          </button>
          {[0.5, 1, 1.5].map((rate) => (
            <button
              key={rate}
              className={`btn btn--small ${state.playbackRate === rate ? 'btn--active' : ''}`}
              onClick={() => dispatch({ type: 'SET_PLAYBACK_RATE', rate })}
            >
              {rate}x
            </button>
          ))}
        </div>

        <div className="controls-filters">
          <button
            className={`btn btn--small ${state.filters.silenceGap ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'silenceGap' })}
            title="Toggle silence gap filter (G)"
          >
            Gaps ({derived.filterCounts.silenceGap})
          </button>
          <button
            className={`btn btn--small ${state.filters.nearZero ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'nearZero' })}
            title="Toggle near-zero probability filter (Z)"
          >
            Near-zero ({derived.filterCounts.nearZero})
          </button>
          <button
            className={`btn btn--small ${state.filters.duplicates ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'duplicates' })}
            title="Toggle duplicate filter (D)"
          >
            Dupes ({derived.filterCounts.duplicates})
          </button>
        </div>

        <div className="controls-actions">
          <button
            className={`btn btn--small ${state.drawerOpen ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_DRAWER' })}
            title="Toggle notes drawer (Shift+N)"
          >
            Notes ({state.notes.length})
          </button>
        </div>
      </div>

      {/* Segment list */}
      <div
        ref={segmentsContainerRef}
        className={`validator-segments ${state.drawerOpen ? 'drawer-open' : ''}`}
      >
        {state.segments.map((seg, i) => {
          const activeReasons = derived.segmentFilterReasons.get(seg.id) ?? [];
          const allReasons = derived.segmentAllFilterReasons.get(seg.id) ?? [];
          return (
            <SegmentCard
              key={seg.id}
              segment={seg}
              index={i}
              isFiltered={activeReasons.length > 0}
              filterReasons={activeReasons}
              allFilterReasons={allReasons}
              hasNotes={derived.notesBySegment.has(seg.id)}
              speakerNames={state.speakerNames}
              onSeek={handleSeek}
              onContextMenu={handleContextMenu}
              cardRef={getCardRef(seg.id)}
            />
          );
        })}
      </div>

      {/* Context menu (portal) */}
      <ContextMenu
        menu={state.contextMenu}
        onAddNote={handleAddNoteFromMenu}
        onDismiss={handleDismissMenu}
      />

      {/* Note modal (portal) */}
      <NoteModal
        modal={state.noteModal}
        onSave={handleSaveNote}
        onDelete={handleDeleteNote}
        onCancel={handleCancelModal}
      />

      {/* Notes drawer */}
      <NotesDrawer
        open={state.drawerOpen}
        tab={state.drawerTab}
        notes={state.notes}
        segments={state.segments}
        onTabChange={(tab) => dispatch({ type: 'SET_DRAWER_TAB', tab })}
        onJump={handleSeek}
        onEditNote={handleEditNote}
        onDeleteNote={handleDeleteNote}
        onDeleteAll={handleDeleteAllNotes}
      />

      {/* Keyboard hints */}
      <div className="keyboard-hints">
        <span><kbd>Space</kbd> play/pause</span>
        <span><kbd>&larr;</kbd><kbd>&rarr;</kbd> seek</span>
        <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> segments</span>
        <span><kbd>N</kbd> note</span>
        <span><kbd>Shift+N</kbd> drawer</span>
        <span><kbd>G</kbd><kbd>Z</kbd><kbd>D</kbd> filters</span>
      </div>
    </div>
  );
}
