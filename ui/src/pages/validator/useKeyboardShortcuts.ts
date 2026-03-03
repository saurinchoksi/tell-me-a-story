/**
 * useKeyboardShortcuts — global keyboard handler for the validator page.
 *
 * Modal-aware: when a modal is open, only Escape works. All other shortcuts
 * are suppressed to avoid accidental actions while typing notes.
 *
 * Shortcuts:
 *   Space        — play/pause
 *   ArrowLeft    — seek -1s
 *   ArrowRight   — seek +1s
 *   ArrowUp      — previous segment (skip filtered)
 *   ArrowDown    — next segment (skip filtered)
 *   N            — add note for current segment
 *   Shift+N      — toggle notes drawer
 *   G            — toggle silence gap filter
 *   Z            — toggle near-zero filter
 *   D            — toggle duplicates filter
 *   Escape       — close modal/menu
 */

import { useEffect, useCallback } from 'react';
import type { WaveformPlayerHandle } from '../../components/WaveformPlayer';
import type { ValidatorSegment, FilterState } from '../../types';
import type { ValidatorAction } from './useValidatorState';

interface KeyboardConfig {
  waveformRef: React.RefObject<WaveformPlayerHandle | null>;
  segments: ValidatorSegment[];
  activeSegmentIndex: number;
  filters: FilterState;
  duplicateIds: Set<number>;
  modalOpen: boolean;
  dispatch: React.Dispatch<ValidatorAction>;
  onSeek: (time: number) => void;
}

export function useKeyboardShortcuts({
  waveformRef,
  segments,
  activeSegmentIndex,
  filters,
  duplicateIds,
  modalOpen,
  dispatch,
  onSeek,
}: KeyboardConfig) {
  const isFiltered = useCallback(
    (seg: ValidatorSegment): boolean => {
      if (seg.words.length === 1) {
        const w = seg.words[0];
        if (filters.silenceGap && w._speaker?.label === null && w._speaker?.coverage === 0.0) return true;
        if (filters.nearZero && w.probability < 0.01) return true;
      }
      if (filters.duplicates && duplicateIds.has(seg.id)) return true;
      return false;
    },
    [filters, duplicateIds],
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture when typing in inputs
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      // Modal open — only Escape works
      if (modalOpen) {
        if (e.key === 'Escape') {
          e.preventDefault();
          dispatch({ type: 'HIDE_NOTE_MODAL' });
          dispatch({ type: 'HIDE_CONTEXT_MENU' });
        }
        return;
      }

      switch (e.key) {
        case ' ':
          e.preventDefault();
          waveformRef.current?.playPause();
          break;

        case 'Escape':
          e.preventDefault();
          dispatch({ type: 'HIDE_CONTEXT_MENU' });
          dispatch({ type: 'HIDE_NOTE_MODAL' });
          break;

        case 'ArrowLeft':
          e.preventDefault();
          if (waveformRef.current) {
            const t = waveformRef.current.getCurrentTime();
            waveformRef.current.setTime(Math.max(0, t - 1));
          }
          break;

        case 'ArrowRight':
          e.preventDefault();
          if (waveformRef.current) {
            const t = waveformRef.current.getCurrentTime();
            waveformRef.current.setTime(t + 1);
          }
          break;

        case 'ArrowUp': {
          e.preventDefault();
          // Previous non-filtered segment
          let idx = activeSegmentIndex - 1;
          while (idx >= 0 && isFiltered(segments[idx])) idx--;
          if (idx >= 0) onSeek(segments[idx].start);
          break;
        }

        case 'ArrowDown': {
          e.preventDefault();
          // Next non-filtered segment
          let idx = activeSegmentIndex + 1;
          while (idx < segments.length && isFiltered(segments[idx])) idx++;
          if (idx < segments.length) onSeek(segments[idx].start);
          break;
        }

        case 'n':
          // n — add note for current segment (lowercase only; Shift+N produces 'N')
          e.preventDefault();
          if (activeSegmentIndex >= 0) {
            const seg = segments[activeSegmentIndex];
            dispatch({
              type: 'SHOW_NOTE_MODAL',
              modal: {
                visible: true,
                target: {
                  type: 'segment',
                  segmentId: seg.id,
                  segmentIndex: activeSegmentIndex,
                  timestamp: seg.start,
                },
                existingNote: null,
              },
            });
          }
          break;

        case 'N':
          // Shift+N — toggle notes drawer
          e.preventDefault();
          dispatch({ type: 'TOGGLE_DRAWER' });
          break;

        case 'g':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_FILTER', filter: 'silenceGap' });
          break;

        case 'z':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_FILTER', filter: 'nearZero' });
          break;

        case 'd':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_FILTER', filter: 'duplicates' });
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [waveformRef, segments, activeSegmentIndex, filters, duplicateIds, modalOpen, dispatch, onSeek, isFiltered]);
}
