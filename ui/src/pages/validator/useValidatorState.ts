/**
 * useValidatorState — reducer-based state management for the validator page.
 *
 * Centralizes ~15 fields with cross-cutting updates (e.g. filter toggle
 * recomputes which segments are dimmed). Dispatch is used by the page
 * component, keyboard shortcuts, and the notes system.
 */

import { useReducer, useMemo } from 'react';
import type { ValidatorSegment, Note, FilterState, ContextMenuState, NoteModalState, SessionSummary } from '../../types';
import { findDuplicateSegmentIds, getFilterReasons, getAllFilterReasons } from '../../utils/filters';

export interface ValidatorState {
  sessions: SessionSummary[];
  segments: ValidatorSegment[];
  notes: Note[];
  currentTime: number;
  activeSegmentIndex: number;
  duration: number;
  playing: boolean;
  playbackRate: number;
  filters: FilterState;
  duplicateIds: Set<number | string>;
  drawerOpen: boolean;
  drawerTab: 'notes' | 'low-confidence';
  contextMenu: ContextMenuState;
  noteModal: NoteModalState;
  speakerNames: Map<string, string>;
  loading: boolean;
  error: string | null;
}

export type ValidatorAction =
  | { type: 'LOAD_SESSION'; segments: ValidatorSegment[]; notes: Note[]; speakerNames: Map<string, string> }
  | { type: 'SET_TIME'; time: number }
  | { type: 'SET_ACTIVE_SEGMENT'; index: number }
  | { type: 'SET_DURATION'; duration: number }
  | { type: 'SET_PLAYING'; playing: boolean }
  | { type: 'SET_PLAYBACK_RATE'; rate: number }
  | { type: 'TOGGLE_FILTER'; filter: keyof FilterState }
  | { type: 'TOGGLE_DRAWER' }
  | { type: 'SET_DRAWER_TAB'; tab: 'notes' | 'low-confidence' }
  | { type: 'SHOW_CONTEXT_MENU'; menu: ContextMenuState }
  | { type: 'HIDE_CONTEXT_MENU' }
  | { type: 'SHOW_NOTE_MODAL'; modal: NoteModalState }
  | { type: 'HIDE_NOTE_MODAL' }
  | { type: 'SET_NOTES'; notes: Note[] }
  | { type: 'SET_ERROR'; error: string }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'SET_SESSIONS'; sessions: SessionSummary[] };

const initialState: ValidatorState = {
  sessions: [],
  segments: [],
  notes: [],
  currentTime: 0,
  activeSegmentIndex: -1,
  duration: 0,
  playing: false,
  playbackRate: 1,
  filters: { silenceGap: false, nearZero: false, duplicates: false },
  duplicateIds: new Set(),
  drawerOpen: false,
  drawerTab: 'notes',
  contextMenu: { visible: false, x: 0, y: 0, target: null },
  noteModal: { visible: false, target: null, existingNote: null },
  speakerNames: new Map(),
  loading: true,
  error: null,
};

function reducer(state: ValidatorState, action: ValidatorAction): ValidatorState {
  switch (action.type) {
    case 'LOAD_SESSION':
      return {
        ...state,
        segments: action.segments,
        notes: action.notes,
        speakerNames: action.speakerNames,
        duplicateIds: findDuplicateSegmentIds(action.segments),
        loading: false,
        error: null,
      };

    case 'SET_TIME':
      return { ...state, currentTime: action.time };

    case 'SET_ACTIVE_SEGMENT':
      return state.activeSegmentIndex === action.index
        ? state
        : { ...state, activeSegmentIndex: action.index };

    case 'SET_DURATION':
      return { ...state, duration: action.duration };

    case 'SET_PLAYING':
      return { ...state, playing: action.playing };

    case 'SET_PLAYBACK_RATE':
      return { ...state, playbackRate: action.rate };

    case 'TOGGLE_FILTER':
      return {
        ...state,
        filters: {
          ...state.filters,
          [action.filter]: !state.filters[action.filter],
        },
      };

    case 'TOGGLE_DRAWER':
      return { ...state, drawerOpen: !state.drawerOpen };

    case 'SET_DRAWER_TAB':
      return { ...state, drawerTab: action.tab };

    case 'SHOW_CONTEXT_MENU':
      return { ...state, contextMenu: action.menu };

    case 'HIDE_CONTEXT_MENU':
      return state.contextMenu.visible
        ? { ...state, contextMenu: { ...state.contextMenu, visible: false } }
        : state;

    case 'SHOW_NOTE_MODAL':
      return { ...state, noteModal: action.modal, contextMenu: { ...state.contextMenu, visible: false } };

    case 'HIDE_NOTE_MODAL':
      return state.noteModal.visible
        ? { ...state, noteModal: { ...state.noteModal, visible: false } }
        : state;

    case 'SET_NOTES':
      return { ...state, notes: action.notes };

    case 'SET_ERROR':
      return { ...state, error: action.error, loading: false };

    case 'SET_LOADING':
      return { ...state, loading: action.loading };

    case 'SET_SESSIONS':
      return { ...state, sessions: action.sessions };

    default:
      return state;
  }
}

export interface DerivedState {
  filterCounts: { silenceGap: number; nearZero: number; duplicates: number };
  segmentFilterReasons: Map<number | string, string[]>;
  segmentAllFilterReasons: Map<number | string, string[]>;
  notesBySegment: Map<number | string, Note[]>;
}

export function useValidatorState() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const derived = useMemo<DerivedState>(() => {
    const filterCounts = { silenceGap: 0, nearZero: 0, duplicates: 0 };
    const segmentFilterReasons = new Map<number | string, string[]>();
    const segmentAllFilterReasons = new Map<number | string, string[]>();

    for (const seg of state.segments) {
      const active = getFilterReasons(seg, state.duplicateIds, state.filters);
      const all = getAllFilterReasons(seg, state.duplicateIds);

      segmentFilterReasons.set(seg.id, active);
      segmentAllFilterReasons.set(seg.id, all);

      for (const r of all) {
        if (r === 'silence-gap') filterCounts.silenceGap++;
        else if (r === 'near-zero') filterCounts.nearZero++;
        else if (r === 'duplicate') filterCounts.duplicates++;
      }
    }

    const notesBySegment = new Map<number | string, Note[]>();
    for (const note of state.notes) {
      if (note.segmentId !== null) {
        const existing = notesBySegment.get(note.segmentId) ?? [];
        existing.push(note);
        notesBySegment.set(note.segmentId, existing);
      }
    }

    return { filterCounts, segmentFilterReasons, segmentAllFilterReasons, notesBySegment };
  }, [state.segments, state.duplicateIds, state.filters, state.notes]);

  return { state, dispatch, derived };
}
