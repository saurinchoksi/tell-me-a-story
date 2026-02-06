import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js';

// State
const state = {
  filename: null,
  segments: [],
  allSegments: [],  // Unfiltered copy for lookups when filtering
  notes: [],
  currentTime: 0,
  currentSegmentIndex: -1,
  currentWordIndex: -1,
  playbackRate: 1,
  filterEnabled: false,
  userScrolledAway: false,
  scrollTimeout: null,
  // Context menu state
  contextTarget: null,  // { type: 'segment'|'word', segmentIndex, wordIndex }
};

const FILTER_PROBABILITY_THRESHOLD = 0.5;

// DOM elements
const sessionSelect = document.getElementById('session-select');
const segmentsContainer = document.getElementById('segments');
const playBtn = document.getElementById('play-btn');
const playIcon = document.getElementById('play-icon');
const pauseIcon = document.getElementById('pause-icon');
const timeDisplay = document.getElementById('time-display');
const loadingOverlay = document.getElementById('loading');
const noteModal = document.getElementById('note-modal');
const noteSegmentId = document.getElementById('note-segment-id');
const noteText = document.getElementById('note-text');
const noteSave = document.getElementById('note-save');
const noteCancel = document.getElementById('note-cancel');
const contextMenu = document.getElementById('context-menu');
const contextAddNote = document.getElementById('context-add-note');
const noteModalTitle = document.getElementById('note-modal-title');
const notesPanel = document.getElementById('notes-panel');
const notesPanelClose = document.getElementById('notes-panel-close');
const notesToggleBtn = document.getElementById('notes-toggle-btn');
const notesList = document.getElementById('notes-list');
const notesCount = document.getElementById('notes-count');
const notesCountBadge = document.getElementById('notes-count-badge');
const filterToggleBtn = document.getElementById('filter-toggle-btn');

// WaveSurfer instance
let wavesurfer = null;

// Initialize WaveSurfer
function initWaveSurfer() {
  if (wavesurfer) {
    wavesurfer.destroy();
  }

  wavesurfer = WaveSurfer.create({
    container: '#waveform',
    height: 100,
    waveColor: getComputedStyle(document.documentElement).getPropertyValue('--wave-color').trim(),
    progressColor: getComputedStyle(document.documentElement).getPropertyValue('--wave-progress').trim(),
    cursorColor: getComputedStyle(document.documentElement).getPropertyValue('--wave-cursor').trim(),
    minPxPerSec: 50,
    plugins: [
      Minimap.create({
        container: '#minimap',
        height: 36,
        waveColor: getComputedStyle(document.documentElement).getPropertyValue('--wave-color').trim(),
        progressColor: getComputedStyle(document.documentElement).getPropertyValue('--wave-progress').trim(),
      }),
    ],
  });

  wavesurfer.on('timeupdate', (time) => {
    state.currentTime = time;
    updateTimeDisplay();
    updateActiveSegment(time);
  });

  wavesurfer.on('play', () => {
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
  });

  wavesurfer.on('pause', () => {
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
  });

  wavesurfer.on('ready', () => {
    hideLoading();
    updateTimeDisplay();
  });

  wavesurfer.on('error', (error) => {
    hideLoading();
    console.error('Audio load error:', error);
    segmentsContainer.innerHTML = `<div class="empty-state">Error loading audio: ${error}</div>`;
  });

  // Add right-click handler for waveform to create timestamp notes
  document.getElementById('waveform').addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (!wavesurfer) return;
    // Calculate clicked time based on click position
    const waveformRect = document.getElementById('waveform').getBoundingClientRect();
    const clickX = e.clientX - waveformRect.left;
    const waveformWidth = waveformRect.width;
    const duration = wavesurfer.getDuration();
    const clickedTime = (clickX / waveformWidth) * duration;
    showContextMenu(e.clientX, e.clientY, { type: 'timestamp', timestamp: clickedTime });
  });

  return wavesurfer;
}

// Format time as seconds
function formatTime(seconds) {
  return `${seconds.toFixed(1)}s`;
}

// Update time display
function updateTimeDisplay() {
  if (!wavesurfer) return;
  const current = formatTime(state.currentTime);
  const duration = formatTime(wavesurfer.getDuration() || 0);
  timeDisplay.textContent = `${current} / ${duration}`;
}

// Get word probability class
function getWordProbabilityClass(prob) {
  if (prob >= 0.9) return 'prob-high';
  if (prob >= 0.5) return 'prob-mid';
  return 'prob-low';
}

// Check if segment has notes
function segmentHasNotes(segmentId) {
  return state.notes.some(n => n.segmentId === segmentId);
}

// Render segments
function renderSegments() {
  if (!state.segments || state.segments.length === 0) {
    const message = state.filterEnabled
      ? 'No words above confidence threshold'
      : 'No segments found';
    segmentsContainer.innerHTML = `<div class="empty-state">${message}</div>`;
    return;
  }

  segmentsContainer.innerHTML = state.segments.map((segment, index) => {
    const hasHighTemp = segment.temperature === 1.0;
    const hasHighComp = segment.compression_ratio > 2.5;
    const hasNotes = segmentHasNotes(segment.id);

    const badges = [];
    if (hasHighTemp) badges.push('<span class="badge badge-temp">temp=1.0</span>');
    if (hasHighComp) badges.push('<span class="badge badge-comp">high comp</span>');
    if (hasNotes) badges.push('<span class="badge badge-note">note</span>');

    const words = (segment.words || []).map((word, wordIndex) => {
      const probClass = getWordProbabilityClass(word.probability || 0);
      const duration = ((word.end || 0) - (word.start || 0)).toFixed(3);
      const tooltip = `prob: ${(word.probability || 0).toFixed(2)} | ${word.start?.toFixed(2)}s → ${word.end?.toFixed(2)}s | dur: ${duration}s`;
      return `<span class="word ${probClass}" data-segment="${index}" data-word="${wordIndex}" data-start="${word.start}" data-tooltip="${tooltip}">${escapeHtml(word.word)}</span>`;
    }).join(' ');

    return `
      <div class="segment-card" data-segment="${index}" id="segment-${index}" style="animation-delay: ${index * 0.03}s">
        <div class="segment-header" data-start="${segment.start}">
          <span class="segment-id">Segment ${segment.id ?? index}</span>
          <span class="segment-time">${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s</span>
          <div class="segment-badges">${badges.join('')}</div>
        </div>
        <div class="segment-meta">
          temp: ${segment.temperature ?? 'N/A'} | comp: ${(segment.compression_ratio ?? 0).toFixed(2)} | no_speech: ${(segment.no_speech_prob ?? 0).toFixed(3)}
        </div>
        <div class="segment-words">${words || '<em style="color: var(--text-secondary)">No words</em>'}</div>
      </div>
    `;
  }).join('');

  // Add click handlers
  segmentsContainer.querySelectorAll('.segment-header').forEach(header => {
    header.addEventListener('click', () => {
      const start = parseFloat(header.dataset.start);
      if (!isNaN(start)) seekAndPlay(start);
    });
  });

  segmentsContainer.querySelectorAll('.word').forEach(word => {
    word.addEventListener('click', () => {
      const start = parseFloat(word.dataset.start);
      if (!isNaN(start)) seekAndPlay(start);
    });
  });

  // Context menu handlers
  segmentsContainer.querySelectorAll('.segment-card').forEach(card => {
    card.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const segmentIndex = parseInt(card.dataset.segment);
      showContextMenu(e.clientX, e.clientY, { type: 'segment', segmentIndex });
    });
  });

  segmentsContainer.querySelectorAll('.word').forEach(word => {
    word.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const segmentIndex = parseInt(word.dataset.segment);
      const wordIndex = parseInt(word.dataset.word);
      showContextMenu(e.clientX, e.clientY, { type: 'word', segmentIndex, wordIndex });
    });
  });
}

// Seek and play
function seekAndPlay(time) {
  if (!wavesurfer) return;
  wavesurfer.setTime(time);
  wavesurfer.play();
}

// Update active segment and word
function updateActiveSegment(time) {
  // Find current segment
  let newSegmentIndex = -1;
  for (let i = 0; i < state.segments.length; i++) {
    const seg = state.segments[i];
    if (time >= seg.start && time < seg.end) {
      newSegmentIndex = i;
      break;
    }
  }

  // Update segment highlight
  if (newSegmentIndex !== state.currentSegmentIndex) {
    // Remove old highlight
    if (state.currentSegmentIndex >= 0) {
      const oldCard = document.getElementById(`segment-${state.currentSegmentIndex}`);
      if (oldCard) oldCard.classList.remove('active');
    }

    // Add new highlight
    state.currentSegmentIndex = newSegmentIndex;
    if (newSegmentIndex >= 0) {
      const newCard = document.getElementById(`segment-${newSegmentIndex}`);
      if (newCard) {
        newCard.classList.add('active');

        // Auto-scroll if not manually scrolled away
        if (!state.userScrolledAway) {
          newCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }

  // Find and highlight current word
  document.querySelectorAll('.word.active').forEach(w => w.classList.remove('active'));

  if (newSegmentIndex >= 0) {
    const segment = state.segments[newSegmentIndex];
    const words = segment.words || [];
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const nextWord = words[i + 1];
      const wordEnd = nextWord ? nextWord.start : segment.end;

      if (time >= word.start && time < wordEnd) {
        const wordEl = document.querySelector(`.word[data-segment="${newSegmentIndex}"][data-word="${i}"]`);
        if (wordEl) {
          wordEl.classList.add('active');
          state.currentWordIndex = i;
        }
        break;
      }
    }
  }
}

// Reload transcript (used by filter toggle and loadSession)
async function reloadTranscript() {
  if (!state.filename) return;

  const url = state.filterEnabled
    ? `/transcript/${state.filename}?min_prob=${FILTER_PROBABILITY_THRESHOLD}`
    : `/transcript/${state.filename}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to load transcript');
  const data = await res.json();
  state.segments = data.segments || [];
  renderSegments();
  updateActiveSegment(state.currentTime);
}

// Toggle probability filter
async function toggleFilter() {
  state.filterEnabled = !state.filterEnabled;
  filterToggleBtn.classList.toggle('active', state.filterEnabled);

  const wasPlaying = wavesurfer && wavesurfer.isPlaying();
  const currentTime = state.currentTime;

  try {
    await reloadTranscript();
  } catch (error) {
    // Revert state on failure
    state.filterEnabled = !state.filterEnabled;
    filterToggleBtn.classList.toggle('active', state.filterEnabled);
    console.error('Error toggling filter:', error);
    return;
  }

  // Refresh notes panel if open (filtered-out status may have changed)
  if (notesPanel.classList.contains('open')) {
    renderNotesList();
  }

  if (wavesurfer) {
    wavesurfer.setTime(currentTime);
    if (wasPlaying) wavesurfer.play();
  }
}

// Toggle theme
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? null : 'dark';

  if (next) {
    html.setAttribute('data-theme', next);
    localStorage.setItem('validator-theme', next);
  } else {
    html.removeAttribute('data-theme');
    localStorage.removeItem('validator-theme');
  }

  // Reinitialize WaveSurfer with new theme colors if audio is loaded
  if (wavesurfer && state.filename) {
    const wasPlaying = wavesurfer.isPlaying();
    const currentTime = state.currentTime;
    const playbackRate = state.playbackRate;

    initWaveSurfer();
    wavesurfer.load(`/audio/${state.filename}`);
    wavesurfer.once('ready', () => {
      wavesurfer.setTime(currentTime);
      wavesurfer.setPlaybackRate(playbackRate);
      if (wasPlaying) wavesurfer.play();
    });
  }
}

// Load session
async function loadSession(stem) {
  if (!stem) {
    segmentsContainer.innerHTML = '<div class="empty-state">Select a session to begin</div>';
    return;
  }

  showLoading();
  state.filename = stem;
  state.segments = [];
  state.allSegments = [];
  state.notes = [];
  state.currentSegmentIndex = -1;
  state.currentWordIndex = -1;

  try {
    // Load transcript (respects filter state)
    await reloadTranscript();

    // Always load unfiltered segments for timestamp lookups
    const allRes = await fetch(`/transcript/${stem}`);
    if (allRes.ok) {
      const allData = await allRes.json();
      state.allSegments = allData.segments || [];
    }

    // Load notes
    try {
      const notesRes = await fetch(`/notes/${stem}`);
      if (notesRes.ok) {
        const notesData = await notesRes.json();
        state.notes = notesData.notes || [];
      }
    } catch (e) {
      console.warn('Failed to load notes:', e);
      state.notes = [];
    }

    // Re-render with notes and update count
    renderSegments();
    updateNotesCount();

    // Initialize and load audio
    initWaveSurfer();
    wavesurfer.load(`/audio/${stem}`);

  } catch (error) {
    console.error('Error loading session:', error);
    hideLoading();
    segmentsContainer.innerHTML = `<div class="empty-state">Error: ${error.message}</div>`;
  }
}

// Load available files
async function loadFiles() {
  try {
    const res = await fetch('/files');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    sessionSelect.innerHTML = '<option value="">Select a session...</option>';
    data.files.forEach(file => {
      const option = document.createElement('option');
      option.value = file;
      option.textContent = file;
      sessionSelect.appendChild(option);
    });
  } catch (error) {
    console.error('Error loading files:', error);
    sessionSelect.innerHTML = '<option value="">Error loading sessions</option>';
  }
}

// Show/hide loading
function showLoading() {
  loadingOverlay.classList.add('visible');
}

function hideLoading() {
  loadingOverlay.classList.remove('visible');
}

// Note modal functions
function openNoteModal(target = null) {
  // Pause playback when opening note modal
  if (wavesurfer && wavesurfer.isPlaying()) {
    wavesurfer.pause();
  }

  // Use passed target or fall back to current segment, or timestamp at current time
  let noteTarget = target;
  if (!noteTarget && state.currentSegmentIndex >= 0) {
    noteTarget = { type: 'segment', segmentIndex: state.currentSegmentIndex };
  } else if (!noteTarget && wavesurfer) {
    // No segment active, create timestamp note at current playback position
    noteTarget = { type: 'timestamp', timestamp: state.currentTime };
  }

  if (!noteTarget) return;

  state.contextTarget = noteTarget;

  // Update modal title
  if (noteTarget.type === 'timestamp') {
    noteModalTitle.innerHTML = `Add Note at <strong>${noteTarget.timestamp.toFixed(1)}s</strong>`;
  } else if (noteTarget.type === 'word') {
    const word = state.segments[noteTarget.segmentIndex]?.words?.[noteTarget.wordIndex];
    const time = word?.start?.toFixed(2) || '0.00';
    noteModalTitle.innerHTML = `Add Note for Word "<em>${escapeHtml(word?.word || '')}</em>" at ${time}s`;
  } else {
    const segmentId = state.segments[noteTarget.segmentIndex]?.id ?? noteTarget.segmentIndex;
    noteModalTitle.innerHTML = `Add Note for Segment <span id="note-segment-id">${segmentId}</span>`;
  }

  noteText.value = '';

  // Check for existing note
  let existingNote;
  if (noteTarget.type === 'timestamp') {
    // Match timestamp notes within 0.5s tolerance
    existingNote = state.notes.find(n =>
      n.segmentId == null && Math.abs(n.timestamp - noteTarget.timestamp) < 0.5
    );
  } else {
    const targetSegmentId = state.segments[noteTarget.segmentIndex]?.id;
    existingNote = state.notes.find(n =>
      n.segmentId === targetSegmentId &&
      (noteTarget.type === 'word' ? n.wordIndex === noteTarget.wordIndex : n.wordIndex == null)
    );
  }
  if (existingNote) {
    noteText.value = existingNote.text;
  }

  noteModal.classList.add('visible');
  noteText.focus();
}

function closeNoteModal() {
  noteModal.classList.remove('visible');
}

async function saveNote() {
  const text = noteText.value.trim();
  const target = state.contextTarget;

  if (!text || !target) {
    closeNoteModal();
    return;
  }

  // Remove existing note for this target
  const targetSegmentId = state.segments[target.segmentIndex]?.id;
  state.notes = state.notes.filter(n => {
    if (target.type === 'timestamp') {
      // Match timestamp notes within 0.5s tolerance
      return !(n.segmentId == null && Math.abs(n.timestamp - target.timestamp) < 0.5);
    }
    if (target.type === 'word') {
      return !(n.segmentId === targetSegmentId && n.wordIndex === target.wordIndex);
    }
    return !(n.segmentId === targetSegmentId && n.wordIndex == null);
  });

  // Add new note
  const targetSegment = state.segments[target.segmentIndex];
  const targetWord = target.type === 'word' ? targetSegment?.words?.[target.wordIndex] : null;
  const note = {
    id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    segmentId: target.type === 'timestamp' ? null : targetSegment?.id,
    wordIndex: target.type === 'word' ? target.wordIndex : null,
    wordText: targetWord?.word ?? null,
    wordStart: targetWord?.start ?? null,
    timestamp: target.type === 'timestamp' ? target.timestamp : (targetWord?.start ?? targetSegment?.start ?? state.currentTime),
    text: text,
    createdAt: new Date().toISOString(),
  };
  state.notes.push(note);

  // Save to server
  try {
    const res = await fetch(`/notes/${state.filename}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes: state.notes }),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status}`);
  } catch (error) {
    console.error('Error saving note:', error);
    alert('Failed to save note. Please try again.');
    return;
  }

  closeNoteModal();
  renderSegments();
  updateActiveSegment(state.currentTime);

  // Update notes panel if open
  if (notesPanel.classList.contains('open')) {
    renderNotesList();
  } else {
    updateNotesCount();
  }
}

// Context menu functions
function showContextMenu(x, y, target) {
  state.contextTarget = target;

  // Update menu item text
  if (target.type === 'timestamp') {
    contextAddNote.textContent = `Add Note at ${target.timestamp.toFixed(1)}s`;
  } else if (target.type === 'word') {
    const word = state.segments[target.segmentIndex]?.words?.[target.wordIndex];
    contextAddNote.textContent = `Add Note for "${word?.word || 'word'}"`;
  } else {
    const segmentId = state.segments[target.segmentIndex]?.id ?? target.segmentIndex;
    contextAddNote.textContent = `Add/Edit Note for Segment ${segmentId}`;
  }

  // Position menu
  contextMenu.style.left = `${x}px`;
  contextMenu.style.top = `${y}px`;
  contextMenu.classList.add('visible');

  // Adjust if menu goes off screen
  const rect = contextMenu.getBoundingClientRect();
  if (rect.right > window.innerWidth) {
    contextMenu.style.left = `${window.innerWidth - rect.width - 10}px`;
  }
  if (rect.bottom > window.innerHeight) {
    contextMenu.style.top = `${window.innerHeight - rect.height - 10}px`;
  }
}

function hideContextMenu() {
  contextMenu.classList.remove('visible');
}

// Notes panel functions
function toggleNotesPanel() {
  notesPanel.classList.toggle('open');
  const isOpen = notesPanel.classList.contains('open');
  notesToggleBtn.classList.toggle('active', isOpen);
  document.body.classList.toggle('notes-panel-open', isOpen);
  if (isOpen) {
    renderNotesList();
  }
}

function closeNotesPanel() {
  notesPanel.classList.remove('open');
  notesToggleBtn.classList.remove('active');
  document.body.classList.remove('notes-panel-open');
}

function updateNotesCount() {
  const count = state.notes.length;
  notesCount.textContent = count;
  notesCountBadge.textContent = count;
}

function renderNotesList() {
  updateNotesCount();

  if (state.notes.length === 0) {
    notesList.innerHTML = '<div class="notes-list-empty">No notes yet</div>';
    return;
  }

  // Sort notes by timestamp for timestamp notes, then segment index, then word index
  const sortedNotes = [...state.notes].sort((a, b) => {
    // Timestamp-only notes sorted by their timestamp
    const aIsTimestamp = a.segmentId == null;
    const bIsTimestamp = b.segmentId == null;

    // Get effective time for sorting - use allSegments so filtered segments can still be looked up
    const aSegment = a.segmentId != null ? state.allSegments.find(s => s.id === a.segmentId) : null;
    const bSegment = b.segmentId != null ? state.allSegments.find(s => s.id === b.segmentId) : null;
    // For word notes, use word start time; for segment notes, use segment start; fallback to stored timestamp
    const aTime = aIsTimestamp ? a.timestamp
      : (a.wordIndex != null ? (aSegment?.words?.[a.wordIndex]?.start ?? a.timestamp ?? 0)
      : (aSegment?.start ?? a.timestamp ?? 0));
    const bTime = bIsTimestamp ? b.timestamp
      : (b.wordIndex != null ? (bSegment?.words?.[b.wordIndex]?.start ?? b.timestamp ?? 0)
      : (bSegment?.start ?? b.timestamp ?? 0));

    // Primary sort by time
    if (Math.abs(aTime - bTime) > 0.01) {
      return aTime - bTime;
    }

    // If same time, timestamp notes come after segment notes
    if (aIsTimestamp !== bIsTimestamp) {
      return aIsTimestamp ? 1 : -1;
    }

    // For segment notes, sort by word index
    if (!aIsTimestamp && !bIsTimestamp) {
      // Segment notes (wordIndex null) come before word notes
      if (a.wordIndex == null && b.wordIndex != null) return -1;
      if (a.wordIndex != null && b.wordIndex == null) return 1;
      return (a.wordIndex || 0) - (b.wordIndex || 0);
    }

    return 0;
  });

  notesList.innerHTML = sortedNotes.map(note => {
    const segment = note.segmentId != null
      ? state.segments.find(s => s.id === note.segmentId)
      : null;
    // Use allSegments to look up word data even when filtered
    const allSegment = note.segmentId != null
      ? state.allSegments.find(s => s.id === note.segmentId)
      : null;
    // Segment is filtered out if note has a segmentId but segment not found in current state.segments
    const isFilteredOut = note.segmentId != null && !segment;
    let location, time;

    if (note.segmentId == null) {
      // Timestamp-only note (no segment association)
      location = `@ ${formatTime(note.timestamp)}`;
      time = formatTime(note.timestamp);
    } else if (note.wordIndex != null) {
      const word = allSegment?.words?.[note.wordIndex];
      const segmentId = allSegment?.id ?? note.segmentId;
      const wordText = word?.word || note.wordText || '?';
      location = `Seg ${segmentId} | Word "${escapeHtml(wordText)}"`;
      time = formatTime(word?.start ?? note.timestamp ?? 0);
    } else {
      location = `Segment ${allSegment?.id ?? note.segmentId}`;
      time = formatTime(allSegment?.start ?? note.timestamp ?? 0);
    }

    const textPreview = note.text.length > 100 ? note.text.substring(0, 100) + '...' : note.text;

    return `
      <div class="note-item ${isFilteredOut ? 'filtered-out' : ''}" data-note-id="${note.id}">
        <div class="note-item-header">
          <span class="note-item-location">${location}</span>
          ${isFilteredOut ? '<span class="note-filtered-badge">hidden</span>' : ''}
          <span>${time}</span>
        </div>
        <div class="note-item-text">${escapeHtml(textPreview)}</div>
        <div class="note-item-actions">
          <button class="note-action-btn" onclick="jumpToNote('${note.id}')" ${isFilteredOut ? 'disabled' : ''}>Jump</button>
          <button class="note-action-btn" onclick="editNoteFromPanel('${note.id}')" ${isFilteredOut ? 'disabled' : ''}>Edit</button>
          <button class="note-action-btn delete" onclick="deleteNote('${note.id}')">Delete</button>
        </div>
      </div>
    `;
  }).join('');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Make functions globally accessible for onclick handlers
window.jumpToNote = function(noteId) {
  const note = state.notes.find(n => n.id === noteId);
  if (!note) return;

  let seekTime;

  if (note.segmentId == null) {
    // Timestamp-only note - seek directly to timestamp
    seekTime = note.timestamp;
  } else {
    const segment = state.segments.find(s => s.id === note.segmentId);
    const segmentIndex = state.segments.findIndex(s => s.id === note.segmentId);
    if (!segment) return;

    if (note.wordIndex != null) {
      const word = segment.words?.[note.wordIndex];
      seekTime = word?.start ?? segment.start;
    } else {
      seekTime = segment.start;
    }

    // Scroll to segment
    const segmentCard = document.getElementById(`segment-${segmentIndex}`);
    if (segmentCard) {
      segmentCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // Seek audio
  if (wavesurfer) {
    wavesurfer.setTime(seekTime);
  }

  // Close notes panel on mobile for better UX
  if (window.innerWidth < 768) {
    closeNotesPanel();
  }
};

window.editNoteFromPanel = function(noteId) {
  const note = state.notes.find(n => n.id === noteId);
  if (!note) return;

  let target;
  if (note.segmentId == null) {
    // Timestamp-only note
    target = { type: 'timestamp', timestamp: note.timestamp };
  } else {
    const segmentIndex = state.segments.findIndex(s => s.id === note.segmentId);
    target = {
      type: note.wordIndex != null ? 'word' : 'segment',
      segmentIndex: segmentIndex,  // Still need index for openNoteModal
      segmentId: note.segmentId,
      wordIndex: note.wordIndex
    };
  }

  openNoteModal(target);
};

window.deleteNote = async function(noteId) {
  if (!confirm('Delete this note?')) return;

  state.notes = state.notes.filter(n => n.id !== noteId);

  // Save to server
  try {
    const res = await fetch(`/notes/${state.filename}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes: state.notes }),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status}`);
  } catch (error) {
    console.error('Error deleting note:', error);
    alert('Failed to delete note. Please try again.');
    return;
  }

  renderNotesList();
  renderSegments();
  updateActiveSegment(state.currentTime);
};

window.deleteAllNotes = async function() {
  if (state.notes.length === 0) return;

  if (!confirm(`Delete all ${state.notes.length} notes? This cannot be undone.`)) return;

  state.notes = [];

  try {
    const res = await fetch(`/notes/${state.filename}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes: state.notes }),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status}`);
  } catch (error) {
    console.error('Error deleting all notes:', error);
    alert('Failed to delete notes. Please try again.');
    return;
  }

  renderNotesList();
  renderSegments();
  updateActiveSegment(state.currentTime);
};

// Keyboard shortcuts
function handleKeyboard(e) {
  // Don't handle if modal is open and typing
  if (noteModal.classList.contains('visible')) {
    if (e.key === 'Escape') {
      closeNoteModal();
      e.preventDefault();
    }
    return;
  }

  switch (e.key) {
    case ' ':
      e.preventDefault();
      if (wavesurfer) {
        wavesurfer.playPause();
      }
      break;

    case 'ArrowLeft':
      e.preventDefault();
      if (wavesurfer) {
        wavesurfer.setTime(Math.max(0, state.currentTime - 1));
      }
      break;

    case 'ArrowRight':
      e.preventDefault();
      if (wavesurfer) {
        wavesurfer.setTime(Math.min(wavesurfer.getDuration(), state.currentTime + 1));
      }
      break;

    case 'ArrowUp':
      e.preventDefault();
      if (state.currentSegmentIndex > 0) {
        const prevSegment = state.segments[state.currentSegmentIndex - 1];
        seekAndPlay(prevSegment.start);
      }
      break;

    case 'ArrowDown':
      e.preventDefault();
      if (state.currentSegmentIndex < state.segments.length - 1) {
        const nextSegment = state.segments[state.currentSegmentIndex + 1];
        seekAndPlay(nextSegment.start);
      }
      break;

    case 'n':
      e.preventDefault();
      openNoteModal();
      break;

    case 'N':
      e.preventDefault();
      toggleNotesPanel();
      break;

    case 'f':
      e.preventDefault();
      toggleFilter();
      break;

    case 't':
    case 'T':
      e.preventDefault();
      toggleTheme();
      break;
  }
}

// Event listeners
sessionSelect.addEventListener('change', (e) => {
  loadSession(e.target.value);
});

playBtn.addEventListener('click', () => {
  if (wavesurfer) {
    wavesurfer.playPause();
  }
});

document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const speed = parseFloat(btn.dataset.speed);
    state.playbackRate = speed;

    if (wavesurfer) {
      wavesurfer.setPlaybackRate(speed);
    }

    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

noteSave.addEventListener('click', saveNote);
noteCancel.addEventListener('click', closeNoteModal);

notesToggleBtn.addEventListener('click', toggleNotesPanel);
notesPanelClose.addEventListener('click', closeNotesPanel);
filterToggleBtn.addEventListener('click', toggleFilter);

document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

document.addEventListener('keydown', handleKeyboard);

// Context menu event listeners
contextAddNote.addEventListener('click', () => {
  openNoteModal(state.contextTarget);
  hideContextMenu();
});

// Hide context menu on click outside
document.addEventListener('click', () => {
  hideContextMenu();
});

// Scroll detection (single listener, not per-render)
segmentsContainer.addEventListener('scroll', () => {
  hideContextMenu();
  state.userScrolledAway = true;
  clearTimeout(state.scrollTimeout);
  state.scrollTimeout = setTimeout(() => {
    state.userScrolledAway = false;
  }, 5000);
}, { passive: true });

// Initialize
loadFiles();
