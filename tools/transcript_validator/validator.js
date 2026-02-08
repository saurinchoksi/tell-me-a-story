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
  silenceGapEnabled: false,
  nearZeroEnabled: false,
  duplicatesEnabled: false,
  userScrolledAway: false,
  activeDrawerTab: 'notes',
  scrollTimeout: null,
  // Context menu state
  contextTarget: null,  // { type: 'segment'|'word', segmentIndex, wordIndex }
};

// --- Client-side filter predicates (mirrors src/filters.py) ---

function isSilenceGap(segment) {
  const words = segment.words || [];
  if (words.length !== 1) return false;
  const speaker = words[0]._speaker || {};
  return speaker.label === null && speaker.coverage === 0.0;
}

function isNearZeroProbability(segment) {
  const words = segment.words || [];
  if (words.length !== 1) return false;
  const prob = words[0].probability;
  return prob != null && prob < 0.01;
}

function findDuplicateSegmentIds(segments) {
  const seen = {};
  const duplicates = new Set();
  for (const seg of segments) {
    const text = (seg.text || '').trim();
    if (!text) continue;
    if (text in seen) {
      duplicates.add(seg.id);
    } else {
      seen[text] = seg.id;
    }
  }
  return duplicates;
}

function getFilterReasons(segment, duplicateIds) {
  const reasons = [];
  if (state.silenceGapEnabled && isSilenceGap(segment)) reasons.push('silence-gap');
  if (state.nearZeroEnabled && isNearZeroProbability(segment)) reasons.push('near-zero');
  if (state.duplicatesEnabled && duplicateIds.has(segment.id)) reasons.push('duplicate');
  return reasons;
}

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
const notesCountBadge = document.getElementById('notes-count-badge');
const silenceGapToggleBtn = document.getElementById('silence-gap-toggle-btn');
const nearZeroToggleBtn = document.getElementById('near-zero-toggle-btn');
const duplicatesToggleBtn = document.getElementById('duplicates-toggle-btn');
const tabContentNotes = document.getElementById('tab-content-notes');
const tabContentLowConfidence = document.getElementById('tab-content-low-confidence');
const lowConfidenceList = document.getElementById('low-confidence-list');
const tabNotesCount = document.getElementById('tab-notes-count');
const tabLcCount = document.getElementById('tab-lc-count');
const deleteAllBtn = document.getElementById('delete-all-notes');

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
    minPxPerSec: 100,
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
    segmentsContainer.innerHTML = '<div class="empty-state">No segments found</div>';
    return;
  }

  state._cachedDuplicateIds = findDuplicateSegmentIds(state.segments);

  const filterBadgeMap = {
    'silence-gap': '<span class="badge badge-filter badge-filter-silence-gap">silence gap</span>',
    'near-zero': '<span class="badge badge-filter badge-filter-near-zero">near zero</span>',
    'duplicate': '<span class="badge badge-filter badge-filter-duplicate">duplicate</span>',
  };

  segmentsContainer.innerHTML = state.segments.map((segment, index) => {
    const hasHighTemp = segment.temperature === 1.0;
    const hasHighComp = segment.compression_ratio > 2.5;
    const hasNotes = segmentHasNotes(segment.id);

    const badges = [];
    if (hasHighTemp) badges.push('<span class="badge badge-temp">temp=1.0</span>');
    if (hasHighComp) badges.push('<span class="badge badge-comp">high comp</span>');
    if (hasNotes) badges.push('<span class="badge badge-note">note</span>');

    const filterReasons = getFilterReasons(segment, state._cachedDuplicateIds);
    const isFiltered = filterReasons.length > 0;
    filterReasons.forEach(r => badges.push(filterBadgeMap[r]));

    const words = (segment.words || []).map((word, wordIndex) => {
      const probClass = getWordProbabilityClass(word.probability || 0);
      const duration = ((word.end || 0) - (word.start || 0)).toFixed(3);
      const tooltip = `prob: ${(word.probability || 0).toFixed(2)} | ${word.start?.toFixed(2)}s → ${word.end?.toFixed(2)}s | dur: ${duration}s`;
      return `<span class="word ${probClass}" data-segment="${index}" data-word="${wordIndex}" data-start="${word.start}" data-tooltip="${tooltip}">${escapeHtml(word.word)}</span>`;
    }).join(' ');

    return `
      <div class="segment-card${isFiltered ? ' filtered' : ''}" data-segment="${index}" id="segment-${index}" style="animation-delay: ${index * 0.03}s">
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

// Reload transcript (used by loadSession)
async function reloadTranscript() {
  if (!state.filename) return;
  const res = await fetch(`/transcript/${state.filename}`);
  if (!res.ok) throw new Error('Failed to load transcript');
  const data = await res.json();
  state.segments = data.segments || [];
  // Same reference — filtering is purely visual (CSS .filtered class), not array-level
  state.allSegments = state.segments;
  renderSegments();
  updateActiveSegment(state.currentTime);
}

// Generic filter toggle helper
function toggleFilterState(key, btn) {
  state[key] = !state[key];
  btn.classList.toggle('active', state[key]);
  renderSegments();
  updateActiveSegment(state.currentTime);

  if (notesPanel.classList.contains('open')) {
    renderNotesList();
    renderLowConfidenceList();
  }
}

function toggleSilenceGap() { toggleFilterState('silenceGapEnabled', silenceGapToggleBtn); }
function toggleNearZero() { toggleFilterState('nearZeroEnabled', nearZeroToggleBtn); }
function toggleDuplicates() { toggleFilterState('duplicatesEnabled', duplicatesToggleBtn); }

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
  localStorage.setItem('validator-last-session', stem);
  state.segments = [];
  state.allSegments = [];
  state.notes = [];
  state.currentSegmentIndex = -1;
  state.currentWordIndex = -1;

  try {
    await reloadTranscript();

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
    const lastSession = localStorage.getItem('validator-last-session');
    if (lastSession && data.files.includes(lastSession)) {
      sessionSelect.value = lastSession;
      loadSession(lastSession);
    }
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
    renderLowConfidenceList();
    // Defer pill position calculation until panel has rendered
    requestAnimationFrame(() => switchDrawerTab(state.activeDrawerTab));
  }
}

function closeNotesPanel() {
  notesPanel.classList.remove('open');
  notesToggleBtn.classList.remove('active');
  document.body.classList.remove('notes-panel-open');
}

function switchDrawerTab(tabName) {
  state.activeDrawerTab = tabName;
  document.querySelectorAll('.drawer-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  tabContentNotes.classList.toggle('active', tabName === 'notes');
  tabContentLowConfidence.classList.toggle('active', tabName === 'low-confidence');
  deleteAllBtn.style.display = tabName === 'notes' ? '' : 'none';

  updateTabPill();
}

function updateTabPill() {
  const tabsContainer = document.querySelector('.drawer-tabs');
  const activeTab = tabsContainer?.querySelector('.drawer-tab.active');
  if (tabsContainer && activeTab) {
    const containerRect = tabsContainer.getBoundingClientRect();
    const tabRect = activeTab.getBoundingClientRect();
    const left = tabRect.left - containerRect.left;
    tabsContainer.style.setProperty('--tab-left', left + 'px');
    tabsContainer.style.setProperty('--tab-width', tabRect.width + 'px');
  }
}

function renderLowConfidenceList() {
  const groups = getFilteredWords();
  const totalWords = groups.reduce((sum, g) => sum + g.words.length, 0);
  tabLcCount.textContent = `(${totalWords})`;

  if (groups.length === 0) {
    lowConfidenceList.innerHTML = `<div class="notes-list-empty">
  <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="24" cy="24" r="20"/>
    <path d="M16 24l5 5 11-11"/>
  </svg>
  <span class="empty-title">All words confident</span>
  <span class="empty-hint">Whisper was sure about every word</span>
</div>`;
    return;
  }

  lowConfidenceList.innerHTML = groups.map(group => {
    const parentSegment = state.segments.find(s => s.id === group.segmentId);
    const isFilteredOut = parentSegment ? getFilterReasons(parentSegment, state._cachedDuplicateIds || new Set()).length > 0 : false;
    const chips = group.words.map(w =>
      `<button class="filtered-word-chip${isFilteredOut ? ' filter-active' : ''}" data-start="${w.start}">` +
        `${escapeHtml(w.word)} <span class="word-prob">${w.probability.toFixed(2)}</span>` +
      `</button>`
    ).join('');

    return `
      <div class="note-item${isFilteredOut ? ' filtered-out' : ''}">
        <div class="note-item-header">
          <span class="note-item-location">Segment ${group.segmentId}</span>
          <span>${formatTime(group.segmentStart)}</span>
        </div>
        <div class="filtered-word-chips">${chips}</div>
      </div>
    `;
  }).join('');

  // Attach click-to-seek listeners
  lowConfidenceList.querySelectorAll('.filtered-word-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const time = parseFloat(chip.dataset.start);
      if (!isNaN(time)) {
        state.userScrolledAway = false;
        seekAndPlay(time);
      }
    });
  });
}

function updateNotesCount() {
  const count = state.notes.length;
  tabNotesCount.textContent = `(${count})`;
  notesCountBadge.textContent = count;
}

function getFilteredWords() {
  const results = [];
  for (const segment of state.allSegments) {
    const lowConfWords = [];
    for (const word of (segment.words || [])) {
      if (word.probability < 0.5) {
        lowConfWords.push({
          word: word.word,
          probability: word.probability,
          start: word.start,
        });
      }
    }
    if (lowConfWords.length > 0) {
      results.push({
        segmentId: segment.id,
        segmentStart: segment.start,
        words: lowConfWords,
      });
    }
  }
  return results;
}

function renderNotesList() {
  updateNotesCount();

  if (state.notes.length === 0) {
    notesList.innerHTML = `<div class="notes-list-empty">
  <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M8 6h24l8 8v28a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z"/>
    <path d="M32 6v8h8"/>
    <path d="M16 24h16"/>
    <path d="M16 32h10"/>
  </svg>
  <span class="empty-title">No notes yet</span>
  <span class="empty-hint">Right-click a word or press <kbd>N</kbd></span>
</div>`;
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
    const isFilteredOut = note.segmentId != null && segment != null && getFilterReasons(segment, state._cachedDuplicateIds || new Set()).length > 0;
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
          ${isFilteredOut ? '<span class="note-filtered-badge">filtered</span>' : ''}
          <span>${time}</span>
        </div>
        <div class="note-item-text">${escapeHtml(textPreview)}</div>
        <div class="note-item-actions">
          <button class="note-action-btn" onclick="jumpToNote('${note.id}')" title="Jump to location"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7"/><path d="M7 7h10v10"/></svg></button>
          <button class="note-action-btn" onclick="editNoteFromPanel('${note.id}')" title="Edit note"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
          <button class="note-action-btn delete" onclick="deleteNote('${note.id}')" title="Delete note"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg></button>
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
    if (segmentIndex === -1) return;  // Segment no longer in transcript
    target = {
      type: note.wordIndex != null ? 'word' : 'segment',
      segmentIndex: segmentIndex,
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

  // Animate the card out before re-rendering
  const noteEl = document.querySelector(`.note-item[data-note-id="${noteId}"]`);
  if (noteEl) {
    noteEl.style.animation = 'noteSlideOut 0.25s ease forwards';
    await new Promise(r => setTimeout(r, 250));
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
        let prev = state.currentSegmentIndex - 1;
        while (prev >= 0 && getFilterReasons(state.segments[prev], state._cachedDuplicateIds || new Set()).length > 0) {
          prev--;
        }
        if (prev >= 0) seekAndPlay(state.segments[prev].start);
      }
      break;

    case 'ArrowDown':
      e.preventDefault();
      if (state.segments.length > 0) {
        let next = (state.currentSegmentIndex >= 0 ? state.currentSegmentIndex : -1) + 1;
        while (next < state.segments.length && getFilterReasons(state.segments[next], state._cachedDuplicateIds || new Set()).length > 0) {
          next++;
        }
        if (next < state.segments.length) seekAndPlay(state.segments[next].start);
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

    case 'g':
      e.preventDefault();
      toggleSilenceGap();
      break;

    case 'z':
      e.preventDefault();
      toggleNearZero();
      break;

    case 'd':
      e.preventDefault();
      toggleDuplicates();
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
document.querySelectorAll('.drawer-tab').forEach(tab => {
  tab.addEventListener('click', () => switchDrawerTab(tab.dataset.tab));
});
silenceGapToggleBtn.addEventListener('click', toggleSilenceGap);
nearZeroToggleBtn.addEventListener('click', toggleNearZero);
duplicatesToggleBtn.addEventListener('click', toggleDuplicates);

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

// Initialize tab pill position after fonts load
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => {
    if (notesPanel.classList.contains('open')) {
      switchDrawerTab(state.activeDrawerTab);
    }
  });
}

window.addEventListener('resize', () => {
  if (notesPanel.classList.contains('open')) {
    updateTabPill();
  }
});
