/** TypeScript interfaces matching API response shapes. */

/**
 * Segment identifier. Integer for real Whisper segments; string for injected
 * gap segments (e.g. "gap_241.680" from speaker.py gap detection). The union
 * is intentional — consumers must accept both, not narrow to `number`.
 */
export type SegmentId = number | string;

/** Human-review status for a session, set from the Sessions list. */
export type ValidationStatus = 'not_started' | 'in_progress' | 'done';

export interface SessionSummary {
  id: string;
  has_audio: boolean;
  has_transcript: boolean;
  has_diarization: boolean;
  has_embeddings: boolean;
  has_identifications: boolean;
  /** Session-level free-text note; '' when none set. */
  note: string;
  /** Recording duration in seconds; null when the session isn't transcribed. */
  duration_seconds: number | null;
  /** Count of timestamped validation notes. */
  note_count: number;
  /** Human-review status. */
  validation_status: ValidationStatus;
  /** Pipeline stages that errored (from transcript-rich.json _processing); [] when healthy. */
  failed_stages: string[];
}

export interface SessionDetail {
  id: string;
  has_audio: boolean;
  transcript: TranscriptData | null;
  diarization: DiarizationData | null;
  embeddings: EmbeddingsData | null;
  identifications: IdentificationData | null;
}

export interface StoryRegion {
  index: number;
  start_id: SegmentId;
  end_id: SegmentId;
  title: string;
  world: string;
}

export interface TranscriptData {
  segments: TranscriptSegment[];
  _stories?: StoryRegion[];
  [key: string]: unknown;
}

export interface TranscriptSegment {
  id: SegmentId;
  start: number;
  end: number;
  text: string;
  words?: TranscriptWord[];
  [key: string]: unknown;
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  probability?: number;
  _speaker?: { label: string };
  [key: string]: unknown;
}

export interface DiarizationData {
  segments: DiarizationSegment[];
  [key: string]: unknown;
}

export interface DiarizationSegment {
  speaker: string;
  start: number;
  end: number;
}

export interface EmbeddingsData {
  speakers: Record<string, SpeakerEmbedding>;
  _generator_version?: string;
  _dimension?: number;
}

export interface SpeakerEmbedding {
  vector: number[];
  num_segments: number;
  total_duration_s?: number;
}

export interface IdentificationData {
  session_id: string;
  identified_at: string;
  profiles_used: number;
  identifications: SpeakerIdentification[];
}

export interface SpeakerIdentification {
  speaker_key: string;
  status: 'identified' | 'suggested' | 'unknown';
  profile_id: string | null;
  profile_name: string | null;
  confidence: number | null;
  confirmed?: boolean;
  confirmed_action?: DecisionAction;
  confirmed_profile_id?: string;
}

/** Profile as returned by GET /api/profiles (vectors stripped). */
export interface ProfileSummary {
  id: string;
  name: string;
  role: string;
  created: string;
  updated: string;
  embeddings: number;       // count, not the actual vectors
  voice_variants: number;   // count
  latest_match_score?: number | null;
  latest_match_session?: string | null;
  last_seen?: string | null;
}

/** Enriched embedding info from GET /api/profiles/:id. */
export interface ProfileEmbeddingInfo {
  session_id: string | null;
  source_speaker_key: string | null;
  total_duration_s: number | null;
  num_segments: number | null;
}

/** Voice variant metadata from GET /api/profiles/:id. */
export interface ProfileVariantInfo {
  id: string;
  created: string;
  session_id: string | null;
  source_speaker_key: string | null;
}

/** Reference to an audio segment for speaker preview. */
export interface AudioSampleRef {
  session_id: string;
  start: number;
  end: number;
}

/** Most recent identification match for a profile. */
export interface LatestMatch {
  session_id: string;
  confidence: number;
  identified_at: string;
}

/** Full profile detail from GET /api/profiles/:id. */
export interface ProfileDetail {
  id: string;
  name: string;
  role: string;
  created: string;
  updated: string;
  embeddings: ProfileEmbeddingInfo[];
  voice_variants: ProfileVariantInfo[];
  audio_sample: AudioSampleRef | null;
  latest_match: LatestMatch | null;
}

// --- Validator types ---

export interface ValidatorWord {
  word: string;
  start: number;
  end: number;
  probability: number;
  _speaker?: {
    label: string | null;
    coverage: number;
  };
  _original?: string;
  _corrections?: Array<{
    stage: string;
    from: string;
    to: string;
  }>;
}

export interface ValidatorSegment {
  id: SegmentId;
  start: number;
  end: number;
  text: string;
  words: ValidatorWord[];
  temperature?: number;
  compression_ratio?: number;
  no_speech_prob?: number;
  avg_logprob?: number;
  _speaker?: {
    label: string | null;
    coverage: number;
  };
  _source?: string;
  _story?: number;
}

export interface Note {
  id: string;
  segmentId: SegmentId | null;
  wordIndex: number | null;
  wordText: string | null;
  wordStart: number | null;
  timestamp: number;
  text: string;
  createdAt: string;
}

/** Per-segment axial codes for EMP step 5 counting. Multiple codes allowed per segment. */
export type AxialCode = 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M6' | 'M7' | 'M8' | 'M9' | 'M10' | 'NotA';

export interface AxialLabel {
  segmentId: SegmentId;
  codes: AxialCode[];
  createdAt: string;
  updatedAt: string;
}

export interface FilterState {
  silenceGap: boolean;
  nearZero: boolean;
  duplicates: boolean;
}

export interface ContextTarget {
  type: 'segment' | 'word' | 'timestamp';
  segmentId?: SegmentId;
  segmentIndex?: number;
  wordIndex?: number;
  wordText?: string;
  wordStart?: number;
  timestamp?: number;
}

export interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  target: ContextTarget | null;
}

export interface NoteModalState {
  visible: boolean;
  target: ContextTarget | null;
  existingNote: Note | null;
}

// --- Detections (Monitor) ---

/** Registry metadata for one failure-mode detector. */
export interface DetectorInfo {
  id: string;
  label: string;
  failure_mode: string;
  version: string;
}

/** One detector's run summary on one session (rollup view). */
export interface DetectionRunSummary {
  n_flags: number;
  run_at: string;
  detector_version: string;
}

export interface DetectionsRollupSession {
  session_id: string;
  duration_seconds: number | null;
  /** Keyed by detector id. A missing key means "never scanned" — distinct from 0 flags. */
  results: Record<string, DetectionRunSummary>;
  /** The transcript changed since this session was scanned — re-scan to refresh. */
  stale: boolean;
}

export interface DetectionsRollup {
  detectors: DetectorInfo[];
  sessions: DetectionsRollupSession[];
  totals: Record<string, number>;
  /** Set when a scan ran code-only because the LLM judge's venv was absent. */
  warning?: string;
}

/** Fields every detector's flag carries — the anchor + the server-side join. */
export interface DetectionFlagBase {
  segment_id: SegmentId;
  word_index: number;
  start: number | null;
  end: number | null;
  token: string;
  cleaned: string;
  dm_codes: string[];
  segment_text: string | null;
  segment_start: number | null;
  segment_end: number | null;
  segment_speaker: string | null;
}

/** M9a — a token matched against the family-name roster. */
export interface FamilyNameFlag extends DetectionFlagBase {
  match_type: 'phonetic' | 'alias';
  matched_person_ids: string[];
  matched_canonicals: string[];
}

/** M9b — a token in a name spelled inconsistently across the session. Only the
 *  deviating (non-majority) spellings are flagged; `majority_spelling` is what this
 *  recording mostly used (the odd token differs from it), or null when no spelling
 *  dominates. It is a within-recording reference, NOT a global canonical. */
export interface NameConsistencyFlag extends DetectionFlagBase {
  cluster_id: string;
  cluster_spellings: string[];
  n_cluster_occurrences: number;
  majority_spelling: string | null;
  majority_count: number | null;
}

/** M9c — a sourced-canon name (Thomas / Mahabharata) spelled wrong, from the
 *  per-story reader. Carries the correct spelling and the story's inferred world. */
export interface CanonNameFlag extends DetectionFlagBase {
  case: 'M9c';
  canonical: string;
  card_id: number;
  all_spellings: string[];
  wrong_cleaned: string[];
  evidence: string;
  story_id: number;
  story_world: string;
  /** Confidence in the SUGGESTED spelling: true when the heard token sounds like the
   *  canonical (shared Double-Metaphone code). false = a "best guess" (e.g. Urzi→Urjani).
   *  Optional: pre-0.4.0 flags lack it. */
  suggestion_confident?: boolean;
  /** How the catch was made — informational, not the confidence driver. */
  methods?: string[];
  /** Order-robust judge: rounds (of vote_rounds) that agreed this is a misspelled canon name. */
  vote_count?: number;
  vote_rounds?: number;
}

/** Discriminated by a field unique to each shape: `cluster_spellings` → m9b,
 *  `matched_canonicals` → m9a, `canonical` → m9c. Render code must narrow on one
 *  of these and fall back gracefully, never assuming a single shape. */
export type DetectionFlag = FamilyNameFlag | NameConsistencyFlag | CanonNameFlag;

export interface SessionDetectorResult {
  label: string;
  failure_mode: string;
  detector_version: string;
  run_at: string;
  n_word_tokens: number;
  n_flags: number;
  flags: DetectionFlag[];
  /** The transcript changed since this section was scanned. */
  stale: boolean;
  /** Whether the LLM judge ran on this section (M9b full pass). */
  judge_applied?: boolean;
}

export interface SessionDetectionsData {
  session_id: string;
  /** Whether an audio file exists — gates the per-flag "play clip" control. */
  has_audio: boolean;
  /** Keyed by detector id; {} when the session was never scanned. */
  detectors: Record<string, SessionDetectorResult>;
  /** Set when a scan ran code-only because the LLM judge's venv was absent. */
  warning?: string;
}

// --- Speaker confirmation (Task 6) ---

export type DecisionAction = 'confirm' | 'confirm_variant' | 'create' | 'reassign' | 'skip';

export interface Decision {
  speaker_key: string;
  action: DecisionAction;
  profile_id?: string;
  new_name?: string;
  new_role?: string;
}

export interface ConfirmSpeakersResponse {
  applied: number;
  skipped: number;
  created_profiles: string[];
  identifications: IdentificationData;
}
