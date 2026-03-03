/** TypeScript interfaces matching API response shapes. */

export interface SessionSummary {
  id: string;
  has_audio: boolean;
  has_transcript: boolean;
  has_diarization: boolean;
  has_embeddings: boolean;
  has_identifications: boolean;
}

export interface SessionDetail {
  id: string;
  has_audio: boolean;
  transcript: TranscriptData | null;
  diarization: DiarizationData | null;
  embeddings: EmbeddingsData | null;
  identifications: IdentificationData | null;
}

export interface TranscriptData {
  segments: TranscriptSegment[];
  [key: string]: unknown;
}

export interface TranscriptSegment {
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
  _generator?: string;
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
