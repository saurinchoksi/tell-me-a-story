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
}
