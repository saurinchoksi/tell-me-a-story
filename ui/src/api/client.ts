/**
 * Typed fetch wrapper for the tell-me-a-story API.
 *
 * All endpoints go through Vite's dev proxy (/api → localhost:5002),
 * so no absolute URLs or CORS configuration needed.
 */

import type {
  SessionSummary,
  SessionDetail,
  ValidationStatus,
  IdentificationData,
  ProfileSummary,
  ProfileDetail,
  Decision,
  ConfirmSpeakersResponse,
  Note,
  AxialLabel,
  DetectionsRollup,
  SessionDetectionsData,
  NameVerdict,
} from '../types';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(body.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// --- Sessions ---

export async function listSessions(): Promise<SessionSummary[]> {
  const data = await fetchJSON<{ sessions: SessionSummary[] }>('/api/sessions');
  return data.sessions;
}

export async function getSession(id: string): Promise<SessionDetail> {
  return fetchJSON<SessionDetail>(`/api/sessions/${id}`);
}

export async function identifySpeakers(sessionId: string): Promise<IdentificationData> {
  return fetchJSON<IdentificationData>(`/api/sessions/${sessionId}/identify`, {
    method: 'POST',
  });
}

export function audioURL(sessionId: string): string {
  return `/api/sessions/${sessionId}/audio`;
}

export async function saveSessionNote(
  sessionId: string,
  note: string,
): Promise<{ note: string; updatedAt: string }> {
  return fetchJSON<{ note: string; updatedAt: string }>(
    `/api/sessions/${sessionId}/note`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note }),
    },
  );
}

export async function setValidationStatus(
  sessionId: string,
  status: ValidationStatus,
): Promise<{ validationStatus: ValidationStatus; updatedAt: string }> {
  return fetchJSON<{ validationStatus: ValidationStatus; updatedAt: string }>(
    `/api/sessions/${sessionId}/validation-status`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    },
  );
}

// --- Speakers ---

export async function confirmSpeakers(
  sessionId: string,
  decisions: Decision[],
): Promise<ConfirmSpeakersResponse> {
  return fetchJSON<ConfirmSpeakersResponse>(
    `/api/sessions/${sessionId}/confirm-speakers`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decisions }),
    },
  );
}

// --- Profiles ---

export async function listProfiles(): Promise<ProfileSummary[]> {
  const data = await fetchJSON<{ profiles: ProfileSummary[] }>('/api/profiles');
  return data.profiles;
}

export async function createProfile(name: string, role: string): Promise<string> {
  const data = await fetchJSON<{ profile_id: string }>('/api/profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, role }),
  });
  return data.profile_id;
}

export async function updateProfile(
  profileId: string,
  updates: { name?: string; role?: string },
): Promise<void> {
  await fetchJSON(`/api/profiles/${profileId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
}

export async function getProfile(id: string): Promise<ProfileDetail> {
  return fetchJSON<ProfileDetail>(`/api/profiles/${id}`);
}

export async function deleteProfile(id: string): Promise<void> {
  await fetchJSON(`/api/profiles/${id}`, { method: 'DELETE' });
}

export async function refreshCentroid(id: string): Promise<void> {
  await fetchJSON(`/api/profiles/${id}/refresh-centroid`, { method: 'POST' });
}

export async function removeEmbedding(
  profileId: string,
  sessionId: string,
): Promise<{ embeddings_remaining: number }> {
  return fetchJSON(`/api/profiles/${profileId}/embeddings/${sessionId}`, {
    method: 'DELETE',
  });
}

// --- Detections (Monitor) ---

export async function getDetectionsRollup(): Promise<DetectionsRollup> {
  return fetchJSON<DetectionsRollup>('/api/detections');
}

export async function getSessionDetections(
  sessionId: string,
): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(`/api/sessions/${sessionId}/detections`);
}

/** Full re-scan of one session (code detectors + the M9b LLM judge). Slow. */
export async function scanSession(sessionId: string): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(
    `/api/sessions/${sessionId}/detections/scan`, { method: 'POST' });
}

/** Toggle a human name verdict (re-sending an identical one removes it). Returns fresh detail. */
export async function saveNameVerdict(
  sessionId: string,
  verdict: NameVerdict,
): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(
    `/api/sessions/${sessionId}/name-verdicts`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(verdict),
    },
  );
}

/** Re-scan every session whose results are missing or stale (full pass). */
export async function scanAllDetections(): Promise<DetectionsRollup> {
  return fetchJSON<DetectionsRollup>('/api/detections/scan', { method: 'POST' });
}

// --- Notes ---

export async function getNotes(sessionId: string): Promise<Note[]> {
  const data = await fetchJSON<{ notes: Note[] }>(`/api/sessions/${sessionId}/notes`);
  return data.notes;
}

export async function saveNotes(sessionId: string, notes: Note[]): Promise<{ saved: number }> {
  return fetchJSON<{ saved: number }>(`/api/sessions/${sessionId}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  });
}

// --- Axial labels ---

/**
 * Server returns labels in one of two shapes (legacy single-code or current
 * multi-code). This shape captures both for the normalizer below.
 */
type RawAxialLabel = {
  segmentId: AxialLabel['segmentId'];
  createdAt: string;
  updatedAt: string;
} & ({ code: AxialLabel['codes'][number]; codes?: never } | { codes: AxialLabel['codes']; code?: never });

/**
 * Normalize a label entry into the multi-code shape. Single-code legacy entries
 * (`code: 'M3'`) become `codes: ['M3']`. Entries with neither field are corrupt
 * and we fail loud — silently dropping a label would lose data.
 */
function normalizeAxialLabel(raw: RawAxialLabel): AxialLabel {
  if ('codes' in raw && Array.isArray(raw.codes)) {
    return {
      segmentId: raw.segmentId,
      codes: raw.codes,
      createdAt: raw.createdAt,
      updatedAt: raw.updatedAt,
    };
  }
  if ('code' in raw && typeof raw.code === 'string') {
    return {
      segmentId: raw.segmentId,
      codes: [raw.code],
      createdAt: raw.createdAt,
      updatedAt: raw.updatedAt,
    };
  }
  throw new Error(
    `Corrupt axial label entry: missing both 'code' and 'codes' (segmentId=${raw.segmentId})`,
  );
}

export async function getAxialLabels(sessionId: string): Promise<AxialLabel[]> {
  const data = await fetchJSON<{ labels: RawAxialLabel[] }>(
    `/api/sessions/${sessionId}/axial-labels`,
  );
  return data.labels.map(normalizeAxialLabel);
}

export async function saveAxialLabels(
  sessionId: string,
  labels: AxialLabel[],
): Promise<{ saved: number }> {
  return fetchJSON<{ saved: number }>(`/api/sessions/${sessionId}/axial-labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels }),
  });
}
