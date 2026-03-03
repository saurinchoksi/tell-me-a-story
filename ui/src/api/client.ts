/**
 * Typed fetch wrapper for the tell-me-a-story API.
 *
 * All endpoints go through Vite's dev proxy (/api → localhost:5002),
 * so no absolute URLs or CORS configuration needed.
 */

import type {
  SessionSummary,
  SessionDetail,
  IdentificationData,
  ProfileSummary,
  ProfileDetail,
  Decision,
  ConfirmSpeakersResponse,
  Note,
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
