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
