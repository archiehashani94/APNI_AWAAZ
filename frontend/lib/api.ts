export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function wsUrl(path: string): string {
  const base = API_URL.replace(/\/$/, "");
  if (base.startsWith("https://")) return `wss://${base.slice("https://".length)}${path}`;
  if (base.startsWith("http://")) return `ws://${base.slice("http://".length)}${path}`;
  return `ws://${base}${path}`;
}

export async function api(
  path: string,
  sessionId: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("X-Session-Id", sessionId);
  return fetch(`${API_URL}${path}`, { ...init, headers });
}

export type Emotion = "warm" | "urgent" | "playful" | "calm";

export type Reply = { text: string; emotion: Emotion };

export type Favourite = { id: string; text: string; emotion: Emotion };
