// Shared request plumbing for every domain module under `web/src/api/`.
// `request` never trusts the response shape itself — each caller passes a
// `parse` function that narrows the `unknown` body into its real type, which
// is what keeps this file free of `any` and `as`.

import { isRecord } from "./validate";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (isRecord(body) && typeof body.detail === "string") {
    return body.detail;
  }
  return response.statusText || `Request failed with status ${response.status}`;
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorDetail(response));
  }
  return response;
}

/** Sends a request and hands the parsed JSON body to `parse`. */
export async function request<T>(
  path: string,
  parse: (data: unknown) => T,
  init?: RequestInit,
): Promise<T> {
  const response = await send(path, init);
  const data: unknown = await response.json();
  return parse(data);
}

/** Sends a request whose response body carries nothing worth reading (204). */
export async function requestVoid(path: string, init?: RequestInit): Promise<void> {
  await send(path, init);
}

/**
 * Streams an NDJSON response (one JSON object per line) through `onLine`.
 *
 * NDJSON rather than Server-Sent Events: the client needs the final message id
 * and the real error text, and SSE carries neither without a second channel —
 * here every line is a plain JSON object on one response.
 *
 * The incomplete last line is buffered between chunks: a JSON object can be
 * cut in half across two network chunks, and parsing that half throws. Only
 * whole lines are ever parsed. `decoder.decode(value, { stream: true })` for
 * the same reason — a multi-byte character can also be split across chunks.
 */
export async function streamNdjson(
  path: string,
  body: object,
  signal: AbortSignal,
  onLine: (data: unknown) => void,
): Promise<void> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorDetail(response));
  }
  if (!response.body) {
    throw new Error("The server sent no response body to stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const parseLine = (line: string): void => {
    const trimmed = line.trim();
    if (!trimmed) return;
    onLine(JSON.parse(trimmed));
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // The last element is the unterminated remainder, if any: it waits for
    // the next chunk instead of being parsed half-done.
    buffer = lines.pop() ?? "";
    for (const line of lines) parseLine(line);
  }
  buffer += decoder.decode();
  if (!buffer.trim()) return;
  try {
    parseLine(buffer);
  } catch {
    // A trailing fragment means the stream was cut mid-object. Everything the
    // server yielded was already persisted server-side before the line was
    // sent, so dropping the half-parsed remnant loses nothing.
  }
}
