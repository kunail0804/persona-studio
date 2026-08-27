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
