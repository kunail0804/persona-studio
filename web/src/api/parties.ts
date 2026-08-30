import { request, requestVoid, streamNdjson } from "./client";
import { expectArray, expectNumber, expectString, isRecord } from "./validate";

export type PartyRole = "user" | "assistant";

export interface PartyMessage {
  id: number;
  role: PartyRole;
  content: string;
  ts: number;
}

export interface PartySummary {
  id: string;
  scenarioId: string;
  scenarioTitle: string;
  label: string;
  createdAt: number;
  updatedAt: number;
}

export interface Party extends PartySummary {
  messages: PartyMessage[];
}

function parsePartyRole(value: unknown, field: string): PartyRole {
  const role = expectString(value, field);
  if (role !== "user" && role !== "assistant") {
    throw new Error(`Expected "user" or "assistant" for "${field}"`);
  }
  return role;
}

function parsePartyMessage(data: unknown): PartyMessage {
  if (!isRecord(data)) throw new Error("Expected a party message object");
  return {
    id: expectNumber(data.id, "id"),
    role: parsePartyRole(data.role, "role"),
    content: expectString(data.content, "content"),
    ts: expectNumber(data.ts, "ts"),
  };
}

function parsePartySummary(data: unknown): PartySummary {
  if (!isRecord(data)) throw new Error("Expected a party object");
  return {
    id: expectString(data.id, "id"),
    scenarioId: expectString(data.scenario_id, "scenario_id"),
    scenarioTitle: expectString(data.scenario_title, "scenario_title"),
    label: expectString(data.label, "label"),
    createdAt: expectNumber(data.created_at, "created_at"),
    updatedAt: expectNumber(data.updated_at, "updated_at"),
  };
}

function parseParty(data: unknown): Party {
  const summary = parsePartySummary(data);
  if (!isRecord(data)) throw new Error("Expected a party object");
  return {
    ...summary,
    messages: expectArray(data.messages, "messages").map((item: unknown) =>
      parsePartyMessage(item),
    ),
  };
}

function parsePartyList(data: unknown): PartySummary[] {
  return expectArray(data, "parties").map((item: unknown) => parsePartySummary(item));
}

export function listParties(): Promise<PartySummary[]> {
  return request("/parties", parsePartyList);
}

export function getParty(id: string, signal?: AbortSignal): Promise<Party> {
  return request(`/parties/${id}`, parseParty, { signal });
}

export function createParty(scenarioId: string): Promise<PartySummary> {
  return request(`/scenarios/${scenarioId}/parties`, parsePartySummary, {
    method: "POST",
    body: JSON.stringify({ label: "" }),
  });
}

export function renameParty(id: string, label: string): Promise<PartySummary> {
  return request(`/parties/${id}`, parsePartySummary, {
    method: "PATCH",
    body: JSON.stringify({ label }),
  });
}

export function deleteParty(id: string): Promise<void> {
  return requestVoid(`/parties/${id}`, { method: "DELETE" });
}

/** One line of a turn's NDJSON stream, narrowed to what the server sends. */
export type TurnEvent =
  | { kind: "delta"; text: string }
  | { kind: "error"; message: string }
  | { kind: "done"; messageId: number | null };

function parseTurnEvent(data: unknown): TurnEvent {
  if (!isRecord(data)) throw new Error("Expected a turn event object");
  if (typeof data.delta === "string") {
    return { kind: "delta", text: data.delta };
  }
  if (typeof data.error === "string") {
    return { kind: "error", message: data.error };
  }
  if (data.done === true) {
    return {
      kind: "done",
      messageId: data.message_id === null ? null : expectNumber(data.message_id, "message_id"),
    };
  }
  throw new Error("Unexpected turn event shape");
}

/**
 * Plays one turn: resolves once the response headers arrive (the player's
 * turn is persisted server-side by then), hands each stream event to
 * `onEvent`, and resolves for good after the final `done` line. Aborting
 * `signal` rejects with an AbortError; whatever the server had already
 * streamed is persisted there either way.
 */
export async function sendTurn(
  id: string,
  content: string,
  onEvent: (event: TurnEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  await streamNdjson(`/parties/${id}/messages`, { content }, signal, (data: unknown) =>
    onEvent(parseTurnEvent(data)),
  );
}
