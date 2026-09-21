import { request, requestVoid, streamNdjson } from "./client";
import {
  expectArray,
  expectBoolean,
  expectInteger,
  expectNumber,
  expectRecord,
  expectString,
  isRecord,
} from "./validate";

export type PartyRole = "user" | "assistant";
export type MessageKind = "text" | "image";
export type MessageStatus = "pending" | "done" | "error";

export interface MessageVariant {
  id: number;
  content: string;
  active: boolean;
}

export interface PartyMessage {
  id: number;
  role: PartyRole;
  content: string;
  ts: number;
  variants: MessageVariant[];
  /** Image messages only: the generation's state, straight from the schema. */
  kind: MessageKind;
  imageId: string | null;
  status: MessageStatus | null;
  /** When the generation started; the elapsed time is computed from it. */
  startedAt: number | null;
  error: string | null;
}

export interface PartySummary {
  id: string;
  scenarioId: string;
  scenarioTitle: string;
  label: string;
  createdAt: number;
  updatedAt: number;
}

/**
 * How full the next turn's prompt is against the configured window. Ollama
 * truncates an over-long prompt in silence, from the front, system prompt
 * first — so the narrator would forget the scenario with nothing on screen to
 * explain it.
 */
export interface ContextUsage {
  estimatedTokens: number;
  numCtx: number;
  nearLimit: boolean;
}

export interface Party extends PartySummary {
  messages: PartyMessage[];
  summaryText: string;
  summaryUpto: number | null;
  worldState: Record<string, unknown>;
  context: ContextUsage;
}

function parseContextUsage(data: unknown): ContextUsage {
  if (!isRecord(data)) throw new Error("Expected a context usage object");
  return {
    estimatedTokens: expectInteger(data.estimated_tokens, "estimated_tokens"),
    numCtx: expectInteger(data.num_ctx, "num_ctx"),
    nearLimit: expectBoolean(data.near_limit, "near_limit"),
  };
}

function parsePartyRole(value: unknown, field: string): PartyRole {
  const role = expectString(value, field);
  if (role !== "user" && role !== "assistant") {
    throw new Error(`Expected "user" or "assistant" for "${field}"`);
  }
  return role;
}

function parseMessageVariant(data: unknown): MessageVariant {
  if (!isRecord(data)) throw new Error("Expected a message variant object");
  return {
    id: expectInteger(data.id, "id"),
    content: expectString(data.content, "content"),
    active: expectBoolean(data.active, "active"),
  };
}

function parseMessageStatus(value: unknown, field: string): MessageStatus | null {
  if (value === null || value === undefined) return null;
  const status = expectString(value, field);
  if (status !== "pending" && status !== "done" && status !== "error") {
    throw new Error(`Expected "pending", "done" or "error" for "${field}"`);
  }
  return status;
}

function parseMessageKind(value: unknown, field: string): MessageKind {
  const kind = expectString(value, field);
  if (kind !== "text" && kind !== "image") {
    throw new Error(`Expected "text" or "image" for "${field}"`);
  }
  return kind;
}

export function parsePartyMessage(data: unknown): PartyMessage {
  if (!isRecord(data)) throw new Error("Expected a party message object");
  return {
    id: expectInteger(data.id, "id"),
    role: parsePartyRole(data.role, "role"),
    content: expectString(data.content, "content"),
    ts: expectNumber(data.ts, "ts"),
    variants: expectArray(data.variants, "variants").map((item: unknown) =>
      parseMessageVariant(item),
    ),
    kind: parseMessageKind(data.kind, "kind"),
    imageId: data.image_id === null || data.image_id === undefined ? null : expectString(data.image_id, "image_id"),
    status: parseMessageStatus(data.status, "status"),
    startedAt: data.started_at === null || data.started_at === undefined ? null : expectNumber(data.started_at, "started_at"),
    error: data.error === null || data.error === undefined ? null : expectString(data.error, "error"),
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
    summaryText: expectString(data.summary_text, "summary_text"),
    summaryUpto: data.summary_upto === null ? null : expectInteger(data.summary_upto, "summary_upto"),
    worldState: expectRecord(data.world_state, "world_state"),
    context: parseContextUsage(data.context),
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

/**
 * Asks for another reply to one narrator message, streamed exactly like a
 * turn. Server-side the old reply is archived as a variant and stays
 * reachable; the replacement is committed only once text has arrived, so an
 * empty stream leaves the message as it was.
 */
export async function regenerateMessage(
  id: string,
  messageId: number,
  onEvent: (event: TurnEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  await streamNdjson(
    `/parties/${id}/messages/${messageId}/regenerate`,
    {},
    signal,
    (data: unknown) => onEvent(parseTurnEvent(data)),
  );
}

/** Corrects a message's text in place, keeping the message's id. */
export function editMessage(
  partyId: string,
  messageId: number,
  content: string,
): Promise<PartyMessage> {
  return request(`/parties/${partyId}/messages/${messageId}`, parsePartyMessage, {
    method: "PATCH",
    body: JSON.stringify({ content }),
  });
}

/** Makes one archived variant the message's active text. */
export function setMessageVariant(
  partyId: string,
  messageId: number,
  variantId: number,
): Promise<PartyMessage> {
  return request(`/parties/${partyId}/messages/${messageId}/variant`, parsePartyMessage, {
    method: "PUT",
    body: JSON.stringify({ variant_id: variantId }),
  });
}
