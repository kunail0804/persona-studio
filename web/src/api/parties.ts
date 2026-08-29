import { request, requestVoid } from "./client";
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
