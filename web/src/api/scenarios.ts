import { request, requestVoid } from "./client";
import { expectArray, expectNumber, expectString, isRecord } from "./validate";

export interface ScenarioSummary {
  id: string;
  title: string;
  synopsis: string;
  characterCount: number;
}

export interface Scenario {
  id: string;
  title: string;
  synopsis: string;
  /** The two biggest levers a scenario has; the prompt omits either when blank. */
  worldRules: string;
  arcs: string;
  createdAt: number;
  updatedAt: number;
}

/** A whole scenario, for a create. */
export interface ScenarioInput {
  title: string;
  synopsis: string;
}

/**
 * A PATCH. Every field is optional and the server writes only the ones sent,
 * so an omitted field keeps its stored value rather than being reset.
 */
export interface ScenarioUpdate {
  title?: string;
  synopsis?: string;
  worldRules?: string;
  arcs?: string;
}

function parseScenarioSummary(data: unknown): ScenarioSummary {
  if (!isRecord(data)) throw new Error("Expected a scenario summary object");
  return {
    id: expectString(data.id, "id"),
    title: expectString(data.title, "title"),
    synopsis: expectString(data.synopsis, "synopsis"),
    characterCount: expectNumber(data.character_count, "character_count"),
  };
}

function parseScenario(data: unknown): Scenario {
  if (!isRecord(data)) throw new Error("Expected a scenario object");
  return {
    id: expectString(data.id, "id"),
    title: expectString(data.title, "title"),
    synopsis: expectString(data.synopsis, "synopsis"),
    worldRules: expectString(data.world_rules, "world_rules"),
    arcs: expectString(data.arcs, "arcs"),
    createdAt: expectNumber(data.created_at, "created_at"),
    updatedAt: expectNumber(data.updated_at, "updated_at"),
  };
}

function parseScenarioList(data: unknown): ScenarioSummary[] {
  return expectArray(data, "scenarios").map((item: unknown) => parseScenarioSummary(item));
}

export function listScenarios(): Promise<ScenarioSummary[]> {
  return request("/scenarios", parseScenarioList);
}

export function getScenario(id: string, signal?: AbortSignal): Promise<Scenario> {
  return request(`/scenarios/${id}`, parseScenario, { signal });
}

export function createScenario(input: ScenarioInput): Promise<Scenario> {
  return request("/scenarios", parseScenario, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateScenario(id: string, input: ScenarioUpdate): Promise<Scenario> {
  // Only the keys present are sent, which is what makes the PATCH partial on
  // the wire as well as on the server.
  const body: Record<string, string> = {};
  if (input.title !== undefined) body.title = input.title;
  if (input.synopsis !== undefined) body.synopsis = input.synopsis;
  if (input.worldRules !== undefined) body.world_rules = input.worldRules;
  if (input.arcs !== undefined) body.arcs = input.arcs;
  return request(`/scenarios/${id}`, parseScenario, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export function deleteScenario(id: string): Promise<void> {
  return requestVoid(`/scenarios/${id}`, { method: "DELETE" });
}
