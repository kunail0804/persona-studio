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
  createdAt: number;
  updatedAt: number;
}

export interface ScenarioInput {
  title: string;
  synopsis: string;
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

export function updateScenario(id: string, input: ScenarioInput): Promise<Scenario> {
  return request(`/scenarios/${id}`, parseScenario, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deleteScenario(id: string): Promise<void> {
  return requestVoid(`/scenarios/${id}`, { method: "DELETE" });
}
