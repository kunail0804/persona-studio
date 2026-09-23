import { request, requestVoid } from "./client";
import { expectArray, expectInteger, expectString, isRecord } from "./validate";

/**
 * A place of the scenario. The description is visual — it is what an image
 * prompt receives in place of the name. The atmosphere is for the narrator.
 */
export interface Place {
  id: number;
  name: string;
  description: string;
  atmosphere: string;
}

export interface PlaceInput {
  name: string;
  description: string;
  atmosphere: string;
}

/**
 * A lorebook entry: its text reaches the narrator only on a turn whose recent
 * messages use one of its keywords.
 */
export interface LoreEntry {
  id: number;
  keywords: string[];
  text: string;
}

export interface LoreInput {
  keywords: string[];
  text: string;
}

function parsePlace(data: unknown): Place {
  if (!isRecord(data)) throw new Error("Expected a place object");
  return {
    id: expectInteger(data.id, "id"),
    name: expectString(data.name, "name"),
    description: expectString(data.description, "description"),
    atmosphere: expectString(data.atmosphere, "atmosphere"),
  };
}

function parseLoreEntry(data: unknown): LoreEntry {
  if (!isRecord(data)) throw new Error("Expected a lore entry object");
  return {
    id: expectInteger(data.id, "id"),
    keywords: expectArray(data.keywords, "keywords").map((item) => expectString(item, "keyword")),
    text: expectString(data.text, "text"),
  };
}

export function listPlaces(scenarioId: string, signal?: AbortSignal): Promise<Place[]> {
  return request(
    `/scenarios/${scenarioId}/places`,
    (data: unknown) => expectArray(data, "places").map(parsePlace),
    { signal },
  );
}

export function createPlace(scenarioId: string, input: PlaceInput): Promise<Place> {
  return request(`/scenarios/${scenarioId}/places`, parsePlace, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updatePlace(scenarioId: string, placeId: number, input: PlaceInput): Promise<Place> {
  return request(`/scenarios/${scenarioId}/places/${placeId}`, parsePlace, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deletePlace(scenarioId: string, placeId: number): Promise<void> {
  return requestVoid(`/scenarios/${scenarioId}/places/${placeId}`, { method: "DELETE" });
}

export function listLore(scenarioId: string, signal?: AbortSignal): Promise<LoreEntry[]> {
  return request(
    `/scenarios/${scenarioId}/lore`,
    (data: unknown) => expectArray(data, "lore").map(parseLoreEntry),
    { signal },
  );
}

export function createLore(scenarioId: string, input: LoreInput): Promise<LoreEntry> {
  return request(`/scenarios/${scenarioId}/lore`, parseLoreEntry, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateLore(scenarioId: string, entryId: number, input: LoreInput): Promise<LoreEntry> {
  return request(`/scenarios/${scenarioId}/lore/${entryId}`, parseLoreEntry, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deleteLore(scenarioId: string, entryId: number): Promise<void> {
  return requestVoid(`/scenarios/${scenarioId}/lore/${entryId}`, { method: "DELETE" });
}
