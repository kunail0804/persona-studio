import { request, requestVoid } from "./client";
import { expectArray, expectInteger, expectString, isRecord } from "./validate";

/**
 * Who owns a list of places and lore entries. A scenario owns the model its
 * editor writes; a party owns its own copy, edited while playing, which the
 * scenario and the other parties never see. The routes have the same shape
 * under both prefixes.
 */
export type WorldOwner = { kind: "scenario"; id: string } | { kind: "party"; id: string };

function ownerPath(owner: WorldOwner): string {
  switch (owner.kind) {
    case "scenario":
      return `/scenarios/${owner.id}`;
    case "party":
      return `/parties/${owner.id}`;
    default: {
      const _exhaustive: never = owner;
      return _exhaustive;
    }
  }
}

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

export function parsePlace(data: unknown): Place {
  if (!isRecord(data)) throw new Error("Expected a place object");
  return {
    id: expectInteger(data.id, "id"),
    name: expectString(data.name, "name"),
    description: expectString(data.description, "description"),
    atmosphere: expectString(data.atmosphere, "atmosphere"),
  };
}

export function parseLoreEntry(data: unknown): LoreEntry {
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

export function createPlace(owner: WorldOwner, input: PlaceInput): Promise<Place> {
  return request(`${ownerPath(owner)}/places`, parsePlace, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updatePlace(owner: WorldOwner, placeId: number, input: PlaceInput): Promise<Place> {
  return request(`${ownerPath(owner)}/places/${placeId}`, parsePlace, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deletePlace(owner: WorldOwner, placeId: number): Promise<void> {
  return requestVoid(`${ownerPath(owner)}/places/${placeId}`, { method: "DELETE" });
}

export function listLore(scenarioId: string, signal?: AbortSignal): Promise<LoreEntry[]> {
  return request(
    `/scenarios/${scenarioId}/lore`,
    (data: unknown) => expectArray(data, "lore").map(parseLoreEntry),
    { signal },
  );
}

export function createLore(owner: WorldOwner, input: LoreInput): Promise<LoreEntry> {
  return request(`${ownerPath(owner)}/lore`, parseLoreEntry, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateLore(owner: WorldOwner, entryId: number, input: LoreInput): Promise<LoreEntry> {
  return request(`${ownerPath(owner)}/lore/${entryId}`, parseLoreEntry, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deleteLore(owner: WorldOwner, entryId: number): Promise<void> {
  return requestVoid(`${ownerPath(owner)}/lore/${entryId}`, { method: "DELETE" });
}
