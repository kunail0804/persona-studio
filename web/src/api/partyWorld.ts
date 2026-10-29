import type { CharacterInput } from "./characters";
import { request, requestVoid } from "./client";
import { expectArray, expectInteger, expectString, isRecord } from "./validate";
import type { LoreEntry, Place } from "./world";
import { parseLoreEntry, parsePlace } from "./world";

/**
 * A character of one party. The party received a copy of its scenario's
 * characters when it was created; this is that copy, edited while playing.
 */
export interface PartyCharacter extends CharacterInput {
  id: number;
}

/** Everything a party's world panel shows, in one read. */
export interface PartyWorld {
  characters: PartyCharacter[];
  places: Place[];
  lore: LoreEntry[];
}

function parsePartyCharacter(data: unknown): PartyCharacter {
  if (!isRecord(data)) throw new Error("Expected a character object");
  return {
    id: expectInteger(data.id, "id"),
    name: expectString(data.name, "name"),
    appearance: expectString(data.appearance, "appearance"),
    personality: expectString(data.personality, "personality"),
    story: expectString(data.story, "story"),
    relationships: expectString(data.relationships, "relationships"),
    secrets: expectString(data.secrets, "secrets"),
  };
}

function parsePartyWorld(data: unknown): PartyWorld {
  if (!isRecord(data)) throw new Error("Expected a party world object");
  return {
    characters: expectArray(data.characters, "characters").map(parsePartyCharacter),
    places: expectArray(data.places, "places").map(parsePlace),
    lore: expectArray(data.lore, "lore").map(parseLoreEntry),
  };
}

export function getPartyWorld(partyId: string, signal?: AbortSignal): Promise<PartyWorld> {
  return request(`/parties/${partyId}/world`, parsePartyWorld, { signal });
}

export function createPartyCharacter(partyId: string, input: CharacterInput): Promise<PartyCharacter> {
  return request(`/parties/${partyId}/characters`, parsePartyCharacter, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updatePartyCharacter(
  partyId: string,
  characterId: number,
  input: CharacterInput,
): Promise<PartyCharacter> {
  return request(`/parties/${partyId}/characters/${characterId}`, parsePartyCharacter, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deletePartyCharacter(partyId: string, characterId: number): Promise<void> {
  return requestVoid(`/parties/${partyId}/characters/${characterId}`, { method: "DELETE" });
}
