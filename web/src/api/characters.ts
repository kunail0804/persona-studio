import { request, requestVoid } from "./client";
import { expectArray, expectInteger, expectString, isRecord } from "./validate";

export interface Character {
  id: number;
  scenarioId: string;
  position: number;
  name: string;
  appearance: string;
  personality: string;
  story: string;
  relationships: string;
  secrets: string;
}

export interface CharacterInput {
  name: string;
  appearance: string;
  personality: string;
  story: string;
  relationships: string;
  secrets: string;
}

function parseCharacter(data: unknown): Character {
  if (!isRecord(data)) throw new Error("Expected a character object");
  return {
    id: expectInteger(data.id, "id"),
    scenarioId: expectString(data.scenario_id, "scenario_id"),
    position: expectInteger(data.position, "position"),
    name: expectString(data.name, "name"),
    appearance: expectString(data.appearance, "appearance"),
    personality: expectString(data.personality, "personality"),
    story: expectString(data.story, "story"),
    relationships: expectString(data.relationships, "relationships"),
    secrets: expectString(data.secrets, "secrets"),
  };
}

function parseCharacterList(data: unknown): Character[] {
  return expectArray(data, "characters").map((item: unknown) => parseCharacter(item));
}

export function listCharacters(scenarioId: string): Promise<Character[]> {
  return request(`/scenarios/${scenarioId}/characters`, parseCharacterList);
}

export function createCharacter(scenarioId: string, input: CharacterInput): Promise<Character> {
  return request(`/scenarios/${scenarioId}/characters`, parseCharacter, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateCharacter(
  scenarioId: string,
  characterId: number,
  input: CharacterInput,
): Promise<Character> {
  return request(`/scenarios/${scenarioId}/characters/${characterId}`, parseCharacter, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deleteCharacter(scenarioId: string, characterId: number): Promise<void> {
  return requestVoid(`/scenarios/${scenarioId}/characters/${characterId}`, {
    method: "DELETE",
  });
}

export function reorderCharacters(scenarioId: string, order: number[]): Promise<Character[]> {
  return request(`/scenarios/${scenarioId}/characters/order`, parseCharacterList, {
    method: "PUT",
    body: JSON.stringify({ order }),
  });
}
