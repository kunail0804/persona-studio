import { request, requestVoid } from "./client";
import { expectArray, expectBoolean, expectNumber, expectString, isRecord } from "./validate";

export interface Persona {
  id: string;
  name: string;
  description: string;
  appearance: string;
  traits: string;
  createdAt: number;
  isActive: boolean;
}

export interface PersonaInput {
  name: string;
  description: string;
  appearance: string;
  traits: string;
}

function parsePersona(data: unknown): Persona {
  if (!isRecord(data)) throw new Error("Expected a persona object");
  return {
    id: expectString(data.id, "id"),
    name: expectString(data.name, "name"),
    description: expectString(data.description, "description"),
    appearance: expectString(data.appearance, "appearance"),
    traits: expectString(data.traits, "traits"),
    createdAt: expectNumber(data.created_at, "created_at"),
    isActive: expectBoolean(data.is_active, "is_active"),
  };
}

function parsePersonaList(data: unknown): Persona[] {
  return expectArray(data, "personas").map((item: unknown) => parsePersona(item));
}

export function listPersonas(signal?: AbortSignal): Promise<Persona[]> {
  return request("/personas", parsePersonaList, { signal });
}

export function createPersona(input: PersonaInput): Promise<Persona> {
  return request("/personas", parsePersona, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updatePersona(id: string, input: PersonaInput): Promise<Persona> {
  return request(`/personas/${id}`, parsePersona, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deletePersona(id: string): Promise<void> {
  return requestVoid(`/personas/${id}`, { method: "DELETE" });
}

export function setActivePersona(id: string): Promise<Persona> {
  return request("/personas/active", parsePersona, {
    method: "PUT",
    body: JSON.stringify({ id }),
  });
}
