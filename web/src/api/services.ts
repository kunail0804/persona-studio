import { request } from "./client";
import { expectArray, expectInteger, expectRecord, expectString } from "./validate";

export interface LoadedModel {
  name: string;
  sizeVram: number;
}

export interface ComfyMemory {
  ramTotal: number;
  ramFree: number;
  vramTotal: number;
  vramFree: number;
}

export interface ServicesMemory {
  ollama: { models: LoadedModel[]; error: string | null };
  gateway: { memory: ComfyMemory | null; busy: string | null; error: string | null };
}

function optionalString(value: unknown, field: string): string | null {
  return value === null || value === undefined ? null : expectString(value, field);
}

function parseLoadedModel(data: unknown): LoadedModel {
  const record = expectRecord(data, "model");
  return {
    name: expectString(record.name, "name"),
    sizeVram: expectInteger(record.size_vram, "size_vram"),
  };
}

function parseComfyMemory(data: unknown): ComfyMemory | null {
  if (data === null || data === undefined) return null;
  const record = expectRecord(data, "memory");
  return {
    ramTotal: expectInteger(record.ram_total, "ram_total"),
    ramFree: expectInteger(record.ram_free, "ram_free"),
    vramTotal: expectInteger(record.vram_total, "vram_total"),
    vramFree: expectInteger(record.vram_free, "vram_free"),
  };
}

function parseServicesMemory(data: unknown): ServicesMemory {
  const record = expectRecord(data, "services memory");
  const ollama = expectRecord(record.ollama, "ollama");
  const gateway = expectRecord(record.gateway, "gateway");
  return {
    ollama: {
      models: expectArray(ollama.models, "models").map(parseLoadedModel),
      error: optionalString(ollama.error, "error"),
    },
    gateway: {
      memory: parseComfyMemory(gateway.memory),
      busy: optionalString(gateway.busy, "busy"),
      error: optionalString(gateway.error, "error"),
    },
  };
}

export function getServicesMemory(signal?: AbortSignal): Promise<ServicesMemory> {
  return request("/services/memory", parseServicesMemory, { signal });
}

function parseMessage(data: unknown): string {
  return expectString(expectRecord(data, "action").message, "message");
}

function post(path: string): Promise<string> {
  return request(path, parseMessage, { method: "POST" });
}

export const unloadOllama = (): Promise<string> => post("/services/ollama/unload");
export const freeComfy = (): Promise<string> => post("/services/gateway/free");
export const stopGateway = (): Promise<string> => post("/services/gateway/stop");
export const restartGateway = (): Promise<string> => post("/services/gateway/restart");
