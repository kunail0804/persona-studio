import { request } from "./client";
import { expectBoolean, expectString, isRecord } from "./validate";

/** One dependency: whether it answered, and why it did not. */
export interface ServiceState {
  reachable: boolean;
  detail: string | null;
}

export interface ServiceStatus {
  ollama: ServiceState;
  comfyui: ServiceState;
}

function parseServiceState(data: unknown, field: string): ServiceState {
  if (!isRecord(data)) throw new Error(`Expected a service state object for "${field}"`);
  return {
    reachable: expectBoolean(data.reachable, "reachable"),
    detail: data.detail === null || data.detail === undefined ? null : expectString(data.detail, "detail"),
  };
}

export function getServiceStatus(signal?: AbortSignal): Promise<ServiceStatus> {
  return request(
    "/status",
    (data: unknown) => {
      if (!isRecord(data)) throw new Error("Expected a service status object");
      return {
        ollama: parseServiceState(data.ollama, "ollama"),
        comfyui: parseServiceState(data.comfyui, "comfyui"),
      };
    },
    { signal },
  );
}
