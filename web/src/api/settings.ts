import { request } from "./client";
import { expectArray, expectBoolean, expectInteger, expectString, isRecord } from "./validate";

export interface LlmSettings {
  model: string | null;
  numCtx: number;
  /** Bounds the backend enforces on write; the page reads them, never copies them. */
  minNumCtx: number;
  maxNumCtx: number;
  /** null when Ollama could not be asked — "not installed" is then unknown. */
  installedModels: string[] | null;
  modelMissing: boolean | null;
  ollamaError: string | null;
}

export interface LlmSettingsInput {
  model: string | null;
  numCtx: number;
}

function parseLlmSettings(data: unknown): LlmSettings {
  if (!isRecord(data)) throw new Error("Expected an LLM settings object");
  const installed = data.installed_models;
  return {
    model: data.model === null ? null : expectString(data.model, "model"),
    numCtx: expectInteger(data.num_ctx, "num_ctx"),
    minNumCtx: expectInteger(data.min_num_ctx, "min_num_ctx"),
    maxNumCtx: expectInteger(data.max_num_ctx, "max_num_ctx"),
    installedModels:
      installed === null
        ? null
        : expectArray(installed, "installed_models").map((name: unknown) =>
            expectString(name, "installed_models"),
          ),
    modelMissing:
      data.model_missing === null ? null : expectBoolean(data.model_missing, "model_missing"),
    ollamaError: data.ollama_error === null ? null : expectString(data.ollama_error, "ollama_error"),
  };
}

export function getLlmSettings(signal?: AbortSignal): Promise<LlmSettings> {
  return request("/settings/llm", parseLlmSettings, { signal });
}

export function updateLlmSettings(input: LlmSettingsInput): Promise<LlmSettings> {
  return request("/settings/llm", parseLlmSettings, {
    method: "PUT",
    body: JSON.stringify({ model: input.model, num_ctx: input.numCtx }),
  });
}
