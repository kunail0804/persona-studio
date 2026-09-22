import { request, requestVoid } from "./client";
import { expectArray, expectBoolean, expectNumber, expectString, isRecord } from "./validate";

/**
 * A master prompt for the image-prompt composition — not for the narrator.
 * The built-in text is Stable Diffusion's grammar ("a short comma-separated
 * list of English keywords"); an image model that wants prose wants another.
 */
export interface ImagePreset {
  id: string;
  name: string;
  instruction: string;
  createdAt: number;
  isActive: boolean;
}

export interface ImagePresetInput {
  name: string;
  instruction: string;
}

function parseImagePreset(data: unknown): ImagePreset {
  if (!isRecord(data)) throw new Error("Expected an image preset object");
  return {
    id: expectString(data.id, "id"),
    name: expectString(data.name, "name"),
    instruction: expectString(data.instruction, "instruction"),
    createdAt: expectNumber(data.created_at, "created_at"),
    isActive: expectBoolean(data.is_active, "is_active"),
  };
}

function parseImagePresetList(data: unknown): ImagePreset[] {
  return expectArray(data, "image presets").map(parseImagePreset);
}

export function listImagePresets(signal?: AbortSignal): Promise<ImagePreset[]> {
  return request("/image-presets", parseImagePresetList, { signal });
}

/** The built-in instruction, so the editor starts from it rather than blank. */
export function getDefaultInstruction(signal?: AbortSignal): Promise<string> {
  return request(
    "/image-presets/default",
    (data: unknown) => {
      if (!isRecord(data)) throw new Error("Expected a default instruction object");
      return expectString(data.instruction, "instruction");
    },
    { signal },
  );
}

export function createImagePreset(input: ImagePresetInput): Promise<ImagePreset> {
  return request("/image-presets", parseImagePreset, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateImagePreset(
  id: string,
  input: Partial<ImagePresetInput>,
): Promise<ImagePreset> {
  return request(`/image-presets/${id}`, parseImagePreset, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

/** `null` goes back to the built-in instruction, which is a choice of its own. */
export function setActiveImagePreset(id: string | null): Promise<ImagePreset[]> {
  return request("/image-presets/active", parseImagePresetList, {
    method: "PUT",
    body: JSON.stringify({ id }),
  });
}

export function deleteImagePreset(id: string): Promise<void> {
  return requestVoid(`/image-presets/${id}`, { method: "DELETE" });
}
