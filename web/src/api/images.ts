import { request } from "./client";
import { expectString, isRecord } from "./validate";

export interface ImagePrompt {
  prompt: string;
}

function parseImagePrompt(data: unknown): ImagePrompt {
  if (!isRecord(data)) throw new Error("Expected an image prompt object");
  return { prompt: expectString(data.prompt, "prompt") };
}

/**
 * Composes an editable English keyword prompt from what the player wants to
 * see plus the current scene. Nothing is generated here: the text comes back
 * for the player to read and edit.
 */
export function composeImagePrompt(partyId: string, instruction: string): Promise<ImagePrompt> {
  return request(`/parties/${partyId}/image-prompt`, parseImagePrompt, {
    method: "POST",
    body: JSON.stringify({ instruction }),
  });
}
