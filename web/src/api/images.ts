import { request, requestVoid } from "./client";
import { expectNumber, expectInteger, expectString, isRecord } from "./validate";
import { parsePartyMessage } from "./parties";
import type { PartyMessage } from "./parties";

export interface ImagePrompt {
  prompt: string;
}

export interface ImageGeneration {
  /** The message whose row now shows the pending image; cancel targets it. */
  messageId: number;
  /** When the generation started; the elapsed time is computed from it. */
  startedAt: number;
}

function parseImagePrompt(data: unknown): ImagePrompt {
  if (!isRecord(data)) throw new Error("Expected an image prompt object");
  return { prompt: expectString(data.prompt, "prompt") };
}

function parseImageGeneration(data: unknown): ImageGeneration {
  if (!isRecord(data)) throw new Error("Expected an image generation object");
  return {
    messageId: expectInteger(data.message_id, "message_id"),
    startedAt: expectNumber(data.started_at, "started_at"),
  };
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

/**
 * Sends the final prompt — exactly the text the player has on screen — to
 * ComfyUI and returns at once: the render runs in the background and lands
 * in the story as an image message the player can cancel.
 */
export function startImageGeneration(
  partyId: string,
  prompt: string,
  instruction: string,
): Promise<ImageGeneration> {
  return request(`/parties/${partyId}/images`, parseImageGeneration, {
    method: "POST",
    body: JSON.stringify({ prompt, instruction }),
  });
}

/** Cancels a pending generation, interrupting ComfyUI only if it is rendering. */
export function cancelImageGeneration(partyId: string, messageId: number): Promise<PartyMessage> {
  return request(`/parties/${partyId}/images/${messageId}/cancel`, parsePartyMessage, {
    method: "POST",
  });
}
/**
 * Removes an image from the party: its message, its row and its PNG. A
 * generation still running is refused — cancel it first, so its watcher is
 * never left writing to a row that no longer exists.
 */
export function deleteImageMessage(partyId: string, messageId: number): Promise<void> {
  return requestVoid(`/parties/${partyId}/images/${messageId}`, { method: "DELETE" });
}
