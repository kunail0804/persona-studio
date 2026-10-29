import { request } from "./client";
import { expectArray, expectString, isRecord } from "./validate";

/**
 * Which comfy-gateway template images are made with, and its preset. The
 * workflows, their settings and their presets live in comfy-gateway; this
 * application only names them.
 */
export interface ImageGeneration {
  template: string | null;
  preset: string | null;
  /** The templates the gateway offers; empty when it cannot be reached. */
  templates: string[];
  /** The presets of the chosen template. */
  presets: string[];
  /** Why no image can be made with this setup, or null when it is ready. */
  problem: string | null;
  /** Where the templates and presets are edited. */
  gatewayUrl: string;
}

export interface ImageGenerationInput {
  template: string | null;
  preset: string | null;
}

function nullableString(value: unknown, field: string): string | null {
  return value === null ? null : expectString(value, field);
}

function parseImageGeneration(data: unknown): ImageGeneration {
  if (!isRecord(data)) throw new Error("Expected an image generation settings object");
  return {
    template: nullableString(data.template, "template"),
    preset: nullableString(data.preset, "preset"),
    templates: expectArray(data.templates, "templates").map((t) => expectString(t, "templates")),
    presets: expectArray(data.presets, "presets").map((p) => expectString(p, "presets")),
    problem: nullableString(data.problem, "problem"),
    gatewayUrl: expectString(data.gateway_url, "gateway_url"),
  };
}

export function getImageGeneration(signal?: AbortSignal): Promise<ImageGeneration> {
  return request("/image-generation", parseImageGeneration, { signal });
}

export function setImageGeneration(input: ImageGenerationInput): Promise<ImageGeneration> {
  return request("/image-generation", parseImageGeneration, {
    method: "PUT",
    body: JSON.stringify(input),
  });
}
