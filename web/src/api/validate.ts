// Narrowing helpers for turning an API response body (`unknown`) into a typed
// value without `any` or `as`. Each `expect*` throws with the offending field
// name, which is enough to debug a shape mismatch during development.

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function expectRecord(value: unknown, field: string): Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) {
    throw new Error(`Expected an object for "${field}"`);
  }
  return value;
}

export function expectString(value: unknown, field: string): string {
  if (typeof value !== "string") {
    throw new Error(`Expected a string for "${field}"`);
  }
  return value;
}

export function expectNumber(value: unknown, field: string): number {
  if (typeof value !== "number") {
    throw new Error(`Expected a number for "${field}"`);
  }
  return value;
}

export function expectInteger(value: unknown, field: string): number {
  const num = expectNumber(value, field);
  if (!Number.isInteger(num)) {
    throw new Error(`Expected an integer for "${field}"`);
  }
  return num;
}

export function expectBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`Expected a boolean for "${field}"`);
  }
  return value;
}

export function expectArray(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`Expected an array for "${field}"`);
  }
  return value.map((item: unknown) => item);
}
