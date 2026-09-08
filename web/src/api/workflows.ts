import { request, requestVoid } from "./client";
import { expectArray, expectBoolean, expectNumber, expectString, isRecord } from "./validate";

export interface WorkflowFieldOption {
  node: string;
  field: string;
  label: string;
}

export interface Workflow {
  id: string;
  name: string;
  isActive: boolean;
  promptNode: string;
  promptField: string;
  seedNode: string;
  seedField: string;
  promptOptions: WorkflowFieldOption[];
  seedOptions: WorkflowFieldOption[];
  /** Why nothing can be generated with this workflow, or null when it is ready. */
  problem: string | null;
  createdAt: number;
}

export interface WorkflowInput {
  name: string;
  graph: Record<string, unknown>;
}

export interface WorkflowPatch {
  name: string;
  promptNode: string;
  promptField: string;
  seedNode: string;
  seedField: string;
}

function parseFieldOption(data: unknown): WorkflowFieldOption {
  if (!isRecord(data)) throw new Error("Expected a field option object");
  return {
    node: expectString(data.node, "node"),
    field: expectString(data.field, "field"),
    label: expectString(data.label, "label"),
  };
}

function parseWorkflow(data: unknown): Workflow {
  if (!isRecord(data)) throw new Error("Expected a workflow object");
  return {
    id: expectString(data.id, "id"),
    name: expectString(data.name, "name"),
    isActive: expectBoolean(data.is_active, "is_active"),
    promptNode: expectString(data.prompt_node, "prompt_node"),
    promptField: expectString(data.prompt_field, "prompt_field"),
    seedNode: expectString(data.seed_node, "seed_node"),
    seedField: expectString(data.seed_field, "seed_field"),
    promptOptions: expectArray(data.prompt_options, "prompt_options").map(parseFieldOption),
    seedOptions: expectArray(data.seed_options, "seed_options").map(parseFieldOption),
    problem: data.problem === null ? null : expectString(data.problem, "problem"),
    createdAt: expectNumber(data.created_at, "created_at"),
  };
}

function parseWorkflowList(data: unknown): Workflow[] {
  return expectArray(data, "workflows").map(parseWorkflow);
}

export function listWorkflows(signal?: AbortSignal): Promise<Workflow[]> {
  return request("/workflows", parseWorkflowList, { signal });
}

export function importWorkflow(input: WorkflowInput): Promise<Workflow> {
  return request("/workflows", parseWorkflow, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateWorkflow(id: string, input: WorkflowPatch): Promise<Workflow> {
  return request(`/workflows/${id}`, parseWorkflow, {
    method: "PATCH",
    body: JSON.stringify({
      name: input.name,
      prompt_node: input.promptNode,
      prompt_field: input.promptField,
      seed_node: input.seedNode,
      seed_field: input.seedField,
    }),
  });
}

export function deleteWorkflow(id: string): Promise<void> {
  return requestVoid(`/workflows/${id}`, { method: "DELETE" });
}

export function setActiveWorkflow(id: string): Promise<Workflow> {
  return request("/workflows/active", parseWorkflow, {
    method: "PUT",
    body: JSON.stringify({ id }),
  });
}
