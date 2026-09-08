import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import type { Persona, PersonaInput } from "../api/personas";
import { createPersona, deletePersona, listPersonas, setActivePersona, updatePersona } from "../api/personas";
import type { LlmSettings } from "../api/settings";
import { getLlmSettings, updateLlmSettings } from "../api/settings";
import type { Workflow } from "../api/workflows";
import { deleteWorkflow, importWorkflow, listWorkflows, setActiveWorkflow, updateWorkflow } from "../api/workflows";
import { Button } from "../components/Button";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { PersonaForm } from "../components/PersonaForm";
import { TextField } from "../components/TextField";
import { isRecord } from "../api/validate";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

function parseBoundedInt(value: string, min: number, max: number): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= min && parsed <= max ? parsed : null;
}

function optionValue(node: string, field: string): string {
  return `${node}::${field}`;
}

export function SettingsPage() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [llm, setLlm] = useState<LlmSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [addingPersona, setAddingPersona] = useState(false);
  const [editingPersonaId, setEditingPersonaId] = useState<string | null>(null);
  const [confirmDeletePersonaId, setConfirmDeletePersonaId] = useState<string | null>(null);
  const [confirmDeleteWorkflowId, setConfirmDeleteWorkflowId] = useState<string | null>(null);
  const [importingWorkflow, setImportingWorkflow] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [model, setModel] = useState<string | null>(null);
  const [numCtx, setNumCtx] = useState("");
  const [historyWindow, setHistoryWindow] = useState("");
  const [savingLlm, setSavingLlm] = useState(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [personaData, llmData, workflowData] = await Promise.all([
        listPersonas(signal),
        getLlmSettings(signal),
        listWorkflows(signal),
      ]);
      setPersonas(personaData);
      setWorkflows(workflowData);
      setLlm(llmData);
      setModel(llmData.model);
      setNumCtx(String(llmData.numCtx));
      setHistoryWindow(String(llmData.historyWindow));
      setError(null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(messageFor(err, "Impossible de charger les paramètres."));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: load() sets state after an await, not synchronously in the effect body.
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  async function reload() {
    await load();
  }

  async function handleCreatePersona(input: PersonaInput) {
    try {
      await createPersona(input);
      setAddingPersona(false);
      await reload();
    } catch (err) {
      setError(messageFor(err, "Impossible de créer la persona."));
    }
  }

  async function handleUpdatePersona(personaId: string, input: PersonaInput) {
    try {
      await updatePersona(personaId, input);
      setEditingPersonaId(null);
      await reload();
    } catch (err) {
      setError(messageFor(err, "Impossible de modifier la persona."));
    }
  }

  async function handleDeletePersona(personaId: string) {
    try {
      await deletePersona(personaId);
      await reload();
    } catch (err) {
      setError(messageFor(err, "Impossible de supprimer la persona."));
    } finally {
      setConfirmDeletePersonaId(null);
    }
  }

  async function handleActivatePersona(personaId: string) {
    try {
      await setActivePersona(personaId);
      await reload();
    } catch (err) {
      setError(messageFor(err, "Impossible de définir la persona active."));
    }
  }

  async function handleImportWorkflow(file: File) {
    const name = file.name.replace(/\.json$/i, "") || "Workflow";
    let parsed: unknown;
    try {
      parsed = JSON.parse(await file.text());
    } catch {
      setError(`« ${file.name} » n'est pas un fichier JSON valide.`);
      return;
    }
    if (!isRecord(parsed) || Array.isArray(parsed)) {
      setError(
        `« ${file.name} » n'est pas un export de workflow : il faut un objet JSON, l'export « Export (API) » de ComfyUI.`,
      );
      return;
    }
    try {
      setImportingWorkflow(true);
      await importWorkflow({ name, graph: parsed });
      await reload();
      setError(null);
    } catch (err) {
      setError(messageFor(err, `Impossible d'importer « ${file.name} ».`));
    } finally {
      setImportingWorkflow(false);
      // Reset the input so re-selecting the same file fires a change event.
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleUpdateWorkflow(workflowId: string, patch: {
    name: string;
    promptNode: string;
    promptField: string;
    seedNode: string;
    seedField: string;
  }) {
    try {
      await updateWorkflow(workflowId, patch);
      await reload();
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible d'enregistrer la configuration du workflow."));
    }
  }

  async function handleDeleteWorkflow(workflowId: string) {
    try {
      await deleteWorkflow(workflowId);
      await reload();
    } catch (err) {
      setError(messageFor(err, "Impossible de supprimer le workflow."));
    } finally {
      setConfirmDeleteWorkflowId(null);
    }
  }

  async function handleActivateWorkflow(workflowId: string) {
    try {
      await setActiveWorkflow(workflowId);
      await reload();
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible de définir le workflow actif."));
    }
  }

  async function handleSaveLlm() {
    if (!llm) return;
    const parsedNumCtx = parseBoundedInt(numCtx, llm.minNumCtx, llm.maxNumCtx);
    const parsedHistoryWindow = parseBoundedInt(
      historyWindow,
      llm.minHistoryWindow,
      llm.maxHistoryWindow,
    );
    if (parsedNumCtx === null || parsedHistoryWindow === null) {
      setError(
        `La fenêtre de contexte doit être un nombre entier entre ${llm.minNumCtx} et ${llm.maxNumCtx.toLocaleString("fr-FR")}, et la fenêtre d'historique un nombre entier entre ${llm.minHistoryWindow} et ${llm.maxHistoryWindow.toLocaleString("fr-FR")}.`,
      );
      return;
    }
    try {
      setSavingLlm(true);
      const updated = await updateLlmSettings({
        model,
        numCtx: parsedNumCtx,
        historyWindow: parsedHistoryWindow,
      });
      setLlm(updated);
      setModel(updated.model);
      setNumCtx(String(updated.numCtx));
      setHistoryWindow(String(updated.historyWindow));
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible d'enregistrer les paramètres."));
    } finally {
      setSavingLlm(false);
    }
  }

  if (!llm) {
    return <p className="text-neutral-500">{error ?? "Chargement…"}</p>;
  }

  return (
    <div className="flex flex-col gap-8">
      {error ? <p className="text-sm text-red-400">{error}</p> : null}

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Personas</h1>
          {!addingPersona ? (
            <Button onClick={() => setAddingPersona(true)}>Nouvelle persona</Button>
          ) : null}
        </div>

        {addingPersona ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            <PersonaForm
              submitLabel="Créer"
              onSubmit={handleCreatePersona}
              onCancel={() => setAddingPersona(false)}
            />
          </div>
        ) : null}

        {personas.length === 0 ? (
          <p className="text-neutral-500">
            Aucune persona : créez le personnage que vous jouez dans chaque histoire.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {personas.map((persona) => (
              <PersonaListItem
                key={persona.id}
                persona={persona}
                isEditing={editingPersonaId === persona.id}
                onEdit={() => setEditingPersonaId(persona.id)}
                onCancelEdit={() => setEditingPersonaId(null)}
                onSave={(input) => handleUpdatePersona(persona.id, input)}
                onDelete={() => setConfirmDeletePersonaId(persona.id)}
                onActivate={() => void handleActivatePersona(persona.id)}
              />
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Workflow ComfyUI</h2>
          <label className="inline-flex cursor-pointer items-center rounded bg-sky-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500">
            {importingWorkflow ? "Import…" : "Importer un workflow"}
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              className="hidden"
              disabled={importingWorkflow}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleImportWorkflow(file);
              }}
            />
          </label>
        </div>
        <p className="text-sm text-neutral-500">
          Importez l'export <span className="font-medium">Export (API)</span> de ComfyUI, puis
          choisissez dans quel nœud et quel champ le prompt est injecté.
        </p>

        {workflows.length === 0 ? (
          <p className="text-neutral-500">
            Aucun workflow : importez-en un pour que la génération d'images puisse fonctionner.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {workflows.map((workflow) => (
              <WorkflowListItem
                key={workflow.id}
                workflow={workflow}
                onSelect={() => void handleActivateWorkflow(workflow.id)}
                onPatch={(patch) => void handleUpdateWorkflow(workflow.id, patch)}
                onDelete={() => setConfirmDeleteWorkflowId(workflow.id)}
              />
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-semibold">Modèle de narration</h2>

        {llm.ollamaError ? <p className="text-sm text-red-400">{llm.ollamaError}</p> : null}
        {llm.modelMissing ? (
          <p className="text-sm text-amber-400">
            Le modèle configuré n'est plus installé dans Ollama. Choisissez-en un autre ci-dessous
            ou réinstallez-le.
          </p>
        ) : null}

        <label className="flex flex-col gap-1 text-sm text-neutral-300">
          <span className="font-medium">Modèle</span>
          <select
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100 focus:border-sky-500 focus:outline-none"
            value={model ?? ""}
            onChange={(event) => setModel(event.target.value === "" ? null : event.target.value)}
          >
            <option value="">Aucun modèle</option>
            {llm.installedModels?.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
            {llm.model && !llm.installedModels?.includes(llm.model) ? (
              <option value={llm.model}>
                {llm.modelMissing ? `${llm.model} (non installé)` : llm.model}
              </option>
            ) : null}
          </select>
        </label>

        <TextField
          label="Fenêtre de contexte (tokens)"
          hint="Envoyée à Ollama à chaque appel. Trop petite, le début du prompt — le système — est tronqué silencieusement."
          type="number"
          value={numCtx}
          onChange={(event) => setNumCtx(event.target.value)}
        />

        <TextField
          label="Fenêtre d'historique (messages)"
          hint="Seuls les messages les plus récents sont envoyés au narrateur à chaque tour."
          type="number"
          value={historyWindow}
          onChange={(event) => setHistoryWindow(event.target.value)}
        />

        <div className="flex justify-end">
          <Button onClick={() => void handleSaveLlm()} disabled={savingLlm}>
            Enregistrer
          </Button>
        </div>
      </section>

      <ConfirmDialog
        open={confirmDeletePersonaId !== null}
        title="Supprimer cette persona ?"
        description="Cette action est définitive. Si elle était active, plus aucune persona ne le sera."
        onConfirm={() => {
          if (confirmDeletePersonaId !== null) void handleDeletePersona(confirmDeletePersonaId);
        }}
        onCancel={() => setConfirmDeletePersonaId(null)}
      />

      <ConfirmDialog
        open={confirmDeleteWorkflowId !== null}
        title="Supprimer ce workflow ?"
        description="Cette action est définitive. S'il était actif, un autre workflow valide devient actif."
        onConfirm={() => {
          if (confirmDeleteWorkflowId !== null) void handleDeleteWorkflow(confirmDeleteWorkflowId);
        }}
        onCancel={() => setConfirmDeleteWorkflowId(null)}
      />
    </div>
  );
}

interface WorkflowListItemProps {
  workflow: Workflow;
  onSelect: () => void;
  onPatch: (patch: { name: string; promptNode: string; promptField: string; seedNode: string; seedField: string }) => void;
  onDelete: () => void;
}

const selectClasses =
  "rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100 focus:border-sky-500 focus:outline-none";

function WorkflowListItem({ workflow, onSelect, onPatch, onDelete }: WorkflowListItemProps) {
  const promptValue =
    workflow.promptNode && workflow.promptField
      ? optionValue(workflow.promptNode, workflow.promptField)
      : "";
  const seedValue =
    workflow.seedNode && workflow.seedField ? optionValue(workflow.seedNode, workflow.seedField) : "";

  const patch = (promptNode: string, promptField: string, seedNode: string, seedField: string) =>
    onPatch({ name: workflow.name, promptNode, promptField, seedNode, seedField });

  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2">
          <h3 className="font-medium">{workflow.name || "Sans nom"}</h3>
          {workflow.isActive ? (
            <span className="rounded bg-sky-900 px-2 py-0.5 text-xs text-sky-300">Actif</span>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {!workflow.isActive ? (
            <Button variant="secondary" onClick={onSelect}>
              Activer
            </Button>
          ) : null}
          <Button variant="danger" onClick={onDelete}>
            Supprimer
          </Button>
        </div>
      </div>

      {workflow.problem ? <p className="mt-2 text-sm text-amber-400">{workflow.problem}</p> : null}

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-neutral-300">
          <span className="font-medium">Champ du prompt</span>
          <select
            className={selectClasses}
            value={promptValue}
            onChange={(event) => {
              if (event.target.value === "") {
                patch("", "", workflow.seedNode, workflow.seedField);
                return;
              }
              const option = workflow.promptOptions.find(
                (o) => optionValue(o.node, o.field) === event.target.value,
              );
              if (option) patch(option.node, option.field, workflow.seedNode, workflow.seedField);
            }}
          >
            <option value="">Non configuré</option>
            {workflow.promptOptions.map((o) => (
              <option key={optionValue(o.node, o.field)} value={optionValue(o.node, o.field)}>
                {o.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm text-neutral-300">
          <span className="font-medium">Champ de la graine</span>
          <select
            className={selectClasses}
            value={seedValue}
            onChange={(event) => {
              if (event.target.value === "") {
                patch(workflow.promptNode, workflow.promptField, "", "");
                return;
              }
              const option = workflow.seedOptions.find(
                (o) => optionValue(o.node, o.field) === event.target.value,
              );
              if (option) patch(workflow.promptNode, workflow.promptField, option.node, option.field);
            }}
          >
            <option value="">Aucune (graine aléatoire)</option>
            {workflow.seedOptions.map((o) => (
              <option key={optionValue(o.node, o.field)} value={optionValue(o.node, o.field)}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </li>
  );
}

interface PersonaListItemProps {
  persona: Persona;
  isEditing: boolean;
  onEdit: () => void;
  onCancelEdit: () => void;
  onSave: (input: PersonaInput) => Promise<void>;
  onDelete: () => void;
  onActivate: () => void;
}

function PersonaListItem({
  persona,
  isEditing,
  onEdit,
  onCancelEdit,
  onSave,
  onDelete,
  onActivate,
}: PersonaListItemProps) {
  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      {isEditing ? (
        <PersonaForm initial={persona} submitLabel="Enregistrer" onSubmit={onSave} onCancel={onCancelEdit} />
      ) : (
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-medium">{persona.name || "Sans nom"}</h3>
              {persona.isActive ? (
                <span className="rounded bg-sky-900 px-2 py-0.5 text-xs text-sky-300">Active</span>
              ) : null}
            </div>
            {persona.description ? (
              <p className="mt-1 line-clamp-2 text-sm text-neutral-400">{persona.description}</p>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {!persona.isActive ? (
              <Button variant="secondary" onClick={onActivate}>
                Activer
              </Button>
            ) : null}
            <Button variant="secondary" onClick={onEdit}>
              Modifier
            </Button>
            <Button variant="danger" onClick={onDelete}>
              Supprimer
            </Button>
          </div>
        </div>
      )}
    </li>
  );
}
