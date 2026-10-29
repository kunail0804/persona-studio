import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import type { ServicesMemory } from "../api/services";
import {
  freeComfy,
  getServicesMemory,
  restartGateway,
  stopGateway,
  unloadOllama,
} from "../api/services";
import type { ServiceState } from "../api/status";
import { Button } from "./Button";
import { ConfirmDialog } from "./ConfirmDialog";

export type ServiceKind = "ollama" | "gateway";

interface ServiceMenuProps {
  kind: ServiceKind;
  label: string;
  state: ServiceState | null;
  /** Re-reads the pills once an action is done. */
  onChanged: () => void;
}

function pillColour(state: ServiceState | null): string {
  if (state === null) return "bg-neutral-600";
  return state.reachable ? "bg-emerald-500" : "bg-red-500";
}

function pillTitle(label: string, state: ServiceState | null): string {
  if (state === null) return `${label} : état inconnu`;
  if (state.reachable) return `${label} : joignable`;
  return state.detail ?? `${label} : injoignable`;
}

function gigabytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} Go`;
}

function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return "L'action a échoué.";
}

type Confirmable = "stop" | "restart";

/**
 * A pill in the header, and the menu it opens: what the service holds in
 * memory, and the buttons that replace the commands once typed to free it.
 * Ollama runs as a system service Persona Studio may not stop, so its button
 * unloads the models, which frees the same memory.
 */
export function ServiceMenu({ kind, label, state, onChanged }: ServiceMenuProps) {
  const [open, setOpen] = useState(false);
  const [memory, setMemory] = useState<ServicesMemory | null>(null);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null);
  const [confirming, setConfirming] = useState<Confirmable | null>(null);
  const root = useRef<HTMLDivElement>(null);

  const loadMemory = useCallback(async () => {
    try {
      setMemory(await getServicesMemory());
      setMemoryError(null);
    } catch (err) {
      setMemoryError(errorText(err));
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-open: loadMemory() sets state after an await, not synchronously in the effect body.
    void loadMemory();
    const onPointer = (event: MouseEvent) => {
      // A confirmation dialog is drawn outside the menu: a click in it must not close the menu.
      if (confirming !== null) return;
      if (root.current && event.target instanceof Node && !root.current.contains(event.target)) {
        setOpen(false);
      }
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && confirming === null) setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, confirming, loadMemory]);

  const run = async (name: string, action: () => Promise<string>) => {
    setBusyAction(name);
    setResult(null);
    try {
      setResult({ ok: true, text: await action() });
    } catch (err) {
      setResult({ ok: false, text: errorText(err) });
    } finally {
      setBusyAction(null);
      onChanged();
      void loadMemory();
    }
  };

  const confirmAction = () => {
    const which = confirming;
    setConfirming(null);
    if (which === "stop") void run("stop", stopGateway);
    if (which === "restart") void run("restart", restartGateway);
  };

  const reachable = state?.reachable ?? false;
  const disabled = busyAction !== null;

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        onClick={() => {
          setOpen((value) => !value);
          setResult(null);
          onChanged();
        }}
        title={pillTitle(label, state)}
        className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-100"
      >
        <span className={`h-2 w-2 rounded-full ${pillColour(state)}`} />
        {label}
      </button>

      {open && (
        <div className="absolute right-0 top-7 z-30 w-80 rounded-lg border border-neutral-700 bg-neutral-900 p-4 text-sm shadow-xl">
          <p className="font-medium text-neutral-100">{label}</p>
          <p className={`mt-1 text-xs ${reachable ? "text-emerald-400" : "text-red-400"}`}>
            {pillTitle(label, state)}
          </p>

          <div className="mt-3 space-y-1 text-xs text-neutral-400">
            {memoryError !== null && <p className="text-red-400">{memoryError}</p>}
            {memory !== null && kind === "ollama" && <OllamaMemoryView memory={memory} />}
            {memory !== null && kind === "gateway" && <GatewayMemoryView memory={memory} />}
          </div>

          <div className="mt-4 flex flex-col gap-2">
            {kind === "ollama" && (
              <Button
                variant="secondary"
                disabled={disabled}
                onClick={() => void run("unload", unloadOllama)}
              >
                {busyAction === "unload" ? "Déchargement…" : "Décharger les modèles"}
              </Button>
            )}
            {kind === "gateway" && (
              <>
                <Button
                  variant="secondary"
                  disabled={disabled || !reachable}
                  onClick={() => void run("free", freeComfy)}
                >
                  {busyAction === "free" ? "Vidage…" : "Vider ComfyUI"}
                </Button>
                <Button
                  variant="secondary"
                  disabled={disabled}
                  onClick={() => (reachable ? setConfirming("restart") : void run("restart", restartGateway))}
                >
                  {busyAction === "restart"
                    ? "Démarrage…"
                    : reachable
                      ? "Redémarrer le gateway"
                      : "Démarrer le gateway"}
                </Button>
                <Button variant="danger" disabled={disabled} onClick={() => setConfirming("stop")}>
                  {busyAction === "stop" ? "Arrêt…" : "Arrêter le gateway"}
                </Button>
              </>
            )}
          </div>

          {result !== null && (
            <p className={`mt-3 text-xs ${result.ok ? "text-emerald-400" : "text-red-400"}`}>{result.text}</p>
          )}
        </div>
      )}

      <ConfirmDialog
        open={confirming !== null}
        title={confirming === "stop" ? "Arrêter le gateway ?" : "Redémarrer le gateway ?"}
        description={
          confirming === "stop"
            ? "ComfyUI s'arrête avec lui, et un rendu en cours est perdu. Les images ne pourront plus être générées jusqu'à ce que tu le redémarres."
            : "ComfyUI redémarre avec lui, et un rendu en cours est perdu. Il faut compter une minute avant de pouvoir générer à nouveau."
        }
        confirmLabel={confirming === "stop" ? "Arrêter" : "Redémarrer"}
        onConfirm={confirmAction}
        onCancel={() => setConfirming(null)}
      />
    </div>
  );
}

function OllamaMemoryView({ memory }: { memory: ServicesMemory }) {
  const { models, error } = memory.ollama;
  if (error !== null) return <p className="text-red-400">{error}</p>;
  if (models.length === 0) return <p>Aucun modèle chargé.</p>;
  return (
    <>
      <p>Modèles chargés :</p>
      <ul className="ml-3 list-disc">
        {models.map((model) => (
          <li key={model.name}>
            {model.name} · {gigabytes(model.sizeVram)} en VRAM
          </li>
        ))}
      </ul>
    </>
  );
}

function GatewayMemoryView({ memory }: { memory: ServicesMemory }) {
  const { memory: comfy, busy, error } = memory.gateway;
  if (error !== null) return <p className="text-red-400">{error}</p>;
  return (
    <>
      {busy !== null && <p className="text-amber-400">{busy}</p>}
      {comfy !== null && (
        <>
          <p>
            VRAM libre : {gigabytes(comfy.vramFree)} / {gigabytes(comfy.vramTotal)}
          </p>
          <p>
            RAM libre : {gigabytes(comfy.ramFree)} / {gigabytes(comfy.ramTotal)}
          </p>
        </>
      )}
    </>
  );
}
