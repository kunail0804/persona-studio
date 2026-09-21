import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import {
  editMessage,
  getParty,
  regenerateMessage,
  sendTurn,
  setMessageVariant,
} from "../api/parties";
import type { Party, PartyMessage, TurnEvent } from "../api/parties";
import { cancelImageGeneration } from "../api/images";
import { Narration } from "../components/Narration";
import { Button } from "../components/Button";
import { TextArea } from "../components/TextArea";
import { ImagePanel } from "../components/ImagePanel";
import { ImageMessage } from "../components/ImageMessage";

// The bubble re-renders at most this often while fragments arrive. Every
// fragment is still accumulated; only the re-render is throttled. The
// previous version needed exactly this: without it the Markdown bubble
// visibly beat while reparsing itself (see issue #10).
const RENDER_INTERVAL_MS = 150;

// After a stop, the server persists the partial when it notices the
// hang-up, which lands slightly after the abort resolves here; the reload
// waits that long before reading the database.
const RELOAD_AFTER_STOP_MS = 500;

// While an image is rendering, the party endpoint is the progress channel:
// the page re-reads it on this cadence and stops when nothing is pending.
const IMAGE_POLL_INTERVAL_MS = 3000;

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

function replaceMessage(party: Party, messageId: number, updated: PartyMessage): Party {
  return {
    ...party,
    messages: party.messages.map((m) => (m.id === messageId ? updated : m)),
  };
}

export function PartyPage() {
  const { id } = useParams<{ id: string }>();

  const [party, setParty] = useState<Party | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The turn being played: the optimistic player bubble, the growing reply,
  // and why the stream broke, if it broke. `regeneratingId` marks the
  // message a regeneration is replacing instead of appending a new bubble.
  const [input, setInput] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [streamText, setStreamText] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [regeneratingId, setRegeneratingId] = useState<number | null>(null);

  const load = useCallback(async (partyId: string, signal?: AbortSignal): Promise<boolean> => {
    try {
      setParty(await getParty(partyId, signal));
      setError(null);
      return true;
    } catch (err) {
      // A stale request aborted by the effect cleanup below, because `id`
      // changed again before it resolved — not a real failure to report.
      if (err instanceof DOMException && err.name === "AbortError") return false;
      setError(messageFor(err, "Impossible de charger la partie."));
      return false;
    }
  }, []);

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: load() sets state after an await, not synchronously in the effect body.
    void load(id, controller.signal);
    return () => controller.abort();
  }, [id, load]);

  // Reopen at the last message: once the transcript is on screen, scroll to
  // its end. Runs once per loaded party, not on every render.
  const endRef = useRef<HTMLDivElement | null>(null);
  const partyId = party?.id;
  useEffect(() => {
    if (!partyId) return;
    endRef.current?.scrollIntoView();
  }, [partyId]);

  // Keep the growing reply in view while a turn streams. A regeneration
  // replaces its bubble in place — wherever that sits in the transcript — so
  // the view must not jump to the end while it runs.
  useEffect(() => {
    if (streamText !== null && regeneratingId === null) endRef.current?.scrollIntoView();
  }, [streamText, regeneratingId]);

  // The reply accumulates in a ref — the state only mirrors it for rendering,
  // throttled to one pass per RENDER_INTERVAL_MS.
  const bufferRef = useRef("");
  const lastRenderRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const renderNow = useCallback(() => {
    timerRef.current = null;
    lastRenderRef.current = Date.now();
    setStreamText(bufferRef.current);
  }, []);

  const onDelta = useCallback(
    (text: string) => {
      bufferRef.current += text;
      if (timerRef.current !== null) return;
      const elapsed = Date.now() - lastRenderRef.current;
      if (elapsed >= RENDER_INTERVAL_MS) {
        renderNow();
      } else {
        timerRef.current = window.setTimeout(renderNow, RENDER_INTERVAL_MS - elapsed);
      }
    },
    [renderNow],
  );

  const onEvent = useCallback(
    (event: TurnEvent) => {
      if (event.kind === "delta") {
        onDelta(event.text);
      } else if (event.kind === "error") {
        setStreamError(event.message);
      }
      // "done": the reload below reconciles the transcript with the server's
      // saved state, ids included; nothing to render from this line.
    },
    [onDelta],
  );

  // Both run before the early returns below: hooks must not be conditional.
  // They rethrow so a caller (the message editor) can keep its draft.
  const saveEdit = useCallback(
    async (messageId: number, content: string) => {
      if (!id) return;
      try {
        const updated = await editMessage(id, messageId, content);
        setParty((current) => (current ? replaceMessage(current, messageId, updated) : current));
      } catch (err) {
        setError(messageFor(err, "Impossible d'enregistrer la modification."));
        throw err;
      }
    },
    [id],
  );

  const switchVariant = useCallback(
    async (messageId: number, variantId: number) => {
      if (!id) return;
      try {
        const updated = await setMessageVariant(id, messageId, variantId);
        setParty((current) => (current ? replaceMessage(current, messageId, updated) : current));
      } catch (err) {
        setError(messageFor(err, "Impossible de changer de variante."));
        throw err;
      }
    },
    [id],
  );

  const cancelImage = useCallback(
    async (messageId: number) => {
      if (!id) return;
      try {
        const updated = await cancelImageGeneration(id, messageId);
        setParty((current) => (current ? replaceMessage(current, messageId, updated) : current));
      } catch (err) {
        setError(messageFor(err, "Impossible d'annuler la génération."));
      }
    },
    [id],
  );

  // Both run before the early returns below: hooks must not be conditional.
  const hasPendingImage =
    party?.messages.some((m) => m.kind === "image" && m.status === "pending") ?? false;
  useEffect(() => {
    if (!id || !hasPendingImage) return;
    const timer = window.setInterval(() => {
      void load(id);
    }, IMAGE_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [id, hasPendingImage, load]);

  if (!id) {
    return <p className="text-red-400">Partie introuvable.</p>;
  }

  if (!party) {
    return <p className="text-neutral-500">{error ?? "Chargement…"}</p>;
  }

  const finishTurn = async (stopped: boolean) => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    // One final render past the throttle, so nothing is left truncated.
    setStreamText(bufferRef.current);
    controllerRef.current = null;
    if (stopped) {
      await new Promise((resolve) => setTimeout(resolve, RELOAD_AFTER_STOP_MS));
    }
    const reloaded = await load(id);
    if (reloaded) {
      // The saved state is on screen; the transient bubbles can go.
      bufferRef.current = "";
      setStreamText(null);
      setPending(null);
      setRegeneratingId(null);
    }
    setStreaming(false);
  };

  const startStream = (): AbortController => {
    setStreamText(null);
    setStreamError(null);
    setStreaming(true);
    bufferRef.current = "";
    lastRenderRef.current = 0;
    const controller = new AbortController();
    controllerRef.current = controller;
    return controller;
  };

  const send = async () => {
    const content = input.trim();
    if (!content || controllerRef.current !== null) return;
    setInput("");
    setPending(content);
    const controller = startStream();
    let stopped = false;
    try {
      await sendTurn(id, content, onEvent, controller.signal);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // Stopping is not an error: whatever arrived is already saved
        // server-side, and the reload below shows it.
        stopped = true;
      } else if (err instanceof ApiError) {
        // The turn never reached generation, so give the player their text
        // back to fix or resend.
        setInput(content);
        setStreamError(err.detail);
      } else {
        setInput(content);
        setStreamError(messageFor(err, "La connexion au narrateur a été interrompue."));
      }
    } finally {
      await finishTurn(stopped);
    }
  };

  const regenerate = async (messageId: number) => {
    if (controllerRef.current !== null) return;
    setRegeneratingId(messageId);
    const controller = startStream();
    let stopped = false;
    try {
      await regenerateMessage(id, messageId, onEvent, controller.signal);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        stopped = true;
      } else {
        setStreamError(messageFor(err, "La connexion au narrateur a été interrompue."));
      }
    } finally {
      await finishTurn(stopped);
    }
  };

  const stop = () => {
    controllerRef.current?.abort();
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <Link to="/parties" className="text-sm text-neutral-500 hover:text-neutral-300">
          ← Toutes les parties
        </Link>
        <h1 className="text-2xl font-semibold">{party.label}</h1>
        <p className="text-sm text-neutral-500">Scénario&nbsp;: {party.scenarioTitle}</p>
      </div>
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      <details className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 text-sm">
        <summary className="cursor-pointer text-neutral-400 select-none">
          Résumé et état du monde
        </summary>
        <div className="mt-3 flex flex-col gap-3">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Résumé</p>
            {party.summaryText ? (
              <p className="mt-1 whitespace-pre-wrap text-neutral-300">{party.summaryText}</p>
            ) : (
              <p className="mt-1 text-neutral-500">
                Aucun résumé pour l'instant — il apparaît quand l'histoire dépasse la fenêtre de
                mémoire.
              </p>
            )}
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
              État du monde
            </p>
            {Object.keys(party.worldState).length > 0 ? (
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-neutral-300">
                {JSON.stringify(party.worldState, null, 2)}
              </pre>
            ) : (
              <p className="mt-1 text-neutral-500">Aucun état établi pour l'instant.</p>
            )}
          </div>
        </div>
      </details>
      {party.messages.length === 0 && pending === null ? (
        <p className="text-neutral-500">Aucun message pour l'instant.</p>
      ) : null}
      <ul className="flex flex-col gap-4">
        {party.messages.map((message) =>
          message.kind === "image" ? (
            <ImageMessage
              key={message.id}
              message={message}
              disabled={streaming}
              onCancel={(messageTarget) => void cancelImage(messageTarget)}
            />
          ) : (
            <MessageBubble
              key={message.id}
              message={message}
              disabled={streaming}
              regenerating={regeneratingId === message.id}
              // The old reply stays on screen until the first token of the new
              // one arrives: `overrideText` is undefined until then.
              overrideText={
                regeneratingId === message.id && streamText !== null ? streamText : undefined
              }
              onSaveEdit={saveEdit}
              onRegenerate={(messageTarget) => void regenerate(messageTarget)}
              onSwitchVariant={(messageTarget, variantId) =>
                void switchVariant(messageTarget, variantId)
              }
            />
          ),
        )}
        {pending !== null ? <MessageBubble message={pendingMessage(pending)} disabled /> : null}
        {streamText !== null && regeneratingId === null ? (
          <StreamingBubble text={streamText} />
        ) : null}
      </ul>
      <div ref={endRef} />
      {streamError ? <p className="text-sm text-red-400">{streamError}</p> : null}
      {/* The warning issue #6 exists for. Ollama truncates an over-long prompt
          in silence, from the front, system prompt first — the narrator then
          forgets the scenario with nothing on screen to explain it. */}
      {party.context.nearLimit ? (
        <p className="rounded border border-amber-800 bg-amber-950/40 px-3 py-2 text-sm text-amber-300">
          Le prompt approche la fenêtre de contexte&nbsp;:{" "}
          {party.context.estimatedTokens.toLocaleString("fr-FR")} jetons estimés sur{" "}
          {party.context.numCtx.toLocaleString("fr-FR")}. Au-delà, Ollama coupe le début du
          prompt sans le dire — le prompt système d'abord, donc le scénario. Augmentez la
          fenêtre de contexte ou réduisez la fenêtre de mémoire dans les réglages.
        </p>
      ) : null}
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
          placeholder="Que faites-vous&nbsp;?"
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100 placeholder:text-neutral-600 focus:border-sky-700 focus:outline-none disabled:opacity-50"
        />
        {streaming ? (
          <Button type="button" variant="danger" onClick={stop}>
            Arrêter
          </Button>
        ) : (
          <Button type="submit" disabled={!input.trim()}>
            Envoyer
          </Button>
        )}
      </form>
      {/* key: the panel keeps its composed prompt in local state, so a party
          change must remount it rather than show one party's prompt on
          another party's page. */}
      <ImagePanel
        key={party.id}
        partyId={party.id}
        onGenerationStarted={() => void load(party.id)}
      />
    </div>
  );
}

function pendingMessage(content: string): PartyMessage {
  // A stand-in until the reload brings the real row with its id. It exists
  // only while a turn is streaming, so its actions are disabled with the
  // bubble's `disabled` prop — its negative id is never sent anywhere.
  return {
    id: -1,
    role: "user",
    content,
    ts: 0,
    variants: [],
    kind: "text",
    imageId: null,
    status: null,
    startedAt: null,
    error: null,
  };
}

interface MessageBubbleProps {
  message: PartyMessage;
  disabled: boolean;
  regenerating?: boolean;
  overrideText?: string;
  onSaveEdit?: (messageId: number, content: string) => Promise<void>;
  onRegenerate?: (messageId: number) => void;
  onSwitchVariant?: (messageId: number, variantId: number) => void;
}

function MessageBubble({
  message,
  disabled,
  regenerating = false,
  overrideText,
  onSaveEdit,
  onRegenerate,
  onSwitchVariant,
}: MessageBubbleProps) {
  const isPlayer = message.role === "user";
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);

  const startEdit = () => {
    setDraft(message.content);
    setEditing(true);
  };

  const save = async () => {
    const content = draft.trim();
    if (!content || saving || !onSaveEdit) return;
    setSaving(true);
    try {
      await onSaveEdit(message.id, content);
      setEditing(false);
    } catch {
      // The page displays the reason; the editor stays open on the draft.
    } finally {
      setSaving(false);
    }
  };

  const activeIndex = message.variants.findIndex((v) => v.active);
  const previousVariant = activeIndex > 0 ? message.variants[activeIndex - 1] : null;
  const nextVariant =
    activeIndex >= 0 && activeIndex < message.variants.length - 1
      ? message.variants[activeIndex + 1]
      : null;
  const text = overrideText ?? message.content;

  return (
    <li
      className={`rounded-lg border p-4 ${
        isPlayer ? "border-sky-800 bg-sky-950/40" : "border-neutral-800 bg-neutral-900"
      }`}
    >
      <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
        {isPlayer ? "Vous" : "Narration"}
      </p>
      {editing ? (
        <div className="mt-2 flex flex-col gap-2">
          <TextArea
            label="Message"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={saving}
            autoFocus
          />
          <div className="flex gap-2">
            <Button onClick={() => void save()} disabled={saving || !draft.trim()}>
              Enregistrer
            </Button>
            <Button variant="secondary" onClick={() => setEditing(false)} disabled={saving}>
              Annuler
            </Button>
          </div>
        </div>
      ) : (
        <>
          {isPlayer ? (
            // The player's own text, not the narrator's: it stays plain, so what
            // they typed is what they see.
            <p className="mt-2 whitespace-pre-wrap text-neutral-100">{text}</p>
          ) : (
            <Narration text={text} />
          )}
          {regenerating ? <p className="mt-2 text-xs text-neutral-500">Régénération…</p> : null}
          {message.variants.length > 1 ? (
            <div className="mt-2 flex items-center gap-2 text-xs text-neutral-500">
              <Button
                variant="secondary"
                className="text-xs"
                disabled={disabled || previousVariant === null}
                onClick={() => {
                  if (previousVariant && onSwitchVariant) {
                    onSwitchVariant(message.id, previousVariant.id);
                  }
                }}
              >
                ←
              </Button>
              <span>
                {activeIndex + 1} / {message.variants.length}
              </span>
              <Button
                variant="secondary"
                className="text-xs"
                disabled={disabled || nextVariant === null}
                onClick={() => {
                  if (nextVariant && onSwitchVariant) {
                    onSwitchVariant(message.id, nextVariant.id);
                  }
                }}
              >
                →
              </Button>
            </div>
          ) : null}
          <div className="mt-2 flex gap-2">
            <Button
              variant="secondary"
              className="text-xs"
              disabled={disabled || !onSaveEdit}
              onClick={startEdit}
            >
              Modifier
            </Button>
            {!isPlayer && onRegenerate ? (
              <Button
                variant="secondary"
                className="text-xs"
                disabled={disabled}
                onClick={() => onRegenerate(message.id)}
              >
                Régénérer
              </Button>
            ) : null}
          </div>
        </>
      )}
    </li>
  );
}

function StreamingBubble({ text }: { text: string }) {
  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Narration</p>
      <Narration text={text} />
    </li>
  );
}
