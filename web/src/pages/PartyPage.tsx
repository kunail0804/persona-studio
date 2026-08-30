import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { getParty, sendTurn } from "../api/parties";
import type { Party, PartyMessage, TurnEvent } from "../api/parties";
import { Narration } from "../components/Narration";
import { Button } from "../components/Button";

// The bubble re-renders at most this often while fragments arrive. Every
// fragment is still accumulated; only the re-render is throttled. The
// previous version needed exactly this: without it the Markdown bubble
// visibly beat while reparsing itself (see issue #10).
const RENDER_INTERVAL_MS = 150;

// After a stop, the server persists the partial when it notices the
// hang-up, which lands slightly after the abort resolves here; the reload
// waits that long before reading the database.
const RELOAD_AFTER_STOP_MS = 500;

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

export function PartyPage() {
  const { id } = useParams<{ id: string }>();

  const [party, setParty] = useState<Party | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The turn being played: the optimistic player bubble, the growing reply,
  // and why the stream broke, if it broke.
  const [input, setInput] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [streamText, setStreamText] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);

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

  // Keep the growing reply in view while it streams.
  useEffect(() => {
    if (streamText !== null) endRef.current?.scrollIntoView();
  }, [streamText]);

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
    }
    setStreaming(false);
  };

  const send = async () => {
    const content = input.trim();
    if (!content || controllerRef.current !== null) return;
    setInput("");
    setPending(content);
    setStreamText(null);
    setStreamError(null);
    setStreaming(true);
    bufferRef.current = "";
    lastRenderRef.current = 0;
    const controller = new AbortController();
    controllerRef.current = controller;
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
      {party.messages.length === 0 && pending === null ? (
        <p className="text-neutral-500">Aucun message pour l'instant.</p>
      ) : null}
      <ul className="flex flex-col gap-4">
        {party?.messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {pending !== null ? <MessageBubble message={pendingMessage(pending)} /> : null}
        {streamText !== null ? <StreamingBubble text={streamText} /> : null}
      </ul>
      <div ref={endRef} />
      {streamError ? <p className="text-sm text-red-400">{streamError}</p> : null}
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
    </div>
  );
}

function pendingMessage(content: string): PartyMessage {
  // A stand-in until the reload brings the real row with its id.
  return { id: -1, role: "user", content, ts: 0 };
}

function MessageBubble({ message }: { message: PartyMessage }) {
  const isPlayer = message.role === "user";
  return (
    <li
      className={`rounded-lg border p-4 ${
        isPlayer ? "border-sky-800 bg-sky-950/40" : "border-neutral-800 bg-neutral-900"
      }`}
    >
      <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
        {isPlayer ? "Vous" : "Narration"}
      </p>
      {isPlayer ? (
        // The player's own text, not the narrator's: it stays plain, so what
        // they typed is what they see.
        <p className="mt-2 whitespace-pre-wrap text-neutral-100">{message.content}</p>
      ) : (
        <Narration text={message.content} />
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
