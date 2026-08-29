import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { getParty } from "../api/parties";
import type { Party, PartyMessage } from "../api/parties";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

export function PartyPage() {
  const { id } = useParams<{ id: string }>();

  const [party, setParty] = useState<Party | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (partyId: string, signal?: AbortSignal) => {
    try {
      setParty(await getParty(partyId, signal));
      setError(null);
    } catch (err) {
      // A stale request aborted by the effect cleanup below, because `id`
      // changed again before it resolved — not a real failure to report.
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(messageFor(err, "Impossible de charger la partie."));
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

  if (!id) {
    return <p className="text-red-400">Partie introuvable.</p>;
  }

  if (!party) {
    return <p className="text-neutral-500">{error ?? "Chargement…"}</p>;
  }

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
      {party.messages.length === 0 ? (
        <p className="text-neutral-500">Aucun message pour l'instant.</p>
      ) : (
        <ul className="flex flex-col gap-4">
          {party.messages.map((message) => (
            <TranscriptMessage key={message.id} message={message} />
          ))}
        </ul>
      )}
      <div ref={endRef} />
    </div>
  );
}

function TranscriptMessage({ message }: { message: PartyMessage }) {
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
      {/* Plain text for now: Markdown rendering is a later pull request.
          `whitespace-pre-wrap` keeps the narrator's line breaks. */}
      <p className="mt-2 whitespace-pre-wrap text-neutral-100">{message.content}</p>
    </li>
  );
}
