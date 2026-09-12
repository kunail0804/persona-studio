import { useEffect, useState } from "react";
import type { PartyMessage } from "../api/parties";
import { Button } from "./Button";

interface ImageMessageProps {
  message: PartyMessage;
  disabled: boolean;
  onCancel?: (messageId: number) => void;
}

function formatElapsed(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secondsLeft = Math.floor(seconds % 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${pad(secondsLeft)}`;
  }
  return `${minutes}:${pad(secondsLeft)}`;
}

/** Seconds since the generation started, ticking once per second. The time
 * comes from the stored `started_at`, so a reload shows the true age of the
 * job rather than restarting from zero. */
function useElapsed(startedAt: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  if (startedAt === null) return 0;
  return Math.max(0, (now - startedAt * 1000) / 1000);
}

/**
 * One image message in the story, in whatever state its generation is:
 * pending with its elapsed time counting up and a control to cancel it,
 * done as the image itself, error or cancelled as its message. A failed or
 * cancelled generation never pretends to still be working.
 */
export function ImageMessage({ message, disabled, onCancel }: ImageMessageProps) {
  const elapsed = useElapsed(message.startedAt);

  if (message.status === "pending") {
    return (
      <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Image</p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <p className="text-neutral-300">
            Génération de l'image… <span className="tabular-nums">{formatElapsed(elapsed)}</span>
          </p>
          {onCancel ? (
            <Button
              variant="secondary"
              className="text-xs"
              disabled={disabled}
              onClick={() => onCancel(message.id)}
            >
              Annuler
            </Button>
          ) : null}
        </div>
      </li>
    );
  }
  if (message.status === "done" && message.imageId !== null) {
    return (
      <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Image</p>
        <img
          src={`/api/images/${message.imageId}/file`}
          alt={message.content || "Image de la scène"}
          className="mt-2 max-w-full rounded"
        />
      </li>
    );
  }
  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Image</p>
      <p className="mt-2 text-sm text-red-400">{message.error ?? "La génération a échoué."}</p>
    </li>
  );
}