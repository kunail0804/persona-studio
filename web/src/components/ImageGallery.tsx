import { useEffect, useState } from "react";
import type { PartyMessage } from "../api/parties";
import { Button } from "./Button";
import { ImagePanel } from "./ImagePanel";
import { Lightbox } from "./Lightbox";

interface ImageGalleryProps {
  partyId: string;
  /** Every image message of the party, oldest first. */
  images: PartyMessage[];
  /** True while a turn streams or a render is running: composing is blocked. */
  busy: boolean;
  busyReason: string | null;
  onGenerationStarted: () => void;
  onCancel: (messageId: number) => void;
  onDelete: (messageId: number) => void;
}

function formatElapsed(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secondsLeft = Math.floor(seconds % 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (hours > 0) return `${hours}:${pad(minutes)}:${pad(secondsLeft)}`;
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
 * The party's images, beside the story rather than inside it.
 *
 * They were rendered in the transcript, which is where they are still
 * stored — an image is a message, and `narrator.load_history` has always
 * filtered them out, so they never reached the narrator either way. Only the
 * rendering moved: the story reads as text, and the images are composed,
 * watched and managed here.
 */
export function ImageGallery({
  partyId,
  images,
  busy,
  busyReason,
  onGenerationStarted,
  onCancel,
  onDelete,
}: ImageGalleryProps) {
  const [zoomed, setZoomed] = useState<PartyMessage | null>(null);

  return (
    <div className="flex flex-col gap-4">
      <ImagePanel
        partyId={partyId}
        disabled={busy}
        disabledReason={busyReason}
        onGenerationStarted={onGenerationStarted}
      />

      {images.length === 0 ? (
        <p className="text-sm text-neutral-500">Aucune image dans cette partie.</p>
      ) : (
        <ul className="flex flex-col gap-3">
          {[...images].reverse().map((image) => (
            <GalleryItem
              key={image.id}
              image={image}
              onZoom={() => setZoomed(image)}
              onCancel={() => onCancel(image.id)}
              onDelete={() => onDelete(image.id)}
            />
          ))}
        </ul>
      )}

      {zoomed !== null && zoomed.imageId !== null ? (
        <Lightbox
          src={`/api/images/${zoomed.imageId}/file`}
          alt={zoomed.content || "Image de la scène"}
          onClose={() => setZoomed(null)}
        />
      ) : null}
    </div>
  );
}

interface GalleryItemProps {
  image: PartyMessage;
  onZoom: () => void;
  onCancel: () => void;
  onDelete: () => void;
}

function GalleryItem({ image, onZoom, onCancel, onDelete }: GalleryItemProps) {
  const elapsed = useElapsed(image.startedAt);

  if (image.status === "pending") {
    return (
      <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
        <p className="truncate text-xs text-neutral-500" title={image.content}>
          {image.content}
        </p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <p className="text-sm text-neutral-300">
            Génération… <span className="tabular-nums">{formatElapsed(elapsed)}</span>
          </p>
          <Button variant="secondary" className="text-xs" onClick={onCancel}>
            Annuler
          </Button>
        </div>
      </li>
    );
  }

  if (image.status === "done" && image.imageId !== null) {
    return (
      // `group` is what makes the controls appear on hover only: the image is
      // the subject, the buttons are not.
      <li className="group relative overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
        {/* The image is the zoom target, so it is wrapped in a button: an
            <img> with a click handler can be clicked but never reached or
            activated from a keyboard. */}
        <button
          type="button"
          onClick={onZoom}
          aria-label="Afficher l'image en plein écran"
          className="block w-full cursor-zoom-in"
        >
          <img
            src={`/api/images/${image.imageId}/file`}
            alt={image.content || "Image de la scène"}
            className="w-full"
          />
        </button>
        <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-end gap-1 p-2 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <button
            type="button"
            onClick={onZoom}
            aria-label="Afficher en plein écran"
            title="Plein écran"
            className="pointer-events-auto rounded bg-neutral-950/80 px-2 py-1 text-sm text-neutral-200 hover:bg-neutral-800"
          >
            ⤢
          </button>
          <button
            type="button"
            onClick={onDelete}
            aria-label="Supprimer l'image"
            title="Supprimer"
            className="pointer-events-auto rounded bg-neutral-950/80 px-2 py-1 text-sm text-red-400 hover:bg-neutral-800"
          >
            ✕
          </button>
        </div>
        {image.content ? (
          <p className="truncate px-3 py-2 text-xs text-neutral-500" title={image.content}>
            {image.content}
          </p>
        ) : null}
      </li>
    );
  }

  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <p className="truncate text-xs text-neutral-500" title={image.content}>
        {image.content}
      </p>
      <div className="mt-2 flex items-start justify-between gap-2">
        <p className="text-sm text-red-400">{image.error ?? "La génération a échoué."}</p>
        <button
          type="button"
          onClick={onDelete}
          aria-label="Supprimer l'image"
          title="Supprimer"
          className="rounded px-2 py-1 text-sm text-red-400 hover:bg-neutral-800"
        >
          ✕
        </button>
      </div>
    </li>
  );
}
