import { useEffect } from "react";
import { createPortal } from "react-dom";

interface LightboxProps {
  src: string;
  alt: string;
  onClose: () => void;
}

/**
 * One image, full screen. Escape closes it, and so does a click anywhere
 * around the image.
 *
 * The backdrop is a real button rather than a div with a click handler: a
 * div can be clicked but never focused or activated from a keyboard, which
 * left the only way out of the viewer — short of Escape — unreachable
 * without a mouse. The image sits above the backdrop as a sibling, so a click
 * on the image lands on the image and does not close anything; no
 * `stopPropagation` is needed to make that true.
 *
 * Rendered into `document.body` through a portal, not where it is declared.
 * The image panel it opens from has a `backdrop-filter`, and an ancestor with
 * a filter becomes the containing block of every `position: fixed`
 * descendant — so `inset-0` meant "the panel", and the full-screen viewer
 * filled only the panel. A portal makes the viewport its only reference,
 * whatever CSS a future parent carries.
 */
export function Lightbox({ src, alt, onClose }: LightboxProps) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    // The page behind must not scroll while this is open.
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        onClick={onClose}
        aria-label="Fermer l'image"
        className="absolute inset-0 cursor-zoom-out bg-black/90"
      />
      <img src={src} alt={alt} className="relative max-h-full max-w-full rounded object-contain" />
      <button
        type="button"
        onClick={onClose}
        aria-label="Fermer"
        className="absolute right-4 top-4 rounded bg-neutral-900/80 px-3 py-1 text-neutral-200 hover:bg-neutral-800"
      >
        ✕
      </button>
    </div>,
    document.body,
  );
}
