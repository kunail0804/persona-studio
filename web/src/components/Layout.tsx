import { useCallback, useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";
import type { ServiceStatus } from "../api/status";
import { getServiceStatus } from "../api/status";
import { ServiceMenu } from "./ServiceMenu";

// Slow enough not to matter, often enough that a service coming back is
// noticed before the next turn is typed.
const STATUS_POLL_INTERVAL_MS = 30000;

export function Layout() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Sticky: the transcript is the long thing on screen, and the way out
          of it should not require scrolling back to the top. */}
      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-neutral-800 bg-neutral-950/95 px-6 py-4 backdrop-blur">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          Persona Studio
        </Link>
        <div className="flex items-center gap-6">
          <ServicePills />
          <Link to="/parties" className="text-sm text-neutral-400 hover:text-neutral-100">
            Parties
          </Link>
          <Link to="/settings" className="text-sm text-neutral-400 hover:text-neutral-100">
            Paramètres
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}

/**
 * Two pills saying whether Ollama and comfy-gateway (with its ComfyUI) are
 * answering. Until these existed you learned Ollama was down when a turn
 * failed, halfway through writing one. Each opens a menu with its memory and
 * the buttons that free it (`ServiceMenu`).
 *
 * A failed poll is shown as "unknown" rather than as "down": the server not
 * answering says nothing about the services it would have probed, and a pill
 * that cries wolf is a pill nobody reads.
 */
function ServicePills() {
  const [status, setStatus] = useState<ServiceStatus | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      setStatus(await getServiceStatus(signal));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: refresh() sets state after an await, not synchronously in the effect body.
    void refresh(controller.signal);
    const timer = window.setInterval(() => void refresh(), STATUS_POLL_INTERVAL_MS);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [refresh]);

  return (
    <div className="flex items-center gap-3">
      <ServiceMenu
        kind="ollama"
        label="Ollama"
        state={status?.ollama ?? null}
        onChanged={() => void refresh()}
      />
      <ServiceMenu
        kind="gateway"
        label="Gateway"
        state={status?.gateway ?? null}
        onChanged={() => void refresh()}
      />
    </div>
  );
}
