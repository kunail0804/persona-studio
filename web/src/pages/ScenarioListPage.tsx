import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { createScenario, listScenarios } from "../api/scenarios";
import type { ScenarioSummary } from "../api/scenarios";
import { Button } from "../components/Button";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

export function ScenarioListPage() {
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  async function loadScenarios() {
    try {
      setScenarios(await listScenarios());
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible de charger les scénarios."));
    }
  }

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: loadScenarios() sets state after an await, not synchronously in the effect body.
    void loadScenarios();
  }, []);

  async function handleCreate() {
    setCreating(true);
    try {
      const scenario = await createScenario({ title: "Sans titre", synopsis: "" });
      navigate(`/scenarios/${scenario.id}`);
    } catch (err) {
      setError(messageFor(err, "Impossible de créer le scénario."));
      setCreating(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Scénarios</h1>
        <Button onClick={() => void handleCreate()} disabled={creating}>
          Nouveau scénario
        </Button>
      </div>
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      {scenarios.length === 0 ? (
        <p className="text-neutral-500">Aucun scénario pour l'instant.</p>
      ) : (
        <ul className="flex flex-col gap-3">
          {scenarios.map((scenario) => (
            <ScenarioListItem key={scenario.id} scenario={scenario} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ScenarioListItem({ scenario }: { scenario: ScenarioSummary }) {
  return (
    <li>
      <Link
        to={`/scenarios/${scenario.id}`}
        className="block rounded-lg border border-neutral-800 bg-neutral-900 p-4 hover:border-neutral-600"
      >
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-medium">{scenario.title}</h2>
          <span className="shrink-0 text-sm text-neutral-500">
            {scenario.characterCount} personnage{scenario.characterCount > 1 ? "s" : ""}
          </span>
        </div>
        {scenario.synopsis ? (
          <p className="mt-1 line-clamp-2 text-sm text-neutral-400">{scenario.synopsis}</p>
        ) : null}
      </Link>
    </li>
  );
}
