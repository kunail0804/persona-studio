import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import type { Character, CharacterInput } from "../api/characters";
import {
  createCharacter,
  deleteCharacter,
  listCharacters,
  reorderCharacters,
  updateCharacter,
} from "../api/characters";
import { ApiError } from "../api/client";
import { createParty } from "../api/parties";
import type { Scenario } from "../api/scenarios";
import { deleteScenario, getScenario, updateScenario } from "../api/scenarios";
import { Button } from "../components/Button";
import { CharacterForm } from "../components/CharacterForm";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { TextArea } from "../components/TextArea";
import { TextField } from "../components/TextField";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

export function ScenarioEditorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [synopsis, setSynopsis] = useState("");
  const [saving, setSaving] = useState(false);
  const [starting, setStarting] = useState(false);
  const [addingCharacter, setAddingCharacter] = useState(false);
  const [editingCharacterId, setEditingCharacterId] = useState<number | null>(null);
  const [confirmDeleteScenario, setConfirmDeleteScenario] = useState(false);
  const [confirmDeleteCharacterId, setConfirmDeleteCharacterId] = useState<number | null>(null);

  const load = useCallback(async (scenarioId: string, signal?: AbortSignal) => {
    try {
      const [scenarioData, characterData] = await Promise.all([
        getScenario(scenarioId, signal),
        listCharacters(scenarioId, signal),
      ]);
      setScenario(scenarioData);
      setTitle(scenarioData.title);
      setSynopsis(scenarioData.synopsis);
      setCharacters(characterData);
      setError(null);
    } catch (err) {
      // A stale request aborted by the effect cleanup below, because `id`
      // changed again before it resolved — not a real failure to report.
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(messageFor(err, "Impossible de charger le scénario."));
    }
  }, []);

  useEffect(() => {
    if (!id) return;
    // Cancels the previous id's request when `id` changes again before it
    // resolves, so a slow response for a scenario the player has since
    // navigated away from can't overwrite the state of the one now showing.
    const controller = new AbortController();
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: load() sets state after an await, not synchronously in the effect body.
    void load(id, controller.signal);
    return () => controller.abort();
  }, [id, load]);

  if (!id) {
    return <p className="text-red-400">Scénario introuvable.</p>;
  }
  // Reassigned so the closures below see a plain `string`: narrowing the
  // `id` param above doesn't carry into function bodies declared after it.
  const scenarioId = id;

  async function handleSave() {
    try {
      setSaving(true);
      const updated = await updateScenario(scenarioId, { title, synopsis });
      setScenario(updated);
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible d'enregistrer le scénario."));
    } finally {
      setSaving(false);
    }
  }

  async function handleStartParty() {
    // The opening scene is generated server-side before the party exists, so
    // this call can take minutes on a local model: the button stays disabled
    // and the page says so instead of looking frozen.
    setStarting(true);
    try {
      const party = await createParty(scenarioId);
      navigate(`/parties/${party.id}`);
    } catch (err) {
      setError(messageFor(err, "Impossible de démarrer la partie."));
      setStarting(false);
    }
  }

  async function handleDeleteScenario() {
    try {
      await deleteScenario(scenarioId);
      navigate("/");
    } catch (err) {
      setError(messageFor(err, "Impossible de supprimer le scénario."));
      setConfirmDeleteScenario(false);
    }
  }

  async function handleCreateCharacter(input: CharacterInput) {
    try {
      await createCharacter(scenarioId, input);
      setAddingCharacter(false);
      await load(scenarioId);
    } catch (err) {
      setError(messageFor(err, "Impossible de créer le personnage."));
    }
  }

  async function handleUpdateCharacter(characterId: number, input: CharacterInput) {
    try {
      await updateCharacter(scenarioId, characterId, input);
      setEditingCharacterId(null);
      await load(scenarioId);
    } catch (err) {
      setError(messageFor(err, "Impossible de modifier le personnage."));
    }
  }

  async function handleDeleteCharacter(characterId: number) {
    try {
      await deleteCharacter(scenarioId, characterId);
      await load(scenarioId);
    } catch (err) {
      setError(messageFor(err, "Impossible de supprimer le personnage."));
    } finally {
      setConfirmDeleteCharacterId(null);
    }
  }

  async function handleMove(characterId: number, direction: -1 | 1) {
    const index = characters.findIndex((character) => character.id === characterId);
    const targetIndex = index + direction;
    if (index === -1 || targetIndex < 0 || targetIndex >= characters.length) return;

    const order = characters.map((character) => character.id);
    const [moved] = order.splice(index, 1);
    order.splice(targetIndex, 0, moved);

    try {
      setCharacters(await reorderCharacters(scenarioId, order));
    } catch (err) {
      setError(messageFor(err, "Impossible de réordonner les personnages."));
    }
  }

  if (!scenario) {
    return <p className="text-neutral-500">{error ?? "Chargement…"}</p>;
  }

  return (
    <div className="flex flex-col gap-8">
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      {starting ? (
        <p className="text-sm text-neutral-400">
          La scène d'ouverture est en cours de génération. Un modèle local peut
          prendre de quelques secondes à plusieurs minutes.
        </p>
      ) : null}

      <section className="flex flex-col gap-3">
        <TextField label="Titre" value={title} onChange={(event) => setTitle(event.target.value)} />
        <TextArea
          label="Synopsis"
          value={synopsis}
          onChange={(event) => setSynopsis(event.target.value)}
        />
        <div className="flex justify-between">
          <Button variant="danger" onClick={() => setConfirmDeleteScenario(true)}>
            Supprimer le scénario
          </Button>
          <div className="flex gap-2">
            <Button onClick={() => void handleStartParty()} disabled={starting || saving}>
              {starting ? "Génération…" : "Jouer"}
            </Button>
            <Button onClick={() => void handleSave()} disabled={saving || starting}>
              Enregistrer
            </Button>
          </div>
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Personnages</h2>
          {!addingCharacter ? (
            <Button onClick={() => setAddingCharacter(true)}>Ajouter un personnage</Button>
          ) : null}
        </div>

        {addingCharacter ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            <CharacterForm
              submitLabel="Créer"
              onSubmit={handleCreateCharacter}
              onCancel={() => setAddingCharacter(false)}
            />
          </div>
        ) : null}

        <ul className="flex flex-col gap-3">
          {characters.map((character, index) => (
            <CharacterListItem
              key={character.id}
              character={character}
              isFirst={index === 0}
              isLast={index === characters.length - 1}
              isEditing={editingCharacterId === character.id}
              onEdit={() => setEditingCharacterId(character.id)}
              onCancelEdit={() => setEditingCharacterId(null)}
              onSave={(input) => handleUpdateCharacter(character.id, input)}
              onDelete={() => setConfirmDeleteCharacterId(character.id)}
              onMoveUp={() => void handleMove(character.id, -1)}
              onMoveDown={() => void handleMove(character.id, 1)}
            />
          ))}
        </ul>
      </section>

      <ConfirmDialog
        open={confirmDeleteScenario}
        title="Supprimer ce scénario ?"
        description="Cette action supprime aussi ses personnages et toutes ses parties. Elle est définitive."
        onConfirm={() => void handleDeleteScenario()}
        onCancel={() => setConfirmDeleteScenario(false)}
      />
      <ConfirmDialog
        open={confirmDeleteCharacterId !== null}
        title="Supprimer ce personnage ?"
        description="Cette action est définitive."
        onConfirm={() => {
          if (confirmDeleteCharacterId !== null) void handleDeleteCharacter(confirmDeleteCharacterId);
        }}
        onCancel={() => setConfirmDeleteCharacterId(null)}
      />
    </div>
  );
}

interface CharacterListItemProps {
  character: Character;
  isFirst: boolean;
  isLast: boolean;
  isEditing: boolean;
  onEdit: () => void;
  onCancelEdit: () => void;
  onSave: (input: CharacterInput) => Promise<void>;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}

function CharacterListItem({
  character,
  isFirst,
  isLast,
  isEditing,
  onEdit,
  onCancelEdit,
  onSave,
  onDelete,
  onMoveUp,
  onMoveDown,
}: CharacterListItemProps) {
  return (
    <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      {isEditing ? (
        <CharacterForm
          initial={character}
          submitLabel="Enregistrer"
          onSubmit={onSave}
          onCancel={onCancelEdit}
        />
      ) : (
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-medium">{character.name || "Sans nom"}</h3>
            {character.appearance ? (
              <p className="mt-1 line-clamp-2 text-sm text-neutral-400">{character.appearance}</p>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <Button
              variant="secondary"
              onClick={onMoveUp}
              disabled={isFirst}
              aria-label="Déplacer vers le haut"
            >
              ↑
            </Button>
            <Button
              variant="secondary"
              onClick={onMoveDown}
              disabled={isLast}
              aria-label="Déplacer vers le bas"
            >
              ↓
            </Button>
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
