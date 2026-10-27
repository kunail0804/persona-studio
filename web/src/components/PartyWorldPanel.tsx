import { useCallback, useEffect, useState } from "react";
import type { CharacterInput } from "../api/characters";
import { ApiError } from "../api/client";
import type { PartyCharacter, PartyWorld } from "../api/partyWorld";
import type { WorldOwner } from "../api/world";
import {
  createPartyCharacter,
  deletePartyCharacter,
  getPartyWorld,
  updatePartyCharacter,
} from "../api/partyWorld";
import { Button } from "./Button";
import { CharacterForm } from "./CharacterForm";
import { ConfirmDialog } from "./ConfirmDialog";
import { LoreSection, PlacesSection } from "./WorldSections";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

/**
 * The party's own world: its characters, places and lore entries, editable
 * while playing. The party received a copy of the scenario's when it was
 * created, so nothing done here reaches the scenario or another party. An
 * edit reaches the narrator at the next turn and the image prompt at the
 * next composition.
 */
export function PartyWorldPanel({ partyId }: { partyId: string }) {
  const [world, setWorld] = useState<PartyWorld | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const data = await getPartyWorld(partyId, signal);
        setWorld(data);
        setError(null);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(messageFor(err, "Impossible de charger le monde de la partie."));
      }
    },
    [partyId],
  );

  useEffect(() => {
    const controller = new AbortController();
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: load() sets state after an await, not synchronously in the effect body.
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const owner: WorldOwner = { kind: "party", id: partyId };

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-neutral-500">
        Ce monde appartient à cette partie seule&nbsp;: le scénario et les autres parties ne voient
        pas ces changements. Le narrateur les reçoit au prochain tour.
      </p>
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      {world === null ? (
        error ? null : <p className="text-sm text-neutral-500">Chargement…</p>
      ) : (
        <>
          <CharactersSection
            partyId={partyId}
            characters={world.characters}
            onChanged={() => load()}
            onError={setError}
          />
          <PlacesSection
            owner={owner}
            places={world.places}
            onChanged={() => load()}
            onError={setError}
          />
          <LoreSection
            owner={owner}
            entries={world.lore}
            onChanged={() => load()}
            onError={setError}
          />
        </>
      )}
    </div>
  );
}

interface CharactersSectionProps {
  partyId: string;
  characters: PartyCharacter[];
  onChanged: () => Promise<void>;
  onError: (message: string) => void;
}

function CharactersSection({ partyId, characters, onChanged, onError }: CharactersSectionProps) {
  const [adding, setAdding] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  const create = async (input: CharacterInput) => {
    try {
      await createPartyCharacter(partyId, input);
      setAdding(false);
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible de créer le personnage."));
    }
  };

  const update = async (id: number, input: CharacterInput) => {
    try {
      await updatePartyCharacter(partyId, id, input);
      setEditingId(null);
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible d'enregistrer le personnage."));
    }
  };

  const remove = async (id: number) => {
    try {
      await deletePartyCharacter(partyId, id);
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible de supprimer le personnage."));
    } finally {
      setConfirmDeleteId(null);
    }
  };

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Personnages</h2>
        {adding ? null : <Button onClick={() => setAdding(true)}>Ajouter</Button>}
      </div>
      {adding ? (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
          <CharacterForm
            submitLabel="Créer"
            onSubmit={create}
            onCancel={() => setAdding(false)}
          />
        </div>
      ) : null}
      {characters.length === 0 && !adding ? (
        <p className="text-sm text-neutral-500">Aucun personnage.</p>
      ) : null}
      <ul className="flex flex-col gap-3">
        {characters.map((character) => (
          <li key={character.id} className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            {editingId === character.id ? (
              <CharacterForm
                initial={character}
                submitLabel="Enregistrer"
                onSubmit={(input) => update(character.id, input)}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="font-medium">{character.name || "Sans nom"}</h3>
                  {character.personality ? (
                    <p className="mt-1 line-clamp-2 text-sm text-neutral-400">
                      {character.personality}
                    </p>
                  ) : null}
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button variant="secondary" onClick={() => setEditingId(character.id)}>
                    Modifier
                  </Button>
                  <Button variant="danger" onClick={() => setConfirmDeleteId(character.id)}>
                    Supprimer
                  </Button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Supprimer ce personnage ?"
        description="Il disparaît de cette partie seulement : le scénario n'est pas modifié."
        onConfirm={() => {
          if (confirmDeleteId !== null) void remove(confirmDeleteId);
        }}
        onCancel={() => setConfirmDeleteId(null)}
      />
    </section>
  );
}
