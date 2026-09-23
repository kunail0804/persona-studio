import { useState } from "react";
import { ApiError } from "../api/client";
import type { LoreEntry, Place } from "../api/world";
import {
  createLore,
  createPlace,
  deleteLore,
  deletePlace,
  updateLore,
  updatePlace,
} from "../api/world";
import { Button } from "./Button";
import { TextArea } from "./TextArea";
import { TextField } from "./TextField";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

interface PlacesSectionProps {
  scenarioId: string;
  places: Place[];
  onChanged: () => Promise<void>;
  onError: (message: string) => void;
}

/**
 * The scenario's places. Every place rides in every narrator prompt, like a
 * character: the narrator has to know a place exists to take the story there.
 */
export function PlacesSection({ scenarioId, places, onChanged, onError }: PlacesSectionProps) {
  const [adding, setAdding] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const save = async (id: number | null, input: { name: string; description: string; atmosphere: string }) => {
    try {
      if (id === null) {
        await createPlace(scenarioId, input);
        setAdding(false);
      } else {
        await updatePlace(scenarioId, id, input);
        setEditingId(null);
      }
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible d'enregistrer le lieu."));
    }
  };

  const remove = async (id: number) => {
    try {
      await deletePlace(scenarioId, id);
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible de supprimer le lieu."));
    }
  };

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Lieux</h2>
        {adding ? null : <Button onClick={() => setAdding(true)}>Ajouter un lieu</Button>}
      </div>
      <p className="text-sm text-neutral-500">
        Le narrateur connaît tous les lieux à chaque tour. Dans un prompt d'image, le nom d'un lieu
        est toujours remplacé par sa description&nbsp;: tu peux nommer librement, rien ne partira.
      </p>
      {adding ? (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
          <PlaceForm onSubmit={(input) => void save(null, input)} onCancel={() => setAdding(false)} />
        </div>
      ) : null}
      <ul className="flex flex-col gap-3">
        {places.map((place) => (
          <li key={place.id} className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            {editingId === place.id ? (
              <PlaceForm
                initial={place}
                onSubmit={(input) => void save(place.id, input)}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="font-medium">{place.name || "Sans nom"}</h3>
                  {place.description ? (
                    <p className="mt-1 line-clamp-2 text-sm text-neutral-400">{place.description}</p>
                  ) : null}
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button variant="secondary" onClick={() => setEditingId(place.id)}>
                    Modifier
                  </Button>
                  <Button variant="danger" onClick={() => void remove(place.id)}>
                    Supprimer
                  </Button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

interface PlaceFormProps {
  initial?: Place;
  onSubmit: (input: { name: string; description: string; atmosphere: string }) => void;
  onCancel: () => void;
}

function PlaceForm({ initial, onSubmit, onCancel }: PlaceFormProps) {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [atmosphere, setAtmosphere] = useState(initial?.atmosphere ?? "");
  return (
    <div className="flex flex-col gap-3">
      <TextField label="Nom" value={name} onChange={(e) => setName(e.target.value)} required />
      <TextArea
        label="Description"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        hint="Ce qu'on voit : factuel et visuel. C'est ce texte qui remplace le nom dans un prompt d'image."
      />
      <TextArea
        label="Atmosphère"
        value={atmosphere}
        onChange={(e) => setAtmosphere(e.target.value)}
        hint="Ce qu'on ressent : odeurs, sons, lumière. Pour le narrateur seulement."
      />
      <div className="flex gap-2">
        <Button onClick={() => onSubmit({ name, description, atmosphere })} disabled={!name.trim()}>
          Enregistrer
        </Button>
        <Button variant="secondary" onClick={onCancel}>
          Annuler
        </Button>
      </div>
    </div>
  );
}

interface LoreSectionProps {
  scenarioId: string;
  entries: LoreEntry[];
  onChanged: () => Promise<void>;
  onError: (message: string) => void;
}

/**
 * The lorebook. An entry reaches the narrator only on a turn whose recent
 * messages use one of its keywords — which is what lets a world hold many
 * facts without every prompt carrying all of them.
 */
export function LoreSection({ scenarioId, entries, onChanged, onError }: LoreSectionProps) {
  const [adding, setAdding] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const save = async (id: number | null, input: { keywords: string[]; text: string }) => {
    try {
      if (id === null) {
        await createLore(scenarioId, input);
        setAdding(false);
      } else {
        await updateLore(scenarioId, id, input);
        setEditingId(null);
      }
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible d'enregistrer l'entrée."));
    }
  };

  const remove = async (id: number) => {
    try {
      await deleteLore(scenarioId, id);
      await onChanged();
    } catch (err) {
      onError(messageFor(err, "Impossible de supprimer l'entrée."));
    }
  };

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Lorebook</h2>
        {adding ? null : <Button onClick={() => setAdding(true)}>Ajouter une entrée</Button>}
      </div>
      <p className="text-sm text-neutral-500">
        Une entrée n'est envoyée au narrateur que si l'un de ses mots-clés apparaît dans les dix
        derniers messages, et huit au plus par tour. Le panneau « Prompt envoyé au narrateur » d'une
        partie montre lesquelles sont entrées. Un objet à retrouver dans l'histoire est une entrée
        dont le mot-clé est son nom.
      </p>
      {adding ? (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
          <LoreForm onSubmit={(input) => void save(null, input)} onCancel={() => setAdding(false)} />
        </div>
      ) : null}
      <ul className="flex flex-col gap-3">
        {entries.map((entry) => (
          <li key={entry.id} className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            {editingId === entry.id ? (
              <LoreForm
                initial={entry}
                onSubmit={(input) => void save(entry.id, input)}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex flex-wrap gap-1">
                    {entry.keywords.length === 0 ? (
                      <span className="text-xs text-amber-400">
                        Aucun mot-clé : cette entrée n'entrera jamais dans le prompt.
                      </span>
                    ) : (
                      entry.keywords.map((keyword) => (
                        <span
                          key={keyword}
                          className="rounded bg-neutral-800 px-2 py-0.5 text-xs text-neutral-300"
                        >
                          {keyword}
                        </span>
                      ))
                    )}
                  </div>
                  <p className="mt-2 line-clamp-2 text-sm text-neutral-400">{entry.text}</p>
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button variant="secondary" onClick={() => setEditingId(entry.id)}>
                    Modifier
                  </Button>
                  <Button variant="danger" onClick={() => void remove(entry.id)}>
                    Supprimer
                  </Button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

interface LoreFormProps {
  initial?: LoreEntry;
  onSubmit: (input: { keywords: string[]; text: string }) => void;
  onCancel: () => void;
}

function LoreForm({ initial, onSubmit, onCancel }: LoreFormProps) {
  const [keywords, setKeywords] = useState(initial?.keywords.join(", ") ?? "");
  const [text, setText] = useState(initial?.text ?? "");
  // Split on commas here; the server strips, drops blanks and removes
  // repeats, so what comes back is what is stored.
  const parsed = keywords.split(",").map((k) => k.trim()).filter((k) => k !== "");
  return (
    <div className="flex flex-col gap-3">
      <TextField
        label="Mots-clés"
        value={keywords}
        onChange={(e) => setKeywords(e.target.value)}
        hint="Séparés par des virgules. La casse ne compte pas, les mots entiers seulement : « Ana » ne déclenche pas « banana »."
      />
      <TextArea
        label="Texte"
        value={text}
        onChange={(e) => setText(e.target.value)}
        hint="Ce que le narrateur sait quand un mot-clé est prononcé."
      />
      <div className="flex gap-2">
        <Button
          onClick={() => onSubmit({ keywords: parsed, text })}
          disabled={parsed.length === 0 || !text.trim()}
        >
          Enregistrer
        </Button>
        <Button variant="secondary" onClick={onCancel}>
          Annuler
        </Button>
      </div>
    </div>
  );
}
