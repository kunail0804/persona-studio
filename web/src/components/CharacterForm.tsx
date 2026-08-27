import { useState } from "react";
import type { FormEvent } from "react";
import type { CharacterInput } from "../api/characters";
import { Button } from "./Button";
import { TextArea } from "./TextArea";
import { TextField } from "./TextField";

const EMPTY_CHARACTER: CharacterInput = {
  name: "",
  appearance: "",
  personality: "",
  story: "",
  relationships: "",
  secrets: "",
};

interface CharacterFormProps {
  initial?: CharacterInput;
  submitLabel: string;
  onSubmit: (input: CharacterInput) => Promise<void>;
  onCancel: () => void;
}

export function CharacterForm({ initial, submitLabel, onSubmit, onCancel }: CharacterFormProps) {
  const [form, setForm] = useState<CharacterInput>(initial ?? EMPTY_CHARACTER);
  const [submitting, setSubmitting] = useState(false);

  function update<K extends keyof CharacterInput>(key: K, value: CharacterInput[K]) {
    setForm((previous) => ({ ...previous, [key]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    try {
      await onSubmit(form);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={(event) => void handleSubmit(event)} className="flex flex-col gap-3">
      <TextField
        label="Nom"
        value={form.name}
        onChange={(event) => update("name", event.target.value)}
        required
      />
      <TextArea
        label="Apparence"
        hint="Description physique uniquement : c'est la seule source utilisée pour dessiner ce personnage."
        value={form.appearance}
        onChange={(event) => update("appearance", event.target.value)}
      />
      <TextArea
        label="Personnalité"
        value={form.personality}
        onChange={(event) => update("personality", event.target.value)}
      />
      <TextArea
        label="Histoire"
        value={form.story}
        onChange={(event) => update("story", event.target.value)}
      />
      <TextArea
        label="Relations"
        value={form.relationships}
        onChange={(event) => update("relationships", event.target.value)}
      />
      <TextArea
        label="Secrets"
        hint="Réservé au narrateur : jamais révélé directement au joueur."
        value={form.secrets}
        onChange={(event) => update("secrets", event.target.value)}
      />
      <div className="flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onCancel} disabled={submitting}>
          Annuler
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
