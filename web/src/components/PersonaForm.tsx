import { useState } from "react";
import type { FormEvent } from "react";
import type { PersonaInput } from "../api/personas";
import { Button } from "./Button";
import { TextArea } from "./TextArea";
import { TextField } from "./TextField";

const EMPTY_PERSONA: PersonaInput = {
  name: "",
  description: "",
  appearance: "",
  traits: "",
};

interface PersonaFormProps {
  initial?: PersonaInput;
  submitLabel: string;
  onSubmit: (input: PersonaInput) => Promise<void>;
  onCancel: () => void;
}

export function PersonaForm({ initial, submitLabel, onSubmit, onCancel }: PersonaFormProps) {
  const [form, setForm] = useState<PersonaInput>(initial ?? EMPTY_PERSONA);
  const [submitting, setSubmitting] = useState(false);

  function update<K extends keyof PersonaInput>(key: K, value: PersonaInput[K]) {
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
        label="Description"
        value={form.description}
        onChange={(event) => update("description", event.target.value)}
      />
      <TextArea
        label="Apparence"
        hint="Description physique uniquement : nourrit les prompts d'image. Ne mentionnez jamais le nom."
        value={form.appearance}
        onChange={(event) => update("appearance", event.target.value)}
      />
      <TextArea
        label="Traits"
        value={form.traits}
        onChange={(event) => update("traits", event.target.value)}
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
