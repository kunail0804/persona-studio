import { useState } from "react";
import { ApiError } from "../api/client";
import { composeImagePrompt } from "../api/images";
import { Button } from "./Button";
import { TextArea } from "./TextArea";
import { TextField } from "./TextField";

interface ImagePanelProps {
  partyId: string;
}

/**
 * The image panel: the player says what they want to see, the server turns
 * that plus the current scene into English keywords, and the result lands in
 * an editable field. Composing is the whole of it — the prompt stays here,
 * in the interface, until it is sent unchanged.
 */
export function ImagePanel({ partyId }: ImagePanelProps) {
  const [instruction, setInstruction] = useState("");
  const [prompt, setPrompt] = useState("");
  const [composing, setComposing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compose = async () => {
    const request = instruction.trim();
    if (!request || composing) return;
    setComposing(true);
    setError(null);
    try {
      const result = await composeImagePrompt(partyId, request);
      setPrompt(result.prompt);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "La composition du prompt a échoué.");
    } finally {
      setComposing(false);
    }
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Image</p>
      <div className="mt-3 flex flex-col gap-3">
        <div className="flex items-end gap-2">
          <TextField
            label="Ce que vous voulez voir"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            disabled={composing}
            placeholder="Ex.&nbsp;: Elena sur le quai au crépuscule, une lanterne bleue"
            className="flex-1"
          />
          <Button onClick={() => void compose()} disabled={composing || !instruction.trim()}>
            {composing ? "Composition…" : "Composer le prompt"}
          </Button>
        </div>
        {error ? <p className="text-sm text-red-400">{error}</p> : null}
        <TextArea
          label="Prompt"
          hint="Modifiez les mots-clés librement&nbsp;: c'est ce texte qui compte."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Le prompt composé apparaîtra ici."
        />
      </div>
    </div>
  );
}
