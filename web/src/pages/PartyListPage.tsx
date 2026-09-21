import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError } from "../api/client";
import { deleteParty, listParties, renameParty } from "../api/parties";
import type { PartySummary } from "../api/parties";
import { Button } from "../components/Button";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { TextField } from "../components/TextField";

function messageFor(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.detail : fallback;
}

export function PartyListPage() {
  const [parties, setParties] = useState<PartySummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  async function loadParties() {
    try {
      setParties(await listParties());
      setError(null);
    } catch (err) {
      setError(messageFor(err, "Impossible de charger les parties."));
    }
  }

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- fetch-on-mount: loadParties() sets state after an await, not synchronously in the effect body.
    void loadParties();
  }, []);

  async function handleRename(partyId: string, label: string) {
    try {
      await renameParty(partyId, label);
      setRenamingId(null);
      await loadParties();
    } catch (err) {
      setError(messageFor(err, "Impossible de renommer la partie."));
    }
  }

  async function handleDelete(partyId: string) {
    setConfirmDeleteId(null);
    try {
      await deleteParty(partyId);
      await loadParties();
    } catch (err) {
      setError(messageFor(err, "Impossible de supprimer la partie."));
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Parties</h1>
      </div>
      {error ? <p className="text-sm text-red-400">{error}</p> : null}
      {parties.length === 0 ? (
        <p className="text-neutral-500">
          Aucune partie pour l'instant. Ouvrez un scénario pour en démarrer une.
        </p>
      ) : (
        <ul className="flex flex-col gap-3">
          {parties.map((party) => (
            <PartyListItem
              key={party.id}
              party={party}
              isRenaming={renamingId === party.id}
              onStartRename={() => setRenamingId(party.id)}
              onCancelRename={() => setRenamingId(null)}
              onRename={(label) => void handleRename(party.id, label)}
              onDelete={() => setConfirmDeleteId(party.id)}
            />
          ))}
        </ul>
      )}
      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Supprimer cette partie ?"
        description="Cette action supprime aussi tous ses messages. Elle est définitive."
        confirmLabel="Supprimer"
        onConfirm={() => {
          if (confirmDeleteId !== null) void handleDelete(confirmDeleteId);
        }}
        onCancel={() => setConfirmDeleteId(null)}
      />
    </div>
  );
}

interface PartyListItemProps {
  party: PartySummary;
  isRenaming: boolean;
  onStartRename: () => void;
  onCancelRename: () => void;
  onRename: (label: string) => void;
  onDelete: () => void;
}

function PartyListItem({
  party,
  isRenaming,
  onStartRename,
  onCancelRename,
  onRename,
  onDelete,
}: PartyListItemProps) {
  if (isRenaming) {
    return (
      <li className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <PartyRenameForm
          initial={party.label}
          submitLabel="Enregistrer"
          onSubmit={onRename}
          onCancel={onCancelRename}
        />
      </li>
    );
  }
  return (
    <li>
      <div className="flex items-center justify-between gap-4 rounded-lg border border-neutral-800 bg-neutral-900 p-4 hover:border-neutral-600">
        <Link to={`/parties/${party.id}`} className="min-w-0">
          <h2 className="truncate text-lg font-medium">{party.label}</h2>
          <p className="mt-1 truncate text-sm text-neutral-500">{party.scenarioTitle}</p>
        </Link>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="secondary" onClick={onStartRename}>
            Renommer
          </Button>
          <Button variant="danger" onClick={onDelete}>
            Supprimer
          </Button>
        </div>
      </div>
    </li>
  );
}

interface PartyRenameFormProps {
  initial: string;
  submitLabel: string;
  onSubmit: (label: string) => void;
  onCancel: () => void;
}

function PartyRenameForm({ initial, submitLabel, onSubmit, onCancel }: PartyRenameFormProps) {
  const [label, setLabel] = useState(initial);
  return (
    <div className="flex items-end gap-2">
      <TextField
        label="Nom de la partie"
        value={label}
        onChange={(event) => setLabel(event.target.value)}
      />
      <Button onClick={() => onSubmit(label)} disabled={label.trim() === ""}>
        {submitLabel}
      </Button>
      <Button variant="secondary" onClick={onCancel}>
        Annuler
      </Button>
    </div>
  );
}
