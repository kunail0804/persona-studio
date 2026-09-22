"""ComfyUI workflows: import an API-format graph export, pick from a dropdown
where the prompt is injected, and keep exactly one active.

The prompt is the only thing this application writes into a graph. Seeds are
the workflow's own business — see `workflows.prepare_graph` for why the
application used to inject them and why it no longer does.

The active workflow is a `setting`, exactly like the active persona. Every
read path goes through `workflows.resolve_active`, which heals the choice on
read when it no longer holds — so a GET can write; that is deliberate.

This module never talks to ComfyUI over HTTP. Submitting a graph is issue
#17, a later pull request; this one only stores graphs and maps them.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, settings, workflows

router = APIRouter(tags=["workflows"])


class FieldOption(BaseModel):
    node: str
    field: str
    label: str


class WorkflowOut(BaseModel):
    id: str
    name: str
    is_active: bool
    prompt_node: str
    prompt_field: str
    prompt_options: list[FieldOption]
    # Why nothing can be generated with this workflow, or None when it is
    # ready. Surfaced so the interface can say why, not just that.
    problem: str | None
    created_at: float


class WorkflowInput(BaseModel):
    name: str
    graph: dict[str, Any]


class WorkflowPatch(BaseModel):
    name: str
    prompt_node: str = ""
    prompt_field: str = ""


class ActiveWorkflowInput(BaseModel):
    id: str


def _get_workflow_row(con: sqlite3.Connection, workflow_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM workflow WHERE id = ?", (workflow_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id!r} not found")
    return row


def _workflow_item(row: sqlite3.Row, active_id: str | None) -> WorkflowOut:
    prompt_options: list[FieldOption] = []
    try:
        graph = workflows.parse_stored_graph(row["graph"])
    except workflows.InvalidWorkflow:
        # A corrupted graph stays listed: the user can delete it, and the
        # problem text below says why it is unusable.
        graph = None
    if graph is not None:
        prompt_options = [
            FieldOption(node=o.node, field=o.field, label=o.label)
            for o in workflows.field_options(graph)
        ]
    return WorkflowOut(
        id=row["id"],
        name=row["name"],
        is_active=row["id"] == active_id,
        prompt_node=row["prompt_node"],
        prompt_field=row["prompt_field"],
        prompt_options=prompt_options,
        problem=workflows.problem(row),
        created_at=row["created_at"],
    )


@router.get("/workflows", response_model=list[WorkflowOut])
def list_workflows() -> list[WorkflowOut]:
    # resolve_active may heal, a read-check-write: with the default deferred
    # transaction the write lock would only be taken at the heal itself, and a
    # concurrent explicit choice committed in between would be overwritten.
    with db.connect(immediate=True) as con:
        active = workflows.resolve_active(con)
        active_id = active["id"] if active is not None else None
        rows = con.execute("SELECT * FROM workflow ORDER BY created_at").fetchall()
    return [_workflow_item(row, active_id) for row in rows]


@router.post("/workflows", response_model=WorkflowOut, status_code=201)
def create_workflow(body: WorkflowInput) -> WorkflowOut:
    try:
        workflows.parse_graph(body.graph)
    except workflows.InvalidWorkflow as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workflow_id = uuid.uuid4().hex
    now = time.time()
    with db.connect() as con:
        con.execute(
            "INSERT INTO workflow (id, name, graph, created_at) VALUES (?, ?, ?, ?)",
            (workflow_id, body.name, json.dumps(body.graph), now),
        )
        # The first import becomes active by itself: one workflow and nothing
        # active is a dead end the user would have to guess their way out of.
        # Default transaction is safe here: the INSERT above already holds the
        # write lock, so the setting read below cannot go stale — keep the
        # INSERT first if this block is ever reordered.
        if settings.get_active_workflow_id(con) is None:
            settings.set_active_workflow_id(con, workflow_id)
        row = _get_workflow_row(con, workflow_id)
        active = workflows.resolve_active(con)
    active_id = active["id"] if active is not None else None
    return _workflow_item(row, active_id)


@router.patch("/workflows/{workflow_id}", response_model=WorkflowOut)
def update_workflow(workflow_id: str, body: WorkflowPatch) -> WorkflowOut:
    with db.connect() as con:
        row = _get_workflow_row(con, workflow_id)
        try:
            graph = workflows.parse_stored_graph(row["graph"])
        except workflows.InvalidWorkflow as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # The dropdown can go stale between two tabs: reject a mapping that
        # names a node or field the graph does not offer, saying which.
        message = workflows.check_mapping(graph, node=body.prompt_node, field=body.prompt_field)
        if message is not None:
            raise HTTPException(status_code=422, detail=message)

        con.execute(
            "UPDATE workflow SET name = ?, prompt_node = ?, prompt_field = ? WHERE id = ?",
            (body.name, body.prompt_node, body.prompt_field, workflow_id),
        )
        row = _get_workflow_row(con, workflow_id)
        # Default transaction is safe here: the UPDATE above already holds the
        # write lock, so the setting read below cannot go stale.
        active_id = settings.get_active_workflow_id(con)
    return _workflow_item(row, active_id)


@router.put("/workflows/active", response_model=WorkflowOut)
def set_active_workflow(body: ActiveWorkflowInput) -> WorkflowOut:
    # The row check reads, and the setting write depends on it: on the default
    # transaction the row could be deleted in between, so the write lock is
    # taken up front. Activation refuses only what can never be used as it
    # stands — a stored graph that does not parse as an API-format graph. A
    # valid but unmapped workflow is activatable: the first import is unmapped
    # and auto-activated, so refusing it here would make a reachable state
    # unreachable by choice. Whether generation is possible yet stays the
    # read-only `problem` field and the generator's own refusal.
    with db.connect(immediate=True) as con:
        row = _get_workflow_row(con, body.id)
        try:
            workflows.parse_stored_graph(row["graph"])
        except workflows.InvalidWorkflow as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        settings.set_active_workflow_id(con, body.id)
    return _workflow_item(row, body.id)


@router.delete("/workflows/{workflow_id}", status_code=204)
def delete_workflow(workflow_id: str) -> None:
    with db.connect() as con:
        _get_workflow_row(con, workflow_id)
        con.execute("DELETE FROM workflow WHERE id = ?", (workflow_id,))
        # If the deleted one was active, healing picks the next valid
        # workflow right here, in the same transaction.
        if settings.get_active_workflow_id(con) == workflow_id:
            healed = workflows.pick_valid(con)
            settings.set_active_workflow_id(con, healed["id"] if healed is not None else None)
