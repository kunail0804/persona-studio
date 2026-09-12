"""Pure logic over a ComfyUI workflow graph, in API format.

ComfyUI's API format is a flat object keyed by node id:

    {"6": {"class_type": "CLIPTextEncode",
           "inputs": {"text": "a photo of a castle", "clip": ["4", 1]}}}

Two rules the whole module rests on:

- A value that is a **list** is a link to another node's output and ComfyUI
  computes it at run time, so writing to it does nothing. A field is offered
  in the dropdowns only when its value is a literal.
- ComfyUI's other export — the one the "Export" menu writes — is the UI
  format and carries no `class_type`, so it cannot be submitted. Importing
  one fails with the name of the fix, because the export to use is
  "Export (API)" and that mistake is easy to make once a week.

Everything here is FastAPI-free and HTTP-free: validation, the field
inventory the dropdowns are built from, resolution of the active workflow,
and preparation of a graph for generation. `routes/workflows.py` owns the
HTTP surface.
"""

from __future__ import annotations

import copy
import json
import random
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from . import settings

# Stock ComfyUI seed widgets declare 0..2**64-1, but some nodes cap lower;
# 2**32-1 sits inside the declared range of every stock node.
MAX_SEED = 2**32 - 1

# System entropy for the default seed draw, created once at import because a
# parameter default must not perform a call (ruff B008).
_SYSTEM_RANDOM = random.SystemRandom()

# Unmapped seed fields whose names these are get randomised: ComfyUI caches
# on the graph, so an identical graph returns the identical image and an
# unrandomised seed makes "generate again" a silent no-op.
SEED_FIELD_NAMES = ("seed", "noise_seed")


class InvalidWorkflow(ValueError):
    """The stored or uploaded value is not an API-format workflow graph."""


class WorkflowNotReady(Exception):
    """The workflow cannot be used to generate: the message says what to fix."""


@dataclass(frozen=True)
class FieldOption:
    """One entry of a dropdown: a node id, a field on it, and a label to show."""

    node: str
    field: str
    label: str


def parse_graph(raw: Any) -> dict[str, Any]:
    """Validate an API-format graph, returning it unchanged.

    Tolerant by design: a node with no `inputs`, an unknown `class_type`, or
    an unexpected `_meta` is a real node from a real install and must import.
    Only the shape ComfyUI itself needs to submit — object nodes, string
    `class_type` — is enforced.
    """
    if not isinstance(raw, dict) or not raw:
        raise InvalidWorkflow(
            "The file must be a non-empty JSON object: a ComfyUI export in API "
            "format, not an arbitrary file."
        )
    if isinstance(raw.get("nodes"), list):
        raise InvalidWorkflow(
            "This is a UI-format export. In ComfyUI, use 'Export (API)' rather "
            "than 'Export', then import the file again."
        )
    for node_id, node in raw.items():
        if not isinstance(node, dict):
            raise InvalidWorkflow(f"Node {node_id!r} is not a JSON object.")
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            raise InvalidWorkflow(
                f"Node {node_id!r} has no 'class_type' string: the file is not "
                "an API-format export."
            )
    return raw


def parse_stored_graph(json_text: str) -> dict[str, Any]:
    """Parse a graph from its stored JSON text, raising on either failure."""
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise InvalidWorkflow(f"The stored graph is not valid JSON: {exc}") from exc
    return parse_graph(raw)


def _node_sort_key(node_id: str) -> tuple[int, int, str]:
    """Order node ids numerically when they are numbers, so 2 comes before 10."""
    if node_id.isdigit():
        return (0, int(node_id), "")
    return (1, 0, node_id)


def _node_label(graph: dict[str, Any], node_id: str) -> str:
    node = graph[node_id]
    meta = node.get("_meta")
    title = meta.get("title") if isinstance(meta, dict) else None
    if not isinstance(title, str) or not title:
        title = node["class_type"]
    return title


def _iter_fields(graph: dict[str, Any]) -> Iterator[tuple[str, str, Any]]:
    """Yield (node_id, field_name, value) for every literal-free input slot.

    A node without `inputs` — or with a malformed one — simply offers no
    fields; it must not break the inventory.
    """
    for node_id in sorted(graph, key=_node_sort_key):
        inputs = graph[node_id].get("inputs")
        if not isinstance(inputs, dict):
            continue
        for field, value in inputs.items():
            yield node_id, field, value


def _make_option(graph: dict[str, Any], node_id: str, field: str) -> FieldOption:
    return FieldOption(
        node=node_id,
        field=field,
        label=f"{_node_label(graph, node_id)} — {field} (node {node_id})",
    )


def field_options(graph: dict[str, Any]) -> tuple[list[FieldOption], list[FieldOption]]:
    """The two dropdown lists: where a prompt can go, and where a seed can go.

    The list check is the whole rule: a value that is a list is a link to
    another node's output, computed at run time, so writing to it would be a
    silent no-op. Only literal `str` values offer a prompt slot and only
    literal `int` values (never `bool`, a subclass of `int`) offer a seed
    slot — regardless of the node's class name.
    """
    prompt_options: list[FieldOption] = []
    seed_options: list[FieldOption] = []
    for node_id, field, value in _iter_fields(graph):
        if isinstance(value, str):
            prompt_options.append(_make_option(graph, node_id, field))
        elif isinstance(value, int) and not isinstance(value, bool):
            seed_options.append(_make_option(graph, node_id, field))
    return prompt_options, seed_options


def _row_is_valid(row: sqlite3.Row) -> bool:
    try:
        parse_stored_graph(row["graph"])
    except InvalidWorkflow:
        return False
    return True


def pick_valid(con: sqlite3.Connection) -> sqlite3.Row | None:
    """The most recently created workflow whose stored graph is still valid."""
    rows = con.execute("SELECT * FROM workflow ORDER BY created_at DESC").fetchall()
    for row in rows:
        if _row_is_valid(row):
            return row
    return None


def resolve_active(con: sqlite3.Connection) -> sqlite3.Row | None:
    """The active workflow row, healing the choice when it no longer holds.

    Healing happens on read and writes the setting inside the caller's
    transaction — that is deliberate, not a mistake. It is the only way the
    "exactly one active workflow, and a valid one" invariant holds without a
    background sweep, so a GET can write.
    """
    active_id = settings.get_active_workflow_id(con)
    if active_id is not None:
        row = con.execute("SELECT * FROM workflow WHERE id = ?", (active_id,)).fetchone()
        if row is not None and _row_is_valid(row):
            return row

    healed = pick_valid(con)
    if healed is None:
        if active_id is not None:
            settings.set_active_workflow_id(con, None)
        return None
    settings.set_active_workflow_id(con, healed["id"])
    return healed


_NOT_MAPPED = (
    "No prompt field is configured: choose in the settings the node and field "
    "the prompt is injected into."
)


def _mapping_message(
    graph: dict[str, Any], node: str, field: str, literal_check: Any, expected: str
) -> str | None:
    """Why a configured mapping cannot be used, or None when it holds.

    `expected` names the literal type the mapping requires ("a string", "an
    integer"): a value can fail the check in two different ways and the user
    fixes them differently — a list means the field became a link, anything
    else means the stored value drifted to another type.
    """
    node_data = graph.get(node)
    if node_data is None:
        return (
            f"The configured node {node!r} no longer exists in the graph: "
            "re-select the field in the settings."
        )
    inputs = node_data.get("inputs")
    if not isinstance(inputs, dict) or field not in inputs:
        return (
            f"The configured field {field!r} no longer exists on node {node!r}: "
            "re-select it in the settings."
        )
    value = inputs[field]
    if isinstance(value, list):
        return (
            f"The configured field {field!r} of node {node!r} is now linked to "
            "another node's output and can no longer be filled. Re-select it in "
            "the settings."
        )
    if not literal_check(value):
        return (
            f"The configured field {field!r} of node {node!r} no longer holds "
            f"{expected}: its value has another type. Re-select it in the settings."
        )
    return None


def problem(workflow: sqlite3.Row | dict[str, Any]) -> str | None:
    """What makes this workflow unusable for generation, or None when ready.

    `prepare_graph` raises exactly this message, so the rule lives here once
    and both the HTTP surface (which shows it to the user) and the generator
    (which enforces it) read the same answer.
    """
    try:
        graph = parse_stored_graph(workflow["graph"])
    except InvalidWorkflow as exc:
        return str(exc)

    node, field = workflow["prompt_node"], workflow["prompt_field"]
    if not node or not field:
        return _NOT_MAPPED
    message = _mapping_message(graph, node, field, lambda v: isinstance(v, str), "a string")
    if message is not None:
        return message

    seed_node, seed_field = workflow["seed_node"], workflow["seed_field"]
    if not seed_node or not seed_field:
        return None
    return _mapping_message(graph, seed_node, seed_field, _is_int_literal, "an integer")


def _is_int_literal(value: Any) -> bool:
    # The bool exclusion is deliberate, not an oversight: a real seed is an
    # integer, and a bool field named "seed" must stay untouched. Writing a
    # random int into a boolean widget would send ComfyUI a wrong-typed value;
    # leaving the graph byte-identical instead is honest, even though it means
    # ComfyUI's cache then serves the same image. A custom node that exposes a
    # boolean "seed" simply cannot be used as a seed mapping — the user maps
    # the node's real integer field.
    return isinstance(value, int) and not isinstance(value, bool)


def prepare_graph(
    workflow: sqlite3.Row | dict[str, Any],
    prompt: str,
    *,
    seed: int | None = None,
    # A callable over system entropy, not the `random` module's PRNG: an
    # image seed is not a secret and `randint` is not a real weakness, but
    # the switch costs nothing here — one draw per field, never in a loop —
    # and the tests replace it wholesale to stay deterministic.
    rng: Callable[[int, int], int] = _SYSTEM_RANDOM.randint,
) -> dict[str, Any]:
    """A private deep copy of the stored graph, with the prompt injected.

    The copy is what makes the returned graph owned by the caller: it may
    mutate it or keep it around without a later `prepare_graph` call ever
    seeing the changes. (`parse_stored_graph` already allocates a fresh tree
    per call, so today the copy is defence in depth against that ever being
    memoised — the stored JSON text itself cannot be reached by mutation.)
    Raises `WorkflowNotReady` with the message `problem()` returns when the
    workflow cannot be used.

    The seed asymmetry is a decision, not an oversight: with no seed mapping,
    every seed field is randomised — otherwise ComfyUI's graph cache returns
    the identical image and rerolling does nothing. With a seed mapping, the
    user chose to control that one field, so it alone is written and every
    other seed field is left exactly as the graph carries it.
    """
    message = problem(workflow)
    if message is not None:
        raise WorkflowNotReady(message)

    graph = copy.deepcopy(parse_stored_graph(workflow["graph"]))
    graph[workflow["prompt_node"]]["inputs"][workflow["prompt_field"]] = prompt

    seed_node, seed_field = workflow["seed_node"], workflow["seed_field"]
    if seed_node and seed_field:
        value = seed if seed is not None else rng(0, MAX_SEED)
        graph[seed_node]["inputs"][seed_field] = value
    else:
        for node_id in sorted(graph, key=_node_sort_key):
            inputs = graph[node_id].get("inputs")
            if not isinstance(inputs, dict):
                continue
            for field, current in list(inputs.items()):
                if field in SEED_FIELD_NAMES and _is_int_literal(current):
                    inputs[field] = seed if seed is not None else rng(0, MAX_SEED)
    return graph


def check_mapping(
    graph: dict[str, Any],
    *,
    node: str,
    field: str,
    kind: str,
) -> str | None:
    """Whether a mapping proposed by the client names a field the graph offers.

    `kind` is "prompt" or "seed" and decides the literal type required; the
    returned message names the node or field so a stale dropdown is fixable.
    An empty mapping is legal — it means "not configured", not "invalid".
    """
    if not node and not field:
        return None
    if not node or not field:
        return f"The {kind} mapping must name both a node and a field."
    if kind == "prompt":
        return _mapping_message(graph, node, field, lambda v: isinstance(v, str), "a string")
    return _mapping_message(graph, node, field, _is_int_literal, "an integer")
