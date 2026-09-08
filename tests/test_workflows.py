from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, settings, workflows


@pytest.fixture(autouse=True)
def _clean_workflows(client: TestClient) -> None:
    """The active workflow id is global state in the shared test database.

    Depends on `client` so the app's startup migration has run first — the
    tables must exist even when this file runs alone.
    """
    with db.connect() as con:
        con.execute("DELETE FROM workflow")
        con.execute("DELETE FROM setting")


class SequentialRng:
    """A deterministic rng whose randint returns 1, 2, 3, … in order."""

    def __init__(self) -> None:
        self.draws: list[int] = []

    def randint(self, low: int, high: int) -> int:
        assert low == 0 and high == workflows.MAX_SEED
        self.draws.append(len(self.draws) + 1)
        return self.draws[-1]


def api_graph() -> dict[str, Any]:
    """A small hand-written API-format graph exercising every rule."""
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors", "stop_at_clip_layer": -1},
            "_meta": {"title": "Loader"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            # `clip` is a link: it must never be offered, while `text` must be.
            "inputs": {"text": "a castle", "clip": ["4", 1]},
            "_meta": {"title": "Positive prompt"},
        },
        "7": {
            "class_type": "KSampler",
            # `flag` is a bool: `bool` is a subclass of `int`, it must not
            # pass for a seed.
            "inputs": {"seed": 42, "noise_seed": 43, "model": ["4", 0], "flag": True},
        },
        "10": {"class_type": "UnknownCustomNode"},
    }


UI_GRAPH = {
    "id": 1,
    "nodes": [{"id": 6, "type": "CLIPTextEncode"}],
    "links": [],
}


def _workflow_row(workflow_id: str) -> sqlite3.Row:
    with db.connect() as con:
        row = con.execute("SELECT * FROM workflow WHERE id = ?", (workflow_id,)).fetchone()
    assert row is not None
    return row


def _mapped_workflow(graph: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A workflow-like mapping for direct calls to `problem` and `prepare_graph`."""
    row = {
        "graph": json.dumps(graph),
        "prompt_node": "",
        "prompt_field": "",
        "seed_node": "",
        "seed_field": "",
    }
    row.update(overrides)
    return row


def _import(client: TestClient, name: str, graph: Any) -> dict:
    response = client.post("/api/workflows", json={"name": name, "graph": graph})
    assert response.status_code == 201
    return response.json()


def _set_mapping(
    client: TestClient, workflow_id: str, name: str, prompt_node: str, prompt_field: str
) -> None:
    response = client.patch(
        f"/api/workflows/{workflow_id}",
        json={
            "name": name,
            "prompt_node": prompt_node,
            "prompt_field": prompt_field,
            "seed_node": "",
            "seed_field": "",
        },
    )
    assert response.status_code == 200


# --- 1. Import: API format accepted, UI format rejected with the fix ----------


def test_valid_api_graph_imports_and_lists(client: TestClient) -> None:
    created = _import(client, "Krea", api_graph())
    assert created["is_active"] is True
    # A freshly imported workflow is not mapped yet: unusable until the user
    # picks the injection point, and the problem text says exactly that.
    assert created["problem"] is not None
    assert "No prompt field" in created["problem"]

    listed = client.get("/api/workflows").json()
    assert [row["name"] for row in listed] == ["Krea"]


def test_ui_format_export_is_rejected_and_names_the_fix(client: TestClient) -> None:
    response = client.post("/api/workflows", json={"name": "UI", "graph": UI_GRAPH})
    assert response.status_code == 422
    assert "Export (API)" in response.json()["detail"]
    assert client.get("/api/workflows").json() == []


def test_empty_graph_is_rejected(client: TestClient) -> None:
    response = client.post("/api/workflows", json={"name": "Empty", "graph": {}})
    assert response.status_code == 422
    assert "API" in response.json()["detail"]


def test_node_that_is_not_an_object_is_rejected_naming_the_node(client: TestClient) -> None:
    response = client.post(
        "/api/workflows", json={"name": "X", "graph": {"6": ["not", "an object"]}}
    )
    assert response.status_code == 422
    assert "6" in response.json()["detail"]


def test_node_without_class_type_is_rejected(client: TestClient) -> None:
    response = client.post("/api/workflows", json={"name": "X", "graph": {"6": {"inputs": {}}}})
    assert response.status_code == 422
    assert "class_type" in response.json()["detail"]


def test_non_numeric_node_ids_sort_after_numeric_ones() -> None:
    graph = {
        "10": {"class_type": "A", "inputs": {"text": "b"}},
        "2": {"class_type": "B", "inputs": {"text": "a"}},
        "434:393": {"class_type": "C", "inputs": {"text": "c"}},
    }
    prompts, _ = workflows.field_options(graph)
    assert [o.node for o in prompts] == ["2", "10", "434:393"]


def test_half_set_mapping_is_rejected(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())
    response = client.patch(
        f"/api/workflows/{workflow['id']}",
        json={
            "name": "Krea",
            "prompt_node": "6",
            "prompt_field": "",
            "seed_node": "",
            "seed_field": "",
        },
    )
    assert response.status_code == 422
    assert "both a node and a field" in response.json()["detail"]


def test_patch_on_a_corrupted_graph_is_rejected(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())
    with db.connect() as con:
        con.execute("UPDATE workflow SET graph = 'not-json{' WHERE id = ?", (workflow["id"],))

    response = client.patch(
        f"/api/workflows/{workflow['id']}",
        json={
            "name": "Krea",
            "prompt_node": "6",
            "prompt_field": "text",
            "seed_node": "",
            "seed_field": "",
        },
    )
    assert response.status_code == 422
    assert "not valid JSON" in response.json()["detail"]


def test_non_json_payload_is_rejected_by_the_body_model(client: TestClient) -> None:
    response = client.post("/api/workflows", json={"name": "X", "graph": "a-string"})
    assert response.status_code == 422


# --- 2. Linked fields are never offered; bool is never a seed -----------------


def test_linked_fields_are_never_offered(client: TestClient) -> None:
    _import(client, "Krea", api_graph())

    graph = api_graph()
    assert isinstance(graph["6"]["inputs"]["clip"], list)
    prompts, seeds = workflows.field_options(graph)
    offered = {(o.node, o.field) for o in prompts} | {(o.node, o.field) for o in seeds}
    assert ("6", "clip") not in offered
    assert ("7", "model") not in offered

    body = client.get("/api/workflows").json()[0]
    offered_fields = {(o["field"]) for o in body["prompt_options"]}
    assert "clip" not in offered_fields
    # A str literal on the same node is offered, with the node id in the label.
    text_option = next(o for o in body["prompt_options"] if o["field"] == "text")
    assert text_option["node"] == "6"
    assert "Positive prompt" in text_option["label"]
    assert "6" in text_option["label"]

    seed_fields = {o["field"] for o in body["seed_options"]}
    assert "flag" not in seed_fields


def test_seed_options_are_int_literals_only(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())
    seed_options = {(o["node"], o["field"]) for o in workflow["seed_options"]}
    assert seed_options == {("7", "seed"), ("7", "noise_seed"), ("4", "stop_at_clip_layer")}


def test_option_labels_fall_back_to_class_type_and_sort_is_stable(
    client: TestClient,
) -> None:
    workflow = _import(client, "Krea", api_graph())
    # Node 7 has no _meta: the label falls back to class_type.
    seed_label = next(o["label"] for o in workflow["seed_options"] if o["field"] == "seed")
    assert "KSampler" in seed_label
    prompt_nodes = [o["node"] for o in workflow["prompt_options"]]
    assert prompt_nodes == sorted(prompt_nodes, key=lambda n: int(n))


# --- 3. Tolerance: unknown nodes must import and not break the inventory ------


def test_unknown_nodes_import_without_breaking_the_inventory(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())
    # Node 10 has no `inputs` and an unknown class_type: the import did not
    # raise, the only problem is the missing mapping, and the other nodes'
    # fields are still offered.
    assert workflow["problem"] == (
        "No prompt field is configured: choose in the settings the node and "
        "field the prompt is injected into."
    )
    offered_nodes = {o["node"] for o in workflow["prompt_options"]}
    assert offered_nodes == {"4", "6"}


# --- 4. The mapping round-trips through PATCH, stale ones are rejected --------


def test_mapping_round_trips_through_patch(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())
    _set_mapping(client, workflow["id"], "Krea", "6", "text")

    body = client.get("/api/workflows").json()
    row = next(w for w in body if w["id"] == workflow["id"])
    assert row["prompt_node"] == "6"
    assert row["prompt_field"] == "text"
    assert row["problem"] is None


def test_patch_rejects_a_mapping_the_graph_does_not_offer(client: TestClient) -> None:
    workflow = _import(client, "Krea", api_graph())

    def patch(prompt_node: str, prompt_field: str, seed_node: str = "", seed_field: str = ""):
        return client.patch(
            f"/api/workflows/{workflow['id']}",
            json={
                "name": "Krea",
                "prompt_node": prompt_node,
                "prompt_field": prompt_field,
                "seed_node": seed_node,
                "seed_field": seed_field,
            },
        )

    missing_field = patch("6", "title")
    assert missing_field.status_code == 422
    assert "title" in missing_field.json()["detail"]

    missing_node = patch("99", "text")
    assert missing_node.status_code == 422
    assert "99" in missing_node.json()["detail"]

    # A linked field is a real field, but it must still be rejected.
    assert patch("6", "clip").status_code == 422

    # A seed mapping must name an int literal, not a string one.
    assert patch("6", "text", seed_node="6", seed_field="text").status_code == 422


# --- 5. Healing ---------------------------------------------------------------


def test_deleting_the_active_workflow_activates_another(client: TestClient) -> None:
    first = _import(client, "First", api_graph())
    second = _import(client, "Second", api_graph())
    assert client.put("/api/workflows/active", json={"id": second["id"]}).status_code == 200

    assert client.delete(f"/api/workflows/{second['id']}").status_code == 204

    listed = client.get("/api/workflows").json()
    assert next(w for w in listed if w["id"] == first["id"])["is_active"] is True
    with db.connect() as con:
        assert settings.get_active_workflow_id(con) == first["id"]


def test_corrupting_the_active_graph_heals_to_a_valid_one(client: TestClient) -> None:
    first = _import(client, "First", api_graph())
    second = _import(client, "Second", api_graph())
    client.put("/api/workflows/active", json={"id": first["id"]})

    with db.connect() as con:
        con.execute("UPDATE workflow SET graph = 'not-json{' WHERE id = ?", (first["id"],))

    # Healing happens on read and persists the choice.
    listed = client.get("/api/workflows").json()
    row = next(w for w in listed if w["id"] == second["id"])
    assert row["is_active"] is True
    with db.connect() as con:
        assert settings.get_active_workflow_id(con) == second["id"]


def test_corrupted_graphs_stay_listed_with_their_problem(client: TestClient) -> None:
    workflow = _import(client, "Broken", api_graph())
    with db.connect() as con:
        con.execute("UPDATE workflow SET graph = 'not-json{' WHERE id = ?", (workflow["id"],))

    listed = client.get("/api/workflows").json()
    assert listed[0]["id"] == workflow["id"]
    assert listed[0]["problem"] is not None
    assert listed[0]["prompt_options"] == []


def test_no_valid_workflow_left_clears_the_setting_and_the_list_still_answers(
    client: TestClient,
) -> None:
    workflow = _import(client, "Only", api_graph())
    with db.connect() as con:
        con.execute("UPDATE workflow SET graph = 'not-json{' WHERE id = ?", (workflow["id"],))

    listed = client.get("/api/workflows").json()
    assert [row["is_active"] for row in listed] == [False]
    with db.connect() as con:
        assert settings.get_active_workflow_id(con) is None


# --- 6. The first import becomes active on its own ----------------------------


def test_first_import_becomes_active_and_later_ones_do_not_steal_it(
    client: TestClient,
) -> None:
    first = _import(client, "First", api_graph())
    assert first["is_active"] is True

    second = _import(client, "Second", api_graph())
    assert second["is_active"] is False
    listed = client.get("/api/workflows").json()
    assert next(w for w in listed if w["id"] == first["id"])["is_active"] is True


# --- 7. prepare_graph injects the prompt and nothing else ---------------------


def test_prepare_graph_injects_the_prompt_and_touches_nothing_else() -> None:
    workflow = _mapped_workflow(api_graph(), prompt_node="6", prompt_field="text")
    rng = SequentialRng()

    prepared = workflows.prepare_graph(workflow, "un château en ruine", rng=rng.randint)

    expected = api_graph()
    expected["6"]["inputs"]["text"] = "un château en ruine"
    # No seed mapping: the two seed literals were randomised, in draw order.
    expected["7"]["inputs"]["seed"] = 1
    expected["7"]["inputs"]["noise_seed"] = 2
    assert prepared == expected
    # The stored graph is never mutated.
    assert json.loads(workflow["graph"])["6"]["inputs"]["text"] == "a castle"


def test_prepare_graph_deep_copies_linked_values_untouched() -> None:
    workflow = _mapped_workflow(api_graph(), prompt_node="6", prompt_field="text")
    prepared = workflows.prepare_graph(workflow, "new prompt", rng=SequentialRng().randint)
    assert prepared["6"]["inputs"]["clip"] == ["4", 1]
    assert prepared["7"]["inputs"]["model"] == ["4", 0]


# --- 8. Unmapped refuses with the same message prepare would raise ------------


def test_unmapped_workflow_refuses_with_a_message() -> None:
    workflow = _mapped_workflow(api_graph())
    with pytest.raises(workflows.WorkflowNotReady, match="No prompt field"):
        workflows.prepare_graph(workflow, "prompt")
    assert workflows.problem(workflow) is not None
    assert "prompt" in str(workflows.problem(workflow))


def test_mapped_node_missing_from_graph_refuses_the_same_way() -> None:
    workflow = _mapped_workflow(api_graph(), prompt_node="99", prompt_field="text")
    message = workflows.problem(workflow)
    assert message is not None and "99" in message
    with pytest.raises(workflows.WorkflowNotReady, match="99"):
        workflows.prepare_graph(workflow, "prompt")


def test_mapped_field_now_linked_refuses() -> None:
    graph = api_graph()
    graph["6"]["inputs"]["text"] = ["4", 0]
    workflow = _mapped_workflow(graph, prompt_node="6", prompt_field="text")
    assert workflows.problem(workflow) is not None


def test_half_mapped_workflow_refuses() -> None:
    workflow = _mapped_workflow(api_graph(), prompt_node="6", prompt_field="")
    assert workflows.problem(workflow) is not None


# --- 9. No seed mapping randomises every seed field ---------------------------


def test_no_seed_mapping_randomises_every_seed_field() -> None:
    graph = api_graph()
    with db.connect() as con:
        con.execute(
            "INSERT INTO workflow (id, name, graph, created_at) VALUES (?, ?, ?, ?)",
            ("w1", "Krea", json.dumps(graph), 1.0),
        )
    row = _workflow_row("w1")
    workflow = _mapped_workflow(graph, prompt_node="6", prompt_field="text")
    rng = SequentialRng()

    prepared = workflows.prepare_graph(workflow, "new prompt", rng=rng.randint)

    # Both seed literals changed, each drew its own fresh value in order.
    assert prepared["7"]["inputs"]["seed"] == 1
    assert prepared["7"]["inputs"]["noise_seed"] == 2
    assert rng.draws == [1, 2]
    # A seed field holding a link is untouched.
    assert prepared["7"]["inputs"]["model"] == ["4", 0]
    # Everything else is exactly the stored graph plus the prompt.
    expected = api_graph()
    expected["6"]["inputs"]["text"] = "new prompt"
    expected["7"]["inputs"]["seed"] = 1
    expected["7"]["inputs"]["noise_seed"] = 2
    assert prepared == expected
    # The stored graph in the database is unchanged.
    assert json.loads(row["graph"])["7"]["inputs"]["seed"] == 42


def test_seed_field_names_beyond_ksampler_are_randomised() -> None:
    graph = {
        "3": {"class_type": "Custom", "inputs": {"noise_seed": 7}},
        "8": {"class_type": "Other", "inputs": {"seed": 9}},
        "9": {"class_type": "Unrelated", "inputs": {"steps": 20, "text": "hi"}},
    }
    workflow = _mapped_workflow(graph, prompt_node="9", prompt_field="text")
    rng = SequentialRng()
    prepared = workflows.prepare_graph(workflow, "p", rng=rng.randint)
    assert prepared["3"]["inputs"]["noise_seed"] == 1
    assert prepared["8"]["inputs"]["seed"] == 2
    assert prepared["9"]["inputs"]["steps"] == 20


# --- 10. A seed mapping writes only the mapped field --------------------------


def test_seed_mapping_writes_only_the_mapped_field() -> None:
    graph = api_graph()
    workflow = _mapped_workflow(
        graph, prompt_node="6", prompt_field="text", seed_node="7", seed_field="seed"
    )
    rng = SequentialRng()

    prepared = workflows.prepare_graph(workflow, "p", rng=rng.randint)

    assert prepared["7"]["inputs"]["seed"] == 1
    # The other node's seed keeps its original value: the user chose to
    # control this one field, and the rest must not be taken back.
    assert prepared["7"]["inputs"]["noise_seed"] == 43
    assert prepared["4"]["inputs"]["stop_at_clip_layer"] == -1
    assert rng.draws == [1]


def test_stale_seed_mapping_refuses() -> None:
    graph = api_graph()
    del graph["7"]
    workflow = _mapped_workflow(
        graph, prompt_node="6", prompt_field="text", seed_node="7", seed_field="seed"
    )
    assert workflows.problem(workflow) is not None
    with pytest.raises(workflows.WorkflowNotReady):
        workflows.prepare_graph(workflow, "p")


# --- The active-workflow setting -----------------------------------------------


def test_set_active_workflow_must_exist(client: TestClient) -> None:
    assert client.put("/api/workflows/active", json={"id": "ghost"}).status_code == 404


def test_delete_missing_workflow_is_404(client: TestClient) -> None:
    assert client.delete("/api/workflows/ghost").status_code == 404


def test_resolve_active_heals_a_dangling_setting() -> None:
    with db.connect() as con:
        con.execute(
            "INSERT INTO workflow (id, name, graph, created_at) VALUES (?, ?, ?, ?)",
            ("real", "Krea", json.dumps(api_graph()), 1.0),
        )
        settings.set_active_workflow_id(con, "deleted-elsewhere")

        row = workflows.resolve_active(con)
        assert row is not None and row["id"] == "real"
        assert settings.get_active_workflow_id(con) == "real"

        con.execute("DELETE FROM workflow")
        assert workflows.resolve_active(con) is None
        assert settings.get_active_workflow_id(con) is None
