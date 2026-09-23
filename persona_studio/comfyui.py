"""The ComfyUI client: the single boundary every ComfyUI call goes through.

Routes and services call into this module, never `httpx` directly. The same
rule as `ollama.py`, with the same two error meanings: `ComfyUIUnreachable`
for a request that never completed, `ComfyUIError` for a response that broke
the protocol or carries ComfyUI's own failure.

The endpoints used, against the real API's shapes:

- `POST /prompt` with `{"prompt": graph}` answers `{"prompt_id": ...}`;
- `GET /history/{prompt_id}` answers a mapping keyed by prompt id; while a
  job is still queued or running there is no entry at all. An entry carries
  `status.status_str` (`success` or `error`), `status.completed`, and
  `status.messages` — a list of `[type, data]` pairs where a failure appears
  as an `execution_error` entry carrying the real reason. `outputs` maps a
  node id to its `images` list, each item with `filename`, `subfolder` and
  `type`;
- `GET /view?filename=...&subfolder=...&type=...` returns the PNG bytes;
- `GET /queue` answers `{"queue_running": [...], "queue_pending": [...]}`,
  each entry a list whose second element is the prompt id;
- `POST /interrupt` stops whatever is executing now — the caller must check
  the queue first, which is why this module exposes the queue separately;
- `POST /queue` with `{"delete": [prompt_id]}` drops a queued job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

COMFYUI_BASE_URL = "http://127.0.0.1:8188"

# Every call here is short — submit, poll, fetch a local file. The 5 s connect
# timeout is so an unreachable ComfyUI fails fast; the read timeout covers a
# busy ComfyUI answering a poll a little late. The watcher retries nothing:
# it fails the job on the first unusable answer.
TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class ComfyUIError(RuntimeError):
    """ComfyUI answered with an error, or its response was not the expected shape."""


class ComfyUIUnreachable(ComfyUIError):
    """ComfyUI could not be reached. The message carries the underlying reason."""


@dataclass(frozen=True)
class ImageRef:
    """One produced image, as `/view` needs to be asked for it."""

    filename: str
    subfolder: str
    image_type: str


@dataclass(frozen=True)
class HistoryEntry:
    """One job's history entry, narrowed to what the watcher needs.

    `error` carries ComfyUI's real message when the execution failed; the
    images are the outputs to fetch.
    """

    completed: bool
    error: str | None
    images: list[ImageRef]


@dataclass(frozen=True)
class QueueState:
    """The prompt ids currently executing and currently waiting."""

    running: list[str]
    pending: list[str]


def _request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    with httpx.Client(base_url=COMFYUI_BASE_URL, timeout=TIMEOUT) as client:
        try:
            response = client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise ComfyUIUnreachable(
                f"ComfyUI is unreachable at {COMFYUI_BASE_URL}: {exc}"
            ) from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:500] or response.reason_phrase
        raise ComfyUIError(f"ComfyUI returned HTTP {response.status_code}: {detail}")
    if not response.content:
        # /interrupt answers an empty body; that is a success, not a protocol break.
        return None
    try:
        return response.json()
    except ValueError as exc:
        raise ComfyUIError(f"ComfyUI returned a non-JSON response: {response.text[:200]}") from exc


# Same reasoning as `ollama.probe`: a status pill cannot wait on the
# generous timeout a render needs.
PROBE_TIMEOUT = httpx.Timeout(2.0, connect=2.0)

# Read by the probe and by `queue`, written by `delete_queued`.
_QUEUE_PATH = "/queue"


def probe() -> str | None:
    """None when ComfyUI answers, else why it did not.

    `/queue` is the lightest endpoint that proves the server is up, and it is
    read-only — this runs on a timer and must never disturb a render.
    """
    try:
        with httpx.Client(base_url=COMFYUI_BASE_URL, timeout=PROBE_TIMEOUT) as client:
            response = client.get(_QUEUE_PATH)
    except httpx.HTTPError as exc:
        return f"ComfyUI is unreachable at {COMFYUI_BASE_URL}: {exc}"
    if response.status_code >= 400:
        return f"ComfyUI returned HTTP {response.status_code}"
    return None


def submit(graph: dict[str, Any]) -> str:
    """Queue a prepared graph and return the prompt id ComfyUI assigned it."""
    data = _request_json("POST", "/prompt", {"prompt": graph})
    if not isinstance(data, dict) or not isinstance(data.get("prompt_id"), str):
        raise ComfyUIError("Unexpected /prompt response shape")
    return data["prompt_id"]


def _error_from_messages(messages: Any) -> str | None:
    """ComfyUI's real failure reason from the status messages, or None.

    A failed execution reports itself as an `execution_error` message whose
    data carries the node and the exception. The node type rides along: for a
    failed workflow it is the difference between "the sampler ran out of
    memory" and "the loader could not find the checkpoint".
    """
    if not isinstance(messages, list):
        return None
    for message in messages:
        if (
            isinstance(message, list)
            and len(message) == 2
            and message[0] == "execution_error"
            and isinstance(message[1], dict)
        ):
            data = message[1]
            node = data.get("node_type") or data.get("node_id") or "?"
            reason = data.get("exception_message") or data.get("exception_type") or "unknown error"
            return f"{node}: {reason}"
    return None


def history(prompt_id: str) -> HistoryEntry | None:
    """One job's history entry, or None while it is queued or running.

    A present entry with `status_str == "error"` carries the real reason in
    `error`. An entry completed without any output image is a failure too —
    the caller could otherwise wait forever on a job that already ended.
    """
    data = _request_json("GET", f"/history/{prompt_id}")
    if not isinstance(data, dict):
        raise ComfyUIError("Unexpected /history response shape")
    entry = _history_entry(data.get(prompt_id))
    if entry is None:
        return None
    status = _history_status(entry.get("status"))
    completed = status.get("completed") is True
    error = (
        _error_from_messages(status.get("messages"))
        if status.get("status_str") == "error"
        else None
    )
    return HistoryEntry(
        completed=completed, error=error, images=_output_images(entry.get("outputs"))
    )


def _history_entry(entry: Any) -> dict[str, Any] | None:
    """The job's entry in the response, or None while it has none yet."""
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ComfyUIError("Unexpected /history entry shape")
    return entry


def _history_status(status: Any) -> dict[str, Any]:
    """The entry's status object, refusing an entry that carries none."""
    if not isinstance(status, dict):
        raise ComfyUIError("The /history entry carries no status object")
    return status


def _output_images(outputs: Any) -> list[ImageRef]:
    """Every usable image reference across all nodes' outputs.

    Nodes with no output, outputs with no `images` list, and items without a
    filename are skipped rather than trusted; the defaults match what `/view`
    answers for stock output images.
    """
    images: list[ImageRef] = []
    if not isinstance(outputs, dict):
        return images
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images.extend(_view_refs(node_output.get("images")))
    return images


def _view_refs(produced: Any) -> list[ImageRef]:
    """Image refs from one node's `images` list, ignoring malformed items."""
    refs: list[ImageRef] = []
    if not isinstance(produced, list):
        return refs
    for item in produced:
        if isinstance(item, dict) and (ref := _view_ref(item)) is not None:
            refs.append(ref)
    return refs


def _view_ref(item: dict[str, Any]) -> ImageRef | None:
    """One `images` item as `/view` needs to ask for it, or None without a filename."""
    filename = item.get("filename")
    if not isinstance(filename, str) or not filename:
        return None
    subfolder = item.get("subfolder")
    image_type = item.get("type")
    return ImageRef(
        filename=filename,
        subfolder=subfolder if isinstance(subfolder, str) else "",
        image_type=image_type if isinstance(image_type, str) else "output",
    )


def image_bytes(ref: ImageRef) -> bytes:
    """Fetch one produced image's bytes from `/view`."""
    with httpx.Client(base_url=COMFYUI_BASE_URL, timeout=TIMEOUT) as client:
        try:
            response = client.get(
                "/view",
                params={
                    "filename": ref.filename,
                    "subfolder": ref.subfolder,
                    "type": ref.image_type,
                },
            )
        except httpx.HTTPError as exc:
            raise ComfyUIUnreachable(
                f"ComfyUI is unreachable at {COMFYUI_BASE_URL}: {exc}"
            ) from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:500] or response.reason_phrase
        raise ComfyUIError(f"ComfyUI returned HTTP {response.status_code} for /view: {detail}")
    return response.content


def queue() -> QueueState:
    """What the queue is doing right now: the prompt ids running and waiting."""
    data = _request_json("GET", _QUEUE_PATH)
    if not isinstance(data, dict):
        raise ComfyUIError("Unexpected /queue response shape")

    def prompt_ids(entries: Any) -> list[str]:
        # Each queue entry is a list whose second element is the prompt id.
        ids: list[str] = []
        if not isinstance(entries, list):
            return ids
        for entry in entries:
            if isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], str):
                ids.append(entry[1])
        return ids

    return QueueState(
        running=prompt_ids(data.get("queue_running")), pending=prompt_ids(data.get("queue_pending"))
    )


def interrupt() -> None:
    """Stop whatever ComfyUI is executing now — whatever that is.

    The interrupt is not job-targeted: the caller must have confirmed from
    the queue that our own prompt id is the one running.
    """
    _request_json("POST", "/interrupt")


def delete_queued(prompt_id: str) -> None:
    """Remove one waiting job from the queue, leaving the running one alone."""
    _request_json("POST", _QUEUE_PATH, {"delete": [prompt_id]})
