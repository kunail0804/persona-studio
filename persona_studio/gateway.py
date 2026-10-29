"""The comfy-gateway client: the single boundary every image call goes through.

comfy-gateway (a separate local service, `~/Projets/comfy-gateway`) owns the
ComfyUI workflows, their settings, the queue and the conversation with
ComfyUI. This application names a template, sends the prompt as the template's
`prompt` role, and follows the job. It never builds or reads a ComfyUI graph.

Routes and services call into this module, never `httpx` directly. The same
rule as `ollama.py`, with three error meanings: `GatewayUnreachable` for a
request that never completed, `GatewayRefused` for a request the gateway
answered with a reason (an unknown template, a template without a prompt
role), `GatewayError` for anything else that broke the protocol.

The endpoints used, against the gateway's real shapes:

- `GET /api/health` answers `{"ok": true, "comfy_reachable": bool, ...}`;
- `GET /api/templates` lists every upload, `{id, name, ...}`, newest first; a
  name with several versions appears once per version;
- `GET /api/templates/latest?name=...` answers the newest version with its
  `description`, whose `roles` and `role_errors` say whether a prompt can be
  sent; 404 for an unknown name;
- `GET /api/templates/{id}/presets` answers `[{name, updated_at}]`;
- `POST /api/jobs` with `{template, preset, roles, client}` answers the job,
  `{id, status, ...}`; 400 or 404 with `detail` when it is refused;
- `GET /api/jobs/{id}` answers `{status, error, images, ...}`, `status` one of
  `queued`, `running`, `done`, `error`, `cancelled`; `images` are paths on the
  gateway; 404 once the job is deleted;
- `POST /api/jobs/{id}/cancel` stops that job and only that one: the gateway
  cancels its prompt in ComfyUI by id;
- `GET /api/queue` answers `{"running": job | null, "queued": [job, ...]}`,
  each job carrying its `client`;
- `POST /api/comfy/free` with `{"mode": "all"}` unloads ComfyUI's models and
  empties its cache, which frees VRAM and RAM; 503 when ComfyUI does not
  answer. While a job runs, ComfyUI frees once that job is over;
- `GET /api/comfy/memory` answers `{ram_total, ram_free, vram_total,
  vram_free}` in bytes, as ComfyUI reports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

GATEWAY_BASE_URL = "http://127.0.0.1:8190"

# The name this application gives itself on the gateway, so its jobs can be
# told from CharCreator's or the gateway UI's.
CLIENT_NAME = "persona-studio"

# Every call is short: the render itself runs on the gateway, and this client
# only submits and polls. The 5 s connect timeout makes an unreachable gateway
# fail fast; the read timeout covers a submission, which the gateway answers
# only once it has checked the settings against the template.
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

# Same reasoning as `ollama.probe`: a status pill cannot wait on the timeout a
# submission needs.
PROBE_TIMEOUT = httpx.Timeout(2.0, connect=2.0)

FINISHED = ("done", "error", "cancelled")

# "all" unloads ComfyUI's models and empties its cache: VRAM and RAM both
# come back. "vram" would keep the weights in RAM, and RAM full of ComfyUI's
# weights is what pushed the narrator into swap.
FREE_MODE = "all"


class GatewayError(RuntimeError):
    """The gateway's answer was not the expected shape, or it failed."""


class GatewayUnreachable(GatewayError):
    """The gateway could not be reached. The message carries the underlying reason."""


class GatewayRefused(GatewayError):
    """The gateway refused the request and said why: a setup problem to fix."""


@dataclass(frozen=True)
class JobState:
    """One job, narrowed to what the watcher needs."""

    status: str
    error: str | None
    images: list[str]


@dataclass(frozen=True)
class TemplateInfo:
    """The newest version of a template, and why a prompt cannot be sent to it."""

    template_id: int
    name: str
    problem: str | None


def _unreachable(exc: Exception) -> GatewayUnreachable:
    return GatewayUnreachable(f"comfy-gateway is unreachable at {GATEWAY_BASE_URL}: {exc}")


def _detail(response: httpx.Response) -> str:
    """The gateway's own reason, which it puts in `detail`."""
    try:
        data = response.json()
    except ValueError:
        data = None
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    return response.text.strip()[:500] or response.reason_phrase


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    with httpx.Client(base_url=GATEWAY_BASE_URL, timeout=TIMEOUT) as client:
        try:
            return client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise _unreachable(exc) from exc


def _json(response: httpx.Response) -> Any:
    if response.status_code in (400, 404, 422):
        raise GatewayRefused(_detail(response))
    if response.status_code >= 400:
        raise GatewayError(
            f"comfy-gateway returned HTTP {response.status_code}: {_detail(response)}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise GatewayError(
            f"comfy-gateway returned a non-JSON response: {response.text[:200]}"
        ) from exc


def probe() -> str | None:
    """None when images can be made, else why not.

    The gateway answering is not enough: it reports whether ComfyUI answers
    it, and an image needs both. Read-only, so it can run on a timer during a
    render.
    """
    try:
        with httpx.Client(base_url=GATEWAY_BASE_URL, timeout=PROBE_TIMEOUT) as client:
            response = client.get("/api/health")
    except httpx.HTTPError as exc:
        return str(_unreachable(exc))
    if response.status_code >= 400:
        return f"comfy-gateway returned HTTP {response.status_code}"
    try:
        data = response.json()
    except ValueError:
        return "comfy-gateway answered something that is not its health report"
    if not isinstance(data, dict):
        return "comfy-gateway answered something that is not its health report"
    if data.get("comfy_reachable") is not True:
        comfy_url = data.get("comfy_url") or "its configured address"
        return f"comfy-gateway answers, but ComfyUI does not answer it at {comfy_url}"
    return None


def template_names() -> list[str]:
    """One name per workflow, sorted. A job naming a template runs its newest version."""
    data = _json(_request("GET", "/api/templates"))
    if not isinstance(data, list):
        raise GatewayError("Unexpected /api/templates response shape")
    names = {
        item["name"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    return sorted(names)


def template(name: str) -> TemplateInfo:
    """The newest version of a template. Raises `GatewayRefused` for an unknown name."""
    data = _json(_request("GET", "/api/templates/latest", params={"name": name}))
    if not isinstance(data, dict) or not isinstance(data.get("id"), int):
        raise GatewayError("Unexpected /api/templates/latest response shape")
    description = data.get("description")
    roles = description.get("roles") if isinstance(description, dict) else None
    errors = description.get("role_errors") if isinstance(description, dict) else None
    problem = None
    if not isinstance(roles, dict) or "prompt" not in roles:
        reason = errors.get("prompt") if isinstance(errors, dict) else None
        problem = (
            f"The template '{name}' has no prompt role, so the image prompt has "
            f"nowhere to go. Set it in comfy-gateway's Roles panel."
            + (f" ({reason})" if isinstance(reason, str) else "")
        )
    return TemplateInfo(template_id=data["id"], name=name, problem=problem)


def preset_names(template_id: int) -> list[str]:
    """The presets saved in the gateway for a template's name."""
    data = _json(_request("GET", f"/api/templates/{template_id}/presets"))
    if not isinstance(data, list):
        raise GatewayError("Unexpected presets response shape")
    return [
        item["name"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def submit(template_name: str, preset: str | None, prompt: str) -> str:
    """Queue one image and return the gateway's job id, as text.

    The prompt goes to the template's `prompt` role and is the only value
    sent: everything else is the template's, or the preset's. `free_after`
    asks the gateway to empty ComfyUI once the job is over, whatever its
    outcome, so the narrator finds the card and the RAM free afterwards.
    """
    job: dict[str, Any] = {
        "template": template_name,
        "roles": {"prompt": prompt},
        "client": CLIENT_NAME,
        "free_after": FREE_MODE,
    }
    if preset:
        job["preset"] = preset
    data = _json(_request("POST", "/api/jobs", json=job))
    if not isinstance(data, dict) or not isinstance(data.get("id"), int):
        raise GatewayError("Unexpected /api/jobs response shape")
    return str(data["id"])


def job(job_id: str) -> JobState:
    """A job's state. A job deleted from the gateway is a `GatewayRefused`."""
    data = _json(_request("GET", f"/api/jobs/{job_id}"))
    if not isinstance(data, dict) or not isinstance(data.get("status"), str):
        raise GatewayError("Unexpected job response shape")
    error = data.get("error")
    images = data.get("images")
    return JobState(
        status=data["status"],
        error=error if isinstance(error, str) else None,
        images=[i for i in images if isinstance(i, str)] if isinstance(images, list) else [],
    )


def image_bytes(path: str) -> bytes:
    """One image of a finished job, by the path the job gave."""
    response = _request("GET", path)
    if response.status_code >= 400:
        raise GatewayError(
            f"comfy-gateway returned HTTP {response.status_code} for {path}: {_detail(response)}"
        )
    return response.content


def cancel(job_id: str) -> None:
    """Stop one job, queued or running. The gateway touches nothing else."""
    response = _request("POST", f"/api/jobs/{job_id}/cancel")
    if response.status_code >= 400:
        raise GatewayError(f"comfy-gateway could not cancel job {job_id}: {_detail(response)}")


def busy() -> str | None:
    """Why the card is taken by the gateway right now, or None when it is idle.

    A queued job counts as much as a running one: it starts as soon as the
    gateway is free, and a narrator loaded meanwhile would be under it.
    """
    data = _json(_request("GET", "/api/queue"))
    if not isinstance(data, dict):
        raise GatewayError("Unexpected /api/queue response shape")
    running = data.get("running")
    queued = data.get("queued")
    jobs = ([running] if isinstance(running, dict) else []) + (
        [job for job in queued if isinstance(job, dict)] if isinstance(queued, list) else []
    )
    if not jobs:
        return None
    clients = sorted({str(job.get("client") or "unknown client") for job in jobs})
    state = "rendering" if isinstance(running, dict) else "about to render"
    return (
        f"comfy-gateway is {state} {len(jobs)} image(s) for {', '.join(clients)}. "
        "The narrator waits until it is done: on this card the two do not fit in "
        "memory together."
    )


def free_comfy() -> None:
    """Ask the gateway to empty ComfyUI: models unloaded, cache emptied.

    A ComfyUI that does not answer (503) holds nothing to free, so that is
    not an error here.
    """
    response = _request("POST", "/api/comfy/free", json={"mode": FREE_MODE})
    if response.status_code == 503:
        return
    _json(response)


def memory() -> dict[str, int]:
    """Free and total RAM and VRAM in bytes, as ComfyUI reports them."""
    data = _json(_request("GET", "/api/comfy/memory"))
    keys = ("ram_total", "ram_free", "vram_total", "vram_free")
    if not isinstance(data, dict) or not all(isinstance(data.get(k), int) for k in keys):
        raise GatewayError("Unexpected /api/comfy/memory response shape")
    return {k: data[k] for k in keys}
