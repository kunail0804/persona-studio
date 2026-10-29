"""What the header's service menus show and do: memory, and the buttons that
replace the commands typed to free it.

Ollama runs as a system service this application has no right to stop, so
its button unloads the models instead, which frees the same memory. The
gateway runs as the user and can be stopped and started again.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import gateway, gateway_process, ollama

router = APIRouter(prefix="/services", tags=["services"])


class LoadedModelOut(BaseModel):
    name: str
    size_vram: int


class OllamaMemory(BaseModel):
    models: list[LoadedModelOut]
    # Why the list could not be read, or None.
    error: str | None


class ComfyMemory(BaseModel):
    ram_total: int
    ram_free: int
    vram_total: int
    vram_free: int


class GatewayMemory(BaseModel):
    memory: ComfyMemory | None
    # What the gateway is rendering, or None when it is idle.
    busy: str | None
    error: str | None


class ServicesMemory(BaseModel):
    ollama: OllamaMemory
    gateway: GatewayMemory


class ActionOut(BaseModel):
    message: str


def _ollama_memory() -> OllamaMemory:
    try:
        models = ollama.loaded_models()
    except ollama.OllamaError as exc:
        return OllamaMemory(models=[], error=str(exc))
    return OllamaMemory(
        models=[LoadedModelOut(name=m.name, size_vram=m.size_vram) for m in models], error=None
    )


def _gateway_memory() -> GatewayMemory:
    try:
        busy = gateway.busy()
        memory = ComfyMemory(**gateway.memory())
    except gateway.GatewayError as exc:
        return GatewayMemory(memory=None, busy=None, error=str(exc))
    return GatewayMemory(memory=memory, busy=busy, error=None)


@router.get("/memory", response_model=ServicesMemory)
def get_memory() -> ServicesMemory:
    return ServicesMemory(ollama=_ollama_memory(), gateway=_gateway_memory())


@router.post("/ollama/unload", response_model=ActionOut)
def unload_ollama() -> ActionOut:
    """Unload every model Ollama holds, and wait until it has let go."""
    try:
        before = [m.name for m in ollama.loaded_models()]
        remaining = ollama.unload_all()
    except ollama.OllamaUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ollama.OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if remaining:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Ollama still holds {', '.join(remaining)} after "
                f"{ollama.UNLOAD_WAIT_SECONDS:.0f} s: a generation is probably still "
                "running on it. Try again once it is done."
            ),
        )
    if not before:
        return ActionOut(message="Ollama held no model.")
    return ActionOut(message=f"Unloaded: {', '.join(before)}.")


@router.post("/gateway/free", response_model=ActionOut)
def free_comfy() -> ActionOut:
    """Empty ComfyUI. During a render, ComfyUI does it once the render is over."""
    try:
        busy = gateway.busy()
        gateway.free_comfy()
    except gateway.GatewayUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except gateway.GatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if busy is not None:
        return ActionOut(message="A render is running: ComfyUI will be emptied once it is over.")
    return ActionOut(message="ComfyUI is emptied: its models are unloaded and its cache cleared.")


def _process_action(action) -> ActionOut:
    try:
        return ActionOut(message=action().message)
    except gateway_process.GatewayProcessError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/gateway/stop", response_model=ActionOut)
def stop_gateway() -> ActionOut:
    """Stop the gateway and its ComfyUI. A render in progress is lost."""
    return _process_action(gateway_process.stop)


@router.post("/gateway/restart", response_model=ActionOut)
def restart_gateway() -> ActionOut:
    """Stop the gateway if it runs, then start it in its window. Also starts a stopped one."""
    return _process_action(gateway_process.restart)
