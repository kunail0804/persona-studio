"""Who gets the graphics card: the narrator or a render.

On this machine's 12 GB the narration model and an image pipeline do not fit
together. Each side makes room before it starts instead of hoping the other
let go:

- before a model call, ComfyUI is emptied through comfy-gateway, and the call
  is refused while the gateway renders or is about to render, whichever
  application asked for it;
- before a render, every model Ollama holds is unloaded, and the gateway
  empties ComfyUI again once the job is over (`gateway.submit` asks it to).

A service that cannot be reached holds no memory this application can get
back, so an unreachable gateway or Ollama never blocks the other side.
"""

from __future__ import annotations

from . import gateway, ollama


class GpuBusy(Exception):
    """The gateway is rendering. The message says for whom."""


def make_room_for_llm() -> None:
    """Empty ComfyUI before a model call, or raise `GpuBusy` while it renders."""
    try:
        reason = gateway.busy()
    except gateway.GatewayError:
        # Down, it took its own ComfyUI with it; refusing would only stop the
        # story for nothing.
        return
    if reason is not None:
        raise GpuBusy(reason)
    try:
        gateway.free_comfy()
    except gateway.GatewayError:
        return


def make_room_for_render() -> list[str]:
    """Unload every Ollama model before a render. Returns those still loaded.

    Best effort: the player has asked for the image, and an Ollama that
    cannot be reached is holding no VRAM.
    """
    try:
        return ollama.unload_all()
    except ollama.OllamaError:
        return []
