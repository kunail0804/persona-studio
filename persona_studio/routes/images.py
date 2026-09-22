"""Images: composing an image prompt, generating the image, serving it.

The path is party-scoped because the scene comes from the party; the module is
`images.py` because the subject is images. A router does not have to own its
path prefix.

The composer turns a request into an editable prompt and writes nothing. Its
text travels exactly as the player sees it on screen: the generation endpoint
below takes it back as-is, alongside the instruction that produced it, and
`generation.start` does the rest — persist the pending message, submit to
ComfyUI, watch on a background thread. The player keeps playing while it
renders.

Cancelling is a separate endpoint on the message, and the file itself is
served from `data/images/` under a strict id check plus a containment check:
anything that is not a plain image id must not resolve to a path, and the
path that resolves must sit inside the images directory, so nothing can walk
out of it.

Composition follows the create-party discipline: the reads happen in one
connection that closes before the model call, so no SQLite connection is ever
held open while a local model thinks. A failure — Ollama unreachable, an
error, an unusable answer — leaves the database untouched and surfaces its
real reason, the way the settings route does.
"""

from __future__ import annotations

import re
from dataclasses import replace

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import comfyui, db, generation, image_prompt, ollama, settings, workflows
from .parties import PartyMessage, _message_response, _require_model

router = APIRouter(tags=["images"])

# Image ids are `uuid4().hex` — exactly 32 lowercase hex characters. The
# check keeps anything else (paths with separators or dots among them) out of
# the images directory.
_IMAGE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


class ImagePromptInput(BaseModel):
    instruction: str


class ImagePromptOutput(BaseModel):
    prompt: str


@router.post("/parties/{party_id}/image-prompt", response_model=ImagePromptOutput)
def compose_image_prompt(party_id: str, body: ImagePromptInput) -> ImagePromptOutput:
    """Turn what the player wants to see, plus the current scene, into English
    keywords they can edit before anything is generated."""
    instruction = body.instruction.strip()
    if not instruction:
        raise HTTPException(
            status_code=400,
            detail="An image request cannot be empty: say what you want to see.",
        )
    with db.connect() as con:
        try:
            inputs = image_prompt.load_inputs(con, party_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        model = _require_model(con)
        num_ctx = settings.get_num_ctx(con)
        master_prompt = image_prompt.load_active_instruction(con)
    # The request arrives in the body, not the database: it fills the input
    # type here, once, and rides in both the call and the scrub below.
    inputs = replace(inputs, instruction=instruction)
    messages = image_prompt.build_messages(inputs, master_prompt)
    try:
        raw = ollama.chat(model, messages, num_ctx=num_ctx)
    except ollama.OllamaUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ollama.OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    prompt = image_prompt.scrub_names(raw, inputs)
    if not prompt:
        raise HTTPException(
            status_code=502,
            detail="The model's answer carried nothing usable once the proper names "
            "were removed. Try again, or write the prompt yourself — the field is "
            "editable.",
        )
    return ImagePromptOutput(prompt=prompt)


# --- Generating -----------------------------------------------------------------


class ImageGenerationInput(BaseModel):
    prompt: str
    instruction: str


class ImageStartOutput(BaseModel):
    message_id: int
    started_at: float


@router.post("/parties/{party_id}/images", response_model=ImageStartOutput, status_code=201)
def start_image_generation(party_id: str, body: ImageGenerationInput) -> ImageStartOutput:
    """Send the player's final prompt to ComfyUI and return at once.

    The prompt is sent as-is: this is the text the player has on screen, and
    nothing may rewrite it between here and ComfyUI. The response carries the
    message id whose row now shows the pending image, so the player can
    cancel it while it renders.
    """
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="An image prompt cannot be empty.")
    try:
        result = generation.start(party_id, prompt, body.instruction.strip())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except workflows.WorkflowNotReady as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except comfyui.ComfyUIUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except comfyui.ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ImageStartOutput(message_id=result.message_id, started_at=result.started_at)


@router.post(
    "/parties/{party_id}/images/{message_id}/cancel",
    response_model=PartyMessage,
)
def cancel_image_generation(party_id: str, message_id: int) -> PartyMessage:
    """Cancel a pending image generation, interrupting ComfyUI only if our own
    job is the one running — see `generation.cancel`."""
    try:
        row = generation.cancel(party_id, message_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with db.connect() as con:
        return _message_response(con, row)


# --- Serving the file -------------------------------------------------------------


@router.get("/images/{image_id}/file")
def get_image_file(image_id: str) -> FileResponse:
    """The PNG behind an image id, for the interface's `<img>` tags."""
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise HTTPException(status_code=404, detail="Not found")
    # The pattern is an early exit, not a containment proof. Resolving both
    # sides and comparing them keeps the served file inside the images
    # directory even if the pattern is ever loosened or a symlink lands
    # there; the directory is resolved too, so a symlinked data directory is
    # not refused by its own link.
    images_dir = db.IMAGES_DIR.resolve()
    path = db.image_path(image_id).resolve()
    if not path.is_relative_to(images_dir) or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image {image_id!r} not found")
    return FileResponse(path, media_type="image/png")
