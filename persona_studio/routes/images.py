"""Images: composing an image prompt from a request and the current scene.

The path is party-scoped because the scene comes from the party; the module is
`images.py` because the subject is images. A router does not have to own its
path prefix.

This router composes a prompt and nothing else: the text is returned to the
interface, where the player edits it, and nothing is written to the database —
no `image` row, no message, no column. Sending the prompt to ComfyUI is the
next feature and will live beside this endpoint, in this file, under the same
rule: the prompt travels exactly as the player sees it on screen.

Composition follows the create-party discipline: the reads happen in one
connection that closes before the model call, so no SQLite connection is ever
held open while a local model thinks. A failure — Ollama unreachable, an error,
an unusable answer — leaves the database untouched and surfaces its real
reason, the way the settings route does.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, image_prompt, ollama, settings
from .parties import _require_model

router = APIRouter(tags=["images"])


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
    # The request arrives in the body, not the database: it fills the input
    # type here, once, and rides in both the call and the scrub below.
    inputs = replace(inputs, instruction=instruction)
    messages = image_prompt.build_messages(inputs)
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
