"""Which comfy-gateway template images are made with, and its preset.

The workflows, their settings and their presets live in comfy-gateway; this
application only names a template and, optionally, one of its presets. This
module reads the lists from the gateway so the settings page offers real
choices, and says why images cannot be made when the setup no longer holds:
the template was deleted from the gateway, it has no prompt role, its preset
is gone.

Reads happen in a connection that closes before any gateway call, the rule
every route here follows for HTTP calls.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import db, gateway, generation, settings

router = APIRouter(tags=["image-generation"])


class ImageGenerationOut(BaseModel):
    template: str | None
    preset: str | None
    # What the gateway offers: empty when it cannot be reached.
    templates: list[str]
    presets: list[str]
    # Why no image can be made with this setup, or None when it is ready.
    problem: str | None
    # Where the templates and presets are edited.
    gateway_url: str


class ImageGenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str | None = None
    preset: str | None = None


def _describe(template: str | None, preset: str | None) -> ImageGenerationOut:
    templates: list[str] = []
    presets: list[str] = []
    problem: str | None = None
    try:
        templates = gateway.template_names()
        if template is None:
            problem = generation.NO_TEMPLATE_MESSAGE
        elif template not in templates:
            problem = f"The template '{template}' is no longer in comfy-gateway: choose another."
        else:
            info = gateway.template(template)
            presets = gateway.preset_names(info.template_id)
            problem = info.problem
            if problem is None and preset is not None and preset not in presets:
                problem = (
                    f"The preset '{preset}' no longer exists for '{template}' in comfy-gateway: "
                    "choose another, or none."
                )
    except gateway.GatewayError as exc:
        problem = str(exc)
    return ImageGenerationOut(
        template=template,
        preset=preset,
        templates=templates,
        presets=presets,
        problem=problem,
        gateway_url=gateway.GATEWAY_BASE_URL,
    )


@router.get("/image-generation", response_model=ImageGenerationOut)
def get_image_generation() -> ImageGenerationOut:
    with db.connect() as con:
        template = settings.get_image_template(con)
        preset = settings.get_image_gateway_preset(con)
    return _describe(template, preset)


@router.put("/image-generation", response_model=ImageGenerationOut)
def set_image_generation(body: ImageGenerationInput) -> ImageGenerationOut:
    """Choose the template and preset. Both are checked against the gateway
    first: a name it does not have would only fail at the first image."""
    template = body.template or None
    # A preset belongs to a template's name: without a template it means nothing.
    preset = None
    if template is not None and body.preset:
        preset = body.preset
    if template is not None:
        try:
            if template not in gateway.template_names():
                raise HTTPException(
                    status_code=422, detail=f"comfy-gateway has no template named '{template}'."
                )
            if preset is not None:
                info = gateway.template(template)
                if preset not in gateway.preset_names(info.template_id):
                    raise HTTPException(
                        status_code=422,
                        detail=f"comfy-gateway has no preset '{preset}' for '{template}'.",
                    )
        except gateway.GatewayUnreachable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except gateway.GatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    with db.connect() as con:
        settings.set_image_generation(con, template, preset)
    return _describe(template, preset)
