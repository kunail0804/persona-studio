from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, generation
from .routes import (
    characters,
    image_presets,
    images,
    lore,
    parties,
    personas,
    places,
    scenarios,
    settings,
    workflows,
)

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le schéma se met à niveau au démarrage : rien à lancer à la main, et une
    # base absente est créée au premier lancement.
    db.migrate()
    # Image messages still `pending` can never complete: their watcher died
    # with the process. The first startup marks them failed, so none of them
    # claims to be working forever — and asks ComfyUI to stop whatever the
    # dead watcher left queued or rendering.
    generation.recover_pending()
    # A `done` image whose PNG is missing — a death between the commit and
    # the file write — is the one inconsistent state the pending recovery
    # cannot see: it is not pending, so it can never be cancelled either.
    generation.recover_missing_done_files()
    yield


app = FastAPI(title="Persona Studio", lifespan=lifespan)

app.include_router(scenarios.router, prefix="/api")
app.include_router(characters.router, prefix="/api")
app.include_router(places.router, prefix="/api")
app.include_router(lore.router, prefix="/api")
app.include_router(personas.router, prefix="/api")
app.include_router(parties.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(image_presets.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, object]:
    with db.connect() as con:
        version = con.execute("PRAGMA user_version").fetchone()[0]
    return {"status": "ok", "schema": version}


def is_reserved_path(full_path: str) -> bool:
    """True for anything under `/api` or `/assets`, including the bare prefix itself.

    FastAPI's `path` converter strips the leading slash, so the bare prefixes
    arrive as the literal strings `"api"` and `"assets"` rather than `"api/"`
    and `"assets/"` — a `startswith` check alone misses them.
    """
    return full_path in ("api", "assets") or full_path.startswith(("api/", "assets/"))


def register_spa_fallback(app: FastAPI, web_dist: Path) -> None:
    """Serve the built SPA: static assets under `/assets`, everything else `index.html`.

    Only call this when `web_dist` exists, so a checkout without a front-end
    build (e.g. CI's `python` job, which never runs `pnpm build`) never mounts
    a route pointed at a missing directory.
    """
    app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        # React Router owns client-side routes like /scenarios/<id>; without
        # this, a hard refresh or a shared link 404s instead of loading the
        # app. Anything under /api or /assets that reached this far matched
        # no real route, so it stays a 404 rather than getting index.html.
        if is_reserved_path(full_path):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(web_dist / "index.html")

    @app.api_route(
        "/{full_path:path}",
        methods=["POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def spa_not_found(full_path: str) -> None:
        """Anything that is not a GET and reached this far is a 404, not a 405.

        The catch-all above is GET-only, so every other method used to be
        rejected at Starlette's method check: a 405 with no body, which reads
        as "wrong verb on a real endpoint" when the truth is that no such
        endpoint exists. Serving index.html here would be worse still — a
        POST to a client-side route is not a page request. Real API routes are
        registered before this one and match first; only what none of them
        claimed arrives here.
        """
        raise HTTPException(status_code=404, detail="Not found")


# Le front construit par Vite est servi tel quel. En dev, on passe plutôt par
# `pnpm dev` sur :5173, qui relaie /api vers ce serveur.
if WEB_DIST.is_dir():
    register_spa_fallback(app, WEB_DIST)
