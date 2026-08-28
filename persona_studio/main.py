from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .routes import characters, personas, scenarios, settings

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le schéma se met à niveau au démarrage : rien à lancer à la main, et une
    # base absente est créée au premier lancement.
    db.migrate()
    yield


app = FastAPI(title="Persona Studio", lifespan=lifespan)

app.include_router(scenarios.router, prefix="/api")
app.include_router(characters.router, prefix="/api")
app.include_router(personas.router, prefix="/api")
app.include_router(settings.router, prefix="/api")


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


# Le front construit par Vite est servi tel quel. En dev, on passe plutôt par
# `pnpm dev` sur :5173, qui relaie /api vers ce serveur.
if WEB_DIST.is_dir():
    register_spa_fallback(app, WEB_DIST)
