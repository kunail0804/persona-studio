from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le schéma se met à niveau au démarrage : rien à lancer à la main, et une
    # base absente est créée au premier lancement.
    db.migrate()
    yield


app = FastAPI(title="Persona Studio", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, object]:
    with db.connect() as con:
        version = con.execute("PRAGMA user_version").fetchone()[0]
    return {"status": "ok", "schema": version}


# Le front construit par Vite est servi tel quel. En dev, on passe plutôt par
# `pnpm dev` sur :5173, qui relaie /api vers ce serveur.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
