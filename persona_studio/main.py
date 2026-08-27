from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .routes import characters, scenarios

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


@app.get("/api/health")
def health() -> dict[str, object]:
    with db.connect() as con:
        version = con.execute("PRAGMA user_version").fetchone()[0]
    return {"status": "ok", "schema": version}


# Le front construit par Vite est servi tel quel. En dev, on passe plutôt par
# `pnpm dev` sur :5173, qui relaie /api vers ce serveur.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        # React Router owns client-side routes like /scenarios/<id>; without
        # this, a hard refresh or a shared link 404s instead of loading the
        # app. Requests under /api or /assets that reached this far matched
        # no real route, so they stay 404s rather than getting index.html.
        if full_path.startswith("api/") or full_path.startswith("assets/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(WEB_DIST / "index.html")
