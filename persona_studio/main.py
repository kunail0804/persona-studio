from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(title="Persona Studio")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Le front construit par Vite est servi tel quel. En dev, on passe plutôt par
# `pnpm dev` sur :5173, qui relaie /api vers ce serveur.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
