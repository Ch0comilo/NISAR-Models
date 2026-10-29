"""Read-only result service for the team's globe. Inference runs through CLI/Python."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .common import read_json


def create_app(root, origins=(), *, assessment_path=None):
    root = Path(root).resolve()
    supplied_assessment = Path(assessment_path).resolve() if assessment_path is not None else None
    if supplied_assessment is not None:
        if not supplied_assessment.is_file():
            raise FileNotFoundError("The supplied model assessment does not exist")
        # Fail at startup if the explicitly supplied readiness statement is invalid.
        read_json(supplied_assessment)
    app = FastAPI(title="NISAR Water", version="0.1.0")
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_methods=["GET"], allow_headers=["*"])

    def series_files():
        return {(p.parent.relative_to(root).as_posix() if p.parent != root else "default"): p
                for p in root.rglob("series.json")}

    @app.get("/health")
    def health():
        return {"state": "ready", "service": "nisar-water", "schemaVersion": "nisar-water-series-1"}

    @app.get("/water/series")
    def list_series():
        return [{"key": key, "id": read_json(path)["id"], "url": "/water/series/"+key}
                for key, path in series_files().items()]

    @app.get("/water/benchmark")
    def benchmark():
        path = root / "report.json"
        if not path.is_file():
            raise HTTPException(404, "No saved benchmark report at this result root")
        return read_json(path)

    @app.get("/water/assessment")
    def assessment():
        path = supplied_assessment if supplied_assessment is not None else root / "assessment.json"
        if not path.is_file():
            raise HTTPException(404, "No saved model assessment at this result root")
        return read_json(path)

    @app.get("/water/series/{key:path}")
    def get_series(key: str):
        path = series_files().get(key)
        if path is None:
            raise HTTPException(404, "No computed water series with that key")
        data = read_json(path)
        prefix = path.parent.relative_to(root).as_posix()

        def rewrite(obj):
            if isinstance(obj, dict):
                return {k: rewrite(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [rewrite(v) for v in obj]
            if isinstance(obj, str) and obj.startswith("/assets/"):
                return "/assets/" + (prefix+"/" if prefix != "." else "") + obj[len("/assets/"):]
            return obj

        return rewrite(data)

    app.mount("/assets", StaticFiles(directory=root, check_dir=True), name="assets")
    return app

