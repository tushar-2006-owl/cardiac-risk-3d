"""FastAPI entrypoint: schema-driven inference for the clinical dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.service import (
    ArtifactError,
    assert_no_leakage_in_schema,
    load_metrics,
    load_samples,
    load_schema,
    predict,
)
from ml.config import ARTIFACTS_DIR, DISCLAIMER

app = FastAPI(
    title="Cardiac Risk 3D",
    description="Educational CAD / coronary stenosis risk visualization. " + DISCLAIMER,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ARTIFACTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(ARTIFACTS_DIR)), name="plots")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/schema")
def schema() -> dict:
    try:
        items = load_schema()
        assert_no_leakage_in_schema()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"features": items, "disclaimer": DISCLAIMER}


@app.get("/metrics")
def metrics() -> dict:
    try:
        payload = load_metrics()
    except ArtifactError as extra:
        raise HTTPException(status_code=503, detail=str(extra)) from extra
    payload = dict(payload)
    payload["disclaimer"] = DISCLAIMER
    return payload


@app.get("/patients/sample")
def sample_patients() -> dict:
    try:
        samples = load_samples()
    except ArtifactError as extra:
        raise HTTPException(status_code=503, detail=str(extra)) from extra
    return {"patients": samples, "disclaimer": DISCLAIMER, "note": "Demo / educational records only."}


@app.get("/explain/global")
def global_explain() -> dict:
    from backend.service import load_global_shap

    return {"global_shap": load_global_shap(), "disclaimer": DISCLAIMER}


@app.post("/predict")
def predict_endpoint(payload: dict) -> dict:
    try:
        return predict(payload)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.get("/plots-file/{name}")
def plot_file(name: str):
    path = ARTIFACTS_DIR / name
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path)
