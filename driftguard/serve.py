"""FastAPI serving for the current production model, with a health endpoint."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .registry import Registry

app = FastAPI(title="DriftGuard serving", version=__version__)


class PredictRequest(BaseModel):
    features: list[list[float]] = Field(..., description="Rows of 4 features each.")


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: int


def _registry() -> Registry:
    return Registry()


@app.get("/health")
def health() -> dict:
    prod = _registry().production()
    return {
        "status": "ok" if prod else "no_model",
        "version": __version__,
        "production_model_version": prod.version if prod else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    reg = _registry()
    prod = reg.production()
    if not prod:
        raise HTTPException(status_code=503, detail="no production model registered")
    model = reg.load(prod.version)
    X = np.asarray(req.features, dtype=float)
    preds = model.predict(X).tolist()
    return PredictResponse(predictions=preds, model_version=prod.version)


@app.get("/lineage")
def lineage() -> list[dict]:
    return [v.__dict__ for v in _registry().lineage()]
