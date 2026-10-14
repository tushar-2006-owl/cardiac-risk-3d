"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.config import DISCLAIMER, TARGETS, VESSEL_MESH_KEYS, risk_level


class VesselPrediction(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    mesh_key: str


class CadPrediction(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    label: Literal["CAD", "Normal"]
    risk_level: Literal["low", "medium", "high"]


class ShapItem(BaseModel):
    feature: str
    value: Any
    shap: float
    direction: Literal["increases risk", "decreases risk"]
    percent_contribution: float


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: dict[str, Any] | None = None

    def as_features(self) -> dict[str, Any]:
        if self.features:
            return dict(self.features)
        data = self.model_dump(exclude={"features"})
        return {k: v for k, v in data.items() if v is not None}


class PredictResponse(BaseModel):
    cad: CadPrediction
    vessels: dict[str, VesselPrediction]
    explanations: dict[str, list[ShapItem]]
    disclaimer: str = DISCLAIMER


class SamplePatient(BaseModel):
    id: str
    label: str
    risk_hint: str
    demo_only: bool = True
    ground_truth: dict[str, int] | None = None
    features: dict[str, Any]


def label_from_proba(target: str, proba: float) -> str:
    if target == "CAD":
        return "CAD" if proba >= 0.5 else "Normal"
    return "Stenotic" if proba >= 0.5 else "Normal"


def vessel_payload(name: str, proba: float) -> dict[str, Any]:
    if name not in VESSEL_MESH_KEYS:
        raise KeyError(name)
    return {
        "probability": proba,
        "risk_level": risk_level(proba),
        "mesh_key": VESSEL_MESH_KEYS[name],
    }


KNOWN_TARGETS = list(TARGETS)
