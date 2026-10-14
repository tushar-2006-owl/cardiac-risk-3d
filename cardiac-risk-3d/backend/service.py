"""Load artifacts and run calibrated prediction + SHAP."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from ml.config import ARTIFACTS_DIR, DISCLAIMER, MODEL_REGISTRY, TARGETS, risk_level
from ml.explain import explain_patient
from ml.preprocess import feature_columns, load_raw_dataframe, normalize_frame, split_xy


class ArtifactError(RuntimeError):
    pass


def _require(path_name: str):
    path = ARTIFACTS_DIR / path_name
    if not path.exists():
        raise ArtifactError(f"Missing {path}. Run `python -m ml.train` first.")
    return path


@lru_cache(maxsize=1)
def load_schema() -> list[dict[str, Any]]:
    return json.loads(_require("feature_schema.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    return json.loads(_require("metrics.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_samples() -> list[dict[str, Any]]:
    return json.loads(_require("sample_patients.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_global_shap() -> dict[str, Any]:
    path = ARTIFACTS_DIR / "global_shap.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_models() -> dict[str, dict[str, Any]]:
    metrics = load_metrics()
    models: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        models[target] = {
            "calibrated": joblib.load(_require(f"model_{target}_calibrated.joblib")),
            "base": joblib.load(_require(f"model_{target}_base.joblib")),
            "selected": metrics["targets"][target]["selected_model"],
            "family": MODEL_REGISTRY[metrics["targets"][target]["selected_model"]]["family"],
        }
    return models


@lru_cache(maxsize=1)
def background_X() -> pd.DataFrame:
    raw = load_raw_dataframe()
    df, _ = normalize_frame(raw)
    X, _ = split_xy(df, "CAD")
    return X


def empty_feature_row() -> dict[str, Any]:
    return {item["name"]: item.get("default") for item in load_schema()}


def features_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    schema = load_schema()
    names = [item["name"] for item in schema]
    row: dict[str, Any] = {name: None for name in names}
    incoming = payload.get("features") if "features" in payload and isinstance(payload.get("features"), dict) else payload
    unknown = [k for k in incoming if k not in names and k != "features"]
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}. Allowed: {names}")
    for item in schema:
        name = item["name"]
        if name not in incoming or incoming[name] is None or incoming[name] == "":
            continue
        value = incoming[name]
        if item["type"] in {"number", "binary"}:
            try:
                value = float(value)
                if item["type"] == "binary":
                    value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{name}' must be numeric, got {incoming[name]!r}") from exc
        row[name] = value
    return pd.DataFrame([row], columns=names)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    X_row = features_to_frame(payload)
    models = load_models()
    bg = background_X()
    explanations: dict[str, list] = {}
    probs: dict[str, float] = {}
    for target, bundle in models.items():
        proba = float(bundle["calibrated"].predict_proba(X_row)[0, 1])
        probs[target] = max(0.0, min(1.0, proba))
        base = bundle["base"]
        try:
            explanations[target] = explain_patient(
                base.named_steps["prep"],
                base.named_steps["clf"],
                bundle["family"],
                bg,
                X_row,
                top_k=10,
            )
        except Exception:
            explanations[target] = []

    cad_p = probs["CAD"]
    vessels = {}
    for name in ("LAD", "LCX", "RCA"):
        p = probs[name]
        vessels[name] = {
            "probability": p,
            "risk_level": risk_level(p),
            "mesh_key": name,
        }
    return {
        "cad": {
            "probability": cad_p,
            "label": "CAD" if cad_p >= 0.5 else "Normal",
            "risk_level": risk_level(cad_p),
        },
        "vessels": vessels,
        "explanations": explanations,
        "disclaimer": DISCLAIMER,
        "global_shap": load_global_shap(),
        "patient_values": json.loads(X_row.iloc[0].to_json()),
    }


def assert_no_leakage_in_schema() -> None:
    from ml.config import DROP_FEATURES, LEAKAGE_COLUMNS

    names = {item["name"] for item in load_schema()}
    banned = set(LEAKAGE_COLUMNS) | set(DROP_FEATURES)
    leaked = names & banned
    if leaked:
        raise ArtifactError(f"Leakage in feature schema: {sorted(leaked)}")
    # Also check training feature list
    features = set(load_metrics().get("features", []))
    leaked2 = features & banned
    if leaked2:
        raise ArtifactError(f"Leakage in trained feature list: {sorted(leaked2)}")
    _ = feature_columns  # imported for tests that monkeypatch
