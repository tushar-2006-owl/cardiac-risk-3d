"""SHAP explanations on the uncalibrated base estimator with original-feature aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from ml.config import ARTIFACTS_DIR, MODEL_REGISTRY
from ml.preprocess import original_feature_from_transformed


def _explainer(estimator: Any, family: str, background: np.ndarray) -> shap.Explainer:
    if family == "tree":
        return shap.TreeExplainer(estimator)
    return shap.LinearExplainer(estimator, background)


def shap_values_positive(raw: Any) -> np.ndarray:
    """Return SHAP values for the positive class as shape (n_samples, n_features)."""
    values = getattr(raw, "values", raw)
    arr = np.array(values)
    if arr.ndim == 3:
        # (samples, features, classes) or (classes, samples, features)
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[0] == 2:
            return arr[1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def aggregate_by_original(
    shap_row: np.ndarray,
    transformed_names: list[str],
    x_original: pd.Series,
) -> list[dict[str, Any]]:
    grouped: dict[str, float] = {}
    for name, val in zip(transformed_names, shap_row):
        orig = original_feature_from_transformed(name)
        grouped[orig] = grouped.get(orig, 0.0) + float(val)

    abs_total = sum(abs(v) for v in grouped.values()) or 1.0
    rows: list[dict[str, Any]] = []
    for feature, shap_val in grouped.items():
        raw_val = x_original[feature] if feature in x_original.index else None
        if hasattr(raw_val, "item"):
            raw_val = raw_val.item()
        if isinstance(raw_val, (np.floating, float)):
            raw_val = float(raw_val)
        elif isinstance(raw_val, (np.integer, int)):
            raw_val = int(raw_val)
        else:
            raw_val = None if pd.isna(raw_val) else str(raw_val)
        rows.append(
            {
                "feature": feature,
                "value": raw_val,
                "shap": shap_val,
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
                "percent_contribution": 100.0 * abs(shap_val) / abs_total,
            }
        )
    rows.sort(key=lambda r: abs(r["shap"]), reverse=True)
    return rows


def explain_patient(
    preprocessor: Any,
    estimator: Any,
    family: str,
    background_X: pd.DataFrame,
    patient: pd.DataFrame,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    X_bg = preprocessor.transform(background_X)
    X_row = preprocessor.transform(patient)
    names = list(preprocessor.get_feature_names_out())
    explainer = _explainer(estimator, family, X_bg)
    raw = explainer.shap_values(X_row)
    shap_row = shap_values_positive(raw)[0]
    return aggregate_by_original(shap_row, names, patient.iloc[0])[:top_k]


def global_importance(
    preprocessor: Any,
    estimator: Any,
    family: str,
    X: pd.DataFrame,
    max_samples: int = 200,
) -> list[dict[str, Any]]:
    sample = X.sample(n=min(max_samples, len(X)), random_state=42)
    Xt = preprocessor.transform(sample)
    names = list(preprocessor.get_feature_names_out())
    explainer = _explainer(estimator, family, Xt)
    raw = explainer.shap_values(Xt)
    sv = shap_values_positive(raw)
    mean_abs: dict[str, float] = {}
    for i, name in enumerate(names):
        orig = original_feature_from_transformed(name)
        mean_abs[orig] = mean_abs.get(orig, 0.0) + float(np.mean(np.abs(sv[:, i])))
    ranked = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in ranked) or 1.0
    return [
        {"feature": f, "mean_abs_shap": v, "percent": 100.0 * v / total}
        for f, v in ranked
    ]


def save_global_importance(payload: dict[str, list[dict[str, Any]]]) -> Path:
    path = ARTIFACTS_DIR / "global_shap.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
