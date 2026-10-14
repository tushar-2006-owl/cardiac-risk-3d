"""Central registry for paths, targets, leakage rules, models, and feature metadata.

Adding a new target or estimator only requires registering it here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASET_CANDIDATES = [
    DATA_DIR / "z_alizadeh_sani_extension.xlsx",
    DATA_DIR / "extention of Z-Alizadeh sani dataset.xlsx",
]

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
CALIBRATION_FOLDS = 5
CALIBRATION_METHOD = "sigmoid"  # isotonic overfits easily at n≈303

# Risk bands for predicted probability of the positive (disease) class.
RISK_THRESHOLDS = {"low": 0.33, "high": 0.66}  # medium is the interval between

DISCLAIMER = (
    "For decision support and educational purposes only. "
    "Not a substitute for formal diagnostic imaging or clinical judgment."
)

# Columns that must never enter X for ANY target.
LEAKAGE_COLUMNS = [
    "LAD",
    "LCX",
    "RCA",
    "Cath",
    "CAD",
    "Cath_binary",
    "Number of Vessel Involved",
    "LM Proximal",
    "LM Distal",
    "LAD Proximal",
    "LAD Mid",
    "LAD Distal",
    "LCX Proximal",
    "LCX Mid",
    "LCX Distal",
    "RCA Proximal",
    "RCA Mid",
    "RCA Distal",
    "EF- Cath",
    "Number Of lesion PCI",
]

# Zero-variance in this extract (all "N"); documented drop.
DROP_FEATURES = ["Exertional CP"]

# Mesh names used by the 3D viewer (must stay in sync with frontend VESSEL_CONFIG).
VESSEL_MESH_KEYS = {"LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}

LABEL_MAPS: dict[str, dict[Any, int]] = {
    "Cath": {"CAD": 1, "Normal": 0, "cad": 1, "normal": 0},
    "LAD": {"Stenotic": 1, "Normal": 0, "stenotic": 1, "normal": 0},
    "LCX": {"Stenotic": 1, "Normal": 0, "stenotic": 1, "normal": 0},
    "RCA": {"Stenotic": 1, "Normal": 0, "stenotic": 1, "normal": 0},
    "YN": {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0},
}

SEX_MAP = {"Male": "Male", "male": "Male", "Fmale": "Female", "Female": "Female", "female": "Female"}

YN_COLUMNS = [
    "Obesity",
    "CRF",
    "CVA",
    "Airway disease",
    "Thyroid Disease",
    "CHF",
    "DLP",
    "Weak Peripheral Pulse",
    "Lung rales",
    "Systolic Murmur",
    "Diastolic Murmur",
    "Dyspnea",
    "Atypical",
    "Nonanginal",
    "LowTH Ang",
    "LVH",
    "Poor R Progression",
    "Exertional CP",
]

CATEGORICAL_COLUMNS = ["Sex", "BBB"]
ORDINAL_COLUMNS = {
    "VHD": ["N", "mild", "Mild", "Moderate", "Severe"],
}

TARGETS: dict[str, dict[str, Any]] = {
    "CAD": {
        "source_column": "Cath",
        "positive_label": "CAD",
        "display": "Overall CAD",
        "mesh_key": None,
    },
    "LAD": {
        "source_column": "LAD",
        "positive_label": "Stenotic",
        "display": "LAD stenosis",
        "mesh_key": "LAD",
    },
    "LCX": {
        "source_column": "LCX",
        "positive_label": "Stenotic",
        "display": "LCX stenosis",
        "mesh_key": "LCX",
    },
    "RCA": {
        "source_column": "RCA",
        "positive_label": "Stenotic",
        "display": "RCA stenosis",
        "mesh_key": "RCA",
    },
}


def _lr(**kwargs: Any) -> LogisticRegression:
    params = dict(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def _rf(**kwargs: Any) -> RandomForestClassifier:
    params = dict(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def _xgb(scale_pos_weight: float = 1.0, **kwargs: Any) -> XGBClassifier:
    params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logreg": {
        "display": "Logistic Regression",
        "family": "linear",
        "builder": _lr,
        "needs_scale_pos_weight": False,
    },
    "random_forest": {
        "display": "Random Forest",
        "family": "tree",
        "builder": _rf,
        "needs_scale_pos_weight": False,
    },
    "xgboost": {
        "display": "XGBoost",
        "family": "tree",
        "builder": _xgb,
        "needs_scale_pos_weight": True,
    },
}


def make_estimator(name: str, y: np.ndarray | None = None) -> Any:
    """Instantiate a registered model, applying imbalance weights when needed."""
    spec = MODEL_REGISTRY[name]
    builder: Callable[..., Any] = spec["builder"]
    if spec.get("needs_scale_pos_weight") and y is not None:
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        spw = (neg / pos) if pos > 0 else 1.0
        return builder(scale_pos_weight=spw)
    return builder()


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "low"
    if probability > RISK_THRESHOLDS["high"]:
        return "high"
    return "medium"


# Feature form metadata. min/max/default/categories are filled from data at train time
# when missing; reference ranges are clinical typical values, not dataset-derived.
FEATURE_META: dict[str, dict[str, Any]] = {
    "Age": {
        "display_label": "Age",
        "group": "Demographics",
        "type": "number",
        "unit": "years",
        "reference_range": None,
    },
    "Weight": {
        "display_label": "Weight",
        "group": "Demographics",
        "type": "number",
        "unit": "kg",
        "reference_range": None,
    },
    "Length": {
        "display_label": "Height",
        "group": "Demographics",
        "type": "number",
        "unit": "cm",
        "reference_range": None,
    },
    "Sex": {
        "display_label": "Sex",
        "group": "Demographics",
        "type": "categorical",
        "unit": None,
        "reference_range": None,
    },
    "BMI": {
        "display_label": "BMI",
        "group": "Demographics",
        "type": "number",
        "unit": "kg/m²",
        "reference_range": {"low": 18.5, "high": 24.9},
    },
    "DM": {
        "display_label": "Diabetes mellitus",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "HTN": {
        "display_label": "Hypertension",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Current Smoker": {
        "display_label": "Current smoker",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "EX-Smoker": {
        "display_label": "Ex-smoker",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "FH": {
        "display_label": "Family history of CAD",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Obesity": {
        "display_label": "Obesity",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "CRF": {
        "display_label": "Chronic renal failure",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "CVA": {
        "display_label": "Cerebrovascular accident",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Airway disease": {
        "display_label": "Airway disease",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Thyroid Disease": {
        "display_label": "Thyroid disease",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "CHF": {
        "display_label": "Congestive heart failure",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "DLP": {
        "display_label": "Dyslipidemia",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "BP": {
        "display_label": "Systolic blood pressure",
        "group": "Clinical exam",
        "type": "number",
        "unit": "mmHg",
        "reference_range": {"low": 90, "high": 129},
    },
    "PR": {
        "display_label": "Pulse rate",
        "group": "Clinical exam",
        "type": "number",
        "unit": "bpm",
        "reference_range": {"low": 60, "high": 100},
    },
    "Edema": {
        "display_label": "Edema",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Weak Peripheral Pulse": {
        "display_label": "Weak peripheral pulse",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Lung rales": {
        "display_label": "Lung rales",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Systolic Murmur": {
        "display_label": "Systolic murmur",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Diastolic Murmur": {
        "display_label": "Diastolic murmur",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Typical Chest Pain": {
        "display_label": "Typical chest pain",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Dyspnea": {
        "display_label": "Dyspnea",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Function Class": {
        "display_label": "NYHA / function class",
        "group": "Clinical exam",
        "type": "number",
        "unit": None,
        "reference_range": {"low": 0, "high": 1},
    },
    "Atypical": {
        "display_label": "Atypical chest pain",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "Nonanginal": {
        "display_label": "Non-anginal chest pain",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": None,
    },
    "LowTH Ang": {
        "display_label": "Low-threshold angina",
        "group": "Clinical exam",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Q Wave": {
        "display_label": "Q wave",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "St Elevation": {
        "display_label": "ST elevation",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "St Depression": {
        "display_label": "ST depression",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Tinversion": {
        "display_label": "T-wave inversion",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "LVH": {
        "display_label": "Left ventricular hypertrophy",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "Poor R Progression": {
        "display_label": "Poor R-wave progression",
        "group": "ECG",
        "type": "binary",
        "unit": None,
        "reference_range": {"low": 0, "high": 0},
    },
    "BBB": {
        "display_label": "Bundle branch block",
        "group": "ECG",
        "type": "categorical",
        "unit": None,
        "reference_range": None,
    },
    "FBS": {
        "display_label": "Fasting blood sugar",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 70, "high": 99},
    },
    "CR": {
        "display_label": "Creatinine",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 0.6, "high": 1.3},
    },
    "TG": {
        "display_label": "Triglycerides",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 0, "high": 149},
    },
    "LDL": {
        "display_label": "LDL cholesterol",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 0, "high": 99},
    },
    "HDL": {
        "display_label": "HDL cholesterol",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 40, "high": 100},
    },
    "BUN": {
        "display_label": "BUN",
        "group": "Laboratory",
        "type": "number",
        "unit": "mg/dL",
        "reference_range": {"low": 7, "high": 20},
    },
    "ESR": {
        "display_label": "ESR",
        "group": "Laboratory",
        "type": "number",
        "unit": "mm/h",
        "reference_range": {"low": 0, "high": 20},
    },
    "HB": {
        "display_label": "Hemoglobin",
        "group": "Laboratory",
        "type": "number",
        "unit": "g/dL",
        "reference_range": {"low": 12.0, "high": 17.0},
    },
    "K": {
        "display_label": "Potassium",
        "group": "Laboratory",
        "type": "number",
        "unit": "mEq/L",
        "reference_range": {"low": 3.5, "high": 5.0},
    },
    "Na": {
        "display_label": "Sodium",
        "group": "Laboratory",
        "type": "number",
        "unit": "mEq/L",
        "reference_range": {"low": 135, "high": 145},
    },
    "WBC": {
        "display_label": "White blood cells",
        "group": "Laboratory",
        "type": "number",
        "unit": "/µL",
        "reference_range": {"low": 4000, "high": 11000},
    },
    "Lymph": {
        "display_label": "Lymphocytes",
        "group": "Laboratory",
        "type": "number",
        "unit": "%",
        "reference_range": {"low": 20, "high": 40},
    },
    "Neut": {
        "display_label": "Neutrophils",
        "group": "Laboratory",
        "type": "number",
        "unit": "%",
        "reference_range": {"low": 40, "high": 70},
    },
    "PLT": {
        "display_label": "Platelets",
        "group": "Laboratory",
        "type": "number",
        "unit": "×10³/µL",
        "reference_range": {"low": 150, "high": 400},
    },
    "EF-TTE": {
        "display_label": "Ejection fraction (TTE)",
        "group": "Echo",
        "type": "number",
        "unit": "%",
        "reference_range": {"low": 50, "high": 70},
    },
    "Region RWMA": {
        "display_label": "Region with RWMA",
        "group": "Echo",
        "type": "number",
        "unit": "score 0–4",
        "reference_range": {"low": 0, "high": 0},
        "notes": "Clinical echo wall-motion score, not an anatomical lesion coordinate.",
    },
    "VHD": {
        "display_label": "Valvular heart disease",
        "group": "Echo",
        "type": "ordinal",
        "unit": None,
        "reference_range": None,
        "categories": ["N", "mild", "Moderate", "Severe"],
    },
}

GROUP_ORDER = ["Demographics", "Clinical exam", "ECG", "Laboratory", "Echo"]
