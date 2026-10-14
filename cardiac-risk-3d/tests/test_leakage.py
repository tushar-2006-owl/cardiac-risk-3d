"""Fail if angiography / cath labels leak into the feature list."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ml.config import DROP_FEATURES, LEAKAGE_COLUMNS, PROJECT_ROOT, TARGETS
from ml.preprocess import feature_columns, load_raw_dataframe, normalize_frame, split_xy


DERIVED_PATTERNS = [
    r"^cad$",
    r"^cath",
    r"^lad",
    r"^lcx",
    r"^rca",
    r"vessel.?involved",
    r"lesion",
    r"__target$",
]


def _is_forbidden(name: str) -> bool:
    if name in LEAKAGE_COLUMNS or name in DROP_FEATURES:
        return True
    lowered = name.lower().strip()
    if lowered in {t.lower() for t in TARGETS} and name in LEAKAGE_COLUMNS:
        return True
    for pat in DERIVED_PATTERNS:
        if re.search(pat, lowered) and name not in {
            # Region RWMA is a clinical score, not a lesion coordinate.
        }:
            if lowered in {"lad", "lcx", "rca", "cath", "cad"} or lowered.endswith("__target"):
                return True
            if "proximal" in lowered or "distal" in lowered or "mid" in lowered:
                return True
            if "vessel involved" in lowered or "lesion" in lowered:
                return True
    return False


def test_split_xy_excludes_leakage():
    raw = load_raw_dataframe()
    df, _ = normalize_frame(raw)
    cols = feature_columns(df)
    leaked = [c for c in cols if _is_forbidden(c) or c in LEAKAGE_COLUMNS]
    assert leaked == [], f"Leakage columns in features: {leaked}"
    for target in TARGETS:
        X, _ = split_xy(df, target)
        overlap = set(X.columns) & set(LEAKAGE_COLUMNS)
        assert not overlap, overlap
        assert "Cath" not in X.columns
        assert "LAD" not in X.columns
        assert "LCX" not in X.columns
        assert "RCA" not in X.columns


def test_saved_schema_and_metrics_have_no_leakage():
    schema_path = PROJECT_ROOT / "artifacts" / "feature_schema.json"
    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    assert schema_path.exists(), "Run python -m ml.train first"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in schema]
    leaked = [n for n in names if n in LEAKAGE_COLUMNS or n in DROP_FEATURES]
    assert leaked == [], leaked
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    leaked2 = [n for n in metrics["features"] if n in LEAKAGE_COLUMNS]
    assert leaked2 == [], leaked2
