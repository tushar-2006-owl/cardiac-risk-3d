"""Load the Z-Alizadeh Sani extension table and build leakage-safe sklearn pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLUMNS,
    DATASET_CANDIDATES,
    DROP_FEATURES,
    FEATURE_META,
    GROUP_ORDER,
    LABEL_MAPS,
    LEAKAGE_COLUMNS,
    ORDINAL_COLUMNS,
    SEX_MAP,
    TARGETS,
    YN_COLUMNS,
)

logger = logging.getLogger(__name__)


def find_dataset() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    xlsx = list((DATASET_CANDIDATES[0].parent).glob("*.xlsx"))
    csvs = list((DATASET_CANDIDATES[0].parent).glob("*.csv"))
    if xlsx:
        return xlsx[0]
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        "No dataset found in data/. Place the Z-Alizadeh Sani extension CSV/XLSX there. "
        "See README.md for download instructions."
    )


def load_raw_dataframe() -> pd.DataFrame:
    path = find_dataset()
    logger.info("Loading dataset from %s", path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        # Prefer the populated 303-row table over empty header-only sheets.
        best = None
        best_rows = -1
        for sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if len(df) > best_rows:
                best = df
                best_rows = len(df)
        if best is None or best_rows == 0:
            raise ValueError(f"No populated sheet in {path}")
        return best
    return pd.read_csv(path)


def _map_label(series: pd.Series, mapping: dict[Any, int], column: str) -> pd.Series:
    mapped = series.map(lambda v: mapping[v] if v in mapping else pd.NA)
    if mapped.isna().any():
        unknown = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Unmapped labels in {column}: {unknown}. Update LABEL_MAPS in ml/config.py.")
    logger.info("Label mapping for %s: %s", column, mapping)
    return mapped.astype(int)


def normalize_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize encodings and derive binary targets. Returns (df, mapping_log)."""
    df = raw.copy()
    mapping_log: dict[str, Any] = {"sex": SEX_MAP, "yn": LABEL_MAPS["YN"], "targets": {}}

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map(lambda v: SEX_MAP.get(v, v))

    for col in YN_COLUMNS:
        if col in df.columns:
            df[col] = _map_label(df[col], LABEL_MAPS["YN"], col)

    for target_name, spec in TARGETS.items():
        src = spec["source_column"]
        mapping = LABEL_MAPS[src]
        binary_col = f"{target_name}__target"
        df[binary_col] = _map_label(df[src], mapping, src)
        mapping_log["targets"][target_name] = {
            "source_column": src,
            "mapping": {str(k): int(v) for k, v in mapping.items() if not isinstance(k, int)},
            "positive_rate": float(df[binary_col].mean()),
        }
        logger.info(
            "Target %s from %s; positive rate=%.3f",
            target_name,
            src,
            df[binary_col].mean(),
        )
    return df, mapping_log


def feature_columns(df: pd.DataFrame) -> list[str]:
    target_cols = {f"{name}__target" for name in TARGETS}
    leakage = set(LEAKAGE_COLUMNS) | set(DROP_FEATURES) | target_cols
    cols = [c for c in df.columns if c not in leakage]
    leaked = [c for c in cols if c in LEAKAGE_COLUMNS or c.endswith("__target")]
    if leaked:
        raise RuntimeError(f"Leakage columns survived into features: {leaked}")
    return cols


def split_xy(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[f"{target_name}__target"]
    X = df[feature_columns(df)]
    forbidden = set(LEAKAGE_COLUMNS) | {f"{n}__target" for n in TARGETS}
    overlap = forbidden.intersection(X.columns)
    if overlap:
        raise RuntimeError(f"Feature matrix contains leakage columns: {sorted(overlap)}")
    return X, y


def column_roles(X: pd.DataFrame) -> dict[str, list[str]]:
    cat = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    ordinal = [c for c in ORDINAL_COLUMNS if c in X.columns]
    used = set(cat) | set(ordinal)
    numeric = [c for c in X.columns if c not in used]
    return {"numeric": numeric, "categorical": cat, "ordinal": ordinal}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputation + scaling + encoding. Fit only inside CV folds / final training."""
    roles = column_roles(X)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if roles["numeric"]:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                roles["numeric"],
            )
        )
    if roles["categorical"]:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                roles["categorical"],
            )
        )
    if roles["ordinal"]:
        categories = []
        for col in roles["ordinal"]:
            order = ORDINAL_COLUMNS[col]
            present = [v for v in order if v in set(X[col].dropna().unique())]
            extras = [v for v in X[col].dropna().unique() if v not in present]
            categories.append(present + extras)
        transformers.append(
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                roles["ordinal"],
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_feature_schema(X: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for col in X.columns:
        meta = FEATURE_META.get(col, {})
        series = X[col]
        item: dict[str, Any] = {
            "name": col,
            "display_label": meta.get("display_label", col),
            "group": meta.get("group", "Other"),
            "type": meta.get("type", "number" if pd.api.types.is_numeric_dtype(series) else "categorical"),
            "unit": meta.get("unit"),
            "notes": meta.get("notes"),
            "reference_range": meta.get("reference_range"),
        }
        if item["type"] in {"categorical", "ordinal"}:
            cats = meta.get("categories") or sorted(series.dropna().astype(str).unique().tolist())
            item["categories"] = cats
            item["default"] = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else cats[0]
            item["min"] = None
            item["max"] = None
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            item["categories"] = None
            lo = float(numeric.min())
            hi = float(numeric.max())
            item["min"] = lo
            item["max"] = hi
            item["default"] = float(numeric.median())
            if item["type"] == "binary":
                item["categories"] = [0, 1]
                item["min"] = 0
                item["max"] = 1
                item["default"] = int(numeric.mode(dropna=True).iloc[0]) if not numeric.mode(dropna=True).empty else 0
        schema.append(item)

    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    schema.sort(key=lambda s: (group_rank.get(s["group"], 99), s["display_label"]))
    return schema


def original_feature_from_transformed(name: str) -> str:
    """Map ColumnTransformer get_feature_names_out() back to the source column."""
    rest = name.split("__", 1)[-1] if "__" in name else name
    for original in list(FEATURE_META) + CATEGORICAL_COLUMNS + list(ORDINAL_COLUMNS):
        if rest == original or rest.startswith(f"{original}_"):
            return original
    return rest


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
