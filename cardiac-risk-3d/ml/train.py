"""Train one calibrated binary classifier per registered target.

Run: python -m ml.train

Validation: stratified 5-fold CV repeated 3 times (same splits for every model
family). The winner is chosen by mean ROC-AUC. Reported scores are the repeated-CV
metrics of the selected family; choosing among 3 families on the same folds can
add a small optimistic bias. A nested 5-outer / 3-inner selection score is also
stored as a less biased estimate of the selection procedure.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.config import (
    ARTIFACTS_DIR,
    CALIBRATION_FOLDS,
    CALIBRATION_METHOD,
    MODEL_REGISTRY,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    TARGETS,
    make_estimator,
)
from ml.evaluate import binary_metrics, save_target_plots, summarize_folds
from ml.explain import global_importance, save_global_importance
from ml.preprocess import (
    build_feature_schema,
    build_preprocessor,
    ensure_artifacts_dir,
    feature_columns,
    load_raw_dataframe,
    normalize_frame,
    split_xy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def _pipeline_for(model_name: str, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(X)),
            ("clf", make_estimator(model_name, y)),
        ]
    )


def _collect_oof(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    y_arr = y.to_numpy()
    oof_sum = np.zeros(len(y_arr), dtype=float)
    oof_n = np.zeros(len(y_arr), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, test_idx in cv.split(X, y_arr):
        pipe = _pipeline_for(model_name, X.iloc[train_idx], y_arr[train_idx])
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(binary_metrics(y_arr[test_idx], pred, proba))
        oof_sum[test_idx] += proba
        oof_n[test_idx] += 1.0
    oof = oof_sum / np.maximum(oof_n, 1.0)
    oof_pred = (oof >= 0.5).astype(int)
    return fold_metrics, oof, oof_pred


def nested_selection_auc(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Outer 5-fold: pick model by inner 3-fold AUC, score winner on outer test."""
    y_arr = y.to_numpy()
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_aucs: list[float] = []
    winners: list[str] = []
    for train_idx, test_idx in outer.split(X, y_arr):
        X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        best_name, best_auc = None, -1.0
        for name in MODEL_REGISTRY:
            inner_aucs: list[float] = []
            for inn_tr, inn_te in inner.split(X_tr, y_tr):
                pipe = _pipeline_for(name, X_tr.iloc[inn_tr], y_tr[inn_tr])
                pipe.fit(X_tr.iloc[inn_tr], y_tr[inn_tr])
                proba = pipe.predict_proba(X_tr.iloc[inn_te])[:, 1]
                if len(np.unique(y_tr[inn_te])) < 2:
                    continue
                inner_aucs.append(roc_auc_score(y_tr[inn_te], proba))
            mean_auc = float(np.mean(inner_aucs)) if inner_aucs else 0.0
            if mean_auc > best_auc:
                best_auc, best_name = mean_auc, name
        assert best_name is not None
        winners.append(best_name)
        pipe = _pipeline_for(best_name, X_tr, y_tr)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        outer_aucs.append(float(roc_auc_score(y_arr[test_idx], proba)))
    return {
        "mean": float(np.mean(outer_aucs)),
        "std": float(np.std(outer_aucs, ddof=1)),
        "winner_counts": {k: winners.count(k) for k in MODEL_REGISTRY},
    }


def pick_sample_patients(df: pd.DataFrame, X: pd.DataFrame) -> list[dict]:
    """Preset demo cohorts plus a few real rows with ground truth (demo only)."""
    samples: list[dict] = []

    def pack(label: str, risk: str, idx: int) -> dict:
        row = X.iloc[idx]
        features = json.loads(row.to_json())
        truth = {
            "CAD": int(df.iloc[idx]["CAD__target"]),
            "LAD": int(df.iloc[idx]["LAD__target"]),
            "LCX": int(df.iloc[idx]["LCX__target"]),
            "RCA": int(df.iloc[idx]["RCA__target"]),
        }
        return {
            "id": f"demo-{risk}-{idx}",
            "label": label,
            "risk_hint": risk,
            "demo_only": True,
            "ground_truth": truth,
            "features": features,
        }

    cad = df["CAD__target"].to_numpy()
    n_ves = df[["LAD__target", "LCX__target", "RCA__target"]].sum(axis=1).to_numpy()
    low_idx = np.where((cad == 0) & (n_ves == 0))[0]
    mid_idx = np.where((cad == 1) & (n_ves == 1))[0]
    high_idx = np.where(n_ves == 3)[0]
    if len(low_idx):
        samples.append(pack("Low-risk preset (Normal cath, no vessel stenosis)", "low", int(low_idx[0])))
    if len(mid_idx):
        samples.append(pack("Medium-risk preset (single-vessel stenosis)", "medium", int(mid_idx[0])))
    if len(high_idx):
        samples.append(pack("High-risk preset (three-vessel stenosis)", "high", int(high_idx[-1])))
    # Extra real records
    for i in [1, 10, 50]:
        if i < len(df):
            samples.append(pack(f"Real record #{i} (demo only)", "real", i))
    return samples


def train() -> dict:
    ensure_artifacts_dir()
    raw = load_raw_dataframe()
    df, mapping_log = normalize_frame(raw)
    X_all, _ = split_xy(df, next(iter(TARGETS)))
    feature_list = feature_columns(df)
    logger.info("Features (%d): %s", len(feature_list), feature_list)

    schema = build_feature_schema(X_all)
    (ARTIFACTS_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    metrics: dict[str, Any] = {
        "validation": {
            "method": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_STATE,
            "holdout": "none — all labeled rows participate in repeated CV",
            "model_selection": (
                "Each registered family is scored on identical repeated-CV folds. "
                "The family with the highest mean ROC-AUC is selected per target. "
                "Reported repeated-CV scores belong to the selected family and may be "
                "slightly optimistic because the family was chosen on the same folds. "
                "nested_cv_roc_auc is a less biased estimate of that selection procedure."
            ),
            "calibration": (
                f"Final probabilities come from CalibratedClassifierCV "
                f"(method={CALIBRATION_METHOD}, cv={CALIBRATION_FOLDS}) wrapping the "
                "full sklearn Pipeline. Calibration folds are internal and do not reuse "
                "a model already fit on the calibration rows."
            ),
        },
        "label_mapping": mapping_log,
        "n_samples": int(len(df)),
        "features": feature_list,
        "leakage_excluded": sorted(set(df.columns) - set(feature_list) - set(df.filter(regex="__target$").columns)),
        "dropped_zero_variance": ["Exertional CP"],
        "targets": {},
    }

    global_shap: dict[str, list] = {}
    selected_names: dict[str, str] = {}

    for target_name in TARGETS:
        X, y = split_xy(df, target_name)
        y_arr = y.to_numpy()
        pos_rate = float(y.mean())
        logger.info("=== Target %s | n=%d | positive_rate=%.3f ===", target_name, len(y), pos_rate)

        family_scores: dict[str, Any] = {}
        best_name, best_auc = None, -1.0
        best_pack: dict[str, Any] | None = None

        for model_name in MODEL_REGISTRY:
            logger.info("Evaluating %s for %s", model_name, target_name)
            fold_metrics, oof, oof_pred = _collect_oof(model_name, X, y, cv)
            summary = summarize_folds(fold_metrics)
            family_scores[model_name] = {
                "display": MODEL_REGISTRY[model_name]["display"],
                "cv": summary,
            }
            mean_auc = summary["roc_auc"]["mean"]
            logger.info("  %s ROC-AUC %.3f ± %.3f", model_name, mean_auc, summary["roc_auc"]["std"])
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_name = model_name
                best_pack = {
                    "fold_metrics": fold_metrics,
                    "summary": summary,
                    "oof": oof,
                    "oof_pred": oof_pred,
                }

        assert best_name is not None and best_pack is not None
        selected_names[target_name] = best_name
        logger.info("Selected %s for %s", best_name, target_name)

        nested = nested_selection_auc(X, y)
        plots = save_target_plots(target_name, y_arr, best_pack["oof_pred"], best_pack["oof"])

        # Uncalibrated base model on all data (SHAP).
        base_pipe = _pipeline_for(best_name, X, y_arr)
        base_pipe.fit(X, y_arr)

        # Calibrated probabilities: internal CV, not prefit on the same rows.
        calib = CalibratedClassifierCV(
            estimator=_pipeline_for(best_name, X, y_arr),
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_FOLDS,
        )
        calib.fit(X, y_arr)

        joblib.dump(base_pipe, ARTIFACTS_DIR / f"model_{target_name}_base.joblib")
        joblib.dump(calib, ARTIFACTS_DIR / f"model_{target_name}_calibrated.joblib")
        joblib.dump(base_pipe, ARTIFACTS_DIR / f"model_{target_name}.joblib")

        prep = base_pipe.named_steps["prep"]
        clf = base_pipe.named_steps["clf"]
        family = MODEL_REGISTRY[best_name]["family"]
        try:
            global_shap[target_name] = global_importance(prep, clf, family, X)
        except Exception as exc:  # SHAP can fail on some linear edge cases
            logger.warning("Global SHAP failed for %s: %s", target_name, exc)
            global_shap[target_name] = []

        metrics["targets"][target_name] = {
            "positive_rate": pos_rate,
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "selected_model": best_name,
            "selected_model_display": MODEL_REGISTRY[best_name]["display"],
            "cv": best_pack["summary"],
            "candidates": family_scores,
            "nested_cv_roc_auc": nested,
            "plots": plots,
            "oof_brier": float(best_pack["summary"]["brier"]["mean"]),
        }

    save_global_importance(global_shap)
    samples = pick_sample_patients(df, X_all)
    (ARTIFACTS_DIR / "sample_patients.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Console table
    print("\n=== Repeated CV (selected model per target) ===")
    header = f"{'Target':<8} {'Model':<16} {'AUC':>12} {'F1':>12} {'Acc':>12} {'Brier':>12} {'Pos%':>8}"
    print(header)
    print("-" * len(header))
    for t, row in metrics["targets"].items():
        cv = row["cv"]
        print(
            f"{t:<8} {row['selected_model']:<16} "
            f"{cv['roc_auc']['mean']:.3f}±{cv['roc_auc']['std']:.3f}  "
            f"{cv['f1']['mean']:.3f}±{cv['f1']['std']:.3f}  "
            f"{cv['accuracy']['mean']:.3f}±{cv['accuracy']['std']:.3f}  "
            f"{cv['brier']['mean']:.3f}±{cv['brier']['std']:.3f}  "
            f"{100*row['positive_rate']:5.1f}%"
        )
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
