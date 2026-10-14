"""CV metrics and diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import ARTIFACTS_DIR


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = fold_metrics[0].keys()
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        out[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1))}
    return out


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, title: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#b45309", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, title: str, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, "o-", color="#b45309", label="Observed")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_target_plots(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    roc_path = ARTIFACTS_DIR / f"roc_{target}.png"
    cal_path = ARTIFACTS_DIR / f"calibration_{target}.png"
    cm_path = ARTIFACTS_DIR / f"confusion_{target}.png"
    plot_roc(y_true, y_proba, f"{target} ROC (OOF)", roc_path)
    plot_calibration(y_true, y_proba, f"{target} calibration (OOF)", cal_path)
    plot_confusion(y_true, y_pred, f"{target} confusion (OOF)", cm_path)
    return {
        "roc": f"/plots/roc_{target}.png",
        "calibration": f"/plots/calibration_{target}.png",
        "confusion": f"/plots/confusion_{target}.png",
    }
