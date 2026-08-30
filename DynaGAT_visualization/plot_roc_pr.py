from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        probs = np.asarray(data["probs"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        patient = (
            str(np.asarray(data["test_patient"]).item())
            if "test_patient" in data.files
            else path.stem
        )
    return labels, probs, patient


def plot_roc_pr(prediction_files: Sequence[Path], out: Path) -> None:
    pooled_labels = []
    pooled_probs = []
    folds = []

    for path in prediction_files:
        labels, probs, patient = _load_prediction(path)
        if labels.size == 0:
            continue
        pooled_labels.append(labels)
        pooled_probs.append(probs)
        folds.append((labels, probs, patient))

    if not pooled_labels:
        print("[skip] ROC/PR: no held-out predictions")
        return

    labels_all = np.concatenate(pooled_labels)
    probs_all = np.concatenate(pooled_probs)
    if np.unique(labels_all).size < 2:
        print("[skip] ROC/PR requires both positive and negative held-out windows")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    fold_aurocs = []
    for labels, probs, _patient in folds:
        if np.unique(labels).size < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        fold_aurocs.append(float(roc_auc_score(labels, probs)))
        ax.plot(fpr, tpr, linewidth=0.8, alpha=0.22)

    fpr, tpr, _ = roc_curve(labels_all, probs_all)
    pooled_auc = float(roc_auc_score(labels_all, probs_all))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"Pooled AUROC = {pooled_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Chance")
    if fold_aurocs:
        ax.text(
            0.98,
            0.04,
            f"Patient AUROC: {np.mean(fold_aurocs):.3f} ± {np.std(fold_aurocs):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Held-out LOPO Window ROC")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure2_ROC.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure2_ROC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    fold_auprcs = []
    for labels, probs, _patient in folds:
        if np.unique(labels).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels, probs)
        fold_auprcs.append(float(average_precision_score(labels, probs)))
        ax.plot(recall, precision, linewidth=0.8, alpha=0.22)

    precision, recall, _ = precision_recall_curve(labels_all, probs_all)
    pooled_auprc = float(average_precision_score(labels_all, probs_all))
    prevalence = float(np.mean(labels_all))
    ax.plot(recall, precision, linewidth=2.2, label=f"Pooled AUPRC = {pooled_auprc:.3f}")
    ax.axhline(prevalence, linestyle="--", linewidth=1.0, label=f"Prevalence = {prevalence:.4f}")
    if fold_auprcs:
        ax.text(
            0.98,
            0.96,
            f"Patient AUPRC: {np.mean(fold_auprcs):.3f} ± {np.std(fold_auprcs):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out LOPO Precision-Recall")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure2_PR.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure2_PR.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
