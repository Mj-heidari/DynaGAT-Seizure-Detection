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


def _load_pooled(prediction_files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    probs_parts = []
    labels_parts = []
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as data:
            probs_parts.append(np.asarray(data["probs"], dtype=np.float64))
            labels_parts.append(np.asarray(data["labels"], dtype=np.int64))
    return np.concatenate(labels_parts), np.concatenate(probs_parts)


def plot_roc_pr(prediction_files: Sequence[Path], out: Path) -> None:
    labels, probs = _load_pooled(prediction_files)
    if labels.size == 0 or np.unique(labels).size < 2:
        print("[skip] ROC/PR requires both positive and negative held-out windows")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)

    fig = plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr, label=f"Pooled LOPO AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Held-out Window ROC")
    plt.legend()
    plt.tight_layout()
    fig.savefig(Path(out) / "Figure2_ROC.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure2_ROC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    prevalence = float(np.mean(labels))

    fig = plt.figure(figsize=(5.5, 5.0))
    plt.plot(recall, precision, label=f"Pooled LOPO AUPRC = {auprc:.3f}")
    plt.axhline(prevalence, linestyle="--", label=f"Prevalence = {prevalence:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Held-out Window Precision-Recall")
    plt.legend()
    plt.tight_layout()
    fig.savefig(Path(out) / "Figure2_PR.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure2_PR.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
