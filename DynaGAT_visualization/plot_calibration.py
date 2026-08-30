from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ece(probs: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> float:
    value = 0.0
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == 0 else (probs > low) & (probs <= high)
        if not np.any(mask):
            continue
        value += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(probs[mask]))
        )
    return value


def plot_calibration(prediction_files: Sequence[Path], out: Path, n_bins: int = 10) -> None:
    probs_parts = []
    labels_parts = []
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as data:
            probs_parts.append(np.asarray(data["probs"], dtype=np.float64))
            labels_parts.append(np.asarray(data["labels"], dtype=np.float64))

    probs = np.concatenate(probs_parts)
    labels = np.concatenate(labels_parts)
    if probs.size == 0:
        print("[skip] calibration: no predictions")
        return

    # Equal-frequency bins are much more informative than uniform bins for a
    # heavily imbalanced seizure detector because they avoid mostly empty bins.
    quantiles = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(np.clip(quantiles, 0.0, 1.0))
    if edges.size < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] = 0.0
    edges[-1] = 1.0

    centers = []
    observed = []
    counts = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == 0 else (probs > low) & (probs <= high)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(probs[mask])))
        observed.append(float(np.mean(labels[mask])))
        counts.append(int(np.sum(mask)))

    ece = _ece(probs, labels, edges)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.plot(centers, observed, marker="o", linewidth=1.8, label=f"DynaGAT (ECE = {ece:.3f})")
    for x, y, n in zip(centers, observed, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed seizure-window frequency")
    ax.set_title("Held-out Reliability Calibration")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure5_calibration.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure5_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
