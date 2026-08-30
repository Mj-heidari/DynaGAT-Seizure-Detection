from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


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

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    observed = []
    counts = []

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == 0 else (probs > low) & (probs <= high)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(probs[mask])))
        observed.append(float(np.mean(labels[mask])))
        counts.append(int(np.sum(mask)))

    fig = plt.figure(figsize=(5.5, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(centers, observed, marker="o", label="DynaGAT-Onset")
    for x, y, n in zip(centers, observed, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed seizure-window frequency")
    plt.title("Held-out Reliability Calibration")
    plt.legend()
    plt.tight_layout()
    fig.savefig(Path(out) / "Figure5_calibration.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure5_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
