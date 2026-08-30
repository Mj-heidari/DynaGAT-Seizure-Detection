from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lopo_heatmap(csv_file: Path, out: Path) -> None:
    df = pd.read_csv(csv_file)
    if df.empty:
        print("[skip] LOPO heatmap: summary is empty")
        return

    preferred = [
        "auroc",
        "auprc",
        "f1",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "ece",
    ]
    metrics = [name for name in preferred if name in df.columns]
    if not metrics:
        print("[skip] LOPO heatmap: no supported numeric metrics")
        return

    values = df[metrics].to_numpy(dtype=float).T
    labels = (
        df["test_patient"].astype(str).tolist()
        if "test_patient" in df.columns
        else [str(i + 1) for i in range(len(df))]
    )

    fig = plt.figure(figsize=(max(9, len(df) * 0.42), 5.5))
    image = plt.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
    plt.yticks(range(len(metrics)), metrics)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.colorbar(image, label="Metric value")
    plt.title("LOPO Held-out Patient Performance")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                plt.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(Path(out) / "Figure1_LOPO_heatmap.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure1_LOPO_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
