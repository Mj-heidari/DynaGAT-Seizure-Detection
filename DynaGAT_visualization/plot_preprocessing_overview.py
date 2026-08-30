from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_preprocessing_overview(manifest_csv: Path, out: Path) -> None:
    df = pd.read_csv(manifest_csv)
    if df.empty:
        print("[skip] preprocessing overview: manifest is empty")
        return

    df = df.sort_values("subject").reset_index(drop=True)
    subjects = df["subject"].astype(str).tolist()
    hours = df["recording_hours"].to_numpy(dtype=float)
    seizures = df["seizures"].to_numpy(dtype=int)
    skipped = df["skipped_recordings"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(9.5, max(6.0, 0.34 * len(df))))
    y = np.arange(len(df))
    bars = ax.barh(y, hours)
    ax.set_yticks(y, subjects)
    ax.set_xlabel("Usable recording hours")
    ax.set_ylabel("Patient")
    ax.set_title("CHB-MIT v3 Preprocessing Coverage")
    ax.invert_yaxis()

    max_hours = max(float(np.nanmax(hours)), 1.0)
    for i, (bar, n_seiz, n_skip) in enumerate(zip(bars, seizures, skipped)):
        note = f"{n_seiz} seizures"
        if n_skip:
            note += f", {n_skip} skipped"
        ax.text(
            float(bar.get_width()) + max_hours * 0.012,
            float(bar.get_y() + bar.get_height() / 2),
            note,
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, max_hours * 1.28)
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure0_preprocessing_coverage.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure0_preprocessing_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    positive_fraction = df["positive_fraction"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, max(6.0, 0.34 * len(df))))
    bars = ax.barh(y, positive_fraction * 100.0)
    ax.set_yticks(y, subjects)
    ax.set_xlabel("Seizure-positive windows (%)")
    ax.set_ylabel("Patient")
    ax.set_title("Patient-Level Class Imbalance After Preprocessing")
    ax.invert_yaxis()

    max_pct = max(float(np.nanmax(positive_fraction * 100.0)), 0.1)
    for bar, value in zip(bars, positive_fraction):
        ax.text(
            float(bar.get_width()) + max_pct * 0.012,
            float(bar.get_y() + bar.get_height() / 2),
            f"{100.0 * value:.3f}%",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max_pct * 1.28)
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure0_class_imbalance.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure0_class_imbalance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
