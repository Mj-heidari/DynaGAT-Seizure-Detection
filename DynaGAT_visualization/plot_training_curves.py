from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(history_files: Sequence[Path], out: Path) -> None:
    histories = []
    for path in history_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[skip] could not read {path.name}: {exc}")
            continue
        if df.empty or "epoch" not in df.columns:
            continue
        histories.append((path, df))

    if not histories:
        print("[skip] training curves: no fold histories")
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for path, df in histories:
        if "train_loss" not in df.columns:
            continue
        ax.plot(
            df["epoch"],
            df["train_loss"],
            linewidth=1.0,
            alpha=0.45,
        )
        if "is_best" in df.columns and np.any(df["is_best"].to_numpy(dtype=int) == 1):
            best_rows = df[df["is_best"] == 1]
            last_best = best_rows.iloc[-1]
            ax.scatter(
                [last_best["epoch"]],
                [last_best["train_loss"]],
                s=18,
                alpha=0.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training focal loss")
    ax.set_title("LOPO Training Convergence Across Folds")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure6_training_loss.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure6_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    any_val = False
    for path, df in histories:
        if "val_auprc" not in df.columns:
            continue
        mask = np.isfinite(df["val_auprc"].to_numpy(dtype=float))
        if not np.any(mask):
            continue
        any_val = True
        ax.plot(
            df.loc[mask, "epoch"],
            df.loc[mask, "val_auprc"],
            marker="o",
            linewidth=1.0,
            alpha=0.55,
        )

    if any_val:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Quick-validation AUPRC")
        ax.set_ylim(bottom=0.0)
        ax.set_title("Validation Checkpoint Selection Across LOPO Folds")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(out) / "Figure6_validation_auprc.pdf", bbox_inches="tight")
        fig.savefig(Path(out) / "Figure6_validation_auprc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
