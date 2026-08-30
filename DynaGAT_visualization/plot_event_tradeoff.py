from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_event_tradeoff(summary_csv: Path, out: Path) -> None:
    df = pd.read_csv(summary_csv)
    required = {"fa_per_hour", "event_sensitivity", "test_patient"}
    if df.empty or not required.issubset(df.columns):
        print("[skip] event tradeoff: required summary columns are unavailable")
        return

    x = df["fa_per_hour"].to_numpy(dtype=float)
    y = df["event_sensitivity"].to_numpy(dtype=float)
    patients = df["test_patient"].astype(str).tolist()
    gt = (
        df["gt_seizures"].to_numpy(dtype=float)
        if "gt_seizures" in df.columns
        else np.ones(len(df), dtype=float)
    )
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        print("[skip] event tradeoff: no finite patient metrics")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sizes = 45.0 + 18.0 * np.sqrt(np.maximum(gt, 1.0))
    ax.scatter(x[finite], y[finite], s=sizes[finite], alpha=0.8)

    for px, py, patient in zip(x[finite], y[finite], np.asarray(patients)[finite]):
        ax.annotate(patient, (px, py), textcoords="offset points", xytext=(5, 4), fontsize=8)

    median_fa = float(np.nanmedian(x[finite]))
    mean_sens = float(np.nanmean(y[finite]))
    ax.axvline(median_fa, linestyle="--", linewidth=1.0, label=f"Median FA/h = {median_fa:.3f}")
    ax.axhline(mean_sens, linestyle="--", linewidth=1.0, label=f"Mean sensitivity = {mean_sens:.3f}")
    ax.set_xlabel("False alarms per interictal hour")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Held-out Patient Event Operating Profile")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure4_event_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure4_event_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if "median_latency_sec" in df.columns:
        latency = df["median_latency_sec"].to_numpy(dtype=float)
        order = np.argsort(np.nan_to_num(latency, nan=np.inf))
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.bar(np.arange(len(df)), latency[order])
        ax.set_xticks(np.arange(len(df)), np.asarray(patients)[order], rotation=60, ha="right")
        ax.set_ylabel("Median detection latency (s)")
        ax.set_xlabel("Held-out patient")
        ax.set_title("Patient-Level Seizure Detection Latency")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(out) / "Figure4_latency_by_patient.pdf", bbox_inches="tight")
        fig.savefig(Path(out) / "Figure4_latency_by_patient.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
