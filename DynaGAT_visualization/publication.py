from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import DEVELOPMENT_FOLD, WINDOW_STRIDE_SEC


DPI = 600


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _primary_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv).sort_values("fold").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("LOPO summary is empty")
    return df[df["fold"].astype(int) != DEVELOPMENT_FOLD].copy()


def _prediction_files(primary: pd.DataFrame, results_dir: Path) -> list[Path]:
    files = [
        results_dir / f"fold_{int(fold):02d}_test_predictions.npz"
        for fold in primary["fold"].tolist()
    ]
    return [path for path in files if path.exists()]


def _history_files(primary: pd.DataFrame, results_dir: Path) -> list[Path]:
    files = [
        results_dir / f"fold_{int(fold):02d}_training_history.csv"
        for fold in primary["fold"].tolist()
    ]
    return [path for path in files if path.exists()]


def _load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files if name != "recording_metadata_json"}
        if "recording_metadata_json" in data.files:
            result["metadata"] = json.loads(str(np.asarray(data["recording_metadata_json"]).item()))
        else:
            result["metadata"] = {}
    return result


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def plot_preprocessing(manifest_csv: Path, out: Path) -> None:
    if not manifest_csv.exists():
        return
    df = pd.read_csv(manifest_csv)
    subjects = df["subject"].astype(str).tolist()
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=True)
    axes[0].bar(x, df["recording_hours"].to_numpy(dtype=float))
    axes[0].set_ylabel("Hours")
    axes[0].set_title("CHB-MIT preprocessing coverage")
    axes[1].bar(x, df["seizures"].to_numpy(dtype=float))
    axes[1].set_ylabel("Seizures")
    axes[2].bar(x, df["positive_fraction"].to_numpy(dtype=float) * 100.0)
    axes[2].set_ylabel("Positive windows (%)")
    axes[2].set_xticks(x, subjects, rotation=60, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    _save(fig, out, "Figure1_dataset_preprocessing")


def plot_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    blocks = [
        (0.2, 1.7, 1.7, 1.5, "EEG\n2 s windows"),
        (2.2, 1.7, 1.8, 1.5, "20 node\nfeatures"),
        (4.3, 2.7, 1.8, 1.2, "Static\nGATv2"),
        (4.3, 1.0, 1.8, 1.2, "Dynamic\nGATv2"),
        (6.4, 1.7, 1.8, 1.5, "Attention pool\n+ gated fusion"),
        (8.5, 1.7, 1.8, 1.5, "Causal\nmultiscale TCN"),
        (10.6, 1.7, 1.8, 1.5, "Causal\nTransformer"),
    ]
    for x, y, w, h, label in blocks:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04", linewidth=1.2, fill=False
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [
        ((1.9, 2.45), (2.2, 2.45)),
        ((4.0, 2.45), (4.3, 3.3)),
        ((4.0, 2.45), (4.3, 1.6)),
        ((6.1, 3.3), (6.4, 2.75)),
        ((6.1, 1.6), (6.4, 2.15)),
        ((8.2, 2.45), (8.5, 2.45)),
        ((10.3, 2.45), (10.6, 2.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.2))
    ax.text(11.5, 0.65, "h(t) + causal Δh(t) → seizure probability", ha="center", fontsize=10)
    ax.annotate("", xy=(11.5, 1.7), xytext=(11.5, 0.9), arrowprops=dict(arrowstyle="->"))
    ax.set_title("DynaGAT causal seizure-detection architecture", fontsize=13)
    _save(fig, out, "Figure2_model_architecture")


def plot_training(history_files: Sequence[Path], out: Path) -> None:
    histories = [pd.read_csv(path) for path in history_files]
    histories = [df for df in histories if not df.empty]
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    max_epoch = max(int(df["epoch"].max()) for df in histories)
    loss_matrix = np.full((len(histories), max_epoch), np.nan, dtype=float)
    for i, df in enumerate(histories):
        epoch = df["epoch"].to_numpy(dtype=int)
        loss = df["train_loss"].to_numpy(dtype=float)
        axes[0].plot(epoch, loss, alpha=0.22, linewidth=0.8)
        loss_matrix[i, epoch - 1] = loss
        valid = np.isfinite(df["val_auprc"].to_numpy(dtype=float))
        if np.any(valid):
            axes[1].plot(
                df.loc[valid, "epoch"], df.loc[valid, "val_auprc"],
                marker="o", alpha=0.25, linewidth=0.8, markersize=3,
            )
    axes[0].plot(
        np.arange(1, max_epoch + 1), np.nanmean(loss_matrix, axis=0),
        linewidth=2.2, label="Mean across primary folds",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training convergence")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUPRC")
    axes[1].set_title("Checkpoint-selection trajectories")
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, out, "Figure3_training_convergence")


def plot_roc_pr(prediction_files: Sequence[Path], out: Path) -> None:
    folds = []
    for path in prediction_files:
        data = _load_prediction(path)
        labels = np.asarray(data["labels"], dtype=int)
        probs = np.asarray(data["probs"], dtype=float)
        if labels.size and np.unique(labels).size > 1:
            folds.append((labels, probs))
    if not folds:
        return
    labels_all = np.concatenate([item[0] for item in folds])
    probs_all = np.concatenate([item[1] for item in folds])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for labels, probs in folds:
        fpr, tpr, _ = roc_curve(labels, probs)
        axes[0].plot(fpr, tpr, linewidth=0.7, alpha=0.18)
        precision, recall, _ = precision_recall_curve(labels, probs)
        axes[1].plot(recall, precision, linewidth=0.7, alpha=0.18)

    fpr, tpr, _ = roc_curve(labels_all, probs_all)
    pooled_auc = roc_auc_score(labels_all, probs_all)
    axes[0].plot(fpr, tpr, linewidth=2.2, label=f"Pooled AUROC = {pooled_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("Window-level ROC")
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(labels_all, probs_all)
    pooled_ap = average_precision_score(labels_all, probs_all)
    prevalence = float(np.mean(labels_all))
    axes[1].plot(recall, precision, linewidth=2.2, label=f"Pooled AUPRC = {pooled_ap:.3f}")
    axes[1].axhline(prevalence, linestyle="--", linewidth=1.0, label=f"Prevalence = {prevalence:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Window-level precision-recall")
    axes[1].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, out, "Figure4_window_discrimination")


def plot_event_tradeoff(primary: pd.DataFrame, out: Path) -> None:
    x = primary["fa_per_hour"].to_numpy(dtype=float)
    y = primary["event_sensitivity"].to_numpy(dtype=float)
    patients = primary["test_patient"].astype(str).to_numpy()
    gt = primary["gt_seizures"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.scatter(x[finite], y[finite], s=45 + 16 * np.sqrt(np.maximum(gt[finite], 1)), alpha=0.8)
    for px, py, patient in zip(x[finite], y[finite], patients[finite]):
        ax.annotate(patient, (px, py), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(float(np.nanmean(y[finite])), linestyle="--", linewidth=1.0)
    ax.axvline(float(np.nanmedian(x[finite])), linestyle="--", linewidth=1.0)
    ax.set_xlabel("False alarms per interictal hour")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Primary held-out event operating profile")
    ax.grid(alpha=0.2)
    _save(fig, out, "Figure5_event_tradeoff")


def plot_sensitivity_forest(primary: pd.DataFrame, out: Path) -> None:
    rows = []
    for row in primary.itertuples(index=False):
        gt = int(row.gt_seizures)
        detected = int(row.detected_seizures)
        low, high = _wilson(detected, gt)
        rows.append((str(row.test_patient), detected / gt if gt else np.nan, low, high, gt))
    rows.sort(key=lambda item: item[1])
    labels = [item[0] for item in rows]
    values = np.asarray([item[1] for item in rows], dtype=float)
    low = np.asarray([item[2] for item in rows], dtype=float)
    high = np.asarray([item[3] for item in rows], dtype=float)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.32 * len(rows) + 1.5)))
    ax.errorbar(values, y, xerr=[values - low, high - values], fmt="o", capsize=2, linewidth=1.0)
    pooled = float(primary["detected_seizures"].sum() / primary["gt_seizures"].sum())
    ax.axvline(pooled, linestyle="--", linewidth=1.2, label=f"Pooled sensitivity = {pooled:.3f}")
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Event sensitivity (95% Wilson CI)")
    ax.set_ylabel("Held-out patient")
    ax.set_title("Patient-level seizure detection sensitivity")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    _save(fig, out, "Figure6_sensitivity_forest")


def plot_patient_events(primary: pd.DataFrame, out: Path) -> None:
    order = np.argsort(primary["event_sensitivity"].to_numpy(dtype=float))
    df = primary.iloc[order].reset_index(drop=True)
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    axes[0].bar(x, df["gt_seizures"], label="Ground-truth seizures")
    axes[0].bar(x, df["detected_seizures"], label="Detected seizures")
    axes[0].set_ylabel("Events")
    axes[0].legend()
    axes[1].bar(x, df["fa_per_hour"])
    axes[1].set_ylabel("FA/h")
    axes[1].set_xlabel("Held-out patient")
    axes[1].set_xticks(x, df["test_patient"].astype(str), rotation=60, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    _save(fig, out, "Figure7_patient_event_results")


def plot_calibration(prediction_files: Sequence[Path], out: Path, n_bins: int = 10) -> None:
    if not prediction_files:
        return
    probs = np.concatenate([np.asarray(_load_prediction(path)["probs"], dtype=float) for path in prediction_files])
    labels = np.concatenate([np.asarray(_load_prediction(path)["labels"], dtype=float) for path in prediction_files])
    if probs.size == 0:
        return
    edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0
    centers, observed, weights = [], [], []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == 0 else (probs > low) & (probs <= high)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(probs[mask])))
        observed.append(float(np.mean(labels[mask])))
        weights.append(float(np.mean(mask)))
    ece = float(np.sum(np.asarray(weights) * np.abs(np.asarray(observed) - np.asarray(centers))))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.plot(centers, observed, marker="o", linewidth=1.8, label=f"DynaGAT (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed seizure-window frequency")
    ax.set_title("Primary held-out calibration")
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, out, "Figure8_calibration")


def plot_detection_timeline(prediction_files: Sequence[Path], out: Path) -> None:
    best = None
    best_score = -np.inf
    for path in prediction_files:
        data = _load_prediction(path)
        probs = np.asarray(data["probs"], dtype=float)
        recording_ids = np.asarray(data["recording_ids"], dtype=str)
        window_indices = np.asarray(data["window_indices"], dtype=int)
        threshold = float(np.asarray(data["threshold"]).item())
        metadata = data.get("metadata", {})
        for rid in np.unique(recording_ids):
            intervals = metadata.get(str(rid), {}).get("seizure_intervals", [])
            if not intervals:
                continue
            mask = recording_ids == rid
            idx = window_indices[mask]
            p = probs[mask]
            order = np.argsort(idx)
            idx, p = idx[order], p[order]
            for seizure_start, seizure_end in intervals:
                onset_window = int(round(float(seizure_start) / WINDOW_STRIDE_SEC))
                pos = int(np.searchsorted(idx, onset_window))
                left, right = max(0, pos - 30), min(len(p), pos + 90)
                if right <= left:
                    continue
                score = float(np.nanmax(p[left:right]))
                if score > best_score:
                    best_score = score
                    best = (rid, idx, p, threshold, float(seizure_start), float(seizure_end), pos)
    if best is None:
        return
    rid, idx, probs, threshold, seizure_start, seizure_end, pos = best
    left, right = max(0, pos - 90), min(len(idx), pos + 150)
    t = idx[left:right].astype(float) * WINDOW_STRIDE_SEC
    p = probs[left:right]
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.plot(t, p, linewidth=1.5, label="Seizure probability")
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label=f"Validation threshold={threshold:.3f}")
    ax.axvspan(seizure_start, seizure_end, alpha=0.15, label="Annotated seizure")
    ax.axvline(seizure_start, linestyle="--", linewidth=1.0, label="Onset")
    ax.set_xlabel("Recording time (s)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Representative primary held-out detection: {rid}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    _save(fig, out, "Figure9_detection_timeline")


def plot_operating_points(primary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].scatter(primary["threshold"], primary["event_sensitivity"], alpha=0.8)
    axes[0].set_xlabel("Validation-selected threshold")
    axes[0].set_ylabel("Held-out event sensitivity")
    axes[0].set_title("Threshold transfer across patients")
    axes[1].scatter(primary["min_consecutive_windows"], primary["fa_per_hour"], alpha=0.8)
    axes[1].set_xlabel("Persistence windows")
    axes[1].set_ylabel("Held-out FA/h")
    axes[1].set_xticks(sorted(primary["min_consecutive_windows"].dropna().astype(int).unique()))
    axes[1].set_title("Persistence and false-alarm rate")
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, out, "FigureS1_operating_point_transfer")


def plot_metric_distributions(primary: pd.DataFrame, out: Path) -> None:
    metrics = ["auroc", "auprc", "event_sensitivity", "event_f1", "fa_per_hour", "ece"]
    available = [metric for metric in metrics if metric in primary.columns]
    if not available:
        return
    data = [primary[metric].dropna().to_numpy(dtype=float) for metric in available]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.boxplot(data, tick_labels=available, showmeans=True)
    ax.set_ylabel("Metric value")
    ax.set_title("Distribution across primary held-out patients")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out, "FigureS2_metric_distributions")


def generate_publication_figures(
    summary_csv: Path,
    results_dir: Path,
    preprocessing_manifest: Path,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)

    primary = _primary_summary(summary_csv)
    predictions = _prediction_files(primary, results_dir)
    histories = _history_files(primary, results_dir)

    plot_preprocessing(preprocessing_manifest, out_dir)
    plot_architecture(out_dir)
    plot_training(histories, out_dir)
    plot_roc_pr(predictions, out_dir)
    plot_event_tradeoff(primary, out_dir)
    plot_sensitivity_forest(primary, out_dir)
    plot_patient_events(primary, out_dir)
    plot_calibration(predictions, out_dir)
    plot_detection_timeline(predictions, out_dir)
    plot_operating_points(primary, out_dir)
    plot_metric_distributions(primary, out_dir)

    return sorted(out_dir.glob("*.pdf"))
