from __future__ import annotations

import json
import tempfile
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

from config import DEVELOPMENT_FOLD, WINDOW_SEC, WINDOW_STRIDE_SEC
from evaluation.metrics import compute_ece, window_decision_times


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
    width = 0.42
    axes[0].bar(x - width / 2, df["gt_seizures"], width=width, label="Ground-truth seizures")
    axes[0].bar(x + width / 2, df["detected_seizures"], width=width, label="Detected seizures")
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
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed, weights = [], [], []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == 0 else (probs > low) & (probs <= high)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(probs[mask])))
        observed.append(float(np.mean(labels[mask])))
        weights.append(float(np.mean(mask)))
    ece = compute_ece(probs, labels, n_bins=n_bins)
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
    candidates = []
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
                decision_times = window_decision_times(idx)
                pos = int(np.searchsorted(decision_times, float(seizure_start)))
                left, right = max(0, pos - 30), min(len(p), pos + 90)
                if right <= left:
                    continue
                score = float(np.nanmax(p[left:right]))
                candidates.append(
                    (
                        score,
                        str(rid),
                        idx,
                        p,
                        threshold,
                        float(seizure_start),
                        float(seizure_end),
                        pos,
                    )
                )
    if not candidates:
        return
    # Select the median-response annotated seizure rather than cherry-picking
    # the strongest detection as the "representative" example.
    candidates.sort(key=lambda item: (item[0], item[1], item[5]))
    _, rid, idx, probs, threshold, seizure_start, seizure_end, pos = candidates[
        len(candidates) // 2
    ]
    left, right = max(0, pos - 90), min(len(idx), pos + 150)
    t = window_decision_times(idx[left:right])
    p = probs[left:right]
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.plot(t, p, linewidth=1.5, label="Seizure probability")
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label=f"Validation threshold={threshold:.3f}")
    ax.axvspan(seizure_start, seizure_end, alpha=0.15, label="Annotated seizure")
    ax.axvline(seizure_start, linestyle="--", linewidth=1.0, label="Onset")
    ax.set_xlabel("Online decision time (s; window end)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Median-response primary held-out seizure: {rid}")
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
    performance = ["auroc", "auprc", "event_sensitivity", "event_f1", "ece"]
    operational = ["fa_per_hour", "median_latency_sec"]
    perf_available = [metric for metric in performance if metric in primary.columns]
    op_available = [metric for metric in operational if metric in primary.columns]
    if not perf_available and not op_available:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    if perf_available:
        data = [primary[metric].dropna().to_numpy(dtype=float) for metric in perf_available]
        axes[0].boxplot(data, tick_labels=perf_available, showmeans=True)
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("Metric value")
        axes[0].set_title("Discrimination, detection, and calibration")
    else:
        axes[0].axis("off")
    if op_available:
        data = [primary[metric].dropna().to_numpy(dtype=float) for metric in op_available]
        axes[1].boxplot(data, tick_labels=op_available, showmeans=True)
        axes[1].set_ylabel("Native metric units")
        axes[1].set_title("Operational outcomes")
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    _save(fig, out, "FigureS2_metric_distributions")


def plot_patient_metric_heatmap(primary: pd.DataFrame, out: Path) -> None:
    """Annotated patient-by-metric view of bounded performance measures."""
    columns = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("f1", "Window F1"),
        ("event_sensitivity", "Event sens."),
        ("event_precision", "Event prec."),
        ("event_f1", "Event F1"),
    ]
    columns = [(key, label) for key, label in columns if key in primary.columns]
    if not columns:
        return
    df = primary.sort_values("fold").reset_index(drop=True)
    values = df[[key for key, _ in columns]].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(columns)), max(5.5, 0.36 * len(df) + 1.8)))
    image = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(columns)), [label for _, label in columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(df)), df["test_patient"].astype(str).tolist())
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                color = "white" if value < 0.45 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, label="Metric value")
    ax.set_xlabel("Held-out performance metric")
    ax.set_ylabel("Held-out patient")
    ax.set_title("Primary patient-level performance heatmap")
    _save(fig, out, "FigureS3_patient_metric_heatmap")


def plot_window_confusion_heatmap(prediction_files: Sequence[Path], out: Path) -> None:
    """Pooled confusion matrix using each fold's validation-selected threshold."""
    counts = np.zeros((2, 2), dtype=np.int64)
    for path in prediction_files:
        data = _load_prediction(path)
        labels = np.asarray(data["labels"], dtype=np.int64)
        probs = np.asarray(data["probs"], dtype=float)
        if labels.size == 0:
            continue
        threshold = float(np.asarray(data["threshold"]).item())
        predicted = (probs >= threshold).astype(np.int64)
        for truth in (0, 1):
            for pred in (0, 1):
                counts[truth, pred] += int(np.sum((labels == truth) & (predicted == pred)))
    if counts.sum() == 0:
        return
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.maximum(row_totals, 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks([0, 1], ["Non-seizure", "Seizure"])
    ax.set_yticks([0, 1], ["Non-seizure", "Seizure"])
    ax.set_xlabel("Predicted window class")
    ax.set_ylabel("Ground-truth window class")
    ax.set_title("Pooled primary window-level confusion matrix")
    for truth in (0, 1):
        for pred in (0, 1):
            color = "white" if normalized[truth, pred] > 0.5 else "black"
            ax.text(
                pred,
                truth,
                f"{counts[truth, pred]:,}\n{normalized[truth, pred]:.1%}",
                ha="center",
                va="center",
                color=color,
            )
    fig.colorbar(image, ax=ax, label="Within-class proportion")
    _save(fig, out, "FigureS4_window_confusion_heatmap")


def plot_validation_test_transfer_heatmap(primary: pd.DataFrame, out: Path) -> None:
    required = {
        "test_patient",
        "val_event_sensitivity",
        "event_sensitivity",
        "val_fa_per_hour",
        "fa_per_hour",
    }
    if not required.issubset(primary.columns):
        return
    df = primary.sort_values("fold").reset_index(drop=True)
    sensitivity = df[["val_event_sensitivity", "event_sensitivity"]].to_numpy(dtype=float)
    far = df[["val_fa_per_hour", "fa_per_hour"]].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, max(5.5, 0.36 * len(df) + 1.8)), sharey=True)
    panels = [
        (axes[0], sensitivity, "Event sensitivity", "viridis", 0.0, 1.0),
        (axes[1], far, "False alarms per hour", "magma_r", 0.0, None),
    ]
    for ax, values, title, cmap, low, high in panels:
        finite = values[np.isfinite(values)]
        vmax = high if high is not None else (float(np.quantile(finite, 0.95)) if finite.size else 1.0)
        vmax = max(vmax, 1e-6)
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=low, vmax=vmax)
        ax.set_xticks([0, 1], ["Validation", "Held-out test"])
        ax.set_title(title)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = "black" if luminance > 0.55 else "white"
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_yticks(np.arange(len(df)), df["test_patient"].astype(str).tolist())
    axes[0].set_ylabel("Held-out patient")
    fig.suptitle("Validation-to-test operating-point transfer")
    _save(fig, out, "FigureS5_validation_test_transfer_heatmap")


def generate_publication_figures(
    summary_csv: Path,
    results_dir: Path,
    preprocessing_manifest: Path,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = _primary_summary(summary_csv)
    predictions = _prediction_files(primary, results_dir)
    histories = _history_files(primary, results_dir)

    # Generate into a staging directory so a failed export never destroys the
    # last complete set of publication figures.
    with tempfile.TemporaryDirectory(prefix="figure-staging-", dir=out_dir) as tmp:
        staging = Path(tmp)
        plot_preprocessing(preprocessing_manifest, staging)
        plot_architecture(staging)
        plot_training(histories, staging)
        plot_roc_pr(predictions, staging)
        plot_event_tradeoff(primary, staging)
        plot_sensitivity_forest(primary, staging)
        plot_patient_events(primary, staging)
        plot_calibration(predictions, staging)
        plot_detection_timeline(predictions, staging)
        plot_operating_points(primary, staging)
        plot_metric_distributions(primary, staging)
        plot_patient_metric_heatmap(primary, staging)
        plot_window_confusion_heatmap(predictions, staging)
        plot_validation_test_transfer_heatmap(primary, staging)

        generated = sorted(staging.glob("*.png")) + sorted(staging.glob("*.pdf"))
        if not generated:
            raise RuntimeError("Publication figure generation produced no files")
        for pattern in ("*.png", "*.pdf"):
            for path in out_dir.glob(pattern):
                path.unlink(missing_ok=True)
        for path in generated:
            path.replace(out_dir / path.name)

    return sorted(out_dir.glob("*.pdf"))
