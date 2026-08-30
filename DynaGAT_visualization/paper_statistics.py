from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch_geometric
from scipy.stats import chi2

from config import (
    ALARM_REFRACTORY_SEC,
    BANDPASS_HFREQ,
    BANDPASS_LFREQ,
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CACHE_VERSION,
    DEVELOPMENT_FOLD,
    DROPOUT,
    EPOCHS,
    EVENT_PERSISTENCE_CANDIDATES,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GAT_HEADS,
    GRAPH_HIDDEN,
    LEARNING_RATE,
    NODE_FEATURE_DIM,
    NUM_NODES,
    PAPER_RESULTS_DIR,
    PAPER_TABLES_DIR,
    PREPROCESSING_TAG,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    SFREQ,
    TCN_HIDDEN,
    TOP_K_DYNAMIC,
    VALIDATION_FA_PER_HOUR_CAP,
    WEIGHT_DECAY,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)


METRICS = [
    "auroc",
    "auprc",
    "f1",
    "event_sensitivity",
    "event_precision",
    "event_f1",
    "fa_per_hour",
    "median_latency_sec",
    "ece",
]


def _bootstrap_mean_ci(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(x, size=(BOOTSTRAP_REPLICATES, x.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _poisson_rate_interval(events: int, exposure: float, alpha: float = 0.05) -> tuple[float, float]:
    if exposure <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if events == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * events) / exposure
    upper = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (events + 1)) / exposure
    return float(lower), float(upper)


def _format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _write_latex(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(
        df.to_latex(index=index, escape=False, na_rep="--", float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )


def _environment_snapshot() -> dict:
    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "mne": mne.__version__,
        "matplotlib": matplotlib.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snapshot["gpu"] = props.name
        snapshot["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 3)
        snapshot["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return snapshot


def _configuration_snapshot() -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "preprocessing_tag": PREPROCESSING_TAG,
        "sampling_rate_hz": SFREQ,
        "window_sec": WINDOW_SEC,
        "window_stride_sec": WINDOW_STRIDE_SEC,
        "bandpass_hz": [BANDPASS_LFREQ, BANDPASS_HFREQ],
        "nodes": NUM_NODES,
        "node_features": NODE_FEATURE_DIM,
        "dynamic_top_k": TOP_K_DYNAMIC,
        "sequence_length": SEQUENCE_LENGTH,
        "graph_hidden": GRAPH_HIDDEN,
        "gat_heads": GAT_HEADS,
        "temporal_hidden": TCN_HIDDEN,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "validation_fa_per_hour_cap": VALIDATION_FA_PER_HOUR_CAP,
        "persistence_candidates": list(EVENT_PERSISTENCE_CANDIDATES),
        "alarm_refractory_sec": ALARM_REFRACTORY_SEC,
        "random_seed": RANDOM_SEED,
        "development_fold": DEVELOPMENT_FOLD,
    }


def generate_paper_statistics(
    summary_csv: Path,
    preprocessing_manifest: Path | None = None,
    results_dir: Path | None = None,
    tables_dir: Path = PAPER_TABLES_DIR,
    paper_results_dir: Path = PAPER_RESULTS_DIR,
) -> dict[str, Path]:
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv).sort_values("fold").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("LOPO summary is empty")

    df["evaluation_role"] = np.where(
        df["fold"].astype(int) == DEVELOPMENT_FOLD, "development", "primary"
    )
    primary = df[df["evaluation_role"] == "primary"].copy()
    if primary.empty:
        raise RuntimeError("No primary held-out folds are available")

    all_path = paper_results_dir / "all_per_patient_results.csv"
    primary_path = paper_results_dir / "primary_per_patient_results.csv"
    df.to_csv(all_path, index=False)
    primary.to_csv(primary_path, index=False)

    metric_rows = []
    for metric in METRICS:
        if metric not in primary.columns:
            continue
        values = primary[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = _bootstrap_mean_ci(finite)
        metric_rows.append(
            {
                "metric": metric,
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "median": float(np.median(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    metric_summary = pd.DataFrame(metric_rows)
    metric_summary_path = paper_results_dir / "primary_metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)

    detected = int(primary["detected_seizures"].sum())
    total_gt = int(primary["gt_seizures"].sum())
    false_alarms = int(primary["false_alarms"].sum())
    interictal_hours = float(primary["interictal_hours"].sum())
    recording_hours = float(primary["recording_hours"].sum())
    sensitivity = detected / total_gt if total_gt else float("nan")
    precision = detected / (detected + false_alarms) if detected + false_alarms else 0.0
    event_f1 = (
        2.0 * sensitivity * precision / (sensitivity + precision)
        if sensitivity + precision > 0
        else 0.0
    )
    pooled_far = false_alarms / interictal_hours if interictal_hours > 0 else float("nan")
    sens_low, sens_high = _wilson_interval(detected, total_gt)
    far_low, far_high = _poisson_rate_interval(false_alarms, interictal_hours)

    pooled = pd.DataFrame(
        [
            {
                "primary_folds": int(len(primary)),
                "detected_seizures": detected,
                "total_seizures": total_gt,
                "micro_event_sensitivity": sensitivity,
                "micro_event_sensitivity_ci95_low": sens_low,
                "micro_event_sensitivity_ci95_high": sens_high,
                "micro_event_precision": precision,
                "micro_event_f1": event_f1,
                "false_alarms": false_alarms,
                "interictal_hours": interictal_hours,
                "recording_hours": recording_hours,
                "pooled_fa_per_hour": pooled_far,
                "pooled_fa_per_hour_ci95_low": far_low,
                "pooled_fa_per_hour_ci95_high": far_high,
            }
        ]
    )
    pooled_path = paper_results_dir / "pooled_event_summary.csv"
    pooled.to_csv(pooled_path, index=False)

    table2 = metric_summary[["metric", "mean", "std", "median", "ci95_low", "ci95_high"]].copy()
    table2["mean_95ci"] = [
        _format_ci(row.mean, row.ci95_low, row.ci95_high)
        for row in table2.itertuples(index=False)
    ]
    table2_path = tables_dir / "Table2_primary_performance.csv"
    table2.to_csv(table2_path, index=False)
    _write_latex(table2[["metric", "mean_95ci", "std", "median"]], tables_dir / "Table2_primary_performance.tex")

    patient_columns = [
        "fold",
        "test_patient",
        "auroc",
        "auprc",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "fa_per_hour",
        "median_latency_sec",
        "ece",
        "gt_seizures",
        "detected_seizures",
        "false_alarms",
    ]
    patient_table = primary[[column for column in patient_columns if column in primary.columns]].copy()
    patient_table.to_csv(tables_dir / "TableS1_patient_performance.csv", index=False)
    _write_latex(patient_table, tables_dir / "TableS1_patient_performance.tex")

    if preprocessing_manifest is not None and Path(preprocessing_manifest).exists():
        manifest = pd.read_csv(preprocessing_manifest)
        dataset_row = {
            "subjects": int(len(manifest)),
            "edf_files": int(manifest["edf_files"].sum()),
            "valid_recordings": int(manifest["valid_recordings"].sum()),
            "recording_hours": float(manifest["recording_hours"].sum()),
            "seizures": int(manifest["seizures"].sum()),
            "windows": int(manifest["windows"].sum()),
            "positive_windows": int(manifest["positive_windows"].sum()),
            "positive_window_fraction": float(manifest["positive_windows"].sum() / manifest["windows"].sum()),
        }
        table1 = pd.DataFrame([dataset_row])
        table1.to_csv(tables_dir / "Table1_dataset_summary.csv", index=False)
        _write_latex(table1, tables_dir / "Table1_dataset_summary.tex")

    config_snapshot = _configuration_snapshot()
    environment_snapshot = _environment_snapshot()
    (paper_results_dir / "experiment_config.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )
    (paper_results_dir / "environment.json").write_text(
        json.dumps(environment_snapshot, indent=2), encoding="utf-8"
    )

    config_table = pd.DataFrame(
        [{"parameter": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}
         for key, value in config_snapshot.items()]
    )
    config_table.to_csv(tables_dir / "TableS2_experiment_configuration.csv", index=False)
    _write_latex(config_table, tables_dir / "TableS2_experiment_configuration.tex")

    metric_lookup = metric_summary.set_index("metric") if not metric_summary.empty else pd.DataFrame()
    def metric_text(name: str) -> str:
        if metric_lookup.empty or name not in metric_lookup.index:
            return "NA"
        row = metric_lookup.loc[name]
        return _format_ci(float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"]))

    narrative = (
        "# Primary held-out evaluation\n\n"
        f"Primary folds: {len(primary)} (development fold {DEVELOPMENT_FOLD} excluded).\n\n"
        f"Macro AUROC: {metric_text('auroc')}\n\n"
        f"Macro AUPRC: {metric_text('auprc')}\n\n"
        f"Macro event sensitivity: {metric_text('event_sensitivity')}\n\n"
        f"Macro FA/h: {metric_text('fa_per_hour')}\n\n"
        f"Macro median-latency statistic: {metric_text('median_latency_sec')} s\n\n"
        f"Micro event sensitivity: {sensitivity:.3f} [{sens_low:.3f}, {sens_high:.3f}] "
        f"({detected}/{total_gt} seizures).\n\n"
        f"Pooled false-alarm rate: {pooled_far:.3f} [{far_low:.3f}, {far_high:.3f}] FA/h "
        f"over {interictal_hours:.1f} interictal hours.\n"
    )
    narrative_path = paper_results_dir / "results_summary.md"
    narrative_path.write_text(narrative, encoding="utf-8")

    return {
        "all_results": all_path,
        "primary_results": primary_path,
        "metric_summary": metric_summary_path,
        "pooled_summary": pooled_path,
        "results_text": narrative_path,
    }
