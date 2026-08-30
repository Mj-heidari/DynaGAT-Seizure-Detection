from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, VALIDATION_FA_PER_HOUR_CAP
from evaluation.metrics import compute_event_metrics, compute_window_metrics, compute_ece


def _load_prediction_bundle(path: Path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["recording_metadata_json"]))
    return {
        "probs": data["probs"].astype(np.float64, copy=False),
        "labels": data["labels"].astype(np.int64, copy=False),
        "recording_ids": data["recording_ids"].astype(str).tolist(),
        "window_indices": data["window_indices"].astype(np.int64, copy=False),
        "recording_metadata": metadata,
    }


def _select(frontier: pd.DataFrame, objective: str, far_cap: float) -> pd.Series:
    valid = frontier.copy()
    valid = valid[
        np.isfinite(valid["event_sensitivity"].to_numpy(float))
        & np.isfinite(valid["event_f1"].to_numpy(float))
        & np.isfinite(valid["fa_per_hour"].to_numpy(float))
    ]
    feasible = valid[valid["fa_per_hour"] <= far_cap + 1e-12]
    pool = feasible if not feasible.empty else valid

    if objective == "sensitivity":
        return pool.sort_values(
            by=["event_sensitivity", "event_f1", "event_precision", "fa_per_hour", "min_consecutive_windows", "threshold"],
            ascending=[False, False, False, True, False, False],
            kind="mergesort",
        ).iloc[0]
    if objective == "event_f1":
        return pool.sort_values(
            by=["event_f1", "event_sensitivity", "event_precision", "fa_per_hour", "min_consecutive_windows", "threshold"],
            ascending=[False, False, False, True, False, False],
            kind="mergesort",
        ).iloc[0]
    raise ValueError(objective)


def _evaluate(name: str, row: pd.Series, pred: dict) -> dict:
    threshold = float(row["threshold"])
    persistence = int(row["min_consecutive_windows"])
    window = compute_window_metrics(pred["labels"], pred["probs"], threshold)
    event = compute_event_metrics(
        probs=pred["probs"],
        recording_ids=pred["recording_ids"],
        window_indices=pred["window_indices"],
        recording_metadata=pred["recording_metadata"],
        threshold=threshold,
        min_consecutive_windows=persistence,
    )
    return {
        "objective": name,
        "threshold": threshold,
        "min_consecutive_windows": persistence,
        "val_event_sensitivity": float(row["event_sensitivity"]),
        "val_event_precision": float(row["event_precision"]),
        "val_event_f1": float(row["event_f1"]),
        "val_fa_per_hour": float(row["fa_per_hour"]),
        "test_auroc": window["auroc"],
        "test_auprc": window["auprc"],
        "test_f1": window["f1"],
        "test_event_sensitivity": event["event_sensitivity"],
        "test_event_precision": event["event_precision"],
        "test_event_f1": event["event_f1"],
        "test_false_alarms": event["false_alarms"],
        "test_fa_per_hour": event["fa_per_hour"],
        "test_median_latency_sec": event["median_latency_sec"],
        "test_ece": compute_ece(pred["probs"], pred["labels"], n_bins=10),
    }


def main(frontier_path: Path, prediction_path: Path, far_cap: float) -> None:
    frontier = pd.read_csv(frontier_path)
    pred = _load_prediction_bundle(prediction_path)

    rows = []
    for objective in ("sensitivity", "event_f1"):
        chosen = _select(frontier, objective, far_cap)
        rows.append(_evaluate(objective, chosen, pred))

    out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("V5 DEVELOPMENT-FOLD ALARM OBJECTIVE COMPARISON")
    print("=" * 100)
    print(f"Validation FA/h cap: {far_cap:.3f}")
    print(out.to_string(index=False))
    out_path = RESULTS_DIR / "v5_fold_01_alarm_objective_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[+] Comparison: {out_path}")
    print("[*] This script is for development-fold analysis only. Do not tune again on later held-out folds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare sensitivity-first vs event-F1-first validation alarm selection for the frozen v5 development fold"
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=RESULTS_DIR / "fold_01_validation_alarm_frontier.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS_DIR / "fold_01_test_predictions.npz",
    )
    parser.add_argument("--far-cap", type=float, default=VALIDATION_FA_PER_HOUR_CAP)
    args = parser.parse_args()
    if not args.frontier.exists():
        parser.error(f"frontier not found: {args.frontier}")
    if not args.predictions.exists():
        parser.error(f"predictions not found: {args.predictions}")
    if args.far_cap < 0:
        parser.error("--far-cap must be >= 0")
    main(args.frontier, args.predictions, args.far_cap)
