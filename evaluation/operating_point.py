from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from config import (
    EVENT_PERSISTENCE_CANDIDATES,
    EVENT_THRESHOLD_MAX_CANDIDATES,
    VALIDATION_FA_PER_HOUR_CAP,
)
from evaluation.metrics import compute_event_metrics, select_f1_threshold


@dataclass(frozen=True)
class AlarmOperatingPoint:
    threshold: float
    min_consecutive_windows: int
    validation_metrics: Dict[str, float]
    far_cap: float
    feasible_under_cap: bool


def _threshold_candidates(
    labels: np.ndarray,
    probs: np.ndarray,
    max_candidates: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    finite = probs[np.isfinite(probs)]
    fallback = select_f1_threshold(labels, probs)
    if finite.size == 0:
        return np.asarray([float(fallback), 0.5], dtype=np.float64)

    # Dense coverage near the empirical high-probability tail plus a global grid.
    quantiles = np.quantile(finite, np.linspace(0.35, 0.9995, 61))
    linear = np.linspace(0.01, 0.99, 67)
    no_alarm_threshold = np.nextafter(float(np.max(finite)), np.inf)
    candidates = np.unique(
        np.concatenate(
            [
                np.clip(
                    np.concatenate([quantiles, linear, [fallback, 0.5]]),
                    1e-4,
                    1.0 - 1e-4,
                ),
                [no_alarm_threshold],
            ]
        )
    )

    if candidates.size > max_candidates:
        keep = np.linspace(0, candidates.size - 1, max_candidates, dtype=int)
        candidates = np.unique(
            np.concatenate([candidates[keep], [fallback, 0.5, no_alarm_threshold]])
        )
    return candidates.astype(np.float64, copy=False)


def select_validation_operating_point(
    labels: np.ndarray,
    probs: np.ndarray,
    recording_ids: Sequence[str],
    window_indices: np.ndarray,
    recording_metadata: Mapping[str, Mapping],
    far_cap: float = VALIDATION_FA_PER_HOUR_CAP,
    persistence_candidates: Sequence[int] = EVENT_PERSISTENCE_CANDIDATES,
    max_threshold_candidates: int = EVENT_THRESHOLD_MAX_CANDIDATES,
    frontier_path: Path | None = None,
) -> AlarmOperatingPoint:
    """Select threshold and persistence using validation patients only.

    The operating rule is intentionally specified before held-out testing:
    among points satisfying the validation false-alarm-rate cap, maximize event
    sensitivity; break ties by event F1, precision, lower FAR, then the more
    conservative persistence/threshold. If no point satisfies the cap, choose
    the lowest-FAR point and then maximize sensitivity/F1.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    window_indices = np.asarray(window_indices, dtype=np.int64)

    persistences = sorted({int(v) for v in persistence_candidates if int(v) >= 1})
    if not persistences:
        raise ValueError("persistence_candidates must contain at least one integer >= 1")
    if far_cap < 0:
        raise ValueError("far_cap must be >= 0")

    total_gt = sum(
        len(meta.get("seizure_intervals", []))
        for meta in recording_metadata.values()
    )
    candidates = _threshold_candidates(labels, probs, max_threshold_candidates)

    rows = []
    for persistence in persistences:
        for threshold in candidates:
            metrics = compute_event_metrics(
                probs=probs,
                recording_ids=recording_ids,
                window_indices=window_indices,
                recording_metadata=recording_metadata,
                threshold=float(threshold),
                min_consecutive_windows=int(persistence),
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "min_consecutive_windows": int(persistence),
                    **metrics,
                    "within_far_cap": bool(metrics["fa_per_hour"] <= far_cap + 1e-12),
                }
            )

    frontier = pd.DataFrame(rows)
    if frontier_path is not None:
        frontier_path = Path(frontier_path)
        frontier_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = frontier_path.with_suffix(frontier_path.suffix + ".tmp")
        frontier.to_csv(temporary, index=False)
        temporary.replace(frontier_path)

    # Degenerate validation set fallback: keep the window-F1 threshold and the
    # middle persistence candidate rather than using held-out information.
    if total_gt <= 0 or frontier.empty:
        fallback = float(select_f1_threshold(labels, probs))
        persistence = int(persistences[min(1, len(persistences) - 1)])
        metrics = compute_event_metrics(
            probs=probs,
            recording_ids=recording_ids,
            window_indices=window_indices,
            recording_metadata=recording_metadata,
            threshold=fallback,
            min_consecutive_windows=persistence,
        )
        return AlarmOperatingPoint(
            threshold=fallback,
            min_consecutive_windows=persistence,
            validation_metrics=metrics,
            far_cap=float(far_cap),
            feasible_under_cap=bool(metrics["fa_per_hour"] <= far_cap),
        )

    valid = frontier[
        np.isfinite(frontier["event_sensitivity"].to_numpy(dtype=float))
        & np.isfinite(frontier["event_f1"].to_numpy(dtype=float))
        & np.isfinite(frontier["fa_per_hour"].to_numpy(dtype=float))
    ].copy()
    if valid.empty:
        raise RuntimeError("Validation operating-point search produced no finite candidates")

    feasible = valid[valid["within_far_cap"]]
    if not feasible.empty:
        # Fixed FAR budget -> sensitivity is the primary clinical objective.
        ranked = feasible.sort_values(
            by=[
                "event_sensitivity",
                "event_f1",
                "event_precision",
                "fa_per_hour",
                "min_consecutive_windows",
                "threshold",
            ],
            ascending=[False, False, False, True, False, False],
            kind="mergesort",
        )
        feasible_under_cap = True
    else:
        # If the requested budget is unattainable, minimize FAR without peeking at test.
        ranked = valid.sort_values(
            by=[
                "fa_per_hour",
                "event_sensitivity",
                "event_f1",
                "event_precision",
                "min_consecutive_windows",
                "threshold",
            ],
            ascending=[True, False, False, False, False, False],
            kind="mergesort",
        )
        feasible_under_cap = False

    best = ranked.iloc[0]
    metric_keys = [
        "total_gt_seizures",
        "detected_seizures",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "false_alarms",
        "recording_hours",
        "interictal_hours",
        "fa_per_hour",
        "median_latency_sec",
    ]
    metrics = {key: float(best[key]) for key in metric_keys}
    metrics["total_gt_seizures"] = int(best["total_gt_seizures"])
    metrics["detected_seizures"] = int(best["detected_seizures"])
    metrics["false_alarms"] = int(best["false_alarms"])

    return AlarmOperatingPoint(
        threshold=float(best["threshold"]),
        min_consecutive_windows=int(best["min_consecutive_windows"]),
        validation_metrics=metrics,
        far_cap=float(far_cap),
        feasible_under_cap=feasible_under_cap,
    )
