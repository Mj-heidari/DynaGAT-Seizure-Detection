"""
Operating-point selection on validation patients.

an earlier iteration pooled all validation patients, chose the threshold that maximised pooled
sensitivity under a pooled false-alarm cap, and transferred it unchanged. A
pooled cap is dominated by whichever validation patient contributes the most
interictal hours, so the selected point satisfied the cap "on average" while
being far too permissive for a typical patient - which is exactly how a
0.44 FA/h validation point became 0.56 FA/h with 36% sensitivity on test.

the current pipeline selects on the *per-patient* distribution: a candidate is admissible only if
the median validation patient meets the cap and the mean stays within a
tolerance factor of it. Among admissible candidates we maximise the mean
per-patient sensitivity. Selecting on the patient distribution rather than the
pooled totals is what makes the point transfer to an unseen patient.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    ALARM_REFRACTORY_SEC,
    EVENT_THRESHOLD_MAX_CANDIDATES,
    PERSISTENCE_K_OF_M,
    VALIDATION_FA_PER_HOUR_CAP,
)
from evaluation.events import evaluate_events

__all__ = ["OperatingPoint", "select_operating_point"]


@dataclass
class OperatingPoint:
    threshold: float
    k: int
    m: int
    val_mean_sensitivity: float
    val_median_fa_per_hour: float
    val_mean_fa_per_hour: float
    val_pooled_sensitivity: float
    val_pooled_fa_per_hour: float
    admissible: bool
    fallback_used: str = ""

    def as_dict(self) -> Dict:
        return {f"op_{k}": v for k, v in asdict(self).items()}


def _threshold_grid(scores: np.ndarray, n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.array([0.0])
    # Dense in the upper tail, where the decision actually lives.
    qs = np.concatenate(
        [
            np.linspace(0.50, 0.95, max(4, n // 6)),
            np.linspace(0.95, 0.9995, n - max(4, n // 6)),
        ]
    )
    grid = np.unique(np.quantile(s, np.clip(qs, 0.0, 1.0)))
    return grid


def select_operating_point(
    validation: Dict[str, Dict[str, Dict]],
    fa_cap: float = VALIDATION_FA_PER_HOUR_CAP,
    candidates: Sequence[Tuple[int, int]] = PERSISTENCE_K_OF_M,
    n_thresholds: int = EVENT_THRESHOLD_MAX_CANDIDATES,
    mean_tolerance: float = 1.5,
    refractory_sec: float = ALARM_REFRACTORY_SEC,
) -> Tuple[OperatingPoint, pd.DataFrame]:
    """
    Parameters
    ----------
    validation : {patient_id: {recording_id: {'score', 'duration_sec',
                  'seizure_intervals'}}}

    Returns
    -------
    (best operating point, full frontier as a DataFrame)
    """
    pooled_scores = np.concatenate(
        [
            np.asarray(rec["score"], dtype=np.float64).ravel()
            for pat in validation.values()
            for rec in pat.values()
        ]
    ) if validation else np.array([0.0])
    grid = _threshold_grid(pooled_scores, n_thresholds)

    rows: List[Dict] = []
    for k, m in candidates:
        for thr in grid:
            per_pat_sens, per_pat_fa = [], []
            pooled_det = pooled_gt = pooled_fa = 0
            pooled_inter = 0.0
            for patient, recs in validation.items():
                res = evaluate_events(recs, float(thr), k, m, refractory_sec=refractory_sec)
                if res.gt_seizures:
                    per_pat_sens.append(res.sensitivity)
                if res.interictal_hours > 0:
                    per_pat_fa.append(res.fa_per_hour)
                pooled_det += res.detected_seizures
                pooled_gt += res.gt_seizures
                pooled_fa += res.false_alarms
                pooled_inter += res.interictal_hours
            if not per_pat_fa:
                continue
            med_fa = float(np.median(per_pat_fa))
            mean_fa = float(np.mean(per_pat_fa))
            mean_sens = float(np.mean(per_pat_sens)) if per_pat_sens else float("nan")
            rows.append(
                {
                    "threshold": float(thr),
                    "k": int(k),
                    "m": int(m),
                    "mean_sensitivity": mean_sens,
                    "median_fa_per_hour": med_fa,
                    "mean_fa_per_hour": mean_fa,
                    "max_fa_per_hour": float(np.max(per_pat_fa)),
                    "pooled_sensitivity": pooled_det / pooled_gt if pooled_gt else float("nan"),
                    "pooled_fa_per_hour": pooled_fa / pooled_inter if pooled_inter > 0 else float("nan"),
                    "admissible": bool(med_fa <= fa_cap and mean_fa <= mean_tolerance * fa_cap),
                }
            )

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        return (
            OperatingPoint(0.0, 3, 4, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), False, "empty_frontier"),
            frontier,
        )

    ok = frontier[frontier["admissible"] & frontier["mean_sensitivity"].notna()]
    fallback = ""
    if ok.empty:
        # No candidate meets the cap: take the lowest achievable false-alarm rate
        # that still detects something, and record that the cap was not met.
        fallback = "fa_cap_unreachable"
        ok = frontier[frontier["mean_sensitivity"].fillna(0) > 0]
        if ok.empty:
            ok = frontier
            fallback = "no_detection_on_validation"
        ok = ok.sort_values(["median_fa_per_hour", "mean_sensitivity"], ascending=[True, False])
    else:
        ok = ok.sort_values(
            ["mean_sensitivity", "median_fa_per_hour"], ascending=[False, True]
        )

    best = ok.iloc[0]
    return (
        OperatingPoint(
            threshold=float(best["threshold"]),
            k=int(best["k"]),
            m=int(best["m"]),
            val_mean_sensitivity=float(best["mean_sensitivity"]),
            val_median_fa_per_hour=float(best["median_fa_per_hour"]),
            val_mean_fa_per_hour=float(best["mean_fa_per_hour"]),
            val_pooled_sensitivity=float(best["pooled_sensitivity"]),
            val_pooled_fa_per_hour=float(best["pooled_fa_per_hour"]),
            admissible=bool(best["admissible"]),
            fallback_used=fallback,
        ),
        frontier,
    )
