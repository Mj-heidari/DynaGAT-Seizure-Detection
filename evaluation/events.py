"""
Event-level and window-level evaluation for an online seizure detector.

Timing convention
-----------------
Probabilities are timestamped at the END of their analysis window, which is the
first instant an online system has observed every sample used for that
decision. Window t therefore carries timestamp ``t * stride + window_sec``.
The primary protocol allows no pre-onset tolerance: an alarm raised before the
annotated onset is a false alarm. A secondary column with a 10 s pre-onset
tolerance is reported for comparability with papers that use one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import (
    ALARM_REFRACTORY_SEC,
    EVENT_EARLY_TOLERANCE_SEC,
    EVENT_LATE_TOLERANCE_SEC,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)

__all__ = [
    "window_end_times",
    "generate_alarms",
    "match_events",
    "EventResult",
    "evaluate_events",
    "window_metrics",
    "expected_calibration_error",
]


def window_end_times(n_windows: int) -> np.ndarray:
    return np.arange(n_windows, dtype=np.float64) * WINDOW_STRIDE_SEC + WINDOW_SEC


def generate_alarms(
    scores: np.ndarray,
    threshold: float,
    k: int,
    m: int,
    refractory_sec: float = ALARM_REFRACTORY_SEC,
) -> np.ndarray:
    """
    Causal k-of-m persistence with a refractory period.

    An alarm fires at window t when at least ``k`` of the ``m`` windows ending
    at t exceed ``threshold``. After an alarm, further alarms are suppressed for
    ``refractory_sec``. Returns the indices of alarm windows.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    above = (s >= float(threshold)).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(above)])
    starts = np.maximum(0, np.arange(n) - int(m) + 1)
    counts = csum[np.arange(n) + 1] - csum[starts]
    fire = counts >= int(k)

    refractory_windows = int(round(float(refractory_sec) / WINDOW_STRIDE_SEC))
    alarms: List[int] = []
    blocked_until = -1
    for t in np.nonzero(fire)[0]:
        if t <= blocked_until:
            continue
        alarms.append(int(t))
        blocked_until = int(t) + refractory_windows
    return np.asarray(alarms, dtype=np.int64)


def match_events(
    alarm_times: np.ndarray,
    seizures: Sequence[Tuple[float, float]],
    early_tolerance: float = EVENT_EARLY_TOLERANCE_SEC,
    late_tolerance: float = EVENT_LATE_TOLERANCE_SEC,
) -> Tuple[int, int, List[float]]:
    """
    Returns (detected_seizures, false_alarms, latencies).

    An alarm inside [onset - early, offset + late] of some seizure is *in-event*:
    the first such alarm detects that seizure, later ones are neither true nor
    false. Alarms outside every seizure interval are false alarms.
    """
    times = np.asarray(alarm_times, dtype=np.float64).ravel()
    detected = 0
    latencies: List[float] = []
    consumed = np.zeros(times.shape[0], dtype=bool)
    in_event = np.zeros(times.shape[0], dtype=bool)

    for onset, offset in seizures:
        lo = onset - float(early_tolerance)
        hi = offset + float(late_tolerance)
        inside = (times >= lo) & (times <= hi)
        in_event |= inside
        candidates = np.nonzero(inside & ~consumed)[0]
        if candidates.size:
            first = int(candidates[0])
            consumed[first] = True
            detected += 1
            latencies.append(float(times[first] - onset))
    false_alarms = int(np.sum(~in_event))
    return detected, false_alarms, latencies


@dataclass
class EventResult:
    gt_seizures: int = 0
    detected_seizures: int = 0
    false_alarms: int = 0
    total_alarms: int = 0
    recording_hours: float = 0.0
    interictal_hours: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def sensitivity(self) -> float:
        return self.detected_seizures / self.gt_seizures if self.gt_seizures else float("nan")

    @property
    def fa_per_hour(self) -> float:
        return self.false_alarms / self.interictal_hours if self.interictal_hours > 0 else float("nan")

    @property
    def precision(self) -> float:
        return self.detected_seizures / self.total_alarms if self.total_alarms else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.sensitivity
        if not np.isfinite(r) or (p + r) == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def median_latency(self) -> float:
        return float(np.median(self.latencies)) if self.latencies else float("nan")

    def as_dict(self, prefix: str = "") -> Dict:
        return {
            f"{prefix}gt_seizures": self.gt_seizures,
            f"{prefix}detected_seizures": self.detected_seizures,
            f"{prefix}event_sensitivity": self.sensitivity,
            f"{prefix}event_precision": self.precision,
            f"{prefix}event_f1": self.f1,
            f"{prefix}false_alarms": self.false_alarms,
            f"{prefix}total_alarms": self.total_alarms,
            f"{prefix}recording_hours": self.recording_hours,
            f"{prefix}interictal_hours": self.interictal_hours,
            f"{prefix}fa_per_hour": self.fa_per_hour,
            f"{prefix}median_latency_sec": self.median_latency,
            f"{prefix}mean_latency_sec": float(np.mean(self.latencies)) if self.latencies else float("nan"),
        }


def evaluate_events(
    per_recording: Dict[str, Dict],
    threshold: float,
    k: int,
    m: int,
    early_tolerance: float = EVENT_EARLY_TOLERANCE_SEC,
    late_tolerance: float = EVENT_LATE_TOLERANCE_SEC,
    refractory_sec: float = ALARM_REFRACTORY_SEC,
) -> EventResult:
    """
    per_recording maps recording_id -> {
        'score': np.ndarray [T], 'duration_sec': float,
        'seizure_intervals': list[(onset, offset)]
    }
    """
    res = EventResult()
    for rec in per_recording.values():
        scores = np.asarray(rec["score"], dtype=np.float64)
        n = scores.shape[0]
        if n == 0:
            continue
        seizures = [tuple(map(float, s)) for s in rec.get("seizure_intervals", [])]
        alarms = generate_alarms(scores, threshold, k, m, refractory_sec)
        times = window_end_times(n)[alarms] if alarms.size else np.zeros(0)
        det, fa, lat = match_events(times, seizures, early_tolerance, late_tolerance)

        duration = float(rec["duration_sec"])
        blocked = sum(
            min(duration, off + late_tolerance) - max(0.0, on - early_tolerance)
            for on, off in seizures
        )
        res.gt_seizures += len(seizures)
        res.detected_seizures += det
        res.false_alarms += fa
        res.total_alarms += int(alarms.size)
        res.recording_hours += duration / 3600.0
        res.interictal_hours += max(0.0, duration - blocked) / 3600.0
        res.latencies.extend(lat)
    return res


# --------------------------------------------------------------------------- #
# Window-level metrics
# --------------------------------------------------------------------------- #
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < sorted_s.shape[0]:
        j = i
        while j + 1 < sorted_s.shape[0] and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_of = np.empty_like(ranks)
    rank_of[order] = ranks
    return float((rank_of[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, tie-aware (matches sklearn.average_precision_score)."""
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order].astype(np.float64)
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    # Evaluate only at distinct score thresholds so tied samples share a point.
    distinct = np.nonzero(np.diff(ss))[0]
    idx = np.concatenate([distinct, [ss.shape[0] - 1]])
    tp, fp = tp[idx], fp[idx]
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * d_recall))


def window_metrics(labels: np.ndarray, scores: np.ndarray, probs: np.ndarray | None = None) -> Dict:
    y = np.asarray(labels, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    out = {
        "auroc": _auroc(y, s),
        "auprc": _auprc(y, s),
        "positive_fraction": float(y.mean()) if y.size else float("nan"),
        "n_windows": int(y.size),
    }
    if probs is not None:
        p = np.asarray(probs, dtype=np.float64).ravel()
        out["ece"] = expected_calibration_error(y, p)
        out["brier"] = float(np.mean((p - y) ** 2))
    return out


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(labels, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += (sel.mean()) * abs(p[sel].mean() - y[sel].mean())
    return float(ece)
