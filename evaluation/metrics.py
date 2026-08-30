from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from config import (
    ALARM_REFRACTORY_SEC,
    EVENT_THRESHOLD_MAX_CANDIDATES,
    MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)


def compute_window_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    binary = (probs >= threshold).astype(np.int64)

    if np.unique(labels).size >= 2:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(labels, binary, zero_division=0)),
    }


def select_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Fallback window-level threshold used when event validation is impossible."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5

    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.size == 0:
        return float("nan")

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = boundaries[i], boundaries[i + 1]
        if i == 0:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs > low) & (probs <= high)
        if not np.any(in_bin):
            continue
        fraction = float(np.mean(in_bin))
        accuracy = float(np.mean(labels[in_bin]))
        confidence = float(np.mean(probs[in_bin]))
        ece += fraction * abs(accuracy - confidence)
    return float(ece)


def _predicted_events(
    probs: np.ndarray,
    threshold: float,
    min_consecutive: int,
    refractory_windows: int,
) -> List[Tuple[int, int, int]]:
    """
    Return (run_start, run_end, alarm_index).

    alarm_index is the first time an alarm can actually be emitted, i.e. after
    `min_consecutive` positive windows have been observed. Nearby events are then
    merged according to the explicit refractory policy.
    """
    binary = (np.asarray(probs) >= threshold).astype(np.uint8)
    raw_events: List[Tuple[int, int, int]] = []
    run_start = None
    run_length = 0

    for idx, value in enumerate(binary):
        if value:
            if run_start is None:
                run_start = idx
            run_length += 1
        else:
            if run_start is not None and run_length >= min_consecutive:
                alarm_idx = run_start + min_consecutive - 1
                raw_events.append((run_start, idx - 1, alarm_idx))
            run_start = None
            run_length = 0

    if run_start is not None and run_length >= min_consecutive:
        alarm_idx = run_start + min_consecutive - 1
        raw_events.append((run_start, len(binary) - 1, alarm_idx))

    if not raw_events or refractory_windows <= 0:
        return raw_events

    merged: List[Tuple[int, int, int]] = [raw_events[0]]
    for start, end, alarm_idx in raw_events[1:]:
        prev_start, prev_end, prev_alarm = merged[-1]
        if start <= prev_end + refractory_windows:
            merged[-1] = (prev_start, max(prev_end, end), prev_alarm)
        else:
            merged.append((start, end, alarm_idx))
    return merged


def compute_event_metrics(
    probs: np.ndarray,
    recording_ids: Sequence[str],
    window_indices: np.ndarray,
    recording_metadata: Mapping[str, Mapping],
    threshold: float,
    min_consecutive_windows: int = MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    refractory_sec: float = ALARM_REFRACTORY_SEC,
    early_tolerance_sec: float = WINDOW_SEC,
    stride_sec: float = WINDOW_STRIDE_SEC,
    window_sec: float = WINDOW_SEC,
) -> Dict[str, float]:
    """Compute clinically relevant event metrics independently for every EDF."""
    probs = np.asarray(probs, dtype=np.float64)
    window_indices = np.asarray(window_indices, dtype=np.int64)
    recording_ids = np.asarray(recording_ids, dtype=object)

    total_gt = 0
    detected_gt = 0
    false_alarms = 0
    latencies: List[float] = []
    total_duration_sec = 0.0
    refractory_windows = max(0, int(round(refractory_sec / max(stride_sec, 1e-12))))

    for rid, meta in recording_metadata.items():
        total_duration_sec += float(meta.get("duration_sec", 0.0))
        seizures = [tuple(x) for x in meta.get("seizure_intervals", [])]
        total_gt += len(seizures)

        mask = recording_ids == rid
        if not np.any(mask):
            continue

        idx = window_indices[mask]
        rec_probs = probs[mask]
        order = np.argsort(idx)
        idx = idx[order]
        rec_probs = rec_probs[order]

        # A persistence run is never allowed to cross a missing-window gap.
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        if len(idx):
            split_points = np.where(np.diff(idx) != 1)[0] + 1
            segments = list(zip(np.split(idx, split_points), np.split(rec_probs, split_points)))

        pred_intervals: List[Tuple[float, float, float]] = []
        for seg_idx, seg_probs in segments:
            events = _predicted_events(
                seg_probs,
                threshold,
                min_consecutive_windows,
                refractory_windows,
            )
            for start_local, end_local, alarm_local in events:
                first_window = int(seg_idx[start_local])
                last_window = int(seg_idx[end_local])
                alarm_window = int(seg_idx[alarm_local])
                start_sec = first_window * stride_sec
                end_sec = last_window * stride_sec + window_sec
                alarm_sec = alarm_window * stride_sec
                pred_intervals.append((start_sec, end_sec, alarm_sec))

        matched_predictions = set()
        for seizure_start, seizure_end in seizures:
            candidates = []
            for p_idx, (_pred_start, _pred_end, alarm_time) in enumerate(pred_intervals):
                if p_idx in matched_predictions:
                    continue
                # A long-running false alarm that started far before the seizure is
                # not converted into a true detection just because it overlaps it.
                if (
                    alarm_time >= seizure_start - early_tolerance_sec
                    and alarm_time <= seizure_end
                ):
                    candidates.append((alarm_time, p_idx))

            if candidates:
                alarm_time, p_idx = min(candidates, key=lambda item: item[0])
                detected_gt += 1
                matched_predictions.add(p_idx)
                # Small negative offsets can arise from the 2-s analysis window.
                latencies.append(max(0.0, float(alarm_time - seizure_start)))

        false_alarms += max(0, len(pred_intervals) - len(matched_predictions))

    hours = total_duration_sec / 3600.0
    sensitivity = detected_gt / total_gt if total_gt > 0 else float("nan")
    event_precision = (
        detected_gt / (detected_gt + false_alarms)
        if (detected_gt + false_alarms) > 0
        else 0.0
    )
    if np.isfinite(sensitivity) and (event_precision + sensitivity) > 0:
        event_f1 = 2.0 * event_precision * sensitivity / (event_precision + sensitivity)
    elif np.isfinite(sensitivity):
        event_f1 = 0.0
    else:
        event_f1 = float("nan")
    median_latency = float(np.median(latencies)) if latencies else float("nan")

    return {
        "total_gt_seizures": int(total_gt),
        "detected_seizures": int(detected_gt),
        "event_sensitivity": float(sensitivity),
        "event_precision": float(event_precision),
        "event_f1": float(event_f1),
        "false_alarms": int(false_alarms),
        "recording_hours": float(hours),
        "fa_per_hour": float(false_alarms / max(hours, 1e-12)),
        "median_latency_sec": median_latency,
    }


def select_event_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    recording_ids: Sequence[str],
    window_indices: np.ndarray,
    recording_metadata: Mapping[str, Mapping],
    max_candidates: int = EVENT_THRESHOLD_MAX_CANDIDATES,
) -> float:
    """
    Select a threshold only on validation patients, optimizing event-level F1.

    Ties prefer higher sensitivity, then fewer false alarms/hour. Window-F1 is
    retained only as a fallback when validation contains no annotated seizures.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    fallback = select_f1_threshold(labels, probs)

    total_gt = sum(
        len(meta.get("seizure_intervals", [])) for meta in recording_metadata.values()
    )
    finite = probs[np.isfinite(probs)]
    if total_gt <= 0 or finite.size == 0:
        return fallback

    quantiles = np.quantile(finite, np.linspace(0.50, 0.999, 41))
    linear = np.linspace(0.02, 0.98, 49)
    candidates = np.unique(
        np.clip(np.concatenate([quantiles, linear, [fallback, 0.5]]), 1e-4, 1.0 - 1e-4)
    )

    if candidates.size > max_candidates:
        keep = np.linspace(0, candidates.size - 1, max_candidates, dtype=int)
        candidates = np.unique(np.concatenate([candidates[keep], [fallback, 0.5]]))

    best_threshold = float(fallback)
    best_score = (-1.0, -1.0, -float("inf"), -float("inf"))

    for threshold in candidates:
        metrics = compute_event_metrics(
            probs=probs,
            recording_ids=recording_ids,
            window_indices=window_indices,
            recording_metadata=recording_metadata,
            threshold=float(threshold),
        )
        event_f1 = metrics["event_f1"]
        sensitivity = metrics["event_sensitivity"]
        if not np.isfinite(event_f1) or not np.isfinite(sensitivity):
            continue

        score = (
            float(event_f1),
            float(sensitivity),
            -float(metrics["fa_per_hour"]),
            float(threshold),
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold
