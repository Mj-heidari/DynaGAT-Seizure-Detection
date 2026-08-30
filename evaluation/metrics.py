from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from config import MIN_CONSECUTIVE_POSITIVE_WINDOWS, WINDOW_SEC, WINDOW_STRIDE_SEC


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
    """Choose threshold on validation data only."""
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
) -> List[Tuple[int, int]]:
    binary = (np.asarray(probs) >= threshold).astype(np.uint8)
    events: List[Tuple[int, int]] = []
    run_start = None
    run_length = 0

    for idx, value in enumerate(binary):
        if value:
            if run_start is None:
                run_start = idx
            run_length += 1
        else:
            if run_start is not None and run_length >= min_consecutive:
                events.append((run_start, idx - 1))
            run_start = None
            run_length = 0

    if run_start is not None and run_length >= min_consecutive:
        events.append((run_start, len(binary) - 1))
    return events


def compute_event_metrics(
    probs: np.ndarray,
    recording_ids: Sequence[str],
    window_indices: np.ndarray,
    recording_metadata: Mapping[str, Mapping],
    threshold: float,
    min_consecutive_windows: int = MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    stride_sec: float = WINDOW_STRIDE_SEC,
    window_sec: float = WINDOW_SEC,
) -> Dict[str, float]:
    """
    Compute event sensitivity, false alarms/hour, and latency independently per EDF.
    This avoids merging events across recording boundaries.
    """
    probs = np.asarray(probs, dtype=np.float64)
    window_indices = np.asarray(window_indices, dtype=np.int64)
    recording_ids = np.asarray(recording_ids, dtype=object)

    total_gt = 0
    detected_gt = 0
    false_alarms = 0
    latencies: List[float] = []
    total_duration_sec = 0.0

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

        # Evaluation cache is continuous. If a gap nevertheless exists, split at gaps
        # so a persistence run cannot jump across missing windows.
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        if len(idx):
            split_points = np.where(np.diff(idx) != 1)[0] + 1
            idx_parts = np.split(idx, split_points)
            prob_parts = np.split(rec_probs, split_points)
            segments = list(zip(idx_parts, prob_parts))

        pred_intervals: List[Tuple[float, float, float]] = []
        for seg_idx, seg_probs in segments:
            for start_local, end_local in _predicted_events(
                seg_probs, threshold, min_consecutive_windows
            ):
                first_window = int(seg_idx[start_local])
                last_window = int(seg_idx[end_local])
                start_sec = first_window * stride_sec
                end_sec = last_window * stride_sec + window_sec
                pred_intervals.append((start_sec, end_sec, start_sec))

        matched_predictions = set()
        for seizure_start, seizure_end in seizures:
            candidates = []
            for p_idx, (pred_start, pred_end, alarm_time) in enumerate(pred_intervals):
                if p_idx in matched_predictions:
                    continue
                if pred_end >= seizure_start and pred_start <= seizure_end:
                    candidates.append((alarm_time, p_idx))

            if candidates:
                alarm_time, p_idx = min(candidates, key=lambda item: item[0])
                detected_gt += 1
                matched_predictions.add(p_idx)
                latencies.append(max(0.0, float(alarm_time - seizure_start)))

        false_alarms += max(0, len(pred_intervals) - len(matched_predictions))

    hours = total_duration_sec / 3600.0
    sensitivity = detected_gt / total_gt if total_gt > 0 else float("nan")
    median_latency = float(np.median(latencies)) if latencies else float("nan")

    return {
        "total_gt_seizures": int(total_gt),
        "detected_seizures": int(detected_gt),
        "event_sensitivity": float(sensitivity),
        "false_alarms": int(false_alarms),
        "recording_hours": float(hours),
        "fa_per_hour": float(false_alarms / max(hours, 1e-12)),
        "median_latency_sec": median_latency,
    }
