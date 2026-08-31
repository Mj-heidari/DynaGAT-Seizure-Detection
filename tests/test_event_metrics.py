from __future__ import annotations

import numpy as np

from evaluation.metrics import compute_event_metrics, window_decision_times


def _metadata(intervals: list[tuple[float, float]]) -> dict:
    return {"rec": {"duration_sec": 20.0, "seizure_intervals": intervals}}


def test_decision_timestamp_is_window_end() -> None:
    assert np.allclose(window_decision_times([0, 1, 9]), [2.0, 3.0, 11.0])


def test_event_latency_uses_online_available_time() -> None:
    probs = np.zeros(18, dtype=float)
    probs[9] = 0.99
    metrics = compute_event_metrics(
        probs=probs,
        recording_ids=["rec"] * len(probs),
        window_indices=np.arange(len(probs)),
        recording_metadata=_metadata([(10.0, 15.0)]),
        threshold=0.5,
        min_consecutive_windows=1,
    )
    assert metrics["detected_seizures"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["median_latency_sec"] == 1.0


def test_alarm_available_after_seizure_end_is_false_alarm() -> None:
    probs = np.zeros(18, dtype=float)
    probs[9] = 0.99  # decision time = 11 s
    metrics = compute_event_metrics(
        probs=probs,
        recording_ids=["rec"] * len(probs),
        window_indices=np.arange(len(probs)),
        recording_metadata=_metadata([(10.0, 10.5)]),
        threshold=0.5,
        min_consecutive_windows=1,
    )
    assert metrics["detected_seizures"] == 0
    assert metrics["false_alarms"] == 1


def test_persistence_cannot_cross_missing_window_gap() -> None:
    metrics = compute_event_metrics(
        probs=np.asarray([0.9, 0.9]),
        recording_ids=["rec", "rec"],
        window_indices=np.asarray([0, 2]),
        recording_metadata=_metadata([]),
        threshold=0.5,
        min_consecutive_windows=2,
    )
    assert metrics["false_alarms"] == 0

