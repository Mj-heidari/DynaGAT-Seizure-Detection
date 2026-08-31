from __future__ import annotations

import numpy as np

from evaluation.operating_point import select_validation_operating_point


def test_search_always_includes_a_true_no_alarm_candidate() -> None:
    probs = np.ones(12, dtype=np.float64)
    labels = np.zeros(12, dtype=np.int64)
    labels[5:7] = 1
    result = select_validation_operating_point(
        labels=labels,
        probs=probs,
        recording_ids=["rec"] * len(probs),
        window_indices=np.arange(len(probs)),
        recording_metadata={
            "rec": {"duration_sec": 15.0, "seizure_intervals": [(7.0, 9.0)]}
        },
        far_cap=0.0,
        persistence_candidates=[1],
    )
    assert result.feasible_under_cap
    assert result.threshold > 1.0
    assert result.validation_metrics["false_alarms"] == 0
