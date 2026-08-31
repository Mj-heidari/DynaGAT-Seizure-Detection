from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CACHE_VERSION, PREPROCESSING_TAG
from training.trainer import (
    EVALUATION_VERSION,
    MODEL_VERSION,
    RESULTS_SCHEMA_VERSION,
    experiment_signature,
    load_existing_results,
)


def test_signature_is_stable_and_tracks_runtime_configuration() -> None:
    baseline = experiment_signature(epochs=30, batch_size=32)
    assert baseline == experiment_signature(epochs=30, batch_size=32)
    assert baseline != experiment_signature(epochs=24, batch_size=32)
    assert baseline != experiment_signature(epochs=30, batch_size=16)


def test_resume_rejects_stale_experiment_rows(tmp_path: Path) -> None:
    signature = experiment_signature(epochs=30, batch_size=32)
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "fold": 1,
                "model_version": MODEL_VERSION,
                "evaluation_version": EVALUATION_VERSION,
                "results_schema_version": RESULTS_SCHEMA_VERSION,
                "cache_version": CACHE_VERSION,
                "preprocessing_tag": PREPROCESSING_TAG,
                "experiment_signature": signature,
            }
        ]
    ).to_csv(path, index=False)
    assert set(load_existing_results(path, signature)) == {1}
    assert load_existing_results(path, "different-signature") == {}
