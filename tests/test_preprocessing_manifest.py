from __future__ import annotations

from pathlib import Path

import torch

from config import CACHE_VERSION, NODE_FEATURE_DIM, PREPROCESSING_TAG
from dataset.bids_loader import _manifest_row_from_cache


def test_existing_cache_can_reconstruct_complete_manifest_row(tmp_path: Path) -> None:
    cache_path = tmp_path / "sub-01_temporal_graphs.pt"
    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "preprocessing_tag": PREPROCESSING_TAG,
            "node_feature_dim": NODE_FEATURE_DIM,
            "recordings": [
                {"duration_sec": 3600.0, "n_windows": 10, "positive_windows": 2},
                {"duration_sec": 1800.0, "n_windows": 5, "positive_windows": 1},
            ],
            "total_windows": 15,
            "positive_windows": 3,
            "total_seizures": 2,
            "valid_recordings": 2,
            "skipped_recordings": 0,
            "event_files_found": 2,
        },
        cache_path,
    )
    row = _manifest_row_from_cache(
        cache_path,
        "sub-01",
        [tmp_path / "a.edf", tmp_path / "b.edf"],
    )
    assert row["subject"] == "sub-01"
    assert row["edf_files"] == 2
    assert row["valid_recordings"] == 2
    assert row["windows"] == 15
    assert row["seizures"] == 2
    assert row["recording_hours"] == 1.5

