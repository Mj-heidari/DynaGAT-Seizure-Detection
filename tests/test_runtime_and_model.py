from __future__ import annotations

import numpy as np
import torch

from config import NODE_FEATURE_DIM, NUM_NODES, TOP_K_DYNAMIC
from models.dynagat_model import DynaGATOnsetModel
from training.runtime import _keep_largest_causal_context, subject_groups


def test_linked_subjects_are_kept_in_one_group() -> None:
    groups = subject_groups(["sub-01", "sub-02", "sub-21"])
    assert ["sub-01", "sub-21"] in groups
    assert ["sub-02"] in groups


def test_overlapping_clips_keep_largest_causal_context() -> None:
    probs, labels, recording_ids, indices = _keep_largest_causal_context(
        probs=np.asarray([0.1, 0.9]),
        labels=np.asarray([1.0, 1.0]),
        recording_ids=["rec", "rec"],
        window_indices=np.asarray([8, 8]),
        context_positions=np.asarray([0, 8]),
    )
    assert probs.tolist() == [0.9]
    assert labels.tolist() == [1.0]
    assert recording_ids == ["rec"]
    assert indices.tolist() == [8]


def test_future_windows_do_not_change_earlier_logits() -> None:
    torch.manual_seed(11)
    model = DynaGATOnsetModel().eval()
    x = torch.randn(1, 4, NUM_NODES, NODE_FEATURE_DIM)
    changed = x.clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 4.0
    dst = torch.zeros(1, 4, NUM_NODES, TOP_K_DYNAMIC, dtype=torch.long)
    weight = torch.ones(1, 4, NUM_NODES, TOP_K_DYNAMIC)
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.inference_mode():
        original_logits = model(x, dst, weight, valid_mask=valid)
        changed_logits = model(changed, dst, weight, valid_mask=valid)
    assert torch.allclose(
        original_logits[:, :2], changed_logits[:, :2], atol=1e-5, rtol=1e-5
    )

