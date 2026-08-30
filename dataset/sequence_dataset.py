from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from config import (
    EVAL_SEQUENCE_STRIDE,
    MIN_NEGATIVE_CLIPS_PER_EPOCH,
    NEGATIVE_TO_IMPORTANT_RATIO,
    NUM_NODES,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TOP_K_DYNAMIC,
    TRAIN_SEQUENCE_STRIDE,
)


@dataclass(frozen=True)
class ClipRef:
    cache_idx: int
    recording_idx: int
    start: int
    valid_len: int


def load_temporal_cache(path: Path) -> Dict:
    """Load a v2 cache using memory mapping when supported by PyTorch."""
    kwargs = dict(map_location="cpu", weights_only=False)
    try:
        cache = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        cache = torch.load(path, **kwargs)

    if int(cache.get("cache_version", -1)) != 2:
        raise RuntimeError(
            f"{path.name} is not a v2 temporal cache. Rebuild it with dataset/bids_loader.py"
        )
    return cache


def compute_fold_normalization(caches: Sequence[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute feature mean/std only from training subjects, without scanning windows."""
    if not caches:
        raise ValueError("No training caches supplied")

    total_sum = torch.zeros(16, dtype=torch.float64)
    total_sumsq = torch.zeros(16, dtype=torch.float64)
    total_count = 0

    for cache in caches:
        total_sum += cache["feature_sum"].double()
        total_sumsq += cache["feature_sumsq"].double()
        total_count += int(cache["feature_count"])

    if total_count <= 1:
        raise ValueError("Invalid training feature count")

    mean = total_sum / total_count
    variance = total_sumsq / total_count - mean.square()
    std = variance.clamp_min(1e-8).sqrt()
    return mean.float(), std.float()


class TemporalClipDataset(Dataset):
    """
    Produces real, ordered temporal clips:
        x: [T, 18, 16]
        dynamic_dst: [T, 18, K]
        dynamic_weight: [T, 18, K]
        labels: [T]

    Training mode dynamically resamples negative clips each epoch. Evaluation mode
    covers every cached window exactly once (apart from padded positions in the final
    clip of a recording).
    """

    def __init__(
        self,
        caches: Sequence[Dict],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        training: bool,
        sequence_length: int = SEQUENCE_LENGTH,
        train_stride: int = TRAIN_SEQUENCE_STRIDE,
        eval_stride: int = EVAL_SEQUENCE_STRIDE,
        negative_ratio: int = NEGATIVE_TO_IMPORTANT_RATIO,
        min_negative_clips: int = MIN_NEGATIVE_CLIPS_PER_EPOCH,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.caches = list(caches)
        self.mean = feature_mean.view(1, 1, -1).float()
        self.std = feature_std.view(1, 1, -1).float().clamp_min(1e-6)
        self.training = training
        self.sequence_length = int(sequence_length)
        self.train_stride = int(train_stride)
        self.eval_stride = int(eval_stride)
        self.negative_ratio = int(negative_ratio)
        self.min_negative_clips = int(min_negative_clips)
        self.seed = int(seed)

        self.important_refs: List[ClipRef] = []
        self.negative_pool: List[ClipRef] = []
        self.eval_refs: List[ClipRef] = []
        self.refs: List[ClipRef] = []
        self.recording_metadata: Dict[str, Dict] = {}

        self._build_index()
        if self.training:
            self.set_epoch(0)
        else:
            self.refs = self.eval_refs

    def _register_recording_metadata(self, rec: Dict) -> None:
        rid = str(rec["recording_id"])
        self.recording_metadata[rid] = {
            "duration_sec": float(rec["duration_sec"]),
            "seizure_intervals": [tuple(x) for x in rec.get("seizure_intervals", [])],
        }

    @staticmethod
    def _starts_covering_all(n: int, length: int, stride: int) -> List[int]:
        if n <= 0:
            return []
        starts = list(range(0, n, stride))
        return starts

    @staticmethod
    def _training_starts(n: int, length: int, stride: int) -> List[int]:
        if n < length:
            return [0]
        starts = list(range(0, n - length + 1, stride))
        final_start = n - length
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def _build_index(self) -> None:
        for cache_idx, cache in enumerate(self.caches):
            for recording_idx, rec in enumerate(cache["recordings"]):
                self._register_recording_metadata(rec)
                n = int(rec["n_windows"])
                labels = rec["labels"].to(torch.bool)
                bweights = rec["boundary_weights"].float()

                if self.training:
                    starts = set(self._training_starts(n, self.sequence_length, self.train_stride))

                    # Explicitly include onset-centered clips so the model repeatedly
                    # sees the transition boundary even when it falls near a base clip edge.
                    if n > 0:
                        prev = torch.zeros_like(labels)
                        prev[1:] = labels[:-1]
                        onset_indices = torch.nonzero(labels & ~prev, as_tuple=False).flatten().tolist()
                        for onset_idx in onset_indices:
                            centered = max(0, min(n - self.sequence_length, onset_idx - self.sequence_length // 2))
                            starts.add(int(max(0, centered)))

                    for start in sorted(starts):
                        valid_len = min(self.sequence_length, n - start)
                        end = start + valid_len
                        important = bool(labels[start:end].any() or (bweights[start:end] > 1.0).any())
                        ref = ClipRef(cache_idx, recording_idx, int(start), int(valid_len))
                        if important:
                            self.important_refs.append(ref)
                        else:
                            self.negative_pool.append(ref)
                else:
                    for start in self._starts_covering_all(n, self.sequence_length, self.eval_stride):
                        valid_len = min(self.sequence_length, n - start)
                        self.eval_refs.append(
                            ClipRef(cache_idx, recording_idx, int(start), int(valid_len))
                        )

    def set_epoch(self, epoch: int) -> None:
        if not self.training:
            return

        rng = random.Random(self.seed + int(epoch))
        desired_negatives = max(
            self.min_negative_clips,
            self.negative_ratio * max(1, len(self.important_refs)),
        )
        desired_negatives = min(desired_negatives, len(self.negative_pool))

        if desired_negatives < len(self.negative_pool):
            negatives = rng.sample(self.negative_pool, desired_negatives)
        else:
            negatives = list(self.negative_pool)

        self.refs = list(self.important_refs) + negatives
        rng.shuffle(self.refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> Dict:
        ref = self.refs[index]
        rec = self.caches[ref.cache_idx]["recordings"][ref.recording_idx]
        start = ref.start
        end = start + ref.valid_len
        t = self.sequence_length

        x = torch.zeros((t, NUM_NODES, 16), dtype=torch.float32)
        dynamic_dst = torch.zeros((t, NUM_NODES, TOP_K_DYNAMIC), dtype=torch.long)
        dynamic_weight = torch.zeros((t, NUM_NODES, TOP_K_DYNAMIC), dtype=torch.float32)
        labels = torch.zeros(t, dtype=torch.float32)
        boundary_weights = torch.ones(t, dtype=torch.float32)
        valid_mask = torch.zeros(t, dtype=torch.bool)
        window_idx = torch.full((t,), -1, dtype=torch.long)

        raw_x = rec["x"][start:end].float()
        x[: ref.valid_len] = torch.clamp((raw_x - self.mean) / self.std, -5.0, 5.0)
        dynamic_dst[: ref.valid_len] = rec["dynamic_dst"][start:end].long()
        dynamic_weight[: ref.valid_len] = rec["dynamic_weight"][start:end].float()
        labels[: ref.valid_len] = rec["labels"][start:end].float()
        boundary_weights[: ref.valid_len] = rec["boundary_weights"][start:end].float()
        valid_mask[: ref.valid_len] = True
        window_idx[: ref.valid_len] = torch.arange(start, end, dtype=torch.long)

        # Harmless self destinations for padded positions. They are masked from loss
        # and metrics; causal TCN ensures end padding cannot alter earlier outputs.
        if ref.valid_len < t:
            self_nodes = torch.arange(NUM_NODES).view(NUM_NODES, 1).repeat(1, TOP_K_DYNAMIC)
            dynamic_dst[ref.valid_len :] = self_nodes.unsqueeze(0)

        return {
            "x": x,
            "dynamic_dst": dynamic_dst,
            "dynamic_weight": dynamic_weight,
            "labels": labels,
            "boundary_weights": boundary_weights,
            "valid_mask": valid_mask,
            "recording_id": str(rec["recording_id"]),
            "window_idx": window_idx,
        }
