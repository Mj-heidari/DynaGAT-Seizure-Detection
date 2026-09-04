"""
Temporal clip dataset over the the current pipeline window cache.

Each item is an ordered clip of ``SEQUENCE_LENGTH`` consecutive windows from a
single continuous EDF recording, so the temporal model never sees a boundary
between recordings and never sees a future window.

Training epochs are built from every clip that contains ictal or peri-onset
windows plus a resampled pool of purely interictal clips. Unlike an earlier iteration the
resulting *sampling prior* is recorded on the dataset object and undone
analytically at inference time (training/calibration.py), so the deployed
probabilities remain calibrated to the true ~1:400 class ratio.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from config import (
    CACHE_VERSION,
    EVAL_SEQUENCE_STRIDE,
    MIN_NEGATIVE_CLIPS_PER_EPOCH,
    NEGATIVE_TO_IMPORTANT_RATIO,
    NODE_FEATURE_DIM,
    NUM_NODES,
    PREPROCESSING_TAG,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TOP_K_CAUSAL,
    TRAIN_SEQUENCE_STRIDE,
)

__all__ = ["load_cache", "compute_fold_normalization", "TemporalClipDataset", "collate"]


@dataclass(frozen=True)
class ClipRef:
    cache_idx: int
    recording_idx: int
    start: int
    valid_len: int


def load_cache(path: Path) -> Dict:
    """Load and strictly validate one subject cache."""
    kwargs = dict(map_location="cpu", weights_only=False)
    try:
        cache = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        cache = torch.load(path, **kwargs)

    if int(cache.get("cache_version", -1)) != CACHE_VERSION:
        raise RuntimeError(
            f"{path.name}: cache_version={cache.get('cache_version')} != {CACHE_VERSION}. "
            "Rebuild with `python -m dataset.preprocess --overwrite`."
        )
    if str(cache.get("preprocessing_tag", "")) != PREPROCESSING_TAG:
        raise RuntimeError(f"{path.name}: preprocessing tag mismatch, rebuild the cache.")
    if int(cache.get("node_feature_dim", -1)) != NODE_FEATURE_DIM:
        raise RuntimeError(f"{path.name}: feature dim mismatch, rebuild the cache.")
    if int(cache.get("top_k_causal", -1)) != TOP_K_CAUSAL:
        raise RuntimeError(f"{path.name}: top-k mismatch, rebuild the cache.")
    if not cache.get("recordings"):
        raise RuntimeError(f"{path.name}: no recordings")
    for rec in cache["recordings"]:
        x = rec.get("x")
        if x is None or x.ndim != 3 or tuple(x.shape[1:]) != (NUM_NODES, NODE_FEATURE_DIM):
            raise RuntimeError(f"{path.name}: bad recording tensor {None if x is None else x.shape}")
    return cache


def compute_fold_normalization(caches: Sequence[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature mean/std from the training subjects only (no held-out leakage)."""
    if not caches:
        raise ValueError("no training caches")
    s = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
    sq = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
    n = 0
    for cache in caches:
        s += cache["feature_sum"].double()
        sq += cache["feature_sumsq"].double()
        n += int(cache["feature_count"])
    if n <= 1:
        raise ValueError("invalid feature count")
    mean = s / n
    std = (sq / n - mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), std.float()


class TemporalClipDataset(Dataset):
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
        self.training = bool(training)
        self.t = int(sequence_length)
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

        self.total_windows = 0
        self.total_positive_windows = 0
        self._build_index()

        # True deployment prior of the ictal class over these subjects.
        self.true_positive_prior = self.total_positive_windows / max(1, self.total_windows)
        self.sampled_positive_prior = self.true_positive_prior

        if self.training:
            self.set_epoch(0)
        else:
            self.refs = self.eval_refs

    # -- index ------------------------------------------------------------- #
    @staticmethod
    def _eval_starts(n: int, length: int, stride: int) -> List[int]:
        if n <= 0:
            return []
        if n <= length:
            return [0]
        last = n - length
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _train_starts(n: int, length: int, stride: int) -> List[int]:
        if n <= 0:
            return []
        if n < length:
            return [0]
        last = n - length
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _build_index(self) -> None:
        for ci, cache in enumerate(self.caches):
            for ri, rec in enumerate(cache["recordings"]):
                rid = str(rec["recording_id"])
                self.recording_metadata[rid] = {
                    "duration_sec": float(rec["duration_sec"]),
                    "seizure_intervals": [tuple(x) for x in rec.get("seizure_intervals", [])],
                    "n_windows": int(rec["n_windows"]),
                }
                n = int(rec["n_windows"])
                self.total_windows += n
                self.total_positive_windows += int(rec["positive_windows"])
                labels = rec["labels"].to(torch.bool)
                bw = rec["boundary_weights"].float()

                if self.training:
                    starts = set(self._train_starts(n, self.t, self.train_stride))
                    if n > 0:
                        prev = torch.zeros_like(labels)
                        prev[1:] = labels[:-1]
                        for onset in torch.nonzero(labels & ~prev).flatten().tolist():
                            # Place the onset at several positions inside the clip
                            # so the model sees it with varying amounts of context.
                            for frac in (0.25, 0.5, 0.75):
                                s = int(max(0, min(n - self.t, onset - int(self.t * frac))))
                                starts.add(s)
                    for s in sorted(starts):
                        vlen = min(self.t, n - s)
                        if vlen <= 0:
                            continue
                        important = bool(labels[s : s + vlen].any() or (bw[s : s + vlen] > 1.0).any())
                        ref = ClipRef(ci, ri, int(s), int(vlen))
                        (self.important_refs if important else self.negative_pool).append(ref)
                else:
                    for s in self._eval_starts(n, self.t, self.eval_stride):
                        vlen = min(self.t, n - s)
                        if vlen > 0:
                            self.eval_refs.append(ClipRef(ci, ri, int(s), int(vlen)))

    def set_epoch(self, epoch: int) -> None:
        if not self.training:
            return
        rng = random.Random(self.seed + int(epoch))
        want = max(self.min_negative_clips, self.negative_ratio * max(1, len(self.important_refs)))
        want = min(want, len(self.negative_pool))
        negatives = (
            rng.sample(self.negative_pool, want)
            if want < len(self.negative_pool)
            else list(self.negative_pool)
        )
        self.refs = list(self.important_refs) + negatives
        rng.shuffle(self.refs)
        self.sampled_positive_prior = self._estimate_sampled_prior()

    def _estimate_sampled_prior(self) -> float:
        """Positive-window fraction of the currently sampled epoch."""
        pos = 0
        tot = 0
        for ref in self.refs:
            rec = self.caches[ref.cache_idx]["recordings"][ref.recording_idx]
            lab = rec["labels"][ref.start : ref.start + ref.valid_len]
            pos += int(lab.sum())
            tot += int(ref.valid_len)
        return pos / max(1, tot)

    # -- items -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> Dict:
        ref = self.refs[index]
        rec = self.caches[ref.cache_idx]["recordings"][ref.recording_idx]
        s, e, t = ref.start, ref.start + ref.valid_len, self.t

        x = torch.zeros((t, NUM_NODES, NODE_FEATURE_DIM), dtype=torch.float32)
        in_dst = torch.zeros((t, NUM_NODES, TOP_K_CAUSAL), dtype=torch.long)
        in_w = torch.zeros((t, NUM_NODES, TOP_K_CAUSAL), dtype=torch.float32)
        out_dst = torch.zeros((t, NUM_NODES, TOP_K_CAUSAL), dtype=torch.long)
        out_w = torch.zeros((t, NUM_NODES, TOP_K_CAUSAL), dtype=torch.float32)
        labels = torch.zeros(t, dtype=torch.float32)
        bw = torch.ones(t, dtype=torch.float32)
        valid = torch.zeros(t, dtype=torch.bool)
        widx = torch.full((t,), -1, dtype=torch.long)

        raw = rec["x"][s:e].float()
        x[: ref.valid_len] = torch.clamp((raw - self.mean) / self.std, -6.0, 6.0)
        in_dst[: ref.valid_len] = rec["in_dst"][s:e].long()
        in_w[: ref.valid_len] = rec["in_weight"][s:e].float()
        out_dst[: ref.valid_len] = rec["out_dst"][s:e].long()
        out_w[: ref.valid_len] = rec["out_weight"][s:e].float()
        labels[: ref.valid_len] = rec["labels"][s:e].float()
        bw[: ref.valid_len] = rec["boundary_weights"][s:e].float()
        valid[: ref.valid_len] = True
        widx[: ref.valid_len] = torch.arange(s, e, dtype=torch.long)

        if ref.valid_len < t:
            self_nodes = torch.arange(NUM_NODES).view(NUM_NODES, 1).repeat(1, TOP_K_CAUSAL)
            in_dst[ref.valid_len :] = self_nodes.unsqueeze(0)
            out_dst[ref.valid_len :] = self_nodes.unsqueeze(0)

        return {
            "x": x,
            "in_dst": in_dst,
            "in_weight": in_w,
            "out_dst": out_dst,
            "out_weight": out_w,
            "labels": labels,
            "boundary_weights": bw,
            "valid_mask": valid,
            "recording_id": str(rec["recording_id"]),
            "window_idx": widx,
        }


def collate(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for key in (
        "x", "in_dst", "in_weight", "out_dst", "out_weight",
        "labels", "boundary_weights", "valid_mask", "window_idx",
    ):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["recording_id"] = [b["recording_id"] for b in batch]
    return out
