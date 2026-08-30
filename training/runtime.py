from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import LINKED_SUBJECT_GROUPS, MAX_GRAD_NORM
from dataset.sequence_dataset import TemporalClipDataset
from training.losses import BoundaryAwareFocalLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionBundle:
    def __init__(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        recording_ids: List[str],
        window_indices: np.ndarray,
        recording_metadata: Dict[str, Dict],
    ) -> None:
        self.probs = probs
        self.labels = labels
        self.recording_ids = recording_ids
        self.window_indices = window_indices
        self.recording_metadata = recording_metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_groups(subjects: Sequence[str]) -> List[List[str]]:
    remaining = set(subjects)
    groups: List[List[str]] = []
    for linked in LINKED_SUBJECT_GROUPS:
        present = sorted(remaining.intersection(linked))
        if len(present) >= 2:
            groups.append(present)
            remaining.difference_update(present)
    groups.extend([[subject] for subject in sorted(remaining)])
    return groups


def make_loader(dataset: TemporalClipDataset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def to_device(batch: Dict) -> Dict[str, torch.Tensor]:
    return {
        "x": batch["x"].to(DEVICE, non_blocking=True),
        "dynamic_dst": batch["dynamic_dst"].to(DEVICE, non_blocking=True),
        "dynamic_weight": batch["dynamic_weight"].to(DEVICE, non_blocking=True),
        "labels": batch["labels"].to(DEVICE, non_blocking=True),
        "boundary_weights": batch["boundary_weights"].to(DEVICE, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(DEVICE, non_blocking=True),
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BoundaryAwareFocalLoss,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    loader.dataset.set_epoch(epoch)
    total_loss = 0.0
    batches = 0
    pbar = tqdm(loader, desc=f"train {epoch:02d}/{total_epochs:02d}", ncols=100, leave=False)
    for batch in pbar:
        tensors = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            logits = model(
                tensors["x"],
                tensors["dynamic_dst"],
                tensors["dynamic_weight"],
                valid_mask=tensors["valid_mask"],
            )
            loss = criterion(
                logits,
                tensors["labels"],
                tensors["boundary_weights"],
                tensors["valid_mask"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().cpu())
        total_loss += value
        batches += 1
        pbar.set_postfix(loss=f"{value:.4f}")
    return total_loss / max(1, batches)


def _keep_largest_causal_context(
    probs: np.ndarray,
    labels: np.ndarray,
    recording_ids: List[str],
    window_indices: np.ndarray,
    context_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    if probs.size == 0:
        return probs, labels, recording_ids, window_indices
    rid = np.asarray(recording_ids, dtype=str)
    order = np.lexsort((context_positions, window_indices, rid))
    rid_s = rid[order]
    idx_s = window_indices[order]
    probs_s = probs[order]
    labels_s = labels[order]
    is_last = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        is_last[:-1] = (rid_s[:-1] != rid_s[1:]) | (idx_s[:-1] != idx_s[1:])
    return probs_s[is_last], labels_s[is_last], rid_s[is_last].tolist(), idx_s[is_last]


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader) -> PredictionBundle:
    model.eval()
    probs_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    context_parts: List[np.ndarray] = []
    recording_ids: List[str] = []

    for batch in tqdm(loader, desc="evaluate", ncols=100, leave=False):
        x = batch["x"].to(DEVICE, non_blocking=True)
        dst = batch["dynamic_dst"].to(DEVICE, non_blocking=True)
        weight = batch["dynamic_weight"].to(DEVICE, non_blocking=True)
        valid_gpu = batch["valid_mask"].to(DEVICE, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            probs = torch.sigmoid(model(x, dst, weight, valid_mask=valid_gpu)).cpu()

        labels = batch["labels"]
        mask = batch["valid_mask"]
        window_idx = batch["window_idx"]
        for b, rid in enumerate(batch["recording_id"]):
            valid = mask[b]
            positions = torch.nonzero(valid, as_tuple=False).flatten()
            count = int(positions.numel())
            if count == 0:
                continue
            probs_parts.append(probs[b][valid].numpy())
            label_parts.append(labels[b][valid].numpy())
            index_parts.append(window_idx[b][valid].numpy())
            context_parts.append(positions.numpy())
            recording_ids.extend([str(rid)] * count)

    probs_np = np.concatenate(probs_parts) if probs_parts else np.empty(0, dtype=np.float32)
    labels_np = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.float32)
    indices_np = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
    contexts_np = np.concatenate(context_parts) if context_parts else np.empty(0, dtype=np.int64)
    probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=1.0, neginf=0.0)
    probs_np, labels_np, recording_ids, indices_np = _keep_largest_causal_context(
        probs_np, labels_np, recording_ids, indices_np, contexts_np
    )
    return PredictionBundle(
        probs=probs_np,
        labels=labels_np,
        recording_ids=recording_ids,
        window_indices=indices_np,
        recording_metadata=loader.dataset.recording_metadata,
    )


def caches_for_subjects(cache_by_subject: Dict[str, Dict], subjects: Sequence[str]) -> List[Dict]:
    return [cache_by_subject[s] for s in subjects]


def cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def validation_score(metrics: Dict[str, float]) -> float:
    auprc = float(metrics.get("auprc", float("nan")))
    if np.isfinite(auprc):
        return auprc
    auroc = float(metrics.get("auroc", float("nan")))
    return auroc if np.isfinite(auroc) else -float("inf")
