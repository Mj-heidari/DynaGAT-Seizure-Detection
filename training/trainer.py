from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from config import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GAT_HEADS,
    GRAPH_HIDDEN,
    LEARNING_RATE,
    LINKED_SUBJECT_GROUPS,
    MAX_GRAD_NORM,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TCN_HIDDEN,
    WEIGHT_DECAY,
)
from dataset.sequence_dataset import (
    TemporalClipDataset,
    compute_fold_normalization,
    load_temporal_cache,
)
from evaluation.metrics import (
    compute_ece,
    compute_event_metrics,
    compute_window_metrics,
    select_event_threshold,
)
from models.dynagat_model import DynaGATOnsetModel
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
    """Create patient groups while keeping known linked CHB-MIT identities together."""
    remaining = set(subjects)
    groups: List[List[str]] = []

    for linked in LINKED_SUBJECT_GROUPS:
        present = sorted(remaining.intersection(linked))
        if len(present) >= 2:
            groups.append(present)
            remaining.difference_update(present)

    for subject in sorted(remaining):
        groups.append([subject])
    return groups


def make_loader(dataset: TemporalClipDataset, shuffle: bool, batch_size: int) -> DataLoader:
    # num_workers=0 is deliberate on Windows: mmap-backed caches plus worker
    # spawning can duplicate mappings and RAM usage.
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
    model: DynaGATOnsetModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BoundaryAwareFocalLoss,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    loader.dataset.set_epoch(epoch)
    running_loss = 0.0
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

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        batches += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")

    return running_loss / max(1, batches)


def _keep_largest_causal_context(
    probs: np.ndarray,
    labels: np.ndarray,
    recording_ids: List[str],
    window_indices: np.ndarray,
    context_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Resolve overlapped evaluation clips without averaging different contexts."""
    if probs.size == 0:
        return probs, labels, recording_ids, window_indices

    rid = np.asarray(recording_ids, dtype=str)
    # Primary key: recording id, secondary: absolute window index, tertiary:
    # local context position. Keeping the last item therefore keeps the largest
    # amount of available causal history for each physical EEG window.
    order = np.lexsort((context_positions, window_indices, rid))
    rid_s = rid[order]
    idx_s = window_indices[order]
    probs_s = probs[order]
    labels_s = labels[order]

    is_last = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        is_last[:-1] = (rid_s[:-1] != rid_s[1:]) | (idx_s[:-1] != idx_s[1:])

    return (
        probs_s[is_last],
        labels_s[is_last],
        rid_s[is_last].tolist(),
        idx_s[is_last],
    )


@torch.inference_mode()
def predict(model: DynaGATOnsetModel, loader: DataLoader) -> PredictionBundle:
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
            logits = model(x, dst, weight, valid_mask=valid_gpu)
            probs = torch.sigmoid(logits).cpu()

        labels = batch["labels"]
        mask = batch["valid_mask"]
        window_idx = batch["window_idx"]

        for b, rid in enumerate(batch["recording_id"]):
            valid = mask[b]
            valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
            count = int(valid_positions.numel())
            if count == 0:
                continue
            probs_parts.append(probs[b][valid].numpy())
            label_parts.append(labels[b][valid].numpy())
            index_parts.append(window_idx[b][valid].numpy())
            context_parts.append(valid_positions.numpy())
            recording_ids.extend([str(rid)] * count)

    probs_np = np.concatenate(probs_parts) if probs_parts else np.empty(0, dtype=np.float32)
    labels_np = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.float32)
    indices_np = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
    contexts_np = np.concatenate(context_parts) if context_parts else np.empty(0, dtype=np.int64)

    probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=1.0, neginf=0.0)
    probs_np, labels_np, recording_ids, indices_np = _keep_largest_causal_context(
        probs_np,
        labels_np,
        recording_ids,
        indices_np,
        contexts_np,
    )

    return PredictionBundle(
        probs=probs_np,
        labels=labels_np,
        recording_ids=recording_ids,
        window_indices=indices_np,
        recording_metadata=loader.dataset.recording_metadata,
    )


def save_prediction_bundle(
    bundle: PredictionBundle,
    path: Path,
    threshold: float,
    test_subjects: Sequence[str],
) -> None:
    """Persist real held-out predictions for reproducible figures and analysis."""
    np.savez_compressed(
        path,
        probs=bundle.probs.astype(np.float32, copy=False),
        labels=bundle.labels.astype(np.uint8, copy=False),
        recording_ids=np.asarray(bundle.recording_ids, dtype=str),
        window_indices=bundle.window_indices.astype(np.int64, copy=False),
        threshold=np.asarray(float(threshold), dtype=np.float32),
        test_patient=np.asarray("+".join(test_subjects), dtype=str),
    )


def caches_for_subjects(cache_by_subject: Dict[str, Dict], subjects: Sequence[str]) -> List[Dict]:
    return [cache_by_subject[s] for s in subjects]


def run_lopo(max_folds: int | None = None, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    seed_everything(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"[*] Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CUDA not detected. The code will run on CPU but will be very slow.")

    cache_paths = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    if not cache_paths:
        raise FileNotFoundError(
            f"No v2 temporal caches found in {PROCESSED_DATA_DIR}.\n"
            "Run: python run_preprocessing.py"
        )

    print(f"[*] Loading {len(cache_paths)} mmap-backed patient caches...")
    cache_by_subject: Dict[str, Dict] = {}
    for path in tqdm(cache_paths, desc="load caches", ncols=100):
        cache = load_temporal_cache(path)
        cache_by_subject[str(cache["subject"])] = cache

    groups = subject_groups(sorted(cache_by_subject))
    if len(groups) < 3:
        raise RuntimeError(
            "LOPO requires at least 3 independent patient groups: train, validation, and test."
        )

    validation_size = min(4, len(groups) - 2)
    print(f"[*] LOPO patient groups: {len(groups)}")
    print("[*] Test patient is never used for training, threshold selection, or normalization.")
    print(f"[*] Inner validation groups per fold: {validation_size}\n")

    results: List[Dict] = []
    run_groups = groups[:max_folds] if max_folds is not None else groups

    for fold_idx, test_group in enumerate(run_groups):
        validation_indices = [
            (fold_idx + offset) % len(groups)
            for offset in range(1, validation_size + 1)
        ]
        excluded_indices = {fold_idx, *validation_indices}
        train_groups = [g for i, g in enumerate(groups) if i not in excluded_indices]
        val_groups = [groups[i] for i in validation_indices]

        train_subjects = [s for group in train_groups for s in group]
        val_subjects = [s for group in val_groups for s in group]
        test_subjects = list(test_group)

        fold_name = "+".join(test_subjects)
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1:02d}/{len(run_groups):02d} | TEST={test_subjects} | VAL={val_subjects}")
        print("=" * 80)

        train_caches = caches_for_subjects(cache_by_subject, train_subjects)
        val_caches = caches_for_subjects(cache_by_subject, val_subjects)
        test_caches = caches_for_subjects(cache_by_subject, test_subjects)

        mean, std = compute_fold_normalization(train_caches)
        train_ds = TemporalClipDataset(train_caches, mean, std, training=True)
        val_ds = TemporalClipDataset(val_caches, mean, std, training=False)
        test_ds = TemporalClipDataset(test_caches, mean, std, training=False)

        if len(train_ds) == 0:
            raise RuntimeError("Training dataset is empty")

        print(
            f"clips: train={len(train_ds):,} "
            f"(important={len(train_ds.important_refs):,}, negative_pool={len(train_ds.negative_pool):,}) | "
            f"val={len(val_ds):,} | test={len(test_ds):,}"
        )

        train_loader = make_loader(train_ds, shuffle=True, batch_size=batch_size)
        val_loader = make_loader(val_ds, shuffle=False, batch_size=batch_size)
        test_loader = make_loader(test_ds, shuffle=False, batch_size=batch_size)

        model = DynaGATOnsetModel(
            graph_hidden=GRAPH_HIDDEN,
            tcn_hidden=TCN_HIDDEN,
            heads=GAT_HEADS,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=LEARNING_RATE * 0.1
        )
        criterion = BoundaryAwareFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        fold_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, epochs)
            scheduler.step()
            epoch_sec = time.perf_counter() - epoch_start

            gpu_mem = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                gpu_mem = f" | peakVRAM={allocated:.2f}GB"
                torch.cuda.reset_peak_memory_stats()

            print(
                f"epoch {epoch:02d}/{epochs:02d} | loss={loss:.5f} | "
                f"time={epoch_sec:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}{gpu_mem}"
            )

        # Validation is evaluated once after training. The held-out test patient is
        # not touched until normalization, training, and threshold selection finish.
        val_pred = predict(model, val_loader)
        threshold = select_event_threshold(
            labels=val_pred.labels,
            probs=val_pred.probs,
            recording_ids=val_pred.recording_ids,
            window_indices=val_pred.window_indices,
            recording_metadata=val_pred.recording_metadata,
        )
        val_window = compute_window_metrics(val_pred.labels, val_pred.probs, threshold)
        val_event = compute_event_metrics(
            probs=val_pred.probs,
            recording_ids=val_pred.recording_ids,
            window_indices=val_pred.window_indices,
            recording_metadata=val_pred.recording_metadata,
            threshold=threshold,
        )

        test_pred = predict(model, test_loader)
        test_window = compute_window_metrics(test_pred.labels, test_pred.probs, threshold)
        event = compute_event_metrics(
            probs=test_pred.probs,
            recording_ids=test_pred.recording_ids,
            window_indices=test_pred.window_indices,
            recording_metadata=test_pred.recording_metadata,
            threshold=threshold,
        )
        ece = compute_ece(test_pred.probs, test_pred.labels, n_bins=10)

        elapsed = time.perf_counter() - fold_start
        print(
            f"VAL event-F1={val_event['event_f1']:.4f} | "
            f"sens={val_event['event_sensitivity']:.4f} | "
            f"FA/h={val_event['fa_per_hour']:.3f} | threshold={threshold:.4f}"
        )
        print(
            f"TEST {fold_name}: AUROC={test_window['auroc']:.4f} | "
            f"AUPRC={test_window['auprc']:.4f} | F1={test_window['f1']:.4f}"
        )
        print(
            f"Event sensitivity={event['event_sensitivity']:.4f} "
            f"({event['detected_seizures']}/{event['total_gt_seizures']}) | "
            f"precision={event['event_precision']:.4f} | event-F1={event['event_f1']:.4f} | "
            f"FA/h={event['fa_per_hour']:.3f} | "
            f"median latency={event['median_latency_sec']:.2f}s | ECE={ece:.4f}"
        )

        checkpoint_path = RESULTS_DIR / f"dynagat_onset_fold_{fold_idx + 1:02d}_{fold_name}.pt"
        torch.save(
            {
                "model_version": "causal_v3",
                "model_state_dict": model.state_dict(),
                "test_subjects": test_subjects,
                "validation_subjects": val_subjects,
                "feature_mean": mean,
                "feature_std": std,
                "validation_threshold": threshold,
            },
            checkpoint_path,
        )

        prediction_path = RESULTS_DIR / f"fold_{fold_idx + 1:02d}_test_predictions.npz"
        save_prediction_bundle(test_pred, prediction_path, threshold, test_subjects)

        results.append(
            {
                "fold": fold_idx + 1,
                "test_patient": fold_name,
                "validation_patient": "+".join(val_subjects),
                "threshold": threshold,
                "val_auroc": val_window["auroc"],
                "val_auprc": val_window["auprc"],
                "val_event_sensitivity": val_event["event_sensitivity"],
                "val_event_precision": val_event["event_precision"],
                "val_event_f1": val_event["event_f1"],
                "val_fa_per_hour": val_event["fa_per_hour"],
                "auroc": test_window["auroc"],
                "auprc": test_window["auprc"],
                "f1": test_window["f1"],
                "gt_seizures": event["total_gt_seizures"],
                "detected_seizures": event["detected_seizures"],
                "event_sensitivity": event["event_sensitivity"],
                "event_precision": event["event_precision"],
                "event_f1": event["event_f1"],
                "false_alarms": event["false_alarms"],
                "recording_hours": event["recording_hours"],
                "fa_per_hour": event["fa_per_hour"],
                "median_latency_sec": event["median_latency_sec"],
                "ece": ece,
                "elapsed_sec": elapsed,
            }
        )

        out_csv = RESULTS_DIR / "lopo_results_summary.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)

        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        del train_ds, val_ds, test_ds, val_pred, test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("LOPO COMPLETE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    for column in [
        "auroc",
        "auprc",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "fa_per_hour",
        "median_latency_sec",
        "ece",
    ]:
        print(f"mean {column:24s}: {df[column].mean(skipna=True):.4f}")
    print(f"[+] Results: {RESULTS_DIR / 'lopo_results_summary.csv'}")
