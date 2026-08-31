from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from config import (
    BATCH_SIZE,
    CACHE_VERSION,
    DECISION_TIME_REFERENCE,
    DEVELOPMENT_FOLD,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GAT_HEADS,
    GRAPH_HIDDEN,
    LEARNING_RATE,
    MIN_EPOCHS_BEFORE_STOPPING,
    PREPROCESSING_TAG,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SEQUENCE_LENGTH,
    TCN_HIDDEN,
    VALIDATION_CHECK_INTERVAL,
    VALIDATION_FA_PER_HOUR_CAP,
    WEIGHT_DECAY,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)
from dataset.sequence_dataset import TemporalClipDataset, compute_fold_normalization, load_temporal_cache
from evaluation.metrics import compute_ece, compute_event_metrics, compute_window_metrics
from evaluation.operating_point import select_validation_operating_point
from models.dynagat_model import DynaGATOnsetModel
from training.losses import BoundaryAwareFocalLoss
from training.runtime import (
    DEVICE,
    PredictionBundle,
    caches_for_subjects,
    cpu_state_dict,
    make_loader,
    predict,
    seed_everything,
    subject_groups,
    train_epoch,
    validation_score,
)


MODEL_VERSION = "causal_v5_residual_delta"
EVALUATION_VERSION = "window_end_online_v1"
RESULTS_SCHEMA_VERSION = 2
ALARM_OBJECTIVE = "sensitivity_first_under_validation_far_cap"

SIGNATURE_SOURCE_FILES = (
    "config.py",
    "dataset/sequence_dataset.py",
    "evaluation/metrics.py",
    "evaluation/operating_point.py",
    "models/dynagat_model.py",
    "training/losses.py",
    "training/runtime.py",
    "training/trainer.py",
)


def experiment_signature(epochs: int, batch_size: int) -> str:
    """Fingerprint code, runtime settings, and the actual local cache set."""
    root = Path(__file__).resolve().parent.parent
    payload = {
        "model_version": MODEL_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "preprocessing_tag": PREPROCESSING_TAG,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for relative in SIGNATURE_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    manifest_path = PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if manifest_path.exists():
        digest.update(b"preprocessing_manifest.csv")
        digest.update(manifest_path.read_bytes())
    for cache_path in sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt")):
        size = cache_path.stat().st_size
        digest.update(cache_path.name.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        # Sampling both ends catches accidental replacement without hashing many
        # gigabytes of cache tensors before every resume/export check.
        with cache_path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if size > 1024 * 1024:
                handle.seek(max(0, size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()[:20]


def hardware_summary() -> Dict[str, object]:
    info: Dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "device": "cpu",
        "gpu_memory_gb": float("nan"),
    }
    print(f"[*] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"] = props.name
        info["gpu_memory_gb"] = props.total_memory / (1024 ** 3)
        print(f"[*] Device: {props.name}")
        print(f"[*] CUDA runtime: {torch.version.cuda}")
        print(f"[*] GPU memory: {info['gpu_memory_gb']:.2f} GB")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        print("[!] CUDA not detected; training will use CPU.")
    return info


def save_predictions(
    bundle: PredictionBundle,
    path: Path,
    threshold: float,
    persistence: int,
    test_subjects: List[str],
    signature: str,
    epochs: int,
    batch_size: int,
) -> None:
    metadata_json = json.dumps(bundle.recording_metadata, separators=(",", ":"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probs=bundle.probs.astype(np.float32, copy=False),
            labels=bundle.labels.astype(np.uint8, copy=False),
            recording_ids=np.asarray(bundle.recording_ids, dtype=str),
            window_indices=bundle.window_indices.astype(np.int64, copy=False),
            threshold=np.asarray(float(threshold), dtype=np.float64),
            min_consecutive_windows=np.asarray(int(persistence), dtype=np.int16),
            validation_far_cap=np.asarray(
                float(VALIDATION_FA_PER_HOUR_CAP), dtype=np.float32
            ),
            test_patient=np.asarray("+".join(test_subjects), dtype=str),
            model_version=np.asarray(MODEL_VERSION, dtype=str),
            evaluation_version=np.asarray(EVALUATION_VERSION, dtype=str),
            results_schema_version=np.asarray(
                RESULTS_SCHEMA_VERSION, dtype=np.int16
            ),
            experiment_signature=np.asarray(signature, dtype=str),
            max_epochs=np.asarray(int(epochs), dtype=np.int16),
            batch_size=np.asarray(int(batch_size), dtype=np.int16),
            window_stride_sec=np.asarray(
                float(WINDOW_STRIDE_SEC), dtype=np.float32
            ),
            window_sec=np.asarray(float(WINDOW_SEC), dtype=np.float32),
            decision_time_reference=np.asarray(DECISION_TIME_REFERENCE, dtype=str),
            recording_metadata_json=np.asarray(metadata_json, dtype=str),
        )
    temporary.replace(path)


def atomic_torch_save(payload: Dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def selected_fold_indices(
    n_groups: int,
    max_folds: int | None,
    folds: Sequence[int] | None,
) -> List[int]:
    if folds is not None:
        selected = sorted({int(fold) for fold in folds})
        invalid = [fold for fold in selected if fold < 1 or fold > n_groups]
        if not selected or invalid:
            raise ValueError(f"Invalid fold selection for 1..{n_groups}: {selected}")
        return [fold - 1 for fold in selected]
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be >= 1")
        return list(range(min(int(max_folds), n_groups)))
    return list(range(n_groups))


def load_existing_results(summary_path: Path, expected_signature: str) -> Dict[int, Dict]:
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
    except Exception as exc:
        print(f"[warn] Could not read existing summary: {exc}")
        return {}
    required = {
        "fold",
        "model_version",
        "evaluation_version",
        "results_schema_version",
        "cache_version",
        "preprocessing_tag",
        "experiment_signature",
    }
    if not required.issubset(df.columns):
        return {}
    df = df[
        (df["model_version"].astype(str) == MODEL_VERSION)
        & (df["evaluation_version"].astype(str) == EVALUATION_VERSION)
        & (df["results_schema_version"].astype(int) == RESULTS_SCHEMA_VERSION)
        & (df["cache_version"].astype(int) == CACHE_VERSION)
        & (df["preprocessing_tag"].astype(str) == PREPROCESSING_TAG)
        & (df["experiment_signature"].astype(str) == expected_signature)
    ]
    rows: Dict[int, Dict] = {}
    for row in df.to_dict(orient="records"):
        try:
            rows[int(row["fold"])] = row
        except (TypeError, ValueError, KeyError):
            pass
    return rows


def write_results(summary_path: Path, rows_by_fold: Dict[int, Dict]) -> pd.DataFrame:
    if not rows_by_fold:
        return pd.DataFrame()
    df = pd.DataFrame([rows_by_fold[key] for key in sorted(rows_by_fold)])
    atomic_dataframe_csv(df, summary_path)
    return df


def run_lopo(
    max_folds: int | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    folds: Sequence[int] | None = None,
) -> None:
    seed_everything(RANDOM_SEED)
    hw = hardware_summary()
    signature = experiment_signature(epochs=epochs, batch_size=batch_size)

    cache_paths = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    if not cache_paths:
        raise FileNotFoundError(f"No temporal caches found in {PROCESSED_DATA_DIR}")

    cache_by_subject: Dict[str, Dict] = {}
    print(f"[*] Loading {len(cache_paths)} patient caches...")
    for path in cache_paths:
        cache = load_temporal_cache(path)
        cache_by_subject[str(cache["subject"])] = cache

    groups = subject_groups(sorted(cache_by_subject))
    if len(groups) < 3:
        raise RuntimeError("LOPO requires at least three independent patient groups")

    validation_size = min(4, len(groups) - 2)
    fold_indices = selected_fold_indices(len(groups), max_folds, folds)
    summary_path = RESULTS_DIR / "lopo_results_summary.csv"
    results_by_fold = load_existing_results(summary_path, signature)

    print(f"[*] Independent patient groups: {len(groups)}")
    print(f"[*] Selected folds: {[idx + 1 for idx in fold_indices]}")
    print(f"[*] Validation groups per fold: {validation_size}")
    print(f"[*] Validation FA/h cap: {VALIDATION_FA_PER_HOUR_CAP:.3f}")
    print(f"[*] Experiment signature: {signature}")
    print(f"[*] Existing completed folds: {sorted(results_by_fold)}")

    for fold_idx in fold_indices:
        test_group = groups[fold_idx]
        validation_indices = [
            (fold_idx + offset) % len(groups)
            for offset in range(1, validation_size + 1)
        ]
        excluded_indices = {fold_idx, *validation_indices}
        train_groups = [group for i, group in enumerate(groups) if i not in excluded_indices]
        val_groups = [groups[i] for i in validation_indices]

        train_subjects = [subject for group in train_groups for subject in group]
        val_subjects = [subject for group in val_groups for subject in group]
        test_subjects = list(test_group)
        fold_name = "+".join(test_subjects)
        fold_number = fold_idx + 1

        print("\n" + "=" * 88)
        print(
            f"FOLD {fold_number:02d}/{len(groups):02d} | "
            f"TEST={test_subjects} | VAL={val_subjects}"
        )
        print("=" * 88)

        train_caches = caches_for_subjects(cache_by_subject, train_subjects)
        val_caches = caches_for_subjects(cache_by_subject, val_subjects)
        test_caches = caches_for_subjects(cache_by_subject, test_subjects)
        mean, std = compute_fold_normalization(train_caches)

        train_ds = TemporalClipDataset(train_caches, mean, std, training=True)
        val_quick_ds = TemporalClipDataset(
            val_caches, mean, std, training=False, eval_stride=SEQUENCE_LENGTH
        )
        val_ds = TemporalClipDataset(val_caches, mean, std, training=False)
        test_ds = TemporalClipDataset(test_caches, mean, std, training=False)
        if len(train_ds) == 0:
            raise RuntimeError("Training dataset is empty")

        print(
            f"clips: train={len(train_ds):,} "
            f"(important={len(train_ds.important_refs):,}, negative_pool={len(train_ds.negative_pool):,}) | "
            f"val-fast={len(val_quick_ds):,} | val={len(val_ds):,} | test={len(test_ds):,}"
        )

        train_loader = make_loader(train_ds, shuffle=True, batch_size=batch_size)
        val_quick_loader = make_loader(val_quick_ds, shuffle=False, batch_size=batch_size)
        val_loader = make_loader(val_ds, shuffle=False, batch_size=batch_size)
        test_loader = make_loader(test_ds, shuffle=False, batch_size=batch_size)

        model = DynaGATOnsetModel(
            graph_hidden=GRAPH_HIDDEN,
            tcn_hidden=TCN_HIDDEN,
            heads=GAT_HEADS,
            dropout=DROPOUT,
        ).to(DEVICE)
        parameter_count = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=LEARNING_RATE * 0.1
        )
        criterion = BoundaryAwareFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        fold_start = time.perf_counter()
        best_state = None
        best_score = -float("inf")
        best_epoch = 0
        checks_without_improvement = 0
        history: List[Dict] = []
        epochs_ran = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, epochs)
            scheduler.step()
            epoch_sec = time.perf_counter() - epoch_start
            current_lr = float(scheduler.get_last_lr()[0])
            epochs_ran = epoch
            peak_vram = float("nan")
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats()

            row = {
                "epoch": epoch,
                "train_loss": loss,
                "lr": current_lr,
                "epoch_sec": epoch_sec,
                "peak_vram_gb": peak_vram,
                "val_auroc": float("nan"),
                "val_auprc": float("nan"),
                "is_best": 0,
            }
            should_validate = epoch % VALIDATION_CHECK_INTERVAL == 0 or epoch == epochs
            if should_validate:
                quick_pred = predict(model, val_quick_loader)
                quick_metrics = compute_window_metrics(
                    quick_pred.labels, quick_pred.probs, threshold=0.5
                )
                score = validation_score(quick_metrics)
                row["val_auroc"] = quick_metrics["auroc"]
                row["val_auprc"] = quick_metrics["auprc"]
                if score > best_score + 1e-5:
                    best_score = score
                    best_epoch = epoch
                    best_state = cpu_state_dict(model)
                    checks_without_improvement = 0
                    row["is_best"] = 1
                else:
                    checks_without_improvement += 1
                print(
                    f"epoch {epoch:02d}/{epochs:02d} | loss={loss:.5f} | "
                    f"val AUPRC={quick_metrics['auprc']:.4f} | best={best_score:.4f}@{best_epoch:02d} | "
                    f"time={epoch_sec:.1f}s | lr={current_lr:.2e}" +
                    (f" | peakVRAM={peak_vram:.2f}GB" if np.isfinite(peak_vram) else "")
                )
                del quick_pred
                if (
                    epoch >= MIN_EPOCHS_BEFORE_STOPPING
                    and checks_without_improvement >= EARLY_STOPPING_PATIENCE
                ):
                    history.append(row)
                    print(f"[*] Early stopping at epoch {epoch}; restore epoch {best_epoch}.")
                    break
            else:
                print(
                    f"epoch {epoch:02d}/{epochs:02d} | loss={loss:.5f} | time={epoch_sec:.1f}s | "
                    f"lr={current_lr:.2e}" +
                    (f" | peakVRAM={peak_vram:.2f}GB" if np.isfinite(peak_vram) else "")
                )
            history.append(row)

        if best_state is None:
            best_state = cpu_state_dict(model)
            best_epoch = epochs_ran
        model.load_state_dict(best_state)
        del best_state

        atomic_dataframe_csv(
            pd.DataFrame(history),
            RESULTS_DIR / f"fold_{fold_number:02d}_training_history.csv",
        )

        val_pred = predict(model, val_loader)
        frontier_path = RESULTS_DIR / f"fold_{fold_number:02d}_validation_alarm_frontier.csv"
        operating = select_validation_operating_point(
            labels=val_pred.labels,
            probs=val_pred.probs,
            recording_ids=val_pred.recording_ids,
            window_indices=val_pred.window_indices,
            recording_metadata=val_pred.recording_metadata,
            frontier_path=frontier_path,
        )
        threshold = operating.threshold
        persistence = operating.min_consecutive_windows
        val_window = compute_window_metrics(val_pred.labels, val_pred.probs, threshold)
        val_event = compute_event_metrics(
            probs=val_pred.probs,
            recording_ids=val_pred.recording_ids,
            window_indices=val_pred.window_indices,
            recording_metadata=val_pred.recording_metadata,
            threshold=threshold,
            min_consecutive_windows=persistence,
        )

        test_pred = predict(model, test_loader)
        test_window = compute_window_metrics(test_pred.labels, test_pred.probs, threshold)
        event = compute_event_metrics(
            probs=test_pred.probs,
            recording_ids=test_pred.recording_ids,
            window_indices=test_pred.window_indices,
            recording_metadata=test_pred.recording_metadata,
            threshold=threshold,
            min_consecutive_windows=persistence,
        )
        ece = compute_ece(test_pred.probs, test_pred.labels, n_bins=10)
        elapsed = time.perf_counter() - fold_start

        print(
            f"VAL operating point: threshold={threshold:.4f} | persistence={persistence} | "
            f"sens={val_event['event_sensitivity']:.4f} | FA/h={val_event['fa_per_hour']:.3f} | "
            f"event-F1={val_event['event_f1']:.4f}"
        )
        print(
            f"TEST {fold_name}: AUROC={test_window['auroc']:.4f} | "
            f"AUPRC={test_window['auprc']:.4f} | F1={test_window['f1']:.4f}"
        )
        print(
            f"Event sensitivity={event['event_sensitivity']:.4f} "
            f"({event['detected_seizures']}/{event['total_gt_seizures']}) | "
            f"precision={event['event_precision']:.4f} | event-F1={event['event_f1']:.4f} | "
            f"FA/h={event['fa_per_hour']:.3f} | median latency={event['median_latency_sec']:.2f}s | "
            f"ECE={ece:.4f}"
        )

        checkpoint_path = RESULTS_DIR / f"dynagat_fold_{fold_number:02d}_{fold_name}.pt"
        atomic_torch_save(
            {
                "model_version": MODEL_VERSION,
                "evaluation_version": EVALUATION_VERSION,
                "results_schema_version": RESULTS_SCHEMA_VERSION,
                "cache_version": CACHE_VERSION,
                "preprocessing_tag": PREPROCESSING_TAG,
                "experiment_signature": signature,
                "max_epochs": int(epochs),
                "batch_size": int(batch_size),
                "decision_time_reference": DECISION_TIME_REFERENCE,
                "model_state_dict": model.state_dict(),
                "test_subjects": test_subjects,
                "validation_subjects": val_subjects,
                "feature_mean": mean,
                "feature_std": std,
                "validation_threshold": threshold,
                "validation_min_consecutive_windows": persistence,
                "validation_far_cap": VALIDATION_FA_PER_HOUR_CAP,
                "alarm_objective": ALARM_OBJECTIVE,
                "best_epoch": best_epoch,
                "best_quick_val_auprc": best_score,
                "parameter_count": parameter_count,
                "random_seed": RANDOM_SEED,
            },
            checkpoint_path,
        )
        save_predictions(
            test_pred,
            RESULTS_DIR / f"fold_{fold_number:02d}_test_predictions.npz",
            threshold,
            persistence,
            test_subjects,
            signature,
            epochs,
            batch_size,
        )

        result_row = {
            "fold": fold_number,
            "evaluation_role": "development" if fold_number == DEVELOPMENT_FOLD else "primary",
            "model_version": MODEL_VERSION,
            "evaluation_version": EVALUATION_VERSION,
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "cache_version": CACHE_VERSION,
            "preprocessing_tag": PREPROCESSING_TAG,
            "experiment_signature": signature,
            "max_epochs": int(epochs),
            "batch_size": int(batch_size),
            "decision_time_reference": DECISION_TIME_REFERENCE,
            "alarm_objective": ALARM_OBJECTIVE,
            "test_patient": fold_name,
            "validation_patient": "+".join(val_subjects),
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "best_quick_val_auprc": best_score,
            "threshold": threshold,
            "min_consecutive_windows": persistence,
            "validation_far_cap": VALIDATION_FA_PER_HOUR_CAP,
            "val_auroc": val_window["auroc"],
            "val_auprc": val_window["auprc"],
            "val_event_sensitivity": val_event["event_sensitivity"],
            "val_event_precision": val_event["event_precision"],
            "val_event_f1": val_event["event_f1"],
            "val_fa_per_hour": val_event["fa_per_hour"],
            "auroc": test_window["auroc"],
            "auprc": test_window["auprc"],
            "f1": test_window["f1"],
            "test_positive_fraction": float(np.mean(test_pred.labels)) if test_pred.labels.size else float("nan"),
            "gt_seizures": event["total_gt_seizures"],
            "detected_seizures": event["detected_seizures"],
            "event_sensitivity": event["event_sensitivity"],
            "event_precision": event["event_precision"],
            "event_f1": event["event_f1"],
            "false_alarms": event["false_alarms"],
            "recording_hours": event["recording_hours"],
            "interictal_hours": event["interictal_hours"],
            "fa_per_hour": event["fa_per_hour"],
            "median_latency_sec": event["median_latency_sec"],
            "ece": ece,
            "elapsed_sec": elapsed,
            "parameter_count": parameter_count,
            "trainable_parameters": trainable_parameters,
            "device": hw["device"],
            "gpu_memory_gb": hw["gpu_memory_gb"],
            "torch_version": hw["torch_version"],
            "cuda_version": hw["cuda_version"],
            "random_seed": RANDOM_SEED,
        }
        results_by_fold[fold_number] = result_row
        write_results(summary_path, results_by_fold)

        del model, optimizer, scheduler, train_loader, val_quick_loader, val_loader, test_loader
        del train_ds, val_quick_ds, val_ds, test_ds, val_pred, test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = write_results(summary_path, results_by_fold)
    print("\n" + "=" * 88)
    print("LOPO COMPLETE")
    print("=" * 88)
    if df.empty:
        print("No results available.")
        return
    print(df.to_string(index=False))
    print("-" * 88)
    primary = df[df.get("evaluation_role", "primary") == "primary"] if "evaluation_role" in df else df
    if primary.empty:
        primary = df
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
        if column in primary.columns:
            print(f"primary mean {column:18s}: {primary[column].mean(skipna=True):.4f}")
    print(f"[+] Results: {summary_path}")
