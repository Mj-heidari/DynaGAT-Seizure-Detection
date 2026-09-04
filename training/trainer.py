"""
Leave-one-patient-out fold runner for DynaGAT.

Protocol per fold
-----------------
  test        : one held-out patient, never seen in any form
  validation  : NUM_VALIDATION_PATIENTS patients, used for checkpoint selection,
                calibration constants and the alarm operating point
  training    : all remaining patients

Everything that touches the test patient happens exactly once, at the end, with
frozen weights, a frozen calibration and a frozen operating point.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    ADAPTIVE_MIX,
    ADAPTIVE_NORM,
    APPLY_PRIOR_CORRECTION,
    BAG_LOSS_WEIGHT,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EVAL_SEQUENCE_STRIDE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MIN_EPOCHS_BEFORE_STOPPING,
    NUM_VALIDATION_PATIENTS,
    ONSET_AUX_WEIGHT,
    POS_WEIGHT,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SEQUENCE_LENGTH,
    VALIDATION_CHECK_INTERVAL,
    VALIDATION_FA_PER_HOUR_CAP,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    experiment_signature,
)
from dataset.sequence_dataset import (
    TemporalClipDataset,
    collate,
    compute_fold_normalization,
    load_cache,
)
from evaluation.events import evaluate_events, window_metrics
from evaluation.operating_point import select_operating_point
from models.dynagat import DynaGAT
from training.calibration import OnlineScorer, prior_correction_offset
from training.losses import DynaGATLoss

__all__ = ["list_subjects", "make_folds", "run_fold", "FoldConfig"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def list_subjects(cache_dir: Path = PROCESSED_DATA_DIR) -> List[str]:
    return sorted(p.name.replace("_v4.pt", "") for p in cache_dir.glob("sub-*_v4.pt"))


def make_folds(subjects: Sequence[str], n_val: int = NUM_VALIDATION_PATIENTS) -> List[Dict]:
    """Deterministic LOPO folds; validation patients rotate through the cohort."""
    subs = list(subjects)
    n = len(subs)
    if n < n_val + 2:
        raise ValueError(f"need at least {n_val + 2} subjects, got {n}")
    folds = []
    for i, test in enumerate(subs):
        val = [subs[(i + 1 + j) % n] for j in range(n_val)]
        train = [s for s in subs if s != test and s not in val]
        folds.append(
            {"fold": i + 1, "test": test, "validation": val, "train": train}
        )
    return folds


@dataclass
class FoldConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_workers: int = 0
    amp: bool = True
    use_static: bool = True
    use_causal: bool = True
    causal_direction: str = "both"
    graph_mode: str = "graph"
    adaptive_mix: float = ADAPTIVE_MIX
    adaptive_norm: bool = ADAPTIVE_NORM
    prior_correction: bool = APPLY_PRIOR_CORRECTION
    tag: str = "dynagat"
    seed: int = RANDOM_SEED


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_patients(
    model: torch.nn.Module,
    caches: Sequence[Dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    num_workers: int = 0,
    amp: bool = True,
    eval_stride: int = EVAL_SEQUENCE_STRIDE,
) -> Dict[str, Dict[str, Dict]]:
    """
    Full causal inference over every window of every recording.

    A window can appear in several overlapping clips. We keep the occurrence
    with the *most* causal context (largest index inside its clip), which is
    what an online detector would have available at that moment.
    """
    model.eval()
    ds = TemporalClipDataset(
        caches, mean, std, training=False, eval_stride=eval_stride
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=(DEVICE.type == "cuda"),
    )

    store: Dict[str, Dict[str, np.ndarray]] = {}
    for rid, meta in ds.recording_metadata.items():
        n = int(meta["n_windows"])
        store[rid] = {
            "logit": np.full(n, -1e9, dtype=np.float32),
            "context": np.full(n, -1, dtype=np.int32),
        }

    autocast = torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and DEVICE.type == "cuda")
    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)
        with autocast:
            logits = model(
                x,
                batch["in_dst"].to(DEVICE, non_blocking=True),
                batch["in_weight"].to(DEVICE, non_blocking=True),
                batch["out_dst"].to(DEVICE, non_blocking=True),
                batch["out_weight"].to(DEVICE, non_blocking=True),
                batch["valid_mask"].to(DEVICE, non_blocking=True),
            )
        logits = logits.float().cpu().numpy()
        widx = batch["window_idx"].numpy()
        valid = batch["valid_mask"].numpy()
        ctx = np.arange(widx.shape[1], dtype=np.int32)[None, :]
        for b, rid in enumerate(batch["recording_id"]):
            sel = valid[b]
            if not sel.any():
                continue
            idx = widx[b][sel]
            better = ctx[0][sel] > store[rid]["context"][idx]
            target = idx[better]
            store[rid]["logit"][target] = logits[b][sel][better]
            store[rid]["context"][target] = ctx[0][sel][better]

    out: Dict[str, Dict[str, Dict]] = {}
    for cache in caches:
        subject = str(cache["subject"])
        out[subject] = {}
        for rec in cache["recordings"]:
            rid = str(rec["recording_id"])
            n = int(rec["n_windows"])
            logit = store[rid]["logit"]
            unresolved = store[rid]["context"] < 0
            if unresolved.any():                      # should not happen
                logit = logit.copy()
                logit[unresolved] = float(np.median(logit[~unresolved])) if (~unresolved).any() else 0.0
            out[subject][rid] = {
                "logit": logit.astype(np.float32),
                "labels": rec["labels"].numpy().astype(np.int64),
                "duration_sec": float(rec["duration_sec"]),
                "seizure_intervals": [tuple(map(float, s)) for s in rec.get("seizure_intervals", [])],
                "n_windows": n,
            }
    return out


def _apply_scorer(pred: Dict[str, Dict[str, Dict]], scorer: OnlineScorer) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for patient, recs in pred.items():
        out[patient] = {}
        for rid, r in recs.items():
            out[patient][rid] = {
                "score": scorer.score_recording(r["logit"]),
                "prob": scorer.probabilities(r["logit"]),
                "labels": r["labels"],
                "duration_sec": r["duration_sec"],
                "seizure_intervals": r["seizure_intervals"],
            }
    return out


def _flatten(pred: Dict[str, Dict[str, Dict]], key: str) -> np.ndarray:
    parts = [r[key] for recs in pred.values() for r in recs.values()]
    return np.concatenate(parts) if parts else np.zeros(0)


# --------------------------------------------------------------------------- #
# Quick validation used only for checkpoint selection
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def _quick_validation(model, loader, amp: bool) -> float:
    model.eval()
    autocast = torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and DEVICE.type == "cuda")
    logits, labels = [], []
    for batch in loader:
        with autocast:
            out = model(
                batch["x"].to(DEVICE, non_blocking=True),
                batch["in_dst"].to(DEVICE, non_blocking=True),
                batch["in_weight"].to(DEVICE, non_blocking=True),
                batch["out_dst"].to(DEVICE, non_blocking=True),
                batch["out_weight"].to(DEVICE, non_blocking=True),
                batch["valid_mask"].to(DEVICE, non_blocking=True),
            )
        v = batch["valid_mask"].numpy().ravel()
        logits.append(out.float().cpu().numpy().ravel()[v])
        labels.append(batch["labels"].numpy().ravel()[v])
    y = np.concatenate(labels).astype(np.int64)
    s = np.concatenate(logits)
    return window_metrics(y, s)["auprc"]


# --------------------------------------------------------------------------- #
# Fold runner
# --------------------------------------------------------------------------- #
def run_fold(fold: Dict, cfg: FoldConfig, cache_dir: Path = PROCESSED_DATA_DIR,
             results_dir: Path = RESULTS_DIR) -> Dict:
    t_start = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    fid = int(fold["fold"])
    test_sub, val_subs, train_subs = fold["test"], list(fold["validation"]), list(fold["train"])
    print(f"\n{'='*78}\n[fold {fid:02d}] test={test_sub}  val={'+'.join(val_subs)}  "
          f"train={len(train_subs)} patients\n{'='*78}")

    train_caches = [load_cache(cache_dir / f"{s}_v4.pt") for s in train_subs]
    val_caches = [load_cache(cache_dir / f"{s}_v4.pt") for s in val_subs]
    test_caches = [load_cache(cache_dir / f"{test_sub}_v4.pt")]

    mean, std = compute_fold_normalization(train_caches)

    train_ds = TemporalClipDataset(train_caches, mean, std, training=True, seed=cfg.seed)
    quick_val_ds = TemporalClipDataset(
        val_caches, mean, std, training=True, seed=cfg.seed + 1,
        negative_ratio=4, min_negative_clips=512,
    )
    # Freeze the checkpoint-selection subset for the whole fold. It must not be
    # resampled between checks or the per-epoch scores stop being comparable.
    quick_val_ds.set_epoch(0)
    print(f"[fold {fid:02d}] train clips={len(train_ds):,} "
          f"| fixed quick-val clips={len(quick_val_ds):,}  "
          f"(important={len(train_ds.important_refs):,})  "
          f"true prior={train_ds.true_positive_prior:.5f}  "
          f"sampled prior={train_ds.sampled_positive_prior:.5f}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=(DEVICE.type == "cuda"),
    )
    quick_loader = DataLoader(
        quick_val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DynaGAT(
        use_static=cfg.use_static,
        use_causal=cfg.use_causal,
        causal_direction=cfg.causal_direction,
        graph_mode=cfg.graph_mode,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = DynaGATLoss(POS_WEIGHT, LABEL_SMOOTHING, BAG_LOSS_WEIGHT, ONSET_AUX_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and DEVICE.type == "cuda")
    autocast = torch.amp.autocast("cuda", dtype=torch.float16,
                                  enabled=cfg.amp and DEVICE.type == "cuda")

    history: List[Dict] = []
    best_score = -np.inf
    best_state = None
    best_epoch = 0
    bad_checks = 0
    sampled_prior_at_best = train_ds.sampled_positive_prior

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)
        ep_start = time.perf_counter()
        running = 0.0
        parts_acc = {"window": 0.0, "bag": 0.0, "onset": 0.0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits, onset_logits, _ = model(
                    batch["x"].to(DEVICE, non_blocking=True),
                    batch["in_dst"].to(DEVICE, non_blocking=True),
                    batch["in_weight"].to(DEVICE, non_blocking=True),
                    batch["out_dst"].to(DEVICE, non_blocking=True),
                    batch["out_weight"].to(DEVICE, non_blocking=True),
                    batch["valid_mask"].to(DEVICE, non_blocking=True),
                    return_aux=True,
                )
                loss, parts = criterion(
                    logits.float(), onset_logits.float(),
                    batch["labels"].to(DEVICE, non_blocking=True),
                    batch["boundary_weights"].to(DEVICE, non_blocking=True),
                    batch["valid_mask"].to(DEVICE, non_blocking=True),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            for k in parts_acc:
                parts_acc[k] += parts[k]

        n_steps = max(1, len(train_loader))
        row = {
            "epoch": epoch,
            "train_loss": running / n_steps,
            "loss_window": parts_acc["window"] / n_steps,
            "loss_bag": parts_acc["bag"] / n_steps,
            "loss_onset": parts_acc["onset"] / n_steps,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": time.perf_counter() - ep_start,
            "peak_vram_gb": (
                torch.cuda.max_memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0.0
            ),
            "quick_val_auprc": np.nan,
            "is_best": 0,
        }

        due = (epoch % VALIDATION_CHECK_INTERVAL == 0) or (epoch == cfg.epochs)
        if due:
            # The validation subset is fixed once, before training (set_epoch(0)
            # in the construction block above). Resampling it here would change
            # the measurement set at every check, so the AUPRC values would not
            # be comparable across epochs: an early favourable sample sets a bar
            # later epochs cannot beat, early stopping trips, and an
            # under-trained checkpoint is kept. That is exactly what happened in
            # run 1 (best epoch 2-4 in 14 of 22 folds).
            auprc = _quick_validation(model, quick_loader, cfg.amp)
            row["quick_val_auprc"] = auprc
            if np.isfinite(auprc) and auprc > best_score:
                best_score = auprc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                sampled_prior_at_best = train_ds.sampled_positive_prior
                row["is_best"] = 1
                bad_checks = 0
            else:
                bad_checks += 1
        history.append(row)
        print(
            f"  epoch {epoch:02d}/{cfg.epochs}  loss {row['train_loss']:.4f}"
            f"  ({row['epoch_sec']:.1f}s)"
            + (f"  quick-val AUPRC {row['quick_val_auprc']:.4f}"
               f"{'  *' if row['is_best'] else ''}" if due else "")
        )
        if epoch >= MIN_EPOCHS_BEFORE_STOPPING and bad_checks >= EARLY_STOPPING_PATIENCE:
            print(f"  early stop at epoch {epoch} (best {best_epoch}, AUPRC {best_score:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- validation pass: calibration + operating point -------- #
    print(f"[fold {fid:02d}] full validation pass ...")
    val_pred = predict_patients(model, val_caches, mean, std, cfg.batch_size,
                                cfg.num_workers, cfg.amp)
    prior_offset = (
        prior_correction_offset(sampled_prior_at_best, train_ds.true_positive_prior)
        if cfg.prior_correction else 0.0
    )
    val_logits_all = _flatten(val_pred, "logit") + prior_offset
    scorer = OnlineScorer(
        prior_offset=prior_offset,
        logit_mean=float(np.mean(val_logits_all)),
        logit_std=float(np.std(val_logits_all) + 1e-6),
        mix=cfg.adaptive_mix,
        enabled=cfg.adaptive_norm,
    )
    val_scored = _apply_scorer(val_pred, scorer)
    op, frontier = select_operating_point(val_scored, fa_cap=VALIDATION_FA_PER_HOUR_CAP)
    print(f"[fold {fid:02d}] operating point: thr={op.threshold:.4f} k={op.k}/m={op.m} "
          f"| val mean sens {op.val_mean_sensitivity:.3f} "
          f"median FA/h {op.val_median_fa_per_hour:.3f}"
          + (f" [{op.fallback_used}]" if op.fallback_used else ""))

    val_y = _flatten(val_scored, "labels").astype(np.int64)
    val_s = _flatten(val_scored, "score")
    val_p = _flatten(val_scored, "prob")
    val_win = window_metrics(val_y, val_s, val_p)
    val_ev = evaluate_events(
        {rid: r for recs in val_scored.values() for rid, r in recs.items()},
        op.threshold, op.k, op.m,
    )

    # ---------------- test pass -------------------------------------------- #
    print(f"[fold {fid:02d}] test pass on {test_sub} ...")
    test_pred = predict_patients(model, test_caches, mean, std, cfg.batch_size,
                                 cfg.num_workers, cfg.amp)
    test_scored = _apply_scorer(test_pred, scorer)
    test_y = _flatten(test_scored, "labels").astype(np.int64)
    test_s = _flatten(test_scored, "score")
    test_p = _flatten(test_scored, "prob")
    test_win = window_metrics(test_y, test_s, test_p)
    test_recs = {rid: r for recs in test_scored.values() for rid, r in recs.items()}
    test_ev = evaluate_events(test_recs, op.threshold, op.k, op.m)
    from config import SECONDARY_EARLY_TOLERANCE_SEC
    test_ev_tol = evaluate_events(
        test_recs, op.threshold, op.k, op.m,
        early_tolerance=SECONDARY_EARLY_TOLERANCE_SEC,
    )

    elapsed = time.perf_counter() - t_start
    print(
        f"[fold {fid:02d}] TEST {test_sub}: sens {test_ev.sensitivity:.3f} "
        f"({test_ev.detected_seizures}/{test_ev.gt_seizures})  "
        f"FA/h {test_ev.fa_per_hour:.3f}  AUROC {test_win['auroc']:.4f}  "
        f"AUPRC {test_win['auprc']:.4f}  latency {test_ev.median_latency:.1f}s  "
        f"[{elapsed/60:.1f} min]"
    )

    # ---------------- artifacts -------------------------------------------- #
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag
    pd.DataFrame(history).to_csv(results_dir / f"{tag}_fold_{fid:02d}_history.csv", index=False)
    frontier.to_csv(results_dir / f"{tag}_fold_{fid:02d}_val_frontier.csv", index=False)
    np.savez_compressed(
        results_dir / f"{tag}_fold_{fid:02d}_test_predictions.npz",
        **{
            f"{rid}::score": r["score"] for rid, r in test_recs.items()
        },
        **{
            f"{rid}::labels": r["labels"] for rid, r in test_recs.items()
        },
        **{
            f"{rid}::prob": r["prob"] for rid, r in test_recs.items()
        },
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "scorer": scorer.to_dict(),
            "operating_point": op.as_dict(),
            "fold": fold,
            "config": cfg.__dict__,
            "signature": experiment_signature(),
        },
        results_dir / f"{tag}_fold_{fid:02d}_{test_sub}.pt",
    )

    summary = {
        "fold": fid,
        "tag": tag,
        "test_patient": test_sub,
        "validation_patients": "+".join(val_subs),
        "n_train_patients": len(train_subs),
        "signature": experiment_signature(),
        "evaluation_role": "development" if fid == 1 else "primary",
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_quick_val_auprc": float(best_score) if np.isfinite(best_score) else float("nan"),
        "train_true_prior": float(train_ds.true_positive_prior),
        "train_sampled_prior": float(sampled_prior_at_best),
        "prior_offset": float(prior_offset),
        "scorer_logit_mean": float(scorer.logit_mean),
        "scorer_logit_std": float(scorer.logit_std),
        "adaptive_mix": float(scorer.mix),
        "threshold": float(op.threshold),
        "persistence_k": int(op.k),
        "persistence_m": int(op.m),
        "op_admissible": bool(op.admissible),
        "op_fallback": op.fallback_used,
        "val_auroc": val_win["auroc"],
        "val_auprc": val_win["auprc"],
        "val_ece": val_win.get("ece", float("nan")),
        **{f"val_{k}": v for k, v in val_ev.as_dict().items()},
        "auroc": test_win["auroc"],
        "auprc": test_win["auprc"],
        "ece": test_win.get("ece", float("nan")),
        "brier": test_win.get("brier", float("nan")),
        "test_positive_fraction": test_win["positive_fraction"],
        "n_test_windows": test_win["n_windows"],
        **test_ev.as_dict(),
        **test_ev_tol.as_dict(prefix="tol10_"),
        "elapsed_sec": elapsed,
        "parameter_count": int(n_params),
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    return summary
