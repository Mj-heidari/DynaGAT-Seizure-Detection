from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from config import BATCH_SIZE, DROPOUT, GAT_HEADS, GRAPH_HIDDEN, PROCESSED_DATA_DIR, RESULTS_DIR, TCN_HIDDEN
from dataset.sequence_dataset import TemporalClipDataset, load_temporal_cache
from evaluation.metrics import compute_ece, compute_event_metrics, compute_window_metrics
from evaluation.operating_point import select_validation_operating_point
from models.dynagat_model import DynaGATOnsetModel
from training.trainer import make_loader, predict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(checkpoint_path: Path, batch_size: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    test_subjects = [str(x) for x in checkpoint["test_subjects"]]
    val_subjects = [str(x) for x in checkpoint["validation_subjects"]]
    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()

    needed = set(test_subjects + val_subjects)
    caches = {}
    for path in sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt")):
        subject = path.name.replace("_temporal_graphs.pt", "")
        if subject in needed:
            cache = load_temporal_cache(path)
            caches[str(cache["subject"])] = cache

    missing = sorted(needed.difference(caches))
    if missing:
        raise FileNotFoundError(f"Missing v3 caches for: {missing}")

    val_ds = TemporalClipDataset([caches[s] for s in val_subjects], mean, std, training=False)
    test_ds = TemporalClipDataset([caches[s] for s in test_subjects], mean, std, training=False)
    val_loader = make_loader(val_ds, shuffle=False, batch_size=batch_size)
    test_loader = make_loader(test_ds, shuffle=False, batch_size=batch_size)

    model = DynaGATOnsetModel(
        graph_hidden=GRAPH_HIDDEN,
        tcn_hidden=TCN_HIDDEN,
        heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"[*] Re-evaluating legacy checkpoint: {checkpoint_path.name}")
    print(f"[*] Validation subjects: {val_subjects}")
    print(f"[*] Test subjects: {test_subjects}")
    print("[*] No training is performed. Test predictions do not influence selection.")

    val_pred = predict(model, val_loader)
    frontier_path = RESULTS_DIR / "legacy_fold_01_validation_alarm_frontier.csv"
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
    test_event = compute_event_metrics(
        probs=test_pred.probs,
        recording_ids=test_pred.recording_ids,
        window_indices=test_pred.window_indices,
        recording_metadata=test_pred.recording_metadata,
        threshold=threshold,
        min_consecutive_windows=persistence,
    )
    ece = compute_ece(test_pred.probs, test_pred.labels, n_bins=10)

    old_threshold = float(checkpoint.get("validation_threshold", float("nan")))
    print("\n" + "=" * 80)
    print("LEGACY CHECKPOINT RE-EVALUATION")
    print("=" * 80)
    print(f"Old threshold: {old_threshold:.4f} | old persistence: 3")
    print(
        f"New VAL operating point: threshold={threshold:.4f} | persistence={persistence} | "
        f"sens={val_event['event_sensitivity']:.4f} | precision={val_event['event_precision']:.4f} | "
        f"event-F1={val_event['event_f1']:.4f} | FA/h={val_event['fa_per_hour']:.3f}"
    )
    print(
        f"TEST {'+'.join(test_subjects)}: AUROC={test_window['auroc']:.4f} | "
        f"AUPRC={test_window['auprc']:.4f} | F1={test_window['f1']:.4f}"
    )
    print(
        f"Event sensitivity={test_event['event_sensitivity']:.4f} "
        f"({test_event['detected_seizures']}/{test_event['total_gt_seizures']}) | "
        f"precision={test_event['event_precision']:.4f} | event-F1={test_event['event_f1']:.4f} | "
        f"FA/h={test_event['fa_per_hour']:.3f} | "
        f"median latency={test_event['median_latency_sec']:.2f}s | ECE={ece:.4f}"
    )

    row = {
        "checkpoint": checkpoint_path.name,
        "test_patient": "+".join(test_subjects),
        "validation_patient": "+".join(val_subjects),
        "old_threshold": old_threshold,
        "threshold": threshold,
        "min_consecutive_windows": persistence,
        "val_auroc": val_window["auroc"],
        "val_auprc": val_window["auprc"],
        "val_event_sensitivity": val_event["event_sensitivity"],
        "val_event_precision": val_event["event_precision"],
        "val_event_f1": val_event["event_f1"],
        "val_fa_per_hour": val_event["fa_per_hour"],
        "auroc": test_window["auroc"],
        "auprc": test_window["auprc"],
        "f1": test_window["f1"],
        "gt_seizures": test_event["total_gt_seizures"],
        "detected_seizures": test_event["detected_seizures"],
        "event_sensitivity": test_event["event_sensitivity"],
        "event_precision": test_event["event_precision"],
        "event_f1": test_event["event_f1"],
        "false_alarms": test_event["false_alarms"],
        "fa_per_hour": test_event["fa_per_hour"],
        "median_latency_sec": test_event["median_latency_sec"],
        "ece": ece,
    }
    out_path = RESULTS_DIR / "legacy_checkpoint_reevaluation.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[+] Frontier: {frontier_path}")
    print(f"[+] Summary: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-evaluate a legacy DynaGAT checkpoint with the new validation-only alarm policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RESULTS_DIR / "dynagat_onset_fold_01_sub-01.pt",
        help="Legacy checkpoint to re-evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    main(args.checkpoint, args.batch_size)
