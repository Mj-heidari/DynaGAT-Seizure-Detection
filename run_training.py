from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import BATCH_SIZE, EPOCHS, PROCESSED_DATA_DIR, RESULTS_DIR
from training.trainer import subject_groups
from training.trainer_v5 import MODEL_VERSION, run_lopo_v5


def parse_folds(spec: str) -> list[int]:
    """Parse comma-separated fold numbers/ranges, e.g. '2-5,8,10-12'."""
    folds: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid fold range: {token}") from exc
            if start < 1 or end < 1 or end < start:
                raise argparse.ArgumentTypeError(f"Invalid fold range: {token}")
            folds.update(range(start, end + 1))
        else:
            try:
                value = int(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid fold number: {token}") from exc
            if value < 1:
                raise argparse.ArgumentTypeError("Fold numbers must be >= 1")
            folds.add(value)
    if not folds:
        raise argparse.ArgumentTypeError("--folds must select at least one fold")
    return sorted(folds)


def remaining_folds() -> list[int]:
    """Return v5 LOPO folds that do not yet have a completed summary row.

    This makes long unattended runs resume-safe at fold granularity. Each fold is
    written to lopo_results_summary.csv immediately after completion, so rerunning
    with --remaining skips already completed folds and continues with the rest.
    """
    cache_paths = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    if not cache_paths:
        raise FileNotFoundError(
            f"No temporal caches found in {PROCESSED_DATA_DIR}. Run preprocessing first."
        )

    subjects = [path.name.removesuffix("_temporal_graphs.pt") for path in cache_paths]
    n_folds = len(subject_groups(sorted(subjects)))
    all_folds = set(range(1, n_folds + 1))

    summary_path = RESULTS_DIR / "lopo_results_summary.csv"
    completed: set[int] = set()
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            if "fold" in df.columns:
                if "model_version" in df.columns:
                    df = df[df["model_version"].astype(str) == MODEL_VERSION]
                else:
                    df = df.iloc[0:0]
                completed = {
                    int(value)
                    for value in df["fold"].dropna().tolist()
                    if 1 <= int(value) <= n_folds
                }
        except Exception as exc:
            print(f"[warn] Could not read existing LOPO summary: {exc}")

    remaining = sorted(all_folds.difference(completed))
    print(f"[*] Total independent LOPO folds: {n_folds}")
    print(f"[*] Completed v5 folds: {sorted(completed)}")
    print(f"[*] Remaining v5 folds: {remaining}")
    return remaining


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train frozen causal DynaGAT v5 with patient-independent LOPO"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N held-out folds (mainly for smoke tests)",
    )
    selection.add_argument(
        "--folds",
        type=parse_folds,
        default=None,
        metavar="SPEC",
        help="Run explicit 1-based folds, e.g. 2-5 or 2,4,7-9. Existing v5 summary rows are preserved.",
    )
    selection.add_argument(
        "--remaining",
        action="store_true",
        help="Run every v5 LOPO fold not already present in lopo_results_summary.csv; safe to rerun after interruption.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Maximum epochs per fold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Temporal clips per GPU batch",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be >= 1")

    folds = args.folds
    if args.remaining:
        folds = remaining_folds()
        if not folds:
            print("[+] All v5 LOPO folds are already complete. Nothing to run.")
            raise SystemExit(0)

    run_lopo_v5(
        max_folds=args.max_folds,
        folds=folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
