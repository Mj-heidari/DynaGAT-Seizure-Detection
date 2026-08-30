from __future__ import annotations

import argparse

from config import BATCH_SIZE, EPOCHS
from training.trainer_v5 import run_lopo_v5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train causal DynaGAT v5 with patient-independent LOPO"
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N held-out folds (useful for a smoke test)",
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

    run_lopo_v5(
        max_folds=args.max_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
