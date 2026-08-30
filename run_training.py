from __future__ import annotations

import argparse

from config import BATCH_SIZE, EPOCHS
from training.trainer_v5 import run_lopo_v5


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
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
