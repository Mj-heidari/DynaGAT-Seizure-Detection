from __future__ import annotations

import argparse

import pandas as pd

from config import BATCH_SIZE, EPOCHS, PROCESSED_DATA_DIR, RESULTS_DIR
from training.runtime import subject_groups
from training.trainer import MODEL_VERSION, run_lopo


def parse_folds(spec: str) -> list[int]:
    folds: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid fold range: {token}") from exc
            if start < 1 or end < start:
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
        raise argparse.ArgumentTypeError("No folds selected")
    return sorted(folds)


def remaining_folds() -> list[int]:
    cache_paths = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    if not cache_paths:
        raise FileNotFoundError(f"No temporal caches found in {PROCESSED_DATA_DIR}")
    subjects = [path.name.removesuffix("_temporal_graphs.pt") for path in cache_paths]
    n_folds = len(subject_groups(sorted(subjects)))
    all_folds = set(range(1, n_folds + 1))

    completed: set[int] = set()
    summary_path = RESULTS_DIR / "lopo_results_summary.csv"
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            if {"fold", "model_version"}.issubset(df.columns):
                df = df[df["model_version"].astype(str) == MODEL_VERSION]
                completed = {
                    int(value)
                    for value in df["fold"].dropna().tolist()
                    if 1 <= int(value) <= n_folds
                }
        except Exception as exc:
            print(f"[warn] Could not read existing summary: {exc}")

    remaining = sorted(all_folds.difference(completed))
    print(f"[*] Total LOPO folds: {n_folds}")
    print(f"[*] Completed folds: {sorted(completed)}")
    print(f"[*] Remaining folds: {remaining}")
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="DynaGAT patient-independent LOPO training")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-folds", type=int, default=None)
    selection.add_argument("--folds", type=parse_folds, default=None, metavar="SPEC")
    selection.add_argument("--remaining", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be >= 1")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be >= 1")

    folds = args.folds
    if args.remaining:
        folds = remaining_folds()
        if not folds:
            print("[+] All LOPO folds are complete.")
            return

    run_lopo(
        max_folds=args.max_folds,
        folds=folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
