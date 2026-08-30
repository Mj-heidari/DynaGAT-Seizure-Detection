from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(">", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DynaGAT v3 end-to-end pipeline: raw CHB-MIT preprocessing -> "
            "causal LOPO training -> publication figures"
        )
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use existing validated data_cache_v3 caches",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Do not train; useful when regenerating figures",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not generate figures after training",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Rebuild existing v3 subject caches from raw EEG",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Preprocess only the first N subjects",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Train/evaluate only the first N LOPO folds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum epochs per fold from config.py",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training/evaluation batch size from config.py",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for name in ("max_subjects", "max_folds", "epochs", "batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")

    print("DynaGAT causal v3 research pipeline")
    print(f"Project : {PROJECT_ROOT}")
    print(f"Python  : {sys.executable}")

    if not args.skip_preprocessing:
        command = [sys.executable, "run_preprocessing.py"]
        if args.overwrite_cache:
            command.append("--overwrite")
        if args.max_subjects is not None:
            command += ["--max-subjects", str(args.max_subjects)]
        run_step("STEP 1/3 - RAW EEG PREPROCESSING V3", command)
    else:
        print("\n[skip] preprocessing (using validated data_cache_v3)")

    if not args.skip_training:
        command = [sys.executable, "run_training.py"]
        if args.max_folds is not None:
            command += ["--max-folds", str(args.max_folds)]
        if args.epochs is not None:
            command += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            command += ["--batch-size", str(args.batch_size)]
        run_step("STEP 2/3 - CAUSAL LOPO TRAINING", command)
    else:
        print("\n[skip] training")

    if not args.skip_figures:
        run_step(
            "STEP 3/3 - REAL FIGURE EXPORT",
            [sys.executable, "-m", "DynaGAT_visualization.generate_all_figures"],
        )
    else:
        print("\n[skip] figures")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
