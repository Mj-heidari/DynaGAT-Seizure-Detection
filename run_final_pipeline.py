from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import BATCH_SIZE, EPOCHS, RESULTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = RESULTS_DIR / "final_pipeline.log"


def run_step(name: str, command: list[str], log) -> None:
    header = f"\n{'=' * 88}\n{name}\n{'=' * 88}\n> {' '.join(command)}\n"
    print(header, end="")
    log.write(header)
    log.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="DynaGAT final reproducible pipeline")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain every fold even when compatible completed results exist",
    )
    parser.add_argument("--skip-healthcheck", action="store_true")
    parser.add_argument("--allow-partial-export", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be >= 1")
    if args.overwrite_cache and not args.preprocess:
        parser.error("--overwrite-cache requires --preprocess")
    if args.force_retrain and args.skip_training:
        parser.error("--force-retrain cannot be combined with --skip-training")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        if args.preprocess:
            command = [sys.executable, "-u", "run_preprocessing.py"]
            if args.overwrite_cache:
                command.append("--overwrite")
            run_step("PREPROCESSING", command, log)

        if not args.skip_healthcheck:
            run_step("HEALTH CHECK", [sys.executable, "-u", "run_healthcheck.py"], log)

        if not args.skip_training:
            training_command = [
                sys.executable,
                "-u",
                "run_training.py",
            ]
            if not args.force_retrain:
                training_command.append("--remaining")
            training_command.extend(
                [
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
            run_step(
                "FULL LOPO RETRAINING" if args.force_retrain else "REMAINING LOPO TRAINING",
                training_command,
                log,
            )

        export_command = [sys.executable, "-u", "export_paper.py"]
        if args.allow_partial_export:
            export_command.append("--allow-partial")
        run_step("PUBLICATION EXPORT", export_command, log)

    print(f"[+] Pipeline log: {LOG_PATH}")


if __name__ == "__main__":
    main()
