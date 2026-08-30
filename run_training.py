from __future__ import annotations

import argparse

from config import BATCH_SIZE, EPOCHS
from training.trainer import run_lopo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DynaGAT-Onset with patient-independent LOPO")
    parser.add_argument("--max-folds", type=int, default=None, help="Run only the first N folds for a smoke test")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Temporal clips per GPU batch")
    args = parser.parse_args()
    run_lopo(max_folds=args.max_folds, epochs=args.epochs, batch_size=args.batch_size)


# ===============================
# Automatic paper figure generation
# ===============================
try:
    from generate_all_figures import generate_all_figures
    print("[+] Generating paper figures...")
    generate_all_figures()
    print("[+] Paper figures generated.")
except Exception as e:
    print("[!] Figure generation skipped:", e)
