from __future__ import annotations

import argparse

from dataset import bids_loader
from dataset.channel_reconstruction import (
    install_robust_loader,
    rebuild_selected_subjects,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build causal v3 CHB-MIT temporal graph caches from raw BIDS EDF files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild v3 caches that already exist",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Only preprocess the first N subjects for a smoke test",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Rebuild only named BIDS subjects, e.g. --subjects sub-12",
    )
    args = parser.parse_args()

    if args.subjects and args.max_subjects is not None:
        parser.error("--subjects and --max-subjects cannot be used together")

    # Handles direct CHB-MIT bipolar channels and reconstructs the same canonical
    # bipolar derivations from common-reference/monopolar montage changes.
    install_robust_loader()

    if args.subjects:
        rebuild_selected_subjects(args.subjects, overwrite=args.overwrite)
    else:
        bids_loader.build_all_subject_caches(
            overwrite=args.overwrite,
            max_subjects=args.max_subjects,
        )
