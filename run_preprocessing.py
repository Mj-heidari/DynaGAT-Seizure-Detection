from __future__ import annotations

import argparse

from dataset.bids_loader import build_all_subject_caches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build continuous v2 CHB-MIT temporal graph caches")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild caches that already exist")
    parser.add_argument("--max-subjects", type=int, default=None, help="Only preprocess the first N subjects for a smoke test")
    args = parser.parse_args()
    build_all_subject_caches(overwrite=args.overwrite, max_subjects=args.max_subjects)
