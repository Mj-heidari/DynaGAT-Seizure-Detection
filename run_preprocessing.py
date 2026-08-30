from __future__ import annotations

import argparse

from dataset import bids_loader
from dataset import channel_reconstruction as channel_reconstruction


# CHB-MIT spans the historical 10-20 naming transition. Some EDF/BIDS headers
# use the legacy temporal labels T3/T5/T4/T6 while the canonical montage used by
# this project follows the modern equivalents T7/P7/T8/P8. This is a naming
# normalization only; it does not alter or interpolate EEG samples.
_LEGACY_1020_ALIASES = {
    "T3": "T7",
    "T5": "P7",
    "T4": "T8",
    "T6": "P8",
}
_original_fix_electrode_token = channel_reconstruction._fix_electrode_token


def _fix_chbmit_electrode_token(token: str) -> str:
    fixed = _original_fix_electrode_token(token)
    return _LEGACY_1020_ALIASES.get(fixed, fixed)


channel_reconstruction._fix_electrode_token = _fix_chbmit_electrode_token
install_robust_loader = channel_reconstruction.install_robust_loader
rebuild_selected_subjects = channel_reconstruction.rebuild_selected_subjects


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
