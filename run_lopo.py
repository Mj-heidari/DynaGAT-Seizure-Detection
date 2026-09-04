"""
Leave-one-patient-out driver for DynaGAT.

    python run_lopo.py                        # full 23-fold LOPO, main model
    python run_lopo.py --folds 1 2 3          # a subset of folds
    python run_lopo.py --ablation no_causal   # one ablation arm
    python run_lopo.py --all-ablations        # main model + every ablation arm

Completed folds are appended to results/<tag>_lopo_summary.csv as soon as they
finish, and a rerun resumes from there as long as the experiment signature
matches. Change any relevant hyper-parameter and the old rows are treated as
stale rather than silently mixed with new ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DATA_DIR, RESULTS_DIR, experiment_signature
from training.trainer import FoldConfig, list_subjects, make_folds, run_fold

# name -> overrides applied to FoldConfig
ABLATIONS: Dict[str, Dict] = {
    "full":            {},
    "no_causal":       {"use_causal": False, "tag": "abl_no_causal"},
    "no_static":       {"use_static": False, "tag": "abl_no_static"},
    "causal_in_only":  {"causal_direction": "in", "tag": "abl_causal_in_only"},
    "causal_out_only": {"causal_direction": "out", "tag": "abl_causal_out_only"},
    "no_adaptive":     {"adaptive_norm": False, "adaptive_mix": 0.0, "tag": "abl_no_adaptive"},
    "no_prior":        {"prior_correction": False, "tag": "abl_no_prior"},
    "adaptive_only":   {"adaptive_mix": 1.0, "tag": "abl_adaptive_only"},
    "no_graph":        {"graph_mode": "none", "tag": "base_no_graph"},
}


def _summary_path(tag: str) -> Path:
    return RESULTS_DIR / f"{tag}_lopo_summary.csv"


def _load_done(tag: str, signature: str) -> pd.DataFrame:
    path = _summary_path(tag)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "signature" not in df.columns:
        print(f"[warn] {path.name} has no signature column; treating as stale")
        return pd.DataFrame()
    keep = df[df["signature"] == signature]
    if len(keep) != len(df):
        print(f"[warn] {len(df)-len(keep)} stale row(s) in {path.name} will be ignored")
    return keep


def run_arm(name: str, cfg_overrides: Dict, folds: List[Dict], only: List[int] | None,
            base: Dict) -> None:
    cfg = FoldConfig(**{**base, **cfg_overrides})
    tag = cfg.tag
    signature = experiment_signature()
    done = _load_done(tag, signature)
    done_folds = set(done["fold"].astype(int)) if not done.empty else set()

    todo = [f for f in folds if (only is None or f["fold"] in only) and f["fold"] not in done_folds]
    print(f"\n### arm '{name}' (tag={tag}) : {len(todo)} fold(s) to run, "
          f"{len(done_folds)} already complete")
    if not todo:
        return

    rows = [r._asdict() if hasattr(r, "_asdict") else r for r in done.to_dict("records")]
    t0 = time.perf_counter()
    for i, fold in enumerate(todo, 1):
        try:
            summary = run_fold(fold, cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"[error] fold {fold['fold']} ran out of VRAM. "
                  f"Retry with --batch-size {max(8, cfg.batch_size // 2)}.")
            raise
        rows.append(summary)
        df = pd.DataFrame(rows).sort_values("fold")
        df.to_csv(_summary_path(tag), index=False)
        elapsed = time.perf_counter() - t0
        remaining = (elapsed / i) * (len(todo) - i)
        print(f"[progress] arm '{name}': {i}/{len(todo)} folds | "
              f"elapsed {elapsed/60:.1f} min | eta {remaining/60:.1f} min")

    df = pd.DataFrame(rows).sort_values("fold")
    primary = df[df["evaluation_role"] == "primary"]
    if not primary.empty:
        print(
            f"\n[arm '{name}' primary folds, n={len(primary)}]\n"
            f"  event sensitivity : {primary['event_sensitivity'].mean():.3f} "
            f"(median {primary['event_sensitivity'].median():.3f})\n"
            f"  FA / h            : {primary['fa_per_hour'].mean():.3f} "
            f"(median {primary['fa_per_hour'].median():.3f})\n"
            f"  AUROC             : {primary['auroc'].mean():.4f}\n"
            f"  AUPRC             : {primary['auprc'].mean():.4f}\n"
            f"  median latency    : {primary['median_latency_sec'].median():.1f} s"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="DynaGAT LOPO runner")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--all-ablations", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else PROCESSED_DATA_DIR
    subjects = list_subjects(cache_dir)
    if len(subjects) < 8:
        print(
            f"[error] only {len(subjects)} subject cache(s) found in {cache_dir}.\n"
            f"        Build them first:  python -m dataset.preprocess"
        )
        return 1
    folds = make_folds(subjects)
    print(f"[*] {len(subjects)} subjects -> {len(folds)} LOPO folds")
    print(f"[*] signature {experiment_signature()}")
    print(f"[*] device {'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    base: Dict = {"num_workers": args.num_workers, "amp": not args.no_amp}
    if args.epochs is not None:
        base["epochs"] = args.epochs
    if args.batch_size is not None:
        base["batch_size"] = args.batch_size

    arms = sorted(ABLATIONS) if args.all_ablations else [args.ablation]
    if args.all_ablations:
        arms = ["full"] + [a for a in arms if a != "full"]
    for name in arms:
        run_arm(name, ABLATIONS[name], folds, args.folds, base)

    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "signature": experiment_signature(),
                "subjects": subjects,
                "folds": folds,
                "arms": arms,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
