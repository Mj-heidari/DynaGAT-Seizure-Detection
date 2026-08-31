from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    CACHE_VERSION,
    PAPER_FIGURES_DIR,
    PAPER_RESULTS_DIR,
    PAPER_TABLES_DIR,
    PROCESSED_DATA_DIR,
    PREPROCESSING_TAG,
    RESULTS_DIR,
)
from DynaGAT_visualization.paper_statistics import generate_paper_statistics
from DynaGAT_visualization.publication import generate_publication_figures
from training.runtime import subject_groups
from training.trainer import (
    EVALUATION_VERSION,
    MODEL_VERSION,
    RESULTS_SCHEMA_VERSION,
    experiment_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _expected_folds() -> int:
    caches = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    subjects = [path.name.removesuffix("_temporal_graphs.pt") for path in caches]
    return len(subject_groups(sorted(subjects)))


def _git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status": run("status", "--porcelain"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(require_complete: bool = True) -> None:
    summary = RESULTS_DIR / "lopo_results_summary.csv"
    manifest = PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    df = pd.read_csv(summary)
    expected = _expected_folds()
    required_columns = {
        "fold",
        "model_version",
        "evaluation_version",
        "results_schema_version",
        "cache_version",
        "preprocessing_tag",
        "experiment_signature",
        "max_epochs",
        "batch_size",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            "LOPO summary uses a legacy/incomplete schema; rerun training. "
            f"Missing columns: {missing_columns}"
        )

    epochs_values = df["max_epochs"].dropna().astype(int).unique().tolist()
    batch_values = df["batch_size"].dropna().astype(int).unique().tolist()
    if len(epochs_values) != 1 or len(batch_values) != 1:
        raise RuntimeError(
            "LOPO summary mixes incompatible training settings: "
            f"max_epochs={epochs_values}, batch_size={batch_values}"
        )
    signature = experiment_signature(epochs_values[0], batch_values[0])
    compatible = df[
        (df["model_version"].astype(str) == MODEL_VERSION)
        & (df["evaluation_version"].astype(str) == EVALUATION_VERSION)
        & (df["results_schema_version"].astype(int) == RESULTS_SCHEMA_VERSION)
        & (df["cache_version"].astype(int) == CACHE_VERSION)
        & (df["preprocessing_tag"].astype(str) == PREPROCESSING_TAG)
        & (df["experiment_signature"].astype(str) == signature)
    ].copy()
    completed_folds = set(compatible["fold"].dropna().astype(int))
    expected_folds = set(range(1, expected + 1))
    missing_folds = sorted(expected_folds.difference(completed_folds))
    extra_folds = sorted(completed_folds.difference(expected_folds))
    completed = len(completed_folds.intersection(expected_folds))
    if require_complete and (missing_folds or extra_folds):
        raise RuntimeError(
            f"LOPO is incomplete: {completed}/{expected} folds. "
            f"Missing={missing_folds}, extra={extra_folds}. "
            "Run `python run_training.py --remaining` first."
        )
    if compatible.empty:
        raise RuntimeError(
            "No result rows match the current code/configuration signature; rerun training."
        )
    if len(compatible) != len(df):
        raise RuntimeError(
            "The summary mixes current and stale experiment rows. Rerun the selected folds "
            "so lopo_results_summary.csv contains one compatible experiment."
        )

    missing_artifacts = []
    for fold in sorted(completed_folds):
        required_paths = [
            RESULTS_DIR / f"fold_{fold:02d}_training_history.csv",
            RESULTS_DIR / f"fold_{fold:02d}_validation_alarm_frontier.csv",
            RESULTS_DIR / f"fold_{fold:02d}_test_predictions.npz",
        ]
        missing_artifacts.extend(str(path) for path in required_paths if not path.exists())
        if not list(RESULTS_DIR.glob(f"dynagat_fold_{fold:02d}_*.pt")):
            missing_artifacts.append(str(RESULTS_DIR / f"dynagat_fold_{fold:02d}_<patient>.pt"))
    if missing_artifacts:
        raise RuntimeError(
            "Completed fold rows are missing required artifacts:\n- "
            + "\n- ".join(missing_artifacts)
        )

    generate_paper_statistics(
        summary_csv=summary,
        preprocessing_manifest=manifest,
        results_dir=RESULTS_DIR,
        tables_dir=PAPER_TABLES_DIR,
        paper_results_dir=PAPER_RESULTS_DIR,
    )
    generate_publication_figures(
        summary_csv=summary,
        results_dir=RESULTS_DIR,
        preprocessing_manifest=manifest,
        out_dir=PAPER_FIGURES_DIR,
    )

    artifacts = []
    for directory in (PAPER_RESULTS_DIR, PAPER_TABLES_DIR, PAPER_FIGURES_DIR):
        for path in sorted(directory.glob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                artifacts.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lopo_completed_folds": completed,
        "lopo_expected_folds": expected,
        "git": _git_state(),
        "artifacts": artifacts,
    }
    manifest_path = PAPER_RESULTS_DIR / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[+] Paper figures: {PAPER_FIGURES_DIR}")
    print(f"[+] Paper tables: {PAPER_TABLES_DIR}")
    print(f"[+] Paper results: {PAPER_RESULTS_DIR}")
    print(f"[+] Artifact manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export publication-ready DynaGAT results")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    main(require_complete=not args.allow_partial)
