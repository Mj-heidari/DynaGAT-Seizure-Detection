from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    PAPER_FIGURES_DIR,
    PAPER_RESULTS_DIR,
    PAPER_TABLES_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from DynaGAT_visualization.paper_statistics import generate_paper_statistics
from DynaGAT_visualization.publication import generate_publication_figures
from training.runtime import subject_groups


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
    completed = len(set(df["fold"].dropna().astype(int))) if "fold" in df.columns else 0
    if require_complete and completed < expected:
        raise RuntimeError(
            f"LOPO is incomplete: {completed}/{expected} folds. "
            "Run `python run_training.py --remaining` first."
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
