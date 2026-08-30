from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plot_calibration import plot_calibration
from .plot_detection_timeline import plot_detection_timeline
from .plot_lopo_heatmap import plot_lopo_heatmap
from .plot_roc_pr import plot_roc_pr


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _prediction_files_for_current_run(results: Path, summary: Path) -> list[Path]:
    if summary.exists():
        df = pd.read_csv(summary)
        if "fold" in df.columns:
            files = [
                results / f"fold_{int(fold):02d}_test_predictions.npz"
                for fold in df["fold"].tolist()
            ]
            return [path for path in files if path.exists()]
    return sorted(results.glob("fold_*_test_predictions.npz"))


def generate_all_figures(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    results = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results"
    out = Path(out_dir) if out_dir is not None else PROJECT_ROOT / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)

    summary = results / "lopo_results_summary.csv"
    if summary.exists():
        plot_lopo_heatmap(summary, out)
    else:
        print(f"[skip] LOPO summary not found: {summary}")

    prediction_files = _prediction_files_for_current_run(results, summary)
    if not prediction_files:
        print("[skip] No real fold prediction files found; ROC/PR/calibration/timeline not generated.")
        print("       Run training first. Synthetic placeholder figures are intentionally disabled.")
        return

    plot_roc_pr(prediction_files, out)
    plot_calibration(prediction_files, out)
    plot_detection_timeline(prediction_files, out)
    print(f"[+] Figures generated from {len(prediction_files)} current held-out fold(s): {out}")


if __name__ == "__main__":
    generate_all_figures()
