from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plot_calibration import plot_calibration
from .plot_detection_timeline import plot_detection_timeline
from .plot_event_tradeoff import plot_event_tradeoff
from .plot_lopo_heatmap import plot_lopo_heatmap
from .plot_preprocessing_overview import plot_preprocessing_overview
from .plot_roc_pr import plot_roc_pr
from .plot_training_curves import plot_training_curves


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


def _history_files_for_current_run(results: Path, summary: Path) -> list[Path]:
    if summary.exists():
        df = pd.read_csv(summary)
        if "fold" in df.columns:
            files = [
                results / f"fold_{int(fold):02d}_training_history.csv"
                for fold in df["fold"].tolist()
            ]
            return [path for path in files if path.exists()]
    return sorted(results.glob("fold_*_training_history.csv"))


def generate_all_figures(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    results = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results"
    out = Path(out_dir) if out_dir is not None else PROJECT_ROOT / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest = PROJECT_ROOT / "data_cache_v3" / "preprocessing_manifest.csv"
    if manifest.exists():
        plot_preprocessing_overview(manifest, out)
    else:
        print(f"[skip] preprocessing manifest not found: {manifest}")

    summary = results / "lopo_results_summary.csv"
    if summary.exists():
        plot_lopo_heatmap(summary, out)
        plot_event_tradeoff(summary, out)
    else:
        print(f"[skip] LOPO summary not found: {summary}")

    history_files = _history_files_for_current_run(results, summary)
    if history_files:
        plot_training_curves(history_files, out)
    else:
        print("[skip] No training-history CSV files found")

    prediction_files = _prediction_files_for_current_run(results, summary)
    if not prediction_files:
        print("[skip] No real fold prediction files found; prediction figures not generated.")
        print("       Run training first. Synthetic placeholder figures are intentionally disabled.")
        return

    plot_roc_pr(prediction_files, out)
    plot_detection_timeline(prediction_files, out)
    plot_calibration(prediction_files, out)

    generated = sorted(p.name for p in out.glob("*.png"))
    manifest_path = out / "figure_manifest.txt"
    manifest_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
    print(
        f"[+] Figures generated from {len(prediction_files)} current held-out fold(s): {out}"
    )
    print(f"[+] Figure manifest: {manifest_path}")


if __name__ == "__main__":
    generate_all_figures()
