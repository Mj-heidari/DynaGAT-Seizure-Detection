from __future__ import annotations

from config import PAPER_FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from DynaGAT_visualization.publication import generate_publication_figures


def generate_all_figures() -> None:
    generate_publication_figures(
        summary_csv=RESULTS_DIR / "lopo_results_summary.csv",
        results_dir=RESULTS_DIR,
        preprocessing_manifest=PROCESSED_DATA_DIR / "preprocessing_manifest.csv",
        out_dir=PAPER_FIGURES_DIR,
    )


if __name__ == "__main__":
    generate_all_figures()
