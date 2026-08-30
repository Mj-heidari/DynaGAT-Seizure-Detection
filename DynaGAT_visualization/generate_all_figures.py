
from pathlib import Path
from plot_lopo_heatmap import plot_lopo_heatmap
from plot_roc_pr import plot_roc_pr
from plot_calibration import plot_calibration
from plot_detection_timeline import plot_detection_timeline

RESULTS = Path("../results")
OUT = Path("../paper_figures")
OUT.mkdir(exist_ok=True)

plot_lopo_heatmap(RESULTS / "lopo_results_summary.csv", OUT)
plot_roc_pr(RESULTS, OUT)
plot_calibration(RESULTS, OUT)
plot_detection_timeline(RESULTS, OUT)

print("All paper figures generated.")
