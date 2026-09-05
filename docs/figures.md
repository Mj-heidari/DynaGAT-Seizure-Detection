# Figures and tables


`run_figures.py` produces every publication artefact: vector PDF for LaTeX
plus 600 dpi PNG, Elsevier column widths (90 / 140 / 190 mm), Type-42 embedded
fonts, and a `paper_tables/figures.tex` that can be `\input` directly.

```powershell
python -u run_figures.py            # everything available
python -u run_figures.py --list
python -u run_figures.py --only causal_matrix
```

Each figure is skipped with a note if its input is not present yet, so the script
is safe to run at any stage.

| Figure | Content |
|---|---|
| `fig_causal_matrix` | 18x18 directed Granger matrices: interictal, ictal, and their difference; also reports a directional asymmetry index |
| `fig_patient_metric_heatmap` | Patients x metrics, annotated, shaded within column so darker is always better |
| `fig_confusion_matrix` | Pooled window-level confusion, row-normalised with counts |
| `fig_ablation_heatmap` | Ablation and baseline arms x metrics |
| `fig_duration_vs_sensitivity` | Sensitivity against mean seizure duration, coloured by AUROC |
| `fig_forest_sensitivity` | Per-patient sensitivity with Wilson intervals |
| `fig_operating_point_transfer` | Sensitivity vs false-alarm rate, with the validation cap marked |
| `fig_val_test_transfer` | Paired validation and held-out values of the operating point |
| `fig_pooled_roc_pr` | Pooled ROC and precision-recall (logarithmic precision axis) |
| `fig_convergence`, `fig_score_distribution`, `fig_calibration` | Supporting |

Colour handling lives in `paperviz/style.py`: single-hue sequential ramps so that
lightness alone carries magnitude in greyscale, a diverging map with a neutral
midpoint pinned to exactly zero, categorical colours assigned by entity rather
than rank, and per-cell text ink chosen by computed luminance.
