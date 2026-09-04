# DynaGAT — Causal Dual-View Graph Attention for Patient-Independent Seizure Detection

Patient-independent seizure detection on the CHB-MIT scalp EEG corpus. The model
combines a fixed anatomical montage graph with a per-window **directed
Granger-causal** graph, reads the causal graph along both its incoming and
outgoing edges, and feeds the fused representation to a strictly causal temporal
stack. See [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) for the reasoning behind
each design choice.

## Method

**Signal.** 18 canonical bipolar derivations, 256 Hz, causal 4th-order Butterworth
band-pass 0.5–45 Hz plus a causal 60 Hz notch. 4 s windows at 1 s stride,
timestamped at the window **end**.

**Node features (34 per channel).**
26 absolute descriptors — 7 log relative band powers, log total power, three Hjorth
parameters, line length, RMS, zero-crossing rate, Teager energy, spectral entropy /
centroid / 90 % edge / flatness, kurtosis, skewness, order-3 permutation entropy, two
intra-window non-stationarity ratios and two cross-channel spatial-contrast scores —
plus 8 **causal trailing-baseline** descriptors. The second block expresses how far the
current value sits above that recording's own recent past, in robust units, using a
time-varying causal estimator that starts as an expanding-window mean and settles to a
5 min exponential baseline. This is the main mechanism that lets one model serve
patients it has never seen.

**Dual view.**
*Anatomical view* — a fixed montage-adjacency graph with intra-chain, chain-to-chain and
inter-hemispheric homologous edges.
*Causal view* — a per-window **directed Granger-causality graph**. For each window we
estimate `GC[i,j] = log(var(x_i | own past) / var(x_i | own past + x_j past))` for all
306 ordered channel pairs, keep the 5 strongest incoming and 5 strongest outgoing edges
per channel, and normalise each window by its own mean edge strength so the graph is
invariant to per-patient signal-to-noise. Because seizure spread is directional, the
model reads the in-edges and out-edges through separate GATv2 stacks.

The two views are fused by a learned per-feature gate, pooled by attention over channels,
and passed to a strictly causal temporal stack (dilated causal TCN with dilations
1/2/4/8, then a 2-layer causal Transformer over 32 s of context). ~1.06 M parameters.

**Decision layer.** Training epochs oversample ictal clips; the resulting sampling prior is
recorded and undone analytically (`logit_true = logit_sampled − logit(π_sampled) + logit(π_true)`),
then the logit is converted to a causal, per-recording adaptive score. The alarm threshold
and k-of-m persistence are chosen on the *per-patient distribution* of the validation
patients under a 0.5 FA/h cap, not on their pooled totals.

Every operation is causal in time. `run_selftest.py` verifies numerically that a logit at
window *t* is bit-identical when all inputs after *t* are replaced with noise.

## Install

```powershell
conda activate GNN_pytorch_gpu
# install the CUDA build matching your driver first, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

No `torch-geometric` is required — graph attention uses gathered fixed-degree
neighbourhoods, which is both lighter on VRAM and faster for an 18-node graph.

## Run

```powershell
$env:CHBMIT_BIDS_ROOT="BIDS_CHB-MIT"

python -u run_selftest.py            # 1. correctness, no dataset needed (~3 min)
python -u run_healthcheck.py         # 2. environment + measured speed projection
python -u -m dataset.preprocess      # 3. build the cache (one-off)
python -u run_lopo.py                # 4. 23-fold leave-one-patient-out
python -u -m baselines.classical     # 5. classical baseline (optional)
python -u run_lopo.py --all-ablations
python -u make_paper_figures.py      # 6. figures + tables for the paper
```

Or run everything with `.\run_all.ps1`. VS Code launch configurations for each step are in
`.vscode/launch.json` (select `env` as the interpreter, then pick a
configuration from the Run and Debug panel).

Start with a two-subject smoke test before committing to the full preprocessing pass:

```powershell
python -u -m dataset.preprocess --max-subjects 2
```

**Resuming.** `run_lopo.py` appends each fold to `results/<tag>_lopo_summary.csv` the moment
it finishes and skips folds already present, provided the experiment signature matches. Any
change to a hyper-parameter that affects results changes the signature, so old rows are
treated as stale rather than silently mixed with new ones.

**If a fold runs out of VRAM:** `python run_lopo.py --batch-size 32`.

## Figures and tables

`make_paper_figures.py` produces every publication artefact: vector PDF for LaTeX
plus 600 dpi PNG, Elsevier column widths (90 / 140 / 190 mm), Type-42 embedded
fonts, and a `paper_tables/figures.tex` that can be `\input` directly.

```powershell
python -u make_paper_figures.py            # everything available
python -u make_paper_figures.py --list
python -u make_paper_figures.py --only causal_matrix
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

## Repository layout

```
config.py               all hyper-parameters and the experiment signature
dataset/
  io_edf.py             BIDS discovery, montage resolution, causal filtering
  features.py           34 node features incl. the causal trailing baseline
  causal_graph.py       batched directed Granger causality
  preprocess.py         cache builder
  sequence_dataset.py   temporal clip dataset
models/dynagat.py    dual-view graph attention + causal temporal stack
training/
  calibration.py        prior correction and online adaptive scoring
  losses.py             window + multiple-instance + onset objective
  trainer.py            leave-one-patient-out fold runner
evaluation/
  events.py             alarm generation and event metrics
  operating_point.py    per-patient operating-point selection
baselines/classical.py  gradient-boosted-tree baseline, identical protocol
paperviz/style.py       figure style and palette
run_selftest.py         correctness checks, no dataset required
run_healthcheck.py      environment and throughput check
run_lopo.py             LOPO driver with resume and ablation arms
make_paper_figures.py   figures and tables
run_export.py           summary statistics export
```

## Ablation arms

| `--ablation` | Question it answers |
|---|---|
| `full` | the complete model |
| `no_causal` | does the directed Granger view add anything over anatomy alone? |
| `no_static` | is the anatomical prior still needed? |
| `causal_in_only` / `causal_out_only` | does edge direction matter? |
| `no_adaptive` | how much of the transfer comes from online adaptation? |
| `no_prior` | how much does the sampling-prior correction contribute? |
| `adaptive_only` | is the absolute score term still contributing? |
| `no_graph` | graph-free control: same features, same temporal stack, no message passing |

`no_prior` and `no_adaptive` isolate the decision layer from the representation
and belong in the paper.

## Outputs

```
results/            per-fold history, validation frontier, test predictions, checkpoints,
                    <tag>_lopo_summary.csv
paper_results/      per-patient results, arm summary with bootstrap CIs, environment.json
paper_tables/       dataset / main-comparison / patient-level tables as CSV and LaTeX
paper_figures/      vector PDF + 600 dpi PNG
```


