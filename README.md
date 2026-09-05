# DynaGAT

### Causal Dual-View Graph Attention for Patient-Independent Seizure Detection

DynaGAT is a research pipeline for EEG seizure detection on the CHB-MIT scalp EEG
corpus. It combines anatomical and directed Granger-causality graphs with graph
attention, a causal temporal encoder, and validation-selected alarm thresholds.

[Setup and reproduction](docs/reproducibility.md) · [Method](docs/method.md) ·
[Figures and tables](docs/figures.md) · [Design history](docs/design_notes.md) ·
[Paper release](docs/release_checklist.md)

## Quick start

Use Python 3.10 or newer in an isolated environment. Install a PyTorch build
appropriate for your hardware, then install the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch
python -m pip install -r requirements.txt
python -u run_selftest.py
```

The self-test uses synthetic data. Full preprocessing and training require your
BIDS-formatted CHB-MIT data; these data and trained weights are not bundled.
For CUDA training, install the appropriate CUDA-enabled PyTorch build before
running the pipeline. The health check requires CUDA.

Set the data location (an absolute path is recommended):

```bash
export CHBMIT_BIDS_ROOT="/path/to/BIDS_CHB-MIT"
```

```powershell
$env:CHBMIT_BIDS_ROOT = "C:\path\to\BIDS_CHB-MIT"
```

Run from the repository root:

```bash
python -u -m dataset.preprocess
python -u run_healthcheck.py
python -u run_lopo.py
python -u run_export.py
python -u run_figures.py
```

Preprocessing precedes the health check because that check validates built caches.
See the [reproduction guide](docs/reproducibility.md) for data format, smoke tests,
ablations, resuming runs, environment settings, and known limitations.

## Method at a glance

| Component | Configuration |
| --- | --- |
| Input | 18 bipolar EEG derivations at 256 Hz |
| Analysis windows | 4 seconds, 1-second stride; window-end timestamps |
| Node features | 26 absolute descriptors and 8 causal baseline-relative descriptors |
| Graph views | Anatomical adjacency and directed Granger-causality neighborhoods |
| Temporal encoder | Causal dilated TCN and causal Transformer |
| Evaluation | Leave-one-patient-out; fold 1 reserved for development |
| Alarm selection | Validation-patient sensitivity and false-alarm constraints |

Configuration values are defined in [config.py](config.py). The
[method guide](docs/method.md) describes the model and ablation arms. Synthetic
checks establish software behavior; they do not establish clinical performance.

## Repository structure

| Path | Purpose |
| --- | --- |
| `config.py` | Paths, model settings, and experiment signature |
| `dataset/` | EEG loading, features, causal graphs, and cache construction |
| `models/` | DynaGAT architecture |
| `training/` | Fold training, loss functions, and score calibration |
| `evaluation/` | Event metrics and operating-point selection |
| `baselines/` | Classical comparison model |
| `paperviz/` | Shared publication figure styles |
| `run_*.py` | Self-test, health check, training, export, and figure commands |
| `scripts/` | Optional PowerShell pipeline and VS Code setup |
| `docs/` | Method, reproduction, figures, and release documentation |
| `.github/workflows/` | Automated source and synthetic checks |

`make_paper_figures.py` remains a compatibility alias for `run_figures.py`.
The [migration notes](docs/migration.md) list renamed helper files.

## Generated outputs

| Directory | Contents |
| --- | --- |
| `data_cache_v4/` | Preprocessed subject caches |
| `results/` | Fold histories, predictions, checkpoints, and LOPO summaries |
| `paper_results/` | Per-patient results, summaries, and environment information |
| `paper_tables/` | CSV and LaTeX tables |
| `paper_figures/` | PDF and PNG figures |

Generated outputs and raw data are excluded from Git. Their directory names are
retained so existing caches and analysis workflows continue to work.

## Citation and release status

Use GitHub's **Cite this repository** button or [CITATION.cff](CITATION.cff) for
software metadata. The paper title, author list, DOI, and software release version
must be finalized before adding a paper citation. No publication DOI or benchmark
result is implied by this repository.

For a manuscript, record the exact commit used for the reported experiments and
link a tagged release. Follow the [paper-release checklist](docs/release_checklist.md)
before freezing that version.

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes and bug reports. A software
license has not yet been selected by the owner; public visibility alone does not
provide an explicit open-source license. Dataset terms are separate from the code.
