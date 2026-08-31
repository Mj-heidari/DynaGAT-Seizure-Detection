# DynaGAT Seizure Detection

DynaGAT is a causal patient-independent EEG seizure-detection pipeline evaluated on the CHB-MIT scalp EEG dataset. The implementation combines static anatomical and dynamic functional graphs with residual GATv2 encoders, attention pooling, gated graph fusion, a causal multi-scale temporal convolutional encoder, and a causal Transformer.

## Method

- 18 bipolar EEG derivations
- 256 Hz sampling rate
- 0.5–45 Hz causal IIR band-pass filtering
- 2 s windows with 1 s stride
- 20 features per channel
- Static anatomical graph and dynamic hybrid wPLI/correlation graph
- Residual dual-view GATv2 encoder
- Attention pooling and feature-wise graph fusion
- Causal multi-scale temporal convolution
- Causal Transformer encoder
- Boundary-aware focal loss
- Patient-independent leave-one-patient-out evaluation
- Validation-only checkpoint and alarm operating-point selection

The alarm threshold and persistence are selected jointly on validation patients under a fixed validation false-alarm-rate cap of 0.5 FA/h. The selected operating point is applied unchanged to the held-out patient.

All alarm and latency timestamps refer to the **end of the 2 s analysis window**, which is the first time an online system has observed every sample used for that probability. The primary protocol uses no pre-onset tolerance.

## Evaluation protocol

Fold 1 was used during method development and is labelled `development`. Primary paper statistics exclude this fold. All subsequent held-out folds are labelled `primary` and are exported separately for final reporting.

## Dataset

The code expects the BIDS-formatted CHB-MIT dataset. Set the dataset path with:

```powershell
$env:CHBMIT_BIDS_ROOT="D:\EEG_Dataset\CHB_MIT\BIDS_CHB-MIT\BIDS_CHB-MIT"
```

The preprocessing stage validates recording coverage, seizure annotations, channel availability, cache version, and feature dimensions. CHB-MIT montage changes are reconstructed to the canonical bipolar derivations when the required electrode pairs share a valid reference.

## Installation

```powershell
python -m pip install -r requirements.txt
```

The validated training environment uses PyTorch 2.6.0+cu124 on an NVIDIA GeForce RTX 3060 Laptop GPU with 48 GB system RAM. GPU name, GPU memory, CUDA/PyTorch versions, package versions, and platform information are recorded automatically in the result artifacts.

## Final execution

First validate the local CUDA environment, full CHB-MIT manifest, caches, model forward pass, causal masking, and event timing:

```powershell
python -u run_healthcheck.py
```

If that passes, start or resume the complete pipeline:

```powershell
python -u run_final_pipeline.py
```

This command:

1. runs every unfinished LOPO fold;
2. writes each completed fold immediately to the merged results file;
3. resumes safely only from folds with the same code/configuration fingerprint;
4. exports publication statistics, tables, figures, configuration, environment information, and checksums after LOPO completion.

An old summary without the current fingerprint is treated as stale, so it cannot be silently mixed with corrected window-end metrics. To intentionally retrain every fold even when compatible results exist:

```powershell
python -u run_final_pipeline.py --force-retrain
```

Existing validated v3 caches can be reused; corrected event timing does not change the cached EEG features. If the health check reports a missing cache or an incomplete manifest, build only missing caches and reconstruct the complete manifest with:

```powershell
python -u run_preprocessing.py
```

To rebuild preprocessing from raw EEG first:

```powershell
python -u run_final_pipeline.py --preprocess --overwrite-cache
```

To regenerate paper artifacts without training:

```powershell
python -u run_final_pipeline.py --skip-training
```

## Main outputs

### Training and evaluation

`results/`

- `lopo_results_summary.csv`
- `fold_XX_training_history.csv`
- `fold_XX_validation_alarm_frontier.csv`
- `fold_XX_test_predictions.npz`
- `dynagat_fold_XX_<patient>.pt`
- `final_pipeline.log`

### Paper statistics

`paper_results/`

- `primary_per_patient_results.csv`
- `all_per_patient_results.csv`
- `primary_metric_summary.csv`
- `pooled_event_summary.csv`
- `results_summary.md`
- `experiment_config.json`
- `environment.json`
- `artifact_manifest.json`

Macro metric confidence intervals use deterministic patient-level bootstrap resampling. Primary pooled confidence intervals use patient-cluster bootstrap resampling so repeated seizures from the same patient are not treated as independent. Conditional Wilson and Poisson intervals are retained in `pooled_event_summary.csv` as supplementary statistics.

### Paper tables

`paper_tables/`

- dataset summary in CSV and LaTeX
- primary performance summary in CSV and LaTeX
- patient-level supplementary results in CSV and LaTeX
- experiment configuration in CSV and LaTeX

### Paper figures

`paper_figures/`

Figures are exported as vector PDF and 600 dpi PNG. The final exporter produces:

- dataset/preprocessing coverage
- model architecture schematic
- training and validation convergence
- pooled and patient-level ROC/precision-recall curves
- event sensitivity versus false-alarm-rate profile
- patient-level sensitivity forest plot with 95% intervals
- detected versus ground-truth seizure counts and FA/h
- reliability calibration
- representative seizure-detection timeline
- operating-point transfer analysis
- cross-patient metric distributions
- annotated patient-by-metric performance heatmap
- pooled window-level confusion heatmap using validation-selected thresholds
- validation-to-test sensitivity and false-alarm transfer heatmap

## Reproducibility

The final export records the Git commit, repository status, actual runtime epochs/batch size, experiment fingerprint, Python/package versions, GPU information, and SHA-256 checksums for generated paper artifacts. Export stops if folds come from mixed code/configurations or if a completed fold is missing its checkpoint, prediction, history, or validation-frontier artifact. The random seeds used for training and bootstrap statistics are fixed in `config.py`.
