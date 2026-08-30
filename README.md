# DynaGAT Seizure Detection

Patient-independent seizure-onset detection on CHB-MIT using dynamic graph attention and a strictly causal temporal pipeline.

## Current pipeline

Raw CHB-MIT BIDS EDF -> canonical 18-channel montage -> causal 0.5-45 Hz Butterworth filtering -> microvolt scaling -> 20 node features -> hybrid wPLI/correlation dynamic graph -> static + dynamic GATv2 -> causal multi-scale temporal encoder -> causal Transformer -> event-level seizure alarm evaluation.

The complete signal path is causal. An output at time `t` cannot use future EEG samples or future temporal windows. Threshold selection and checkpoint selection use validation patients only; the held-out LOPO test patient is never used for normalization, training, model selection, or threshold tuning.

## v3 preprocessing

`data_cache_v3` is a new cache schema and is intentionally incompatible with all previous caches.

Important v3 changes:

- raw EDF files are processed again from scratch;
- the 18 canonical bipolar channels are selected with CHB-MIT suffix tolerance;
- unexpected sampling rates are rejected instead of silently resampled;
- filtering is forward-only causal Butterworth filtering;
- signals are converted from volts to microvolts before amplitude-domain feature extraction;
- each node has 20 features: five relative band powers, six Hjorth/time statistics, four spectral-shape statistics, and five log-covariance summaries;
- dynamic edges use a wPLI-dominant hybrid functional-connectivity score with an absolute-correlation component;
- continuous 2-second windows use a 1-second stride;
- a `preprocessing_manifest.csv` file records usable recordings, skipped recordings, recording hours, seizure counts, positive windows, and class imbalance for every subject.

The loader validates `cache_version`, preprocessing tag, and feature dimension. A v2 cache cannot accidentally enter v3 training.

## Main files

- `config.py` - experiment paths and defaults
- `run_preprocessing.py` - raw CHB-MIT -> v3 temporal graph caches
- `run_training.py` - causal patient-independent LOPO training/evaluation
- `run_final_pipeline.py` - end-to-end orchestration
- `dataset/` - BIDS loading, feature extraction, cache validation, temporal datasets
- `models/` - dual-view GATv2 + causal temporal model
- `training/` - focal loss, validation checkpoint selection, LOPO trainer
- `evaluation/` - window/event metrics and validation-only threshold selection
- `DynaGAT_visualization/` - data QA, training, and held-out result figures

## Windows / VS Code setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the BIDS dataset location:

```powershell
$env:CHBMIT_BIDS_ROOT="D:\path\to\BIDS_CHB-MIT"
```

You can also edit `_DEFAULT_BIDS_ROOT` in `config.py`.

## Recommended workflow

### 1. Optional one-subject preprocessing smoke test

```powershell
python run_preprocessing.py --max-subjects 1 --overwrite
```

This is only a code/hardware check. Do not train the LOPO model from a one-subject cache set.

### 2. Build the complete v3 dataset from raw EEG

```powershell
python run_preprocessing.py --overwrite
```

After preprocessing, inspect:

```text
data_cache_v3/preprocessing_manifest.csv
```

Check especially `valid_recordings`, `skipped_recordings`, `seizures`, `recording_hours`, and `positive_fraction` before starting the full experiment.

### 3. Training smoke test

After all patient caches exist:

```powershell
python run_training.py --max-folds 1 --epochs 5
```

### 4. Full LOPO experiment

```powershell
python run_training.py
```

### 5. Generate all real figures

```powershell
python -m DynaGAT_visualization.generate_all_figures
```

Or run the whole pipeline in one command:

```powershell
python run_final_pipeline.py --overwrite-cache
```

## RTX 3060 / 48 GB RAM defaults

The defaults are deliberately conservative enough for an RTX 3060 12 GB while using the GPU more effectively than the older pipeline:

- preprocessing chunk: 256 EEG windows;
- training batch size: 32 temporal clips;
- mixed-precision CUDA training;
- mmap-backed compact fp16 caches;
- Windows DataLoader workers: 0 to avoid multi-process duplication of large mmap cache objects;
- maximum training: 30 epochs;
- quick validation every 5 epochs;
- validation-only best-checkpoint restoration and early stopping after the minimum training period.

You can override batch size from the command line, for example:

```powershell
python run_training.py --max-folds 1 --epochs 5 --batch-size 40
```

Do not increase it for the full run until a smoke test confirms stable VRAM usage.

## Training outputs

Training writes:

- `results/lopo_results_summary.csv`
- `results/dynagat_onset_fold_*.pt`
- `results/fold_*_test_predictions.npz`
- `results/fold_*_training_history.csv`

Held-out prediction files contain real probabilities, labels, recording IDs, absolute window indices, the validation-selected threshold, model version, time stride, and recording/seizure metadata used by the visualization pipeline.

## Visualization outputs

Generated files are written to `paper_figures/` and include:

- preprocessing coverage and skipped-recording QA;
- patient-level class imbalance;
- LOPO patient metric heatmap;
- pooled and per-fold ROC curves;
- pooled and per-fold precision-recall curves;
- a true held-out seizure timeline;
- event sensitivity versus false alarms/hour;
- patient-level detection latency;
- reliability calibration / ECE;
- fold training-loss curves;
- validation AUPRC checkpoint-selection curves;
- `figure_manifest.txt` listing the generated PNG files.

Synthetic placeholder figures are disabled.

## Evaluation safeguards

- The same physical subject is not allowed to leak across train/test through known CHB-MIT identity linkage.
- Feature normalization is computed from training patients only.
- Raw filtering is causal.
- Temporal convolutions and Transformer attention are causal.
- Validation checkpoint selection does not use the test patient.
- Validation threshold selection optimizes event-level performance without test-patient tuning.
- Evaluation clips overlap for context, but each physical EEG window is counted once using the prediction with the largest available causal history.
- Alarm latency is measured from the time the persistence condition is actually satisfied.
- False alarms/hour uses interictal recording time rather than including seizure duration in the denominator.
