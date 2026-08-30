# DynaGAT Seizure Detection

Patient-independent seizure-onset detection on CHB-MIT using dynamic graph attention and causal temporal modeling.

## Current research pipeline

EEG windows -> static + dynamic GATv2 graph reasoning -> causal multi-scale temporal encoder -> causal Transformer -> seizure probability -> event-level alarm evaluation.

The temporal path is strictly causal: an output at time `t` cannot attend to future EEG windows. Threshold selection is performed only on validation patients, never on the held-out LOPO test patient.

## Main files

- `config.py` - paths and experiment defaults
- `run_preprocessing.py` - build continuous temporal graph caches from CHB-MIT BIDS
- `run_training.py` - causal patient-independent LOPO training/evaluation
- `run_final_pipeline.py` - end-to-end orchestration
- `dataset/` - BIDS loading, feature extraction, temporal datasets
- `models/` - DynaGAT causal model
- `training/` - loss and LOPO trainer
- `evaluation/` - window/event metrics and validation threshold selection
- `DynaGAT_visualization/` - figures generated only from real held-out predictions

## VS Code / Windows setup

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the CHB-MIT BIDS path either in `config.py` or, preferably, as an environment variable:

```powershell
$env:CHBMIT_BIDS_ROOT="D:\path\to\BIDS_CHB-MIT"
```

## Preprocessing

Build all subject caches:

```powershell
python run_preprocessing.py
```

One-subject preprocessing smoke test:

```powershell
python run_preprocessing.py --max-subjects 1
```

Existing `data_cache_v2` caches remain compatible with the causal model. Old sparse `*_graphs.pt` caches must not be reused.

## Training smoke test

If the full patient caches already exist:

```powershell
python run_training.py --max-folds 1 --epochs 2
```

You need at least three independent patient groups in the cache set so train, validation, and test groups remain separated.

## Full experiment

```powershell
python run_final_pipeline.py --skip-preprocessing
```

If caches do not exist yet, run the complete pipeline:

```powershell
python run_final_pipeline.py
```

Useful overrides:

```powershell
python run_final_pipeline.py --skip-preprocessing --max-folds 2 --epochs 4 --batch-size 24
```

Regenerate figures from existing predictions without retraining:

```powershell
python run_final_pipeline.py --skip-preprocessing --skip-training
```

## Outputs

Training writes:

- `results/lopo_results_summary.csv`
- `results/dynagat_onset_fold_*.pt`
- `results/fold_*_test_predictions.npz`

Prediction `.npz` files contain real held-out probabilities, labels, recording IDs, absolute window indices, and the validation-selected threshold. They are used to generate ROC, PR, calibration, and seizure timeline figures.

Generated figures are written to `paper_figures/`.

## Evaluation safeguards

- LOPO test patients are excluded from normalization, training, and threshold selection.
- Known linked CHB-MIT subject identities are grouped to reduce identity leakage.
- Temporal convolutions and Transformer attention are causal.
- Evaluation clips overlap for context, but each physical EEG window is counted once using the prediction with the largest available causal history.
- Alarm latency is measured from the time the persistence condition is actually satisfied, not from the first positive window.
- Validation threshold selection optimizes event-level F1 and breaks ties toward higher sensitivity and lower false alarms/hour.
- Synthetic placeholder publication figures are disabled.

## Compatibility note

The current model checkpoint format is tagged `causal_v3`. Older model checkpoints should not be treated as equivalent to this architecture; retrain the model for valid comparisons.
