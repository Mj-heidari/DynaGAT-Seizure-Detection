# DynaGAT Seizure Detection

Patient-independent seizure-onset detection on CHB-MIT using dynamic graph attention and a strictly causal temporal pipeline.

## Current pipeline

Raw CHB-MIT BIDS EDF -> canonical 18-channel montage -> causal 0.5-45 Hz Butterworth filtering -> microvolt scaling -> 20 node features -> hybrid wPLI/correlation dynamic graph -> residual static + dynamic GATv2 -> attention pooling -> feature-wise graph fusion -> causal residual multi-scale temporal encoder -> causal Transformer -> causal onset-delta classifier -> validation-only alarm operating-point selection.

The complete signal path is causal. An output at time `t` cannot use future EEG samples or future temporal windows. Checkpoint selection, threshold selection, and alarm-persistence selection use validation patients only; the held-out LOPO test patient is never used for normalization, training, model selection, or alarm tuning.

## Verified CHB-MIT preprocessing coverage

The current v3 preprocessing has been validated on the complete local BIDS tree:

- EDF files exposed by the BIDS tree: **686**
- valid recordings: **686 / 686**
- parsed annotated seizures: **198**
- usable EEG duration: **982.94 h**
- `sub-12` montage-change recordings are retained by reconstructing the same canonical bipolar derivations from common-reference/monopolar channels rather than discarding seizure-containing recordings.

The often-quoted 664-file count predates the additional `chb24` recordings. The current PhysioNet record list contains 686 EDF paths.

## v3 preprocessing

`data_cache_v3` is intentionally incompatible with all previous caches.

Important v3 changes:

- raw EDF files are processed from scratch;
- the 18 canonical bipolar channels are selected with CHB-MIT suffix and legacy temporal-electrode alias handling;
- common-reference/monopolar montage changes are reconstructed mathematically when the canonical bipolar derivation is available from shared-reference channels;
- unexpected sampling rates are rejected instead of silently resampled;
- filtering is forward-only causal Butterworth filtering;
- signals are converted from volts to microvolts before amplitude-domain feature extraction;
- each node has 20 features: five relative band powers, six Hjorth/time statistics, four spectral-shape statistics, and five log-covariance summaries;
- dynamic edges use a wPLI-dominant hybrid functional-connectivity score with an absolute-correlation component;
- continuous 2-second windows use a 1-second stride;
- `preprocessing_manifest.csv` records usable/skipped recordings, hours, seizure counts, positive windows, and class imbalance for every subject.

The loader validates cache version, preprocessing tag, and feature dimension. An older cache cannot accidentally enter current training.

## DynaGAT v5 model

The v5 model keeps the original causal dual-view concept but improves representation learning without simply increasing width:

- residual connections and LayerNorm inside both static and dynamic GATv2 branches;
- learned node-attention pooling instead of unconditional mean pooling;
- feature-wise fusion between anatomical/static and functional/dynamic graph embeddings;
- residual gated multi-scale causal temporal convolution;
- causal Transformer refinement;
- an explicit backward-looking temporal-delta feature `h_t - h_(t-1)` in the classifier to emphasize onset transitions without future information.

The main hidden dimensions remain 96, so the change targets optimization and onset sensitivity rather than brute-force parameter growth.

## Validation-only alarm operating point

For each LOPO fold, the final alarm rule is selected exclusively on the validation patients.

The search jointly evaluates:

- threshold candidates;
- persistence of 1, 2, or 3 consecutive positive windows.

The pre-specified validation false-alarm budget is **0.5 FA/h**. Among candidates satisfying that cap, selection prioritizes event sensitivity, then event F1/precision and lower FAR. The selected threshold and persistence are frozen before evaluating the held-out patient.

Each fold saves the complete validation sensitivity/FAR frontier to:

```text
results/fold_XX_validation_alarm_frontier.csv
```

This makes the operating-point decision auditable and suitable for paper figures/ablation analysis.

## Main files

- `config.py` - paths, preprocessing/model defaults, and locked alarm-policy constants
- `run_preprocessing.py` - raw CHB-MIT -> v3 temporal graph caches
- `run_training.py` - DynaGAT v5 patient-independent LOPO training/evaluation
- `run_reevaluate_checkpoint.py` - re-evaluate a legacy checkpoint with the new alarm policy without retraining
- `run_final_pipeline.py` - end-to-end orchestration
- `dataset/` - BIDS loading, montage reconstruction, features, cache validation, temporal datasets
- `models/dynagat_model.py` - legacy causal model retained for checkpoint compatibility
- `models/dynagat_model_v5.py` - current residual attention-pooled v5 model
- `training/trainer.py` - legacy trainer/helper functions
- `training/trainer_v5.py` - current v5 LOPO trainer
- `evaluation/metrics.py` - window/event metrics
- `evaluation/operating_point.py` - validation-only threshold+persistence selection
- `DynaGAT_visualization/` - data QA, training, and held-out result figures

## Windows / VS Code environment

The experiment has been run in the existing Conda environment:

```text
GNN_pytorch_gpu
```

Verified runtime hardware/software in the current development system:

- GPU: **NVIDIA GeForce RTX 3060 Laptop GPU**
- system RAM: **48 GB**
- PyTorch: **2.6.0+cu124**
- CUDA available: **True**

The code does not hard-code a desktop RTX 3060 or a 12 GB VRAM assumption. `trainer_v5.py` queries and prints the actual GPU name, CUDA runtime, compute capability, and physical VRAM at runtime. Peak allocated VRAM is also recorded every epoch.

Set the BIDS dataset location if needed:

```powershell
$env:CHBMIT_BIDS_ROOT="D:\path\to\BIDS_CHB-MIT"
```

## Recommended workflow

### 1. Build/rebuild preprocessing

```powershell
python run_preprocessing.py --overwrite
```

For a selective subject rebuild:

```powershell
python run_preprocessing.py --subjects sub-12 --overwrite
```

Inspect:

```text
data_cache_v3/preprocessing_manifest.csv
```

A complete current run should report 686/686 valid recordings and 198 seizures.

### 2. Re-evaluate the existing Fold-1 legacy checkpoint without retraining

If `results/dynagat_onset_fold_01_sub-01.pt` exists:

```powershell
python run_reevaluate_checkpoint.py --batch-size 32
```

This isolates the effect of the new alarm operating policy from the effect of the v5 architecture.

### 3. DynaGAT v5 one-fold experiment

```powershell
python run_training.py --max-folds 1 --epochs 30 --batch-size 32
```

Do not use a short 2-epoch smoke run to judge scientific performance; it is only a pipeline check.

### 4. Full locked-protocol LOPO

After the v5 one-fold development run is accepted, freeze the protocol and run:

```powershell
python run_training.py --epochs 30 --batch-size 32
```

Avoid changing architecture, loss, FAR cap, persistence candidates, or threshold policy after examining additional held-out folds if those folds will be reported as final test results.

### 5. Generate real figures

```powershell
python -m DynaGAT_visualization.generate_all_figures
```

Or run the complete pipeline:

```powershell
python run_final_pipeline.py --overwrite-cache
```

## Training defaults

Current defaults are intentionally modest and hardware-neutral:

- preprocessing chunk: 256 EEG windows;
- training batch size: 32 temporal clips;
- mixed-precision CUDA training;
- mmap-backed compact fp16 caches;
- Windows DataLoader workers: 0 to avoid process duplication of large cache objects;
- maximum training: 30 epochs;
- quick validation every 5 epochs;
- validation AUPRC checkpoint selection;
- early stopping after the minimum training period;
- joint validation-only threshold/persistence search under the fixed FAR cap.

Batch size can be overridden from the command line. Actual peak VRAM should determine safe throughput, not a hard-coded GPU-memory assumption.

## Training outputs

Current v5 training writes:

- `results/lopo_results_summary.csv`
- `results/dynagat_v5_fold_*.pt`
- `results/fold_*_test_predictions.npz`
- `results/fold_*_training_history.csv`
- `results/fold_*_validation_alarm_frontier.csv`

Held-out prediction files include real probabilities, labels, recording IDs, absolute window indices, validation-selected threshold, validation-selected persistence, model version, stride, and recording/seizure metadata.

## Visualization outputs

Generated files are written to `paper_figures/` and include:

- preprocessing coverage and class-balance QA;
- LOPO patient metric heatmap;
- pooled and per-fold ROC curves;
- pooled and per-fold precision-recall curves;
- true held-out seizure timeline;
- event sensitivity versus false alarms/hour;
- patient-level detection latency;
- reliability calibration / ECE;
- fold training-loss curves;
- validation AUPRC checkpoint-selection curves;
- `figure_manifest.txt` listing generated figures.

Synthetic placeholder figures are disabled.

## Evaluation safeguards

- Known same-patient CHB-MIT identities are grouped to prevent patient leakage.
- Feature normalization is computed from training patients only.
- Raw filtering is causal.
- Temporal convolutions, temporal-delta features, and Transformer attention are causal.
- Validation checkpoint selection never uses the held-out patient.
- Threshold and persistence selection never use the held-out patient.
- Evaluation clips can overlap for causal context, but each physical EEG window is counted once using the prediction with the largest available past context.
- Alarm latency is measured from the time the persistence condition is actually satisfied.
- False alarms/hour uses interictal recording time rather than including seizure duration in the denominator.
