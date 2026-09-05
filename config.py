"""Paths and scientific configuration for DynaGAT.

Set local data paths through environment variables. Method development history
is documented in docs/design_notes.md. Preserve the configuration and exact
source revision with each reported experiment.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_BIDS_ROOT = PROJECT_ROOT / "data" / "BIDS_CHB-MIT"
BIDS_ROOT = Path(os.environ.get("CHBMIT_BIDS_ROOT", _DEFAULT_BIDS_ROOT))

CACHE_VERSION = 4
PREPROCESSING_TAG = "causal_v4_feat34_granger_k5"

PROCESSED_DATA_DIR = Path(
    os.environ.get("DYNAGAT_CACHE_DIR", str(PROJECT_ROOT / "data_cache_v4"))
)
RESULTS_DIR = Path(os.environ.get("DYNAGAT_RESULTS_DIR", str(PROJECT_ROOT / "results")))
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper_figures"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper_tables"
PAPER_RESULTS_DIR = PROJECT_ROOT / "paper_results"
for _d in (PROCESSED_DATA_DIR, RESULTS_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR, PAPER_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Signal / windowing
# --------------------------------------------------------------------------- #
SFREQ = 256.0
WINDOW_SEC = 4.0          # an earlier iteration used 2 s; 4 s gives a usable MVAR sample size
WINDOW_STRIDE_SEC = 1.0
BANDPASS_LFREQ = 0.5
BANDPASS_HFREQ = 45.0
FILTER_IIR_ORDER = 4
NOTCH_FREQ_HZ = 60.0      # CHB-MIT was recorded in the United States
NOTCH_Q = 30.0

WINDOW_SAMPLES = int(round(WINDOW_SEC * SFREQ))
STRIDE_SAMPLES = int(round(WINDOW_STRIDE_SEC * SFREQ))

CHANNELS_18 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
NUM_NODES = 18

BANDS = [
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 16.0),  # sigma
    (16.0, 24.0),  # beta-1
    (24.0, 32.0),  # beta-2
    (32.0, 45.0),  # low gamma
]
NUM_BANDS = len(BANDS)

# --------------------------------------------------------------------------- #
# Node features: 26 absolute/shape + 8 causal trailing-baseline relative
# --------------------------------------------------------------------------- #
ABS_FEATURE_DIM = 26
REL_FEATURE_DIM = 8
NODE_FEATURE_DIM = ABS_FEATURE_DIM + REL_FEATURE_DIM      # 34

# Indices (into the absolute block) that also get a causal relative version.
REL_SOURCE_INDICES = (7, 8, 9, 11, 12, 17, 1, 6)
# 7 log total power, 8 Hjorth activity, 9 Hjorth mobility, 11 log line length,
# 12 log rms, 17 spectral edge, 1 theta rel-power, 6 gamma rel-power

# Causal exponential baseline: ~5 min time constant at 1 s stride.
BASELINE_TAU_WINDOWS = 300.0
BASELINE_WARMUP_WINDOWS = 60          # first minute is flagged low-confidence
REL_CLIP = 8.0

# --------------------------------------------------------------------------- #
# Static (anatomical) view
# --------------------------------------------------------------------------- #
STATIC_EDGE_INDEX = [
    # within-chain neighbours (left temporal, left parasagittal, right
    # parasagittal, right temporal, midline)
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
    # chain-to-chain anterior / posterior links
    (0, 4), (12, 8), (3, 7), (15, 11),
    (4, 16), (8, 16), (7, 17), (11, 17),
    # inter-hemispheric homologues
    (0, 12), (1, 13), (2, 14), (3, 15),
    (4, 8), (5, 9), (6, 10), (7, 11),
]

# --------------------------------------------------------------------------- #
# Directed causal (Granger) view
# --------------------------------------------------------------------------- #
GC_FS = 128.0             # decimation used only for the MVAR fit
GC_ORDER = 6              # 6 lags @128 Hz ~= 47 ms of conduction delay
GC_RIDGE = 1e-3           # Tikhonov term on the normal equations
TOP_K_CAUSAL = 5          # retained directed edges per node, per direction
GC_CHUNK_WINDOWS = 512    # windows processed per GPU batch during preprocessing

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH = 32      # 32 s of causal context at 1 s stride
TRAIN_SEQUENCE_STRIDE = 12
EVAL_SEQUENCE_STRIDE = 16
NODE_EMBED = 64
GRAPH_HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 2
GRAPH_OUT = 128
TCN_HIDDEN = 128
TCN_DILATIONS = (1, 2, 4, 8)
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
DROPOUT = 0.2

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_EPOCHS = 2
VALIDATION_CHECK_INTERVAL = 2
# Run 1 stopped 14 of 22 folds at exactly MIN_EPOCHS_BEFORE_STOPPING with the
# best checkpoint at epoch 2-4, i.e. the models were selected before they had
# converged. The proximate cause was a resampled validation subset (fixed in
# training/trainer.py); these bounds additionally stop the schedule being cut
# off before the cosine decay has done anything.
MIN_EPOCHS_BEFORE_STOPPING = 16
EARLY_STOPPING_PATIENCE = 4

# Negative:positive clip ratio used to build each training epoch. Unlike an earlier iteration
# this value is *recorded* and undone analytically at inference time (see
# training/calibration.py), so the deployed probabilities stay calibrated.
NEGATIVE_TO_IMPORTANT_RATIO = 8
MIN_NEGATIVE_CLIPS_PER_EPOCH = 1024
APPLY_PRIOR_CORRECTION = True

POS_WEIGHT = 3.0          # mild; the sampler already carries most of the work
LABEL_SMOOTHING = 0.02
BOUNDARY_WEIGHT_MAX = 2.5
BAG_LOSS_WEIGHT = 0.3     # multiple-instance term: each ictal clip needs a peak
ONSET_AUX_WEIGHT = 0.2

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Online detector post-processing
# --------------------------------------------------------------------------- #
# Detector output is converted to a causal, recording-adaptive score before
# thresholding. This removes the per-patient probability offset that made an earlier iteration's
# validation-selected threshold untransferable.
ADAPTIVE_NORM = True
ADAPTIVE_TAU_WINDOWS = 900.0       # 15 min causal baseline of the logit
ADAPTIVE_WARMUP_WINDOWS = 120
ADAPTIVE_MIX = 0.5                 # score = (1-m) * z_abs + m * z_adaptive

# Persistence candidates, capped at a 6 s decision window.
#
# Run 1 allowed up to 6-of-10 and the validation search selected it in 16 of 22
# folds, because on validation patients with long seizures a longer window buys
# a large false-alarm reduction for no sensitivity cost. On the held-out patient
# it is catastrophic when seizures are short: mean seizure duration in this
# cohort ranges from 9 s (sub-16) to 270 s (sub-11), and a 9 s seizure yields
# only ~7 labelled windows, so "6 of the last 10" can essentially never fire.
# Sensitivity was 0.00 for every patient whose mean seizure was under 25 s.
#
# Seizure duration is a property of the held-out patient and therefore cannot be
# tuned on validation. The fix is to bound the persistence window by the
# shortest seizures we intend to detect and control false alarms with the
# threshold instead.
PERSISTENCE_K_OF_M = ((2, 3), (3, 4), (3, 5), (4, 6))
ALARM_REFRACTORY_SEC = 30.0
VALIDATION_FA_PER_HOUR_CAP = 0.5
EVENT_THRESHOLD_MAX_CANDIDATES = 121
EVENT_EARLY_TOLERANCE_SEC = 0.0        # primary protocol: strictly online
EVENT_LATE_TOLERANCE_SEC = 30.0        # alarm inside [onset, offset + 30 s]
SECONDARY_EARLY_TOLERANCE_SEC = 10.0   # reported as a supplementary column
DECISION_TIME_REFERENCE = "window_end"

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
NUM_VALIDATION_PATIENTS = 5
DEVELOPMENT_FOLD = 1
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026

# chb01 and chb21 are the same subject recorded 1.5 years apart. They must
# never be split across train/test.
LINKED_SUBJECT_GROUPS = [
    {"sub-01", "sub-21"},
    {"chb01", "chb21"},
    {"sub-chb01", "sub-chb21"},
]


def get_static_edge_tensor() -> torch.Tensor:
    """Symmetric anatomical edge list with self-loops, shape [2, E]."""
    edges = set()
    for u, v in STATIC_EDGE_INDEX:
        edges.add((u, v))
        edges.add((v, u))
    for n in range(NUM_NODES):
        edges.add((n, n))
    ordered = sorted(edges)
    src = [u for u, _ in ordered]
    dst = [v for _, v in ordered]
    return torch.tensor([src, dst], dtype=torch.long)


def get_static_adjacency() -> torch.Tensor:
    """Dense boolean anatomical adjacency [N, N] (with self-loops)."""
    adj = torch.zeros(NUM_NODES, NUM_NODES, dtype=torch.bool)
    ei = get_static_edge_tensor()
    adj[ei[0], ei[1]] = True
    return adj


def experiment_signature() -> str:
    """Short fingerprint tying results to the code/configuration that made them."""
    import hashlib
    payload = "|".join(
        str(x)
        for x in (
            CACHE_VERSION, PREPROCESSING_TAG, WINDOW_SEC, WINDOW_STRIDE_SEC,
            NODE_FEATURE_DIM, GC_FS, GC_ORDER, TOP_K_CAUSAL, SEQUENCE_LENGTH,
            NODE_EMBED, GRAPH_HIDDEN, GRAPH_OUT, GAT_LAYERS, TCN_HIDDEN,
            TRANSFORMER_LAYERS, EPOCHS, BATCH_SIZE, LEARNING_RATE,
            NEGATIVE_TO_IMPORTANT_RATIO, APPLY_PRIOR_CORRECTION, ADAPTIVE_NORM,
            ADAPTIVE_MIX, VALIDATION_FA_PER_HOUR_CAP, EVENT_EARLY_TOLERANCE_SEC,
            EVENT_LATE_TOLERANCE_SEC, RANDOM_SEED,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
