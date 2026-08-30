from __future__ import annotations

import os
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# Paths / experiment versioning
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Override with environment variable CHBMIT_BIDS_ROOT if your dataset is elsewhere.
_DEFAULT_BIDS_ROOT = r"D:\EEG_Dataset\CHB_MIT\BIDS_CHB-MIT\BIDS_CHB-MIT"
BIDS_ROOT = Path(os.environ.get("CHBMIT_BIDS_ROOT", _DEFAULT_BIDS_ROOT))

# v3 is intentionally isolated from every previous cache. Old caches are never
# considered compatible with the current preprocessing / feature schema.
CACHE_VERSION = 3
PREPROCESSING_TAG = "causal_v3_features20_hybrid_connectivity"
PROCESSED_DATA_DIR = Path(
    os.environ.get("DYNAGAT_CACHE_DIR", str(PROJECT_ROOT / "data_cache_v3"))
)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(os.environ.get("DYNAGAT_RESULTS_DIR", str(PROJECT_ROOT / "results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# EEG preprocessing / graph specification
# -----------------------------------------------------------------------------
# CHB-MIT is natively sampled at 256 Hz. Unexpected sampling rates are rejected
# rather than silently resampled, preserving the strictly causal signal path.
SFREQ = 256.0
WINDOW_SEC = 2.0
WINDOW_STRIDE_SEC = 1.0
BANDPASS_LFREQ = 0.5
BANDPASS_HFREQ = 45.0
FILTER_IIR_ORDER = 4

CHANNELS_18 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

STATIC_EDGE_INDEX = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
    (0, 4), (12, 8), (3, 7), (15, 11),
    (4, 16), (8, 16), (7, 17), (11, 17),
]

TOP_K_DYNAMIC = 4
NUM_NODES = 18

# Per-node feature schema (20 features):
#   5 relative spectral band powers
#   6 Hjorth / time-domain statistics
#   4 spectral-shape statistics
#   5 log-covariance connectivity summaries
NODE_FEATURE_DIM = 20

# Dynamic graph edge score combines phase-lag synchrony (wPLI) with a smaller
# absolute-correlation contribution. This retains wPLI robustness while not
# discarding clinically useful near-zero-lag hypersynchrony.
DYNAMIC_WPLI_WEIGHT = 0.75
DYNAMIC_CORR_WEIGHT = 0.25


def get_static_edge_tensor() -> torch.Tensor:
    edges = set()
    for u, v in STATIC_EDGE_INDEX:
        edges.add((u, v))
        edges.add((v, u))
    ordered = sorted(edges)
    src = [u for u, _ in ordered]
    dst = [v for _, v in ordered]
    return torch.tensor([src, dst], dtype=torch.long)


# -----------------------------------------------------------------------------
# Temporal model / training defaults
# -----------------------------------------------------------------------------
SEQUENCE_LENGTH = 16               # ~17 s span with 1-second window strides
TRAIN_SEQUENCE_STRIDE = 16
# Overlap during evaluation gives each physical window more causal history;
# duplicate predictions are resolved by keeping the largest past context.
EVAL_SEQUENCE_STRIDE = 8

GRAPH_HIDDEN = 96
GAT_HEADS = 4
TCN_HIDDEN = 96
DROPOUT = 0.25

# RTX 3060 12 GB: 32 clips is still conservative for this graph/temporal model
# while improving throughput over 24. Override from the CLI if desired.
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# Validation checkpoint selection. The test patient remains completely untouched.
VALIDATION_CHECK_INTERVAL = 5
MIN_EPOCHS_BEFORE_STOPPING = 15
EARLY_STOPPING_PATIENCE = 3

# Dynamic negative sampling keeps expensive graph computation focused while each
# epoch sees a different background subset.
NEGATIVE_TO_IMPORTANT_RATIO = 6
MIN_NEGATIVE_CLIPS_PER_EPOCH = 512
RANDOM_SEED = 42

# Boundary-aware focal loss
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Event-level alarm policy. Applied identically on validation and held-out test.
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
ALARM_REFRACTORY_SEC = 30.0
EVENT_THRESHOLD_MAX_CANDIDATES = 81

# RTX 3060 preprocessing. The largest temporary tensor is the pairwise phase
# interaction tensor; 256 windows remains comfortably below 12 GB VRAM.
PREPROCESS_CHUNK_WINDOWS = 256

# Known CHB-MIT identity linkage. PhysioNet notes that chb21 and chb01 are the
# same subject recorded at different times; grouping prevents patient leakage.
LINKED_SUBJECT_GROUPS = [
    {"sub-01", "sub-21"},
    {"chb01", "chb21"},
    {"sub-chb01", "sub-chb21"},
]
