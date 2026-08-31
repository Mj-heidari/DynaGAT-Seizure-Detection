from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_BIDS_ROOT = r"D:\EEG_Dataset\CHB_MIT\BIDS_CHB-MIT\BIDS_CHB-MIT"
BIDS_ROOT = Path(os.environ.get("CHBMIT_BIDS_ROOT", _DEFAULT_BIDS_ROOT))

CACHE_VERSION = 3
PREPROCESSING_TAG = "causal_v3_features20_hybrid_connectivity"
PROCESSED_DATA_DIR = Path(
    os.environ.get("DYNAGAT_CACHE_DIR", str(PROJECT_ROOT / "data_cache_v3"))
)
RESULTS_DIR = Path(os.environ.get("DYNAGAT_RESULTS_DIR", str(PROJECT_ROOT / "results")))
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper_figures"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper_tables"
PAPER_RESULTS_DIR = PROJECT_ROOT / "paper_results"
for directory in (PROCESSED_DATA_DIR, RESULTS_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR, PAPER_RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

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
NODE_FEATURE_DIM = 20
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


SEQUENCE_LENGTH = 16
TRAIN_SEQUENCE_STRIDE = 16
EVAL_SEQUENCE_STRIDE = 8
GRAPH_HIDDEN = 96
GAT_HEADS = 4
TCN_HIDDEN = 96
DROPOUT = 0.25

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
VALIDATION_CHECK_INTERVAL = 5
MIN_EPOCHS_BEFORE_STOPPING = 15
EARLY_STOPPING_PATIENCE = 3

NEGATIVE_TO_IMPORTANT_RATIO = 6
MIN_NEGATIVE_CLIPS_PER_EPOCH = 512
RANDOM_SEED = 42
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

EVENT_PERSISTENCE_CANDIDATES = (1, 2, 3)
VALIDATION_FA_PER_HOUR_CAP = 0.5
ALARM_REFRACTORY_SEC = 30.0
# Probabilities are timestamped when their complete analysis window becomes
# available. A strictly online detector therefore has no pre-onset tolerance.
EVENT_EARLY_TOLERANCE_SEC = 0.0
DECISION_TIME_REFERENCE = "window_end"
EVENT_THRESHOLD_MAX_CANDIDATES = 81
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
PREPROCESS_CHUNK_WINDOWS = 256

DEVELOPMENT_FOLD = 1
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026

LINKED_SUBJECT_GROUPS = [
    {"sub-01", "sub-21"},
    {"chb01", "chb21"},
    {"sub-chb01", "sub-chb21"},
]
