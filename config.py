from __future__ import annotations

import os
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Override with environment variable CHBMIT_BIDS_ROOT if your dataset is elsewhere.
_DEFAULT_BIDS_ROOT = r"D:\EEG_Dataset\CHB_MIT\BIDS_CHB-MIT\BIDS_CHB-MIT"
BIDS_ROOT = Path(os.environ.get("CHBMIT_BIDS_ROOT", _DEFAULT_BIDS_ROOT))

# Versioned cache directory. This intentionally does NOT reuse the old sparse
# *_graphs.pt files because those files skipped most background windows and are
# unsuitable for continuous temporal evaluation / FA-per-hour computation.
PROCESSED_DATA_DIR = Path(
    os.environ.get("DYNAGAT_CACHE_DIR", str(PROJECT_ROOT / "data_cache_v2"))
)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(os.environ.get("DYNAGAT_RESULTS_DIR", str(PROJECT_ROOT / "results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# EEG / graph specification
# -----------------------------------------------------------------------------
SFREQ = 256.0
WINDOW_SEC = 2.0
WINDOW_STRIDE_SEC = 1.0
BANDPASS_LFREQ = 0.5
BANDPASS_HFREQ = 45.0
NOTCH_FREQ = 60.0

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
NODE_FEATURE_DIM = 16
NUM_NODES = 18


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
SEQUENCE_LENGTH = 16               # 16 x 1-second strides ~= 17 s temporal span
TRAIN_SEQUENCE_STRIDE = 16         # non-overlapping base clips for training speed
# Evaluation overlaps clips. Duplicate windows are resolved by keeping the
# prediction with the largest amount of causal past context.
EVAL_SEQUENCE_STRIDE = 8

GRAPH_HIDDEN = 96
GAT_HEADS = 4
TCN_HIDDEN = 96
DROPOUT = 0.25

BATCH_SIZE = 24                    # safe starting point for RTX 3060 12 GB
EPOCHS = 24
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# Dynamic negative sampling keeps the expensive graph model focused while each
# epoch sees a different subset of background clips.
NEGATIVE_TO_IMPORTANT_RATIO = 6
MIN_NEGATIVE_CLIPS_PER_EPOCH = 512
RANDOM_SEED = 42

# Boundary-aware focal loss
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Event-level alarm policy. These values are applied identically on validation
# (threshold selection) and on the held-out test patient.
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
ALARM_REFRACTORY_SEC = 30.0
EVENT_THRESHOLD_MAX_CANDIDATES = 81

# Preprocessing batch size. Reduce to 64 if GPU preprocessing runs out of VRAM.
PREPROCESS_CHUNK_WINDOWS = 128

# Known CHB-MIT identity linkage. The BIDS release often already merges these;
# this mapping prevents leakage if both subject IDs happen to exist separately.
LINKED_SUBJECT_GROUPS = [
    {"sub-01", "sub-21"},
    {"chb01", "chb21"},
    {"sub-chb01", "sub-chb21"},
]
