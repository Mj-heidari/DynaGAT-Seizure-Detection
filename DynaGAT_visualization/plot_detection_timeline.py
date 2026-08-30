from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import WINDOW_STRIDE_SEC


def _find_example(prediction_files: Sequence[Path]):
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as data:
            probs = np.asarray(data["probs"], dtype=np.float64)
            labels = np.asarray(data["labels"], dtype=np.uint8)
            recording_ids = np.asarray(data["recording_ids"], dtype=str)
            window_indices = np.asarray(data["window_indices"], dtype=np.int64)
            threshold = float(np.asarray(data["threshold"]).item())

        for rid in np.unique(recording_ids):
            mask = recording_ids == rid
            idx = window_indices[mask]
            y = labels[mask]
            p = probs[mask]
            order = np.argsort(idx)
            idx, y, p = idx[order], y[order], p[order]
            positive = np.flatnonzero(y > 0)
            if positive.size:
                return rid, idx, y, p, threshold, int(positive[0])
    return None


def plot_detection_timeline(prediction_files: Sequence[Path], out: Path) -> None:
    example = _find_example(prediction_files)
    if example is None:
        print("[skip] timeline: no held-out seizure example found")
        return

    rid, idx, labels, probs, threshold, onset_pos = example
    left = max(0, onset_pos - 60)
    right = min(len(idx), onset_pos + 120)

    idx = idx[left:right]
    labels = labels[left:right]
    probs = probs[left:right]
    time_sec = idx.astype(np.float64) * WINDOW_STRIDE_SEC

    fig = plt.figure(figsize=(10.5, 3.8))
    plt.plot(time_sec, probs, label="Seizure probability")
    plt.axhline(threshold, linestyle="--", label=f"Validation threshold = {threshold:.3f}")

    if np.any(labels > 0):
        positive = labels > 0
        starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
        ends = np.flatnonzero(positive & np.r_[~positive[1:], True])
        for i, (start, end) in enumerate(zip(starts, ends)):
            plt.axvspan(
                time_sec[start],
                time_sec[end] + WINDOW_STRIDE_SEC,
                alpha=0.15,
                label="Annotated seizure windows" if i == 0 else None,
            )

    plt.xlabel("Recording time (s)")
    plt.ylabel("Probability")
    plt.ylim(-0.02, 1.02)
    plt.title(f"Held-out Detection Timeline: {rid}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(Path(out) / "Figure3_detection_timeline.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure3_detection_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
