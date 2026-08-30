from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import WINDOW_STRIDE_SEC


def _load_file(path: Path):
    with np.load(path, allow_pickle=False) as data:
        probs = np.asarray(data["probs"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.uint8)
        recording_ids = np.asarray(data["recording_ids"], dtype=str)
        window_indices = np.asarray(data["window_indices"], dtype=np.int64)
        threshold = float(np.asarray(data["threshold"]).item())
        metadata = {}
        if "recording_metadata_json" in data.files:
            metadata = json.loads(str(np.asarray(data["recording_metadata_json"]).item()))
    return probs, labels, recording_ids, window_indices, threshold, metadata


def _find_informative_example(prediction_files: Sequence[Path]):
    best = None
    best_score = -float("inf")

    for path in prediction_files:
        probs, labels, recording_ids, window_indices, threshold, metadata = _load_file(path)
        for rid in np.unique(recording_ids):
            mask = recording_ids == rid
            idx = window_indices[mask]
            y = labels[mask]
            p = probs[mask]
            order = np.argsort(idx)
            idx, y, p = idx[order], y[order], p[order]

            seizure_intervals = metadata.get(str(rid), {}).get("seizure_intervals", [])
            if seizure_intervals:
                for seizure_start, seizure_end in seizure_intervals:
                    onset_window = int(round(float(seizure_start) / WINDOW_STRIDE_SEC))
                    position = int(np.searchsorted(idx, onset_window))
                    left = max(0, position - 20)
                    right = min(len(p), position + 60)
                    if right <= left:
                        continue
                    score = float(np.nanmax(p[left:right]))
                    if score > best_score:
                        best_score = score
                        best = (
                            rid,
                            idx,
                            y,
                            p,
                            threshold,
                            float(seizure_start),
                            float(seizure_end),
                            position,
                        )
            else:
                positive = np.flatnonzero(y > 0)
                if positive.size:
                    position = int(positive[0])
                    score = float(np.nanmax(p[max(0, position - 20) : min(len(p), position + 60)]))
                    if score > best_score:
                        best_score = score
                        onset_sec = float(idx[position] * WINDOW_STRIDE_SEC)
                        best = (
                            rid,
                            idx,
                            y,
                            p,
                            threshold,
                            onset_sec,
                            onset_sec + WINDOW_STRIDE_SEC,
                            position,
                        )
    return best


def plot_detection_timeline(prediction_files: Sequence[Path], out: Path) -> None:
    example = _find_informative_example(prediction_files)
    if example is None:
        print("[skip] timeline: no held-out seizure example found")
        return

    rid, idx, labels, probs, threshold, seizure_start, seizure_end, onset_pos = example
    left = max(0, onset_pos - 90)
    right = min(len(idx), onset_pos + 150)

    idx_view = idx[left:right]
    probs_view = probs[left:right]
    time_sec = idx_view.astype(np.float64) * WINDOW_STRIDE_SEC

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.plot(time_sec, probs_view, linewidth=1.5, label="Seizure probability")
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label=f"Validation threshold = {threshold:.3f}")
    ax.axvline(seizure_start, linestyle="--", linewidth=1.2, label="Annotated seizure onset")
    ax.axvspan(seizure_start, seizure_end, alpha=0.14, label="Annotated seizure")

    above = probs_view >= threshold
    if np.any(above):
        candidate = np.flatnonzero(above & (time_sec >= seizure_start - 2.0))
        if candidate.size:
            first = int(candidate[0])
            ax.scatter([time_sec[first]], [probs_view[first]], s=38, zorder=4, label="First threshold crossing")

    ax.set_xlabel("Recording time (s)")
    ax.set_ylabel("Seizure probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Held-out Detection Timeline: {rid}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure3_detection_timeline.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / "Figure3_detection_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
