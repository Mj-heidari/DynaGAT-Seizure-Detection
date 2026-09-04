"""
Build the DynaGAT window cache from raw CHB-MIT BIDS EDF files.

Per recording the cache stores, for every 1 s-stride 4 s window:
    x           [T, 18, 34]  float16  node features (26 absolute + 8 causal-relative)
    in_dst      [T, 18, 5]   uint8    top-k Granger parents of each node
    in_weight   [T, 18, 5]   float16  normalised parent strengths
    out_dst     [T, 18, 5]   uint8    top-k Granger children of each node
    out_weight  [T, 18, 5]   float16
    labels      [T]          uint8
    boundary_weights [T]     float16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ABS_FEATURE_DIM,
    BIDS_ROOT,
    BOUNDARY_WEIGHT_MAX,
    CACHE_VERSION,
    GC_CHUNK_WINDOWS,
    GC_FS,
    GC_ORDER,
    GC_RIDGE,
    NODE_FEATURE_DIM,
    NUM_NODES,
    PREPROCESSING_TAG,
    PROCESSED_DATA_DIR,
    SFREQ,
    STRIDE_SAMPLES,
    TOP_K_CAUSAL,
    WINDOW_SAMPLES,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)
from dataset.causal_graph import build_causal_topk, granger_causality_batch
from dataset.features import apply_causal_baseline, extract_absolute_features
from dataset.io_edf import (
    EXPECTED_EDF_FILES,
    EXPECTED_SEIZURES,
    events_path_for_edf,
    load_canonical_recording,
    parse_seizure_events,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GC_DECIM = max(1, int(round(SFREQ / GC_FS)))


def _atomic_save(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def make_labels(
    n_windows: int, seizure_intervals: Sequence[Tuple[float, float]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Window labels and onset-boundary weights.

    A window is ictal when at least half of it overlaps an annotated seizure.
    Windows are timestamped at their *end*, so the label of window t refers to
    the interval [t*stride, t*stride + WINDOW_SEC].
    """
    starts = np.arange(n_windows, dtype=np.float64) * WINDOW_STRIDE_SEC
    ends = starts + WINDOW_SEC
    labels = np.zeros(n_windows, dtype=np.uint8)
    weights = np.ones(n_windows, dtype=np.float32)

    for onset, offset in seizure_intervals:
        overlap = np.minimum(ends, offset) - np.maximum(starts, onset)
        ictal = overlap >= (WINDOW_SEC * 0.5)
        labels[ictal] = 1
        near = np.abs(ends - onset) <= 10.0
        weights[near] = np.maximum(weights[near], BOUNDARY_WEIGHT_MAX * 0.7)
        weights[np.logical_and(near, ictal)] = BOUNDARY_WEIGHT_MAX
    return torch.from_numpy(labels), torch.from_numpy(weights)


@torch.inference_mode()
def extract_recording(
    data_uv: np.ndarray, seizure_intervals: Sequence[Tuple[float, float]]
) -> Dict:
    channels, total = data_uv.shape
    if channels != NUM_NODES:
        raise ValueError(f"expected {NUM_NODES} channels, got {channels}")
    n_windows = (total - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1
    if n_windows <= 0:
        raise ValueError("recording shorter than one analysis window")

    view = np.lib.stride_tricks.as_strided(
        data_uv,
        shape=(n_windows, channels, WINDOW_SAMPLES),
        strides=(
            data_uv.strides[1] * STRIDE_SAMPLES,
            data_uv.strides[0],
            data_uv.strides[1],
        ),
        writeable=False,
    )

    abs_parts: List[np.ndarray] = []
    in_dst_parts, in_w_parts, out_dst_parts, out_w_parts = [], [], [], []

    for start in range(0, n_windows, GC_CHUNK_WINDOWS):
        end = min(n_windows, start + GC_CHUNK_WINDOWS)
        wins = torch.from_numpy(np.ascontiguousarray(view[start:end])).to(
            DEVICE, dtype=torch.float32, non_blocking=True
        )
        abs_parts.append(extract_absolute_features(wins).cpu().numpy())

        gc = granger_causality_batch(
            wins[:, :, ::_GC_DECIM].contiguous(), order=GC_ORDER, ridge=GC_RIDGE
        )
        i_dst, i_w, o_dst, o_w = build_causal_topk(gc, k=TOP_K_CAUSAL)
        in_dst_parts.append(i_dst.to(torch.uint8).cpu())
        in_w_parts.append(i_w.to(torch.float16).cpu())
        out_dst_parts.append(o_dst.to(torch.uint8).cpu())
        out_w_parts.append(o_w.to(torch.float16).cpu())
        del wins, gc

    abs_feats = np.concatenate(abs_parts, axis=0)                # [T, 18, 26]
    full_feats = apply_causal_baseline(abs_feats)                # [T, 18, 34]

    labels, boundary = make_labels(n_windows, seizure_intervals)
    feat_t = torch.from_numpy(full_feats)
    return {
        "x": feat_t.to(torch.float16),
        "in_dst": torch.cat(in_dst_parts, dim=0),
        "in_weight": torch.cat(in_w_parts, dim=0),
        "out_dst": torch.cat(out_dst_parts, dim=0),
        "out_weight": torch.cat(out_w_parts, dim=0),
        "labels": labels.to(torch.uint8),
        "boundary_weights": boundary.to(torch.float16),
        "n_windows": int(n_windows),
        "duration_sec": float(total / SFREQ),
        "seizure_intervals": [(float(a), float(b)) for a, b in seizure_intervals],
        "positive_windows": int(labels.sum().item()),
        "feature_sum": feat_t.double().sum(dim=(0, 1)),
        "feature_sumsq": feat_t.double().square().sum(dim=(0, 1)),
        "feature_count": int(n_windows * NUM_NODES),
    }


def _manifest_row(cache: Dict, subject: str, n_edf: int) -> Dict:
    recs = cache["recordings"]
    hours = sum(float(r["duration_sec"]) for r in recs) / 3600.0
    total = int(cache["total_windows"])
    pos = int(cache["positive_windows"])
    return {
        "subject": subject,
        "edf_files": int(n_edf),
        "event_files_found": int(cache.get("event_files_found", 0)),
        "valid_recordings": len(recs),
        "skipped_recordings": int(cache.get("skipped_recordings", 0)),
        "windows": total,
        "positive_windows": pos,
        "positive_fraction": pos / max(1, total),
        "seizures": int(cache.get("total_seizures", 0)),
        "recording_hours": hours,
        "cache_version": CACHE_VERSION,
        "feature_dim": NODE_FEATURE_DIM,
    }


def build_all(overwrite: bool = False, max_subjects: int | None = None,
              only: Sequence[str] | None = None) -> None:
    if not BIDS_ROOT.exists():
        raise FileNotFoundError(
            f"BIDS root does not exist: {BIDS_ROOT}\n"
            "Set CHBMIT_BIDS_ROOT or edit BIDS_ROOT in config.py."
        )
    subject_dirs = sorted(
        d for d in BIDS_ROOT.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    if only:
        wanted = set(only)
        subject_dirs = [d for d in subject_dirs if d.name in wanted]
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    if not subject_dirs:
        raise RuntimeError(f"no sub-* directories found under {BIDS_ROOT}")

    print(f"[*] tag        : {PREPROCESSING_TAG}")
    print(f"[*] cache ver  : {CACHE_VERSION}")
    print(f"[*] features   : {NODE_FEATURE_DIM} ({ABS_FEATURE_DIM} absolute + "
          f"{NODE_FEATURE_DIM - ABS_FEATURE_DIM} causal-relative)")
    print(f"[*] window     : {WINDOW_SEC:g} s / {WINDOW_STRIDE_SEC:g} s stride")
    print(f"[*] Granger    : order {GC_ORDER} @ {GC_FS:g} Hz, top-{TOP_K_CAUSAL} per direction")
    print(f"[*] device     : {DEVICE}")
    print(f"[*] subjects   : {len(subject_dirs)}")
    print(f"[*] output     : {PROCESSED_DATA_DIR}\n")

    rows: List[Dict] = []
    for sub_dir in subject_dirs:
        cache_path = PROCESSED_DATA_DIR / f"{sub_dir.name}_v4.pt"
        edf_files = sorted(sub_dir.rglob("*.edf"))
        if not edf_files:
            print(f"[skip] {sub_dir.name}: no EDF files")
            continue
        if cache_path.exists() and not overwrite:
            existing = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
            rows.append(_manifest_row(existing, sub_dir.name, len(edf_files)))
            print(f"[skip] {sub_dir.name}: cache exists")
            del existing
            continue

        recordings: List[Dict] = []
        f_sum = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
        f_sq = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
        f_n = 0
        total_windows = pos_windows = total_seizures = skipped = events_found = 0

        for edf_path in tqdm(edf_files, desc=sub_dir.name, ncols=100):
            tsv = events_path_for_edf(edf_path)
            if tsv.exists():
                events_found += 1
            intervals = parse_seizure_events(tsv)
            data = load_canonical_recording(edf_path)
            if data is None:
                skipped += 1
                continue
            try:
                rec = extract_recording(data, intervals)
            except Exception as exc:
                skipped += 1
                print(f"[skip] {edf_path.name}: extraction failed: {exc}")
                continue
            rec["file_name"] = edf_path.name
            rec["recording_id"] = f"{sub_dir.name}/{edf_path.name}"
            recordings.append(rec)
            f_sum += rec.pop("feature_sum")
            f_sq += rec.pop("feature_sumsq")
            f_n += rec.pop("feature_count")
            total_windows += rec["n_windows"]
            pos_windows += rec["positive_windows"]
            total_seizures += len(intervals)
            del data

        if not recordings:
            print(f"[error] {sub_dir.name}: no valid recordings")
            continue

        payload = {
            "cache_version": CACHE_VERSION,
            "preprocessing_tag": PREPROCESSING_TAG,
            "node_feature_dim": NODE_FEATURE_DIM,
            "top_k_causal": TOP_K_CAUSAL,
            "subject": sub_dir.name,
            "recordings": recordings,
            "feature_sum": f_sum,
            "feature_sumsq": f_sq,
            "feature_count": int(f_n),
            "total_windows": int(total_windows),
            "positive_windows": int(pos_windows),
            "total_seizures": int(total_seizures),
            "valid_recordings": len(recordings),
            "skipped_recordings": int(skipped),
            "event_files_found": int(events_found),
            "sampling_rate_hz": float(SFREQ),
            "window_sec": float(WINDOW_SEC),
            "stride_sec": float(WINDOW_STRIDE_SEC),
        }
        _atomic_save(payload, cache_path)
        rows.append(_manifest_row(payload, sub_dir.name, len(edf_files)))
        print(
            f"[+] {sub_dir.name}: {len(recordings)}/{len(edf_files)} recordings | "
            f"{total_windows:,} windows | {pos_windows:,} ictal | "
            f"{total_seizures} seizures -> {cache_path.name}"
        )

    if rows:
        manifest = pd.DataFrame(rows)
        name = "preprocessing_manifest.csv" if max_subjects is None and not only else "preprocessing_manifest_partial.csv"
        path = PROCESSED_DATA_DIR / name
        manifest.to_csv(path, index=False)
        print(f"\n[+] manifest: {path}")
        print(
            f"[*] QA: EDF={int(manifest['edf_files'].sum())} "
            f"(expected {EXPECTED_EDF_FILES}), "
            f"seizures={int(manifest['seizures'].sum())} (expected {EXPECTED_SEIZURES}), "
            f"hours={float(manifest['recording_hours'].sum()):.1f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the DynaGAT window cache")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="e.g. --only sub-01 sub-02")
    args = ap.parse_args()
    build_all(overwrite=args.overwrite, max_subjects=args.max_subjects, only=args.only)
