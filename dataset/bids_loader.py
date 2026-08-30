from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from config import (
    BANDPASS_HFREQ,
    BANDPASS_LFREQ,
    BIDS_ROOT,
    CHANNELS_18,
    NUM_NODES,
    PREPROCESS_CHUNK_WINDOWS,
    PROCESSED_DATA_DIR,
    SFREQ,
    TOP_K_DYNAMIC,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)

mne.set_log_level("ERROR")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANONICAL_CHANNELS = [ch.upper().replace(" ", "") for ch in CHANNELS_18]
CACHE_VERSION = 2

BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
]


def parse_seizure_events(tsv_path: Path) -> List[Tuple[float, float]]:
    """Parse seizure intervals from a BIDS events.tsv file."""
    if not tsv_path.exists():
        return []

    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as exc:
        print(f"[warn] could not parse {tsv_path.name}: {exc}")
        return []

    descriptive_cols = [
        c for c in df.columns
        if c.lower() not in {"onset", "duration", "sample"}
    ]
    intervals: List[Tuple[float, float]] = []

    for _, row in df.iterrows():
        try:
            onset = float(row.get("onset", 0.0))
            duration = float(row.get("duration", 0.0))
        except Exception:
            continue
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            continue

        text = " ".join(str(row.get(c, "")).lower() for c in descriptive_cols)
        has_seizure_keyword = any(
            key in text for key in ("seiz", "ictal", "sz", "epil")
        )

        if has_seizure_keyword or not descriptive_cols:
            intervals.append((onset, onset + duration))

    return sorted(intervals)


def _normalize_channel_name(name: str) -> str:
    return (
        name.upper()
        .replace("EEG ", "")
        .replace("-REF", "")
        .replace(" ", "")
    )


def _pick_canonical_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw | None:
    """
    Select the 18 canonical bipolar channels while tolerating common CHB-MIT
    duplicate suffixes such as T8-P8-0 / T8-P8-1.
    """
    normalized = {name: _normalize_channel_name(name) for name in raw.ch_names}
    selected: List[str] = []
    rename: Dict[str, str] = {}
    missing: List[str] = []

    for canonical in CANONICAL_CHANNELS:
        exact = [name for name, norm in normalized.items() if norm == canonical]
        aliases = [
            name
            for name, norm in normalized.items()
            if norm.startswith(canonical + "-")
            and norm[len(canonical) + 1 :].isdigit()
        ]
        candidates = exact if exact else aliases
        if not candidates:
            missing.append(canonical)
            continue
        chosen = candidates[0]
        selected.append(chosen)
        rename[chosen] = canonical

    if missing:
        return None

    picked = raw.copy().pick(selected)
    picked.rename_channels(rename)
    return picked


def clean_raw(edf_path: Path) -> np.ndarray | None:
    """Read one EDF, select canonical channels, resample if needed, and filter."""
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        picked = _pick_canonical_channels(raw)
        if picked is None:
            available = [_normalize_channel_name(ch) for ch in raw.ch_names]
            missing = [ch for ch in CANONICAL_CHANNELS if ch not in available]
            print(f"[skip] {edf_path.name}: missing canonical channels {missing}")
            return None

        sfreq = float(picked.info["sfreq"])
        if not np.isclose(sfreq, SFREQ):
            print(f"[info] {edf_path.name}: resampling {sfreq:g} Hz -> {SFREQ:g} Hz")
            picked.resample(SFREQ, npad="auto", verbose="ERROR")

        picked.filter(
            l_freq=BANDPASS_LFREQ,
            h_freq=BANDPASS_HFREQ,
            method="iir",
            verbose="ERROR",
        )
        return picked.get_data().astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[skip] {edf_path.name}: {exc}")
        return None


def _make_labels(
    n_windows: int,
    seizure_intervals: Sequence[Tuple[float, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = np.arange(n_windows, dtype=np.float64) * WINDOW_STRIDE_SEC
    ends = starts + WINDOW_SEC

    labels = np.zeros(n_windows, dtype=np.uint8)
    boundary_weights = np.ones(n_windows, dtype=np.float32)

    for seizure_start, seizure_end in seizure_intervals:
        overlap = np.minimum(ends, seizure_end) - np.maximum(starts, seizure_start)
        ictal = overlap >= (WINDOW_SEC * 0.5)
        labels[ictal] = 1

        onset_distance = np.abs(starts - seizure_start)
        near_onset = onset_distance <= 10.0
        boundary_weights[near_onset] = np.maximum(boundary_weights[near_onset], 2.0)
        boundary_weights[np.logical_and(near_onset, ictal)] = 3.0

    return torch.from_numpy(labels), torch.from_numpy(boundary_weights)


@torch.inference_mode()
def _extract_feature_chunk(wins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    wins : [B, 18, 512] float32 on GPU/CPU

    Returns
    -------
    node_features : [B, 18, 16] float32
    dynamic_dst   : [B, 18, K] int64
    dynamic_weight: [B, 18, K] float32
    """
    _, channels, win_len = wins.shape
    assert channels == NUM_NODES

    # 1) Relative FFT band power: 5 features/node
    power = torch.abs(torch.fft.rfft(wins, dim=-1)).square()
    freqs = torch.fft.rfftfreq(win_len, d=1.0 / SFREQ).to(wins.device)
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_power = power[..., total_mask].sum(dim=-1, keepdim=True).clamp_min(1e-10)

    bp_features = []
    for low, high in BANDS:
        band_mask = (freqs >= low) & (freqs < high)
        bp = power[..., band_mask].sum(dim=-1, keepdim=True) / total_power
        bp_features.append(torch.log1p(bp))
    bandpower = torch.cat(bp_features, dim=-1)

    # 2) Hjorth + time-domain: 6 features/node
    d1 = torch.diff(wins, dim=-1)
    d2 = torch.diff(d1, dim=-1)
    var0 = torch.var(wins, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-10)
    var1 = torch.var(d1, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-10)
    var2 = torch.var(d2, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-10)

    activity = torch.log1p(var0)
    mobility = torch.sqrt(var1 / var0)
    complexity = torch.sqrt(var2 / var1) / mobility.clamp_min(1e-10)
    line_length = torch.mean(torch.abs(d1), dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(wins.square(), dim=-1, keepdim=True))
    zero_cross = (
        torch.diff(torch.sign(wins), dim=-1).ne(0).sum(dim=-1, keepdim=True).float()
        / float(win_len)
    )
    hjorth_time = torch.cat(
        [activity, mobility, complexity, line_length, rms, zero_cross], dim=-1
    )

    # 3) Log-covariance summaries: 5 features/node
    centered = wins - wins.mean(dim=-1, keepdim=True)
    cov = torch.bmm(centered, centered.transpose(1, 2)) / float(max(1, win_len - 1))
    eye = torch.eye(channels, device=wins.device, dtype=wins.dtype).unsqueeze(0)
    cov = cov + 1e-5 * eye

    eigvals, eigvecs = torch.linalg.eigh(cov)
    log_eigvals = eigvals.clamp_min(1e-6).log()
    log_cov = torch.bmm(
        eigvecs,
        torch.bmm(torch.diag_embed(log_eigvals), eigvecs.transpose(1, 2)),
    )

    diag = torch.diagonal(log_cov, dim1=1, dim2=2).unsqueeze(-1)
    mean_conn = log_cov.mean(dim=-1, keepdim=True)
    std_conn = log_cov.std(dim=-1, keepdim=True, unbiased=False)

    large = torch.eye(channels, device=wins.device, dtype=wins.dtype).unsqueeze(0) * 1e5
    max_conn = (log_cov - large).max(dim=-1, keepdim=True).values
    min_conn = (log_cov + large).min(dim=-1, keepdim=True).values
    riemann = torch.cat([diag, mean_conn, std_conn, max_conn, min_conn], dim=-1)

    node_features = torch.cat([bandpower, hjorth_time, riemann], dim=-1)
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) wPLI: |E[Im(Sxy)]| / E[|Im(Sxy)|]
    spectrum = torch.fft.fft(wins, dim=-1)
    hilbert_filter = torch.zeros(win_len, dtype=wins.dtype, device=wins.device)
    hilbert_filter[0] = 1.0
    if win_len % 2 == 0:
        hilbert_filter[1 : win_len // 2] = 2.0
        hilbert_filter[win_len // 2] = 1.0
    else:
        hilbert_filter[1 : (win_len + 1) // 2] = 2.0

    analytic = torch.fft.ifft(spectrum * hilbert_filter, dim=-1)
    re = analytic.real
    im = analytic.imag

    imag_cross = (
        im.unsqueeze(2) * re.unsqueeze(1)
        - re.unsqueeze(2) * im.unsqueeze(1)
    )
    numerator = imag_cross.mean(dim=-1).abs()
    denominator = imag_cross.abs().mean(dim=-1).clamp_min(1e-10)
    wpli = (numerator / denominator).clamp(0.0, 1.0)

    diagonal = torch.arange(channels, device=wins.device)
    wpli[:, diagonal, diagonal] = 0.0

    dynamic_weight, dynamic_dst = torch.topk(
        wpli, k=TOP_K_DYNAMIC, dim=-1, largest=True, sorted=True
    )

    return node_features, dynamic_dst, dynamic_weight


@torch.inference_mode()
def extract_recording(raw_data: np.ndarray, seizure_intervals: Sequence[Tuple[float, float]]) -> Dict:
    """Extract compact graph tensors for every 1-second-stride window in one EDF."""
    channels, total_samples = raw_data.shape
    win_len = int(round(WINDOW_SEC * SFREQ))
    stride = int(round(WINDOW_STRIDE_SEC * SFREQ))

    if channels != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} channels, got {channels}")

    n_windows = (total_samples - win_len) // stride + 1
    if n_windows <= 0:
        raise ValueError("Recording is shorter than one analysis window")

    shape = (n_windows, channels, win_len)
    strides = (
        raw_data.strides[1] * stride,
        raw_data.strides[0],
        raw_data.strides[1],
    )
    window_view = np.lib.stride_tricks.as_strided(
        raw_data, shape=shape, strides=strides, writeable=False
    )

    x_parts: List[torch.Tensor] = []
    dst_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []

    feature_sum = torch.zeros(16, dtype=torch.float64)
    feature_sumsq = torch.zeros(16, dtype=torch.float64)
    feature_count = 0

    for start in range(0, n_windows, PREPROCESS_CHUNK_WINDOWS):
        end = min(n_windows, start + PREPROCESS_CHUNK_WINDOWS)
        chunk_np = np.ascontiguousarray(window_view[start:end])
        wins = torch.from_numpy(chunk_np).to(DEVICE, dtype=torch.float32)

        features, dynamic_dst, dynamic_weight = _extract_feature_chunk(wins)
        features_cpu = features.cpu()

        feature_sum += features_cpu.sum(dim=(0, 1), dtype=torch.float64)
        feature_sumsq += features_cpu.double().square().sum(dim=(0, 1))
        feature_count += int(features_cpu.shape[0] * features_cpu.shape[1])

        x_parts.append(features_cpu.to(torch.float16))
        dst_parts.append(dynamic_dst.cpu().to(torch.uint8))
        weight_parts.append(dynamic_weight.cpu().to(torch.float16))

        del wins, features, dynamic_dst, dynamic_weight, features_cpu

    labels, boundary_weights = _make_labels(n_windows, seizure_intervals)

    return {
        "x": torch.cat(x_parts, dim=0),
        "dynamic_dst": torch.cat(dst_parts, dim=0),
        "dynamic_weight": torch.cat(weight_parts, dim=0),
        "labels": labels.to(torch.uint8),
        "boundary_weights": boundary_weights.to(torch.float16),
        "n_windows": int(n_windows),
        "duration_sec": float(total_samples / SFREQ),
        "seizure_intervals": [(float(a), float(b)) for a, b in seizure_intervals],
        "feature_sum": feature_sum,
        "feature_sumsq": feature_sumsq,
        "feature_count": int(feature_count),
    }


def build_all_subject_caches(overwrite: bool = False, max_subjects: int | None = None) -> None:
    """Build continuous compact temporal caches for every BIDS subject."""
    if not BIDS_ROOT.exists():
        raise FileNotFoundError(
            f"BIDS root does not exist: {BIDS_ROOT}\n"
            "Set CHBMIT_BIDS_ROOT or edit BIDS_ROOT in config.py."
        )

    subject_dirs = sorted(
        d for d in BIDS_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    if not subject_dirs:
        raise RuntimeError(f"No sub-* directories found under {BIDS_ROOT}")
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"[*] Building continuous temporal caches on {DEVICE} for {len(subject_dirs)} subjects")
    print(f"[*] Output: {PROCESSED_DATA_DIR}\n")

    for sub_dir in subject_dirs:
        cache_path = PROCESSED_DATA_DIR / f"{sub_dir.name}_temporal_graphs.pt"
        if cache_path.exists() and not overwrite:
            print(f"[skip] {sub_dir.name}: cache already exists")
            continue

        edf_files = sorted(sub_dir.rglob("*.edf"))
        if not edf_files:
            print(f"[skip] {sub_dir.name}: no EDF files found")
            continue

        recordings: List[Dict] = []
        subject_sum = torch.zeros(16, dtype=torch.float64)
        subject_sumsq = torch.zeros(16, dtype=torch.float64)
        subject_count = 0
        total_windows = 0
        total_seizures = 0

        pbar = tqdm(edf_files, desc=f"{sub_dir.name}", ncols=100)
        for edf_path in pbar:
            tsv_path = edf_path.parent / edf_path.name.replace("_eeg.edf", "_events.tsv")
            intervals = parse_seizure_events(tsv_path)
            raw_data = clean_raw(edf_path)
            if raw_data is None:
                continue

            try:
                rec = extract_recording(raw_data, intervals)
            except Exception as exc:
                print(f"[skip] {edf_path.name}: extraction failed: {exc}")
                continue

            rec["file_name"] = edf_path.name
            rec["recording_id"] = f"{sub_dir.name}/{edf_path.name}"
            recordings.append(rec)

            subject_sum += rec["feature_sum"]
            subject_sumsq += rec["feature_sumsq"]
            subject_count += rec["feature_count"]
            total_windows += rec["n_windows"]
            total_seizures += len(intervals)

            del raw_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not recordings or subject_count <= 0:
            print(f"[error] {sub_dir.name}: no valid recordings; cache not written")
            if overwrite and cache_path.exists():
                cache_path.unlink()
            continue

        payload = {
            "cache_version": CACHE_VERSION,
            "subject": sub_dir.name,
            "recordings": recordings,
            "feature_sum": subject_sum,
            "feature_sumsq": subject_sumsq,
            "feature_count": int(subject_count),
            "total_windows": int(total_windows),
            "total_seizures": int(total_seizures),
        }
        torch.save(payload, cache_path)
        print(
            f"[+] {sub_dir.name}: {len(recordings)} recordings | "
            f"{total_windows:,} windows | {total_seizures} seizures -> {cache_path.name}"
        )


if __name__ == "__main__":
    build_all_subject_caches(overwrite=False)
