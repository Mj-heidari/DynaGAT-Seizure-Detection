from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from scipy import signal
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from config import (
    BANDPASS_HFREQ,
    BANDPASS_LFREQ,
    BIDS_ROOT,
    CACHE_VERSION,
    CHANNELS_18,
    DYNAMIC_CORR_WEIGHT,
    DYNAMIC_WPLI_WEIGHT,
    FILTER_IIR_ORDER,
    NODE_FEATURE_DIM,
    NUM_NODES,
    PREPROCESSING_TAG,
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

BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
]

# Used only as a full-dataset QA warning, never as a hard requirement. PhysioNet
# CHB-MIT v1.0.0 documents 664 EDF files and 198 annotated seizures.
EXPECTED_CHBMIT_EDF_FILES = 664
EXPECTED_CHBMIT_SEIZURES = 198


def _events_path_for_edf(edf_path: Path) -> Path:
    """Resolve the standard BIDS events.tsv sibling for an EEG EDF file."""
    if edf_path.name.endswith("_eeg.edf"):
        return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
    stem = edf_path.stem
    if stem.endswith("_eeg"):
        stem = stem[:-4]
    return edf_path.with_name(stem + "_events.tsv")


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
    """Select the canonical 18 bipolar channels with CHB-MIT suffix tolerance."""
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

    # raw is local to clean_raw, so mutating it avoids an unnecessary full-data copy.
    raw.pick(selected)
    raw.rename_channels(rename)
    return raw


def _causal_bandpass(data_v: np.ndarray, sfreq: float) -> np.ndarray:
    """Causal Butterworth band-pass with steady-state initialization."""
    sos = signal.butter(
        FILTER_IIR_ORDER,
        [BANDPASS_LFREQ, BANDPASS_HFREQ],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    zi_template = signal.sosfilt_zi(sos)
    filtered = np.empty(data_v.shape, dtype=np.float32)

    for channel in range(data_v.shape[0]):
        x = np.asarray(data_v[channel], dtype=np.float64)
        zi = zi_template * float(x[0])
        y, _ = signal.sosfilt(sos, x, zi=zi)
        filtered[channel] = y.astype(np.float32, copy=False)

    return filtered


def clean_raw(edf_path: Path) -> np.ndarray | None:
    """
    Read one EDF, load only the selected montage, apply causal filtering, and
    return microvolt data for numerically stable feature extraction.
    """
    try:
        # Metadata are cheap to read. Selecting channels before load_data avoids
        # preloading ECG/VNS/dummy channels and reduces RAM/I/O on long recordings.
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        picked = _pick_canonical_channels(raw)
        if picked is None:
            available = [_normalize_channel_name(ch) for ch in raw.ch_names]
            missing = [
                ch
                for ch in CANONICAL_CHANNELS
                if not any(a == ch or a.startswith(ch + "-") for a in available)
            ]
            print(f"[skip] {edf_path.name}: missing canonical channels {missing}")
            return None

        sfreq = float(picked.info["sfreq"])
        if not np.isclose(sfreq, SFREQ, rtol=0.0, atol=1e-6):
            print(
                f"[skip] {edf_path.name}: unexpected sfreq={sfreq:g} Hz; "
                f"expected native CHB-MIT {SFREQ:g} Hz"
            )
            return None

        picked.load_data()
        data_v = picked.get_data()
        filtered_uv = _causal_bandpass(data_v, sfreq)
        filtered_uv *= 1e6

        if not np.isfinite(filtered_uv).all():
            print(f"[skip] {edf_path.name}: non-finite samples after filtering")
            return None
        return np.ascontiguousarray(filtered_uv, dtype=np.float32)
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
def _extract_feature_chunk(
    wins: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract 20 node features and a hybrid dynamic connectivity graph."""
    _, channels, win_len = wins.shape
    if channels != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} channels, got {channels}")

    taper = torch.hann_window(
        win_len, periodic=False, device=wins.device, dtype=wins.dtype
    ).view(1, 1, -1)
    spectrum_r = torch.fft.rfft(wins * taper, dim=-1)
    power = spectrum_r.abs().square()
    freqs = torch.fft.rfftfreq(win_len, d=1.0 / SFREQ).to(wins.device)
    total_mask = (freqs >= BANDPASS_LFREQ) & (freqs <= BANDPASS_HFREQ)
    total_freqs = freqs[total_mask]
    total_band_power = power[..., total_mask].clamp_min(1e-12)
    total_power = total_band_power.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # 1) Five log relative band powers.
    bp_features: List[torch.Tensor] = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        relative = power[..., mask].sum(dim=-1, keepdim=True) / total_power
        bp_features.append(relative.clamp_min(1e-8).log())
    bandpower = torch.cat(bp_features, dim=-1)

    # 2) Six Hjorth / time-domain features in microvolts.
    d1 = torch.diff(wins, dim=-1)
    d2 = torch.diff(d1, dim=-1)
    var0 = torch.var(wins, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    var1 = torch.var(d1, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    var2 = torch.var(d2, dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)

    activity = var0.log()
    mobility = torch.sqrt(var1 / var0)
    complexity = torch.sqrt(var2 / var1) / mobility.clamp_min(1e-8)
    line_length = torch.log1p(torch.mean(torch.abs(d1), dim=-1, keepdim=True))
    rms = torch.log1p(torch.sqrt(torch.mean(wins.square(), dim=-1, keepdim=True)))
    sign = torch.signbit(wins)
    zero_cross = (
        torch.logical_xor(sign[..., 1:], sign[..., :-1])
        .sum(dim=-1, keepdim=True)
        .float()
        / float(max(1, win_len - 1))
    )
    time_features = torch.cat(
        [activity, mobility, complexity, line_length, rms, zero_cross], dim=-1
    )

    # 3) Four spectral-shape features.
    spectral_prob = total_band_power / total_power
    spectral_entropy = -(
        spectral_prob * spectral_prob.clamp_min(1e-12).log()
    ).sum(dim=-1, keepdim=True) / np.log(max(2, spectral_prob.shape[-1]))
    centroid = (
        spectral_prob * total_freqs.view(1, 1, -1)
    ).sum(dim=-1, keepdim=True) / BANDPASS_HFREQ
    cdf = spectral_prob.cumsum(dim=-1)
    edge_idx = (cdf >= 0.90).to(torch.int64).argmax(dim=-1)
    spectral_edge = total_freqs[edge_idx].unsqueeze(-1) / BANDPASS_HFREQ
    flatness = (
        total_band_power.log().mean(dim=-1, keepdim=True).exp()
        / total_band_power.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    spectral_shape = torch.cat(
        [spectral_entropy, centroid, spectral_edge, flatness], dim=-1
    )

    # 4) Five log-covariance features and correlation connectivity.
    centered = wins - wins.mean(dim=-1, keepdim=True)
    cov = torch.bmm(centered, centered.transpose(1, 2)) / float(max(1, win_len - 1))
    diag_mean = torch.diagonal(cov, dim1=1, dim2=2).mean(dim=-1).clamp_min(1e-6)
    eye = torch.eye(channels, device=wins.device, dtype=wins.dtype).unsqueeze(0)
    cov = cov + eye * (diag_mean * 1e-4).view(-1, 1, 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    log_eigvals = eigvals.clamp_min(1e-6).log()
    log_cov = torch.bmm(
        eigvecs,
        torch.bmm(torch.diag_embed(log_eigvals), eigvecs.transpose(1, 2)),
    )

    diag = torch.diagonal(log_cov, dim1=1, dim2=2).unsqueeze(-1)
    mean_conn = log_cov.mean(dim=-1, keepdim=True)
    std_conn = log_cov.std(dim=-1, keepdim=True, unbiased=False)
    large = eye * 1e5
    max_conn = (log_cov - large).max(dim=-1, keepdim=True).values
    min_conn = (log_cov + large).min(dim=-1, keepdim=True).values
    covariance_features = torch.cat(
        [diag, mean_conn, std_conn, max_conn, min_conn], dim=-1
    )

    node_features = torch.cat(
        [bandpower, time_features, spectral_shape, covariance_features], dim=-1
    )
    if node_features.shape[-1] != NODE_FEATURE_DIM:
        raise RuntimeError(
            f"Feature schema produced {node_features.shape[-1]} dims; "
            f"expected {NODE_FEATURE_DIM}"
        )
    node_features = torch.nan_to_num(
        node_features, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-30.0, 30.0)

    std0 = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    z = centered / std0
    corr = torch.bmm(z, z.transpose(1, 2)) / float(win_len)
    abs_corr = corr.abs().clamp(0.0, 1.0)

    # wPLI: |E[Im(Sxy)]| / E[|Im(Sxy)|]
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
    denominator = imag_cross.abs().mean(dim=-1).clamp_min(1e-8)
    wpli = (numerator / denominator).clamp(0.0, 1.0)

    functional = (
        DYNAMIC_WPLI_WEIGHT * wpli
        + DYNAMIC_CORR_WEIGHT * abs_corr
    ).clamp(0.0, 1.0)
    diagonal = torch.arange(channels, device=wins.device)
    functional[:, diagonal, diagonal] = 0.0

    dynamic_weight, dynamic_dst = torch.topk(
        functional, k=TOP_K_DYNAMIC, dim=-1, largest=True, sorted=True
    )
    return node_features, dynamic_dst, dynamic_weight


@torch.inference_mode()
def extract_recording(
    raw_data_uv: np.ndarray,
    seizure_intervals: Sequence[Tuple[float, float]],
) -> Dict:
    """Extract compact graph tensors for every 1-second-stride window in one EDF."""
    channels, total_samples = raw_data_uv.shape
    win_len = int(round(WINDOW_SEC * SFREQ))
    stride = int(round(WINDOW_STRIDE_SEC * SFREQ))

    if channels != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} channels, got {channels}")

    n_windows = (total_samples - win_len) // stride + 1
    if n_windows <= 0:
        raise ValueError("Recording is shorter than one analysis window")

    shape = (n_windows, channels, win_len)
    strides = (
        raw_data_uv.strides[1] * stride,
        raw_data_uv.strides[0],
        raw_data_uv.strides[1],
    )
    window_view = np.lib.stride_tricks.as_strided(
        raw_data_uv, shape=shape, strides=strides, writeable=False
    )

    x_parts: List[torch.Tensor] = []
    dst_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []

    feature_sum = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
    feature_sumsq = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
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
        "positive_windows": int(labels.sum().item()),
        "feature_sum": feature_sum,
        "feature_sumsq": feature_sumsq,
        "feature_count": int(feature_count),
    }


def build_all_subject_caches(
    overwrite: bool = False,
    max_subjects: int | None = None,
) -> None:
    """Build new v3 continuous caches directly from raw CHB-MIT BIDS EDF files."""
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

    print(f"[*] Preprocessing tag : {PREPROCESSING_TAG}")
    print(f"[*] Cache version     : {CACHE_VERSION}")
    print(f"[*] Feature dimension : {NODE_FEATURE_DIM}")
    print(f"[*] Device            : {DEVICE}")
    print(f"[*] Subjects          : {len(subject_dirs)}")
    print(f"[*] Output            : {PROCESSED_DATA_DIR}\n")

    manifest_rows: List[Dict] = []

    for sub_dir in subject_dirs:
        cache_path = PROCESSED_DATA_DIR / f"{sub_dir.name}_temporal_graphs.pt"
        if cache_path.exists() and not overwrite:
            print(f"[skip] {sub_dir.name}: v3 cache already exists")
            continue

        edf_files = sorted(sub_dir.rglob("*.edf"))
        if not edf_files:
            print(f"[skip] {sub_dir.name}: no EDF files found")
            continue

        recordings: List[Dict] = []
        subject_sum = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
        subject_sumsq = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float64)
        subject_count = 0
        total_windows = 0
        positive_windows = 0
        total_seizures = 0
        skipped_recordings = 0
        event_files_found = 0

        pbar = tqdm(edf_files, desc=f"{sub_dir.name}", ncols=100)
        for edf_path in pbar:
            tsv_path = _events_path_for_edf(edf_path)
            if tsv_path.exists():
                event_files_found += 1
            intervals = parse_seizure_events(tsv_path)
            raw_data = clean_raw(edf_path)
            if raw_data is None:
                skipped_recordings += 1
                continue

            try:
                rec = extract_recording(raw_data, intervals)
            except Exception as exc:
                skipped_recordings += 1
                print(f"[skip] {edf_path.name}: extraction failed: {exc}")
                continue

            rec["file_name"] = edf_path.name
            rec["recording_id"] = f"{sub_dir.name}/{edf_path.name}"
            recordings.append(rec)

            subject_sum += rec["feature_sum"]
            subject_sumsq += rec["feature_sumsq"]
            subject_count += rec["feature_count"]
            total_windows += rec["n_windows"]
            positive_windows += rec["positive_windows"]
            total_seizures += len(intervals)
            del raw_data

        if not recordings or subject_count <= 0:
            print(f"[error] {sub_dir.name}: no valid recordings; cache not written")
            if overwrite and cache_path.exists():
                cache_path.unlink()
            continue

        payload = {
            "cache_version": CACHE_VERSION,
            "preprocessing_tag": PREPROCESSING_TAG,
            "node_feature_dim": NODE_FEATURE_DIM,
            "subject": sub_dir.name,
            "recordings": recordings,
            "feature_sum": subject_sum,
            "feature_sumsq": subject_sumsq,
            "feature_count": int(subject_count),
            "total_windows": int(total_windows),
            "positive_windows": int(positive_windows),
            "total_seizures": int(total_seizures),
            "valid_recordings": int(len(recordings)),
            "skipped_recordings": int(skipped_recordings),
            "event_files_found": int(event_files_found),
            "sampling_rate_hz": float(SFREQ),
            "signal_unit": "microvolt",
        }
        torch.save(payload, cache_path)

        duration_hours = sum(float(r["duration_sec"]) for r in recordings) / 3600.0
        manifest_rows.append(
            {
                "subject": sub_dir.name,
                "edf_files": len(edf_files),
                "event_files_found": event_files_found,
                "valid_recordings": len(recordings),
                "skipped_recordings": skipped_recordings,
                "windows": total_windows,
                "positive_windows": positive_windows,
                "positive_fraction": positive_windows / max(1, total_windows),
                "seizures": total_seizures,
                "recording_hours": duration_hours,
                "cache_version": CACHE_VERSION,
                "feature_dim": NODE_FEATURE_DIM,
            }
        )
        print(
            f"[+] {sub_dir.name}: {len(recordings)}/{len(edf_files)} recordings | "
            f"{total_windows:,} windows | {positive_windows:,} positive | "
            f"{total_seizures} seizures -> {cache_path.name}"
        )

    if manifest_rows:
        manifest_path = PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(manifest_path, index=False)
        print(f"\n[+] Preprocessing manifest: {manifest_path}")

        total_edf = int(manifest["edf_files"].sum())
        total_valid = int(manifest["valid_recordings"].sum())
        total_seiz = int(manifest["seizures"].sum())
        total_hours = float(manifest["recording_hours"].sum())
        print(
            f"[*] Dataset QA: EDF={total_edf}, valid={total_valid}, "
            f"seizures={total_seiz}, hours={total_hours:.2f}"
        )

        if max_subjects is None:
            if total_edf != EXPECTED_CHBMIT_EDF_FILES:
                print(
                    f"[warn] Full CHB-MIT reference contains {EXPECTED_CHBMIT_EDF_FILES} EDF files; "
                    f"this BIDS tree exposed {total_edf}. Verify dataset completeness."
                )
            if total_seiz != EXPECTED_CHBMIT_SEIZURES:
                print(
                    f"[warn] CHB-MIT reference contains {EXPECTED_CHBMIT_SEIZURES} seizures; "
                    f"parsed BIDS annotations yielded {total_seiz}. Inspect events.tsv conversion before training."
                )


if __name__ == "__main__":
    build_all_subject_caches(overwrite=False)
