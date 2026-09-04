"""
Node-feature extraction for DynaGAT.

Two blocks per channel and window:

  * 26 absolute / shape descriptors  (spectral, Hjorth, amplitude, complexity,
    intra-window variability, and cross-channel spatial contrast)
  * 8 causal trailing-baseline descriptors, each expressing how far the current
    value of a key descriptor deviates from that *same recording's* own recent
    past, measured in robust units.

The second block is what makes the representation patient-adaptive without
using any label or any future sample: an exponential causal baseline with a
~5 min time constant is subtracted, and the residual is scaled by a causal
estimate of its own dispersion. In an earlier iteration all normalisation came from training
patients only, so a held-out patient was scored under a shifted input
distribution; the relative block removes most of that shift.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import signal as sp_signal

from config import (
    ABS_FEATURE_DIM,
    BANDPASS_HFREQ,
    BANDPASS_LFREQ,
    BANDS,
    BASELINE_TAU_WINDOWS,
    BASELINE_WARMUP_WINDOWS,
    NODE_FEATURE_DIM,
    NUM_NODES,
    REL_CLIP,
    REL_SOURCE_INDICES,
    SFREQ,
)

__all__ = ["extract_absolute_features", "apply_causal_baseline", "FEATURE_NAMES"]

FEATURE_NAMES = (
    [f"logrel_bp_{lo:g}_{hi:g}" for lo, hi in BANDS]
    + [
        "log_total_power",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "log_line_length",
        "log_rms",
        "zero_cross_rate",
        "log_teager_energy",
        "spectral_entropy",
        "spectral_centroid",
        "spectral_edge90",
        "spectral_flatness",
        "log_kurtosis",
        "skewness",
        "perm_entropy_o3",
        "subframe_cv_line_length",
        "subframe_cv_power",
        "spatial_z_line_length",
        "spatial_z_total_power",
    ]
    + [f"rel::{i}" for i in REL_SOURCE_INDICES]
)
assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (len(FEATURE_NAMES), NODE_FEATURE_DIM)


@torch.inference_mode()
def extract_absolute_features(wins: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    wins : [B, N, L] float32 windowed EEG in microvolts.

    Returns
    -------
    [B, N, ABS_FEATURE_DIM] float32
    """
    b, n, length = wins.shape
    if n != NUM_NODES:
        raise ValueError(f"expected {NUM_NODES} channels, got {n}")

    taper = torch.hann_window(
        length, periodic=False, device=wins.device, dtype=wins.dtype
    ).view(1, 1, -1)
    spec = torch.fft.rfft(wins * taper, dim=-1)
    power = spec.abs().square()
    freqs = torch.fft.rfftfreq(length, d=1.0 / SFREQ).to(wins.device)

    band_mask = (freqs >= BANDPASS_LFREQ) & (freqs <= BANDPASS_HFREQ)
    in_band_freqs = freqs[band_mask]
    in_band_power = power[..., band_mask].clamp_min(1e-12)
    total_power = in_band_power.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # (0-6) log relative band powers -------------------------------------- #
    rel_bp = []
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        rel_bp.append(power[..., mask].sum(dim=-1, keepdim=True) / total_power)
    log_rel_bp = torch.cat(rel_bp, dim=-1).clamp_min(1e-8).log()

    # (7) absolute in-band power ------------------------------------------- #
    log_total = total_power.log()

    # (8-10) Hjorth --------------------------------------------------------- #
    d1 = torch.diff(wins, dim=-1)
    d2 = torch.diff(d1, dim=-1)
    v0 = wins.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    v1 = d1.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    v2 = d2.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    activity = v0.log()
    mobility = (v1 / v0).sqrt()
    complexity = (v2 / v1).sqrt() / mobility.clamp_min(1e-6)

    # (11-14) amplitude / waveform ------------------------------------------ #
    line_length = d1.abs().mean(dim=-1, keepdim=True)
    log_line_length = torch.log1p(line_length)
    log_rms = torch.log1p(wins.square().mean(dim=-1, keepdim=True).sqrt())
    sign = torch.signbit(wins)
    zero_cross = (
        torch.logical_xor(sign[..., 1:], sign[..., :-1]).sum(dim=-1, keepdim=True).float()
        / float(max(1, length - 1))
    )
    teager = (wins[..., 1:-1].square() - wins[..., 2:] * wins[..., :-2]).abs()
    log_teager = torch.log1p(teager.mean(dim=-1, keepdim=True))

    # (15-18) spectral shape ------------------------------------------------ #
    prob = in_band_power / total_power
    spectral_entropy = -(prob * prob.clamp_min(1e-12).log()).sum(
        dim=-1, keepdim=True
    ) / float(np.log(max(2, prob.shape[-1])))
    centroid = (prob * in_band_freqs.view(1, 1, -1)).sum(dim=-1, keepdim=True) / BANDPASS_HFREQ
    cdf = prob.cumsum(dim=-1)
    edge_idx = (cdf >= 0.90).to(torch.int64).argmax(dim=-1)
    spectral_edge = in_band_freqs[edge_idx].unsqueeze(-1) / BANDPASS_HFREQ
    flatness = (
        in_band_power.log().mean(dim=-1, keepdim=True).exp()
        / in_band_power.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )

    # (19-20) higher moments ------------------------------------------------ #
    centered = wins - wins.mean(dim=-1, keepdim=True)
    sd = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    z = centered / sd
    kurtosis = z.pow(4).mean(dim=-1, keepdim=True)
    log_kurt = torch.log1p(kurtosis.clamp_min(0.0))
    skewness = z.pow(3).mean(dim=-1, keepdim=True).clamp(-10.0, 10.0)

    # (21) permutation entropy, order 3, delay 4 ---------------------------- #
    delay = 4
    a = wins[..., : length - 2 * delay]
    bb = wins[..., delay : length - delay]
    c = wins[..., 2 * delay :]
    pattern = (
        (bb > a).to(torch.int64) * 4 + (c > a).to(torch.int64) * 2 + (c > bb).to(torch.int64)
    )
    hist = torch.zeros(b, n, 8, device=wins.device, dtype=wins.dtype)
    hist.scatter_add_(2, pattern, torch.ones_like(pattern, dtype=wins.dtype))
    hist = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1.0)
    perm_entropy = -(hist * hist.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True) / float(
        np.log(6.0)
    )

    # (22-23) intra-window non-stationarity --------------------------------- #
    n_sub = 8
    sub_len = length // n_sub
    sub = wins[..., : n_sub * sub_len].reshape(b, n, n_sub, sub_len)
    sub_ll = sub.diff(dim=-1).abs().mean(dim=-1)
    sub_pw = sub.var(dim=-1, unbiased=False).clamp_min(1e-8)
    cv_ll = (sub_ll.std(dim=-1, unbiased=False) / sub_ll.mean(dim=-1).clamp_min(1e-6)).unsqueeze(-1)
    cv_pw = (sub_pw.std(dim=-1, unbiased=False) / sub_pw.mean(dim=-1).clamp_min(1e-6)).unsqueeze(-1)

    # (24-25) spatial contrast across the montage --------------------------- #
    def _spatial_z(v: torch.Tensor) -> torch.Tensor:
        mu = v.mean(dim=1, keepdim=True)
        sdv = v.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (v - mu) / sdv

    spatial_ll = _spatial_z(log_line_length)
    spatial_pw = _spatial_z(log_total)

    feats = torch.cat(
        [
            log_rel_bp, log_total, activity, mobility, complexity,
            log_line_length, log_rms, zero_cross, log_teager,
            spectral_entropy, centroid, spectral_edge, flatness,
            log_kurt, skewness, perm_entropy, cv_ll, cv_pw,
            spatial_ll, spatial_pw,
        ],
        dim=-1,
    )
    if feats.shape[-1] != ABS_FEATURE_DIM:
        raise RuntimeError(
            f"feature schema produced {feats.shape[-1]} dims, expected {ABS_FEATURE_DIM}"
        )
    return torch.nan_to_num(feats, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)


def _causal_adaptive_baseline(x: np.ndarray, tau: float, warmup: int):
    """
    Strictly causal robust baseline with a time-varying rate.

    For t < tau the update rate is 1/(t+1), i.e. an exact expanding-window mean,
    after which it settles to an exponential moving average with time constant
    ``tau``. This removes the long initialisation transient that a fixed-rate
    EMA would introduce at the start of every recording.

    Returns (mean_prev, scale_prev), both using only samples strictly before t.
    """
    t_len = x.shape[0]
    mean_prev = np.empty_like(x)
    scale_prev = np.empty_like(x)
    m = x[0].copy()
    d = np.zeros_like(m)
    inv_tau = 1.0 / float(tau)
    for t in range(t_len):
        mean_prev[t] = m
        scale_prev[t] = d
        a = max(1.0 / (t + 1.0), inv_tau)
        d = (1.0 - a) * d + a * np.abs(x[t] - m)
        m = (1.0 - a) * m + a * x[t]
    return mean_prev, scale_prev


def apply_causal_baseline(abs_feats: np.ndarray) -> np.ndarray:
    """
    Append the causal trailing-baseline block.

    Parameters
    ----------
    abs_feats : [T, N, ABS_FEATURE_DIM] float32, time-ordered windows of one
                continuous recording.

    Returns
    -------
    [T, N, NODE_FEATURE_DIM] float32
    """
    if abs_feats.ndim != 3 or abs_feats.shape[-1] != ABS_FEATURE_DIM:
        raise ValueError(f"expected [T, N, {ABS_FEATURE_DIM}], got {abs_feats.shape}")
    t_len = abs_feats.shape[0]
    src = np.ascontiguousarray(
        abs_feats[:, :, list(REL_SOURCE_INDICES)].astype(np.float64)
    )                                                       # [T, N, R]

    if t_len < 4:
        rel = np.zeros_like(src, dtype=np.float32)
        return np.concatenate([abs_feats, rel], axis=-1).astype(np.float32)

    mean_prev, scale_prev = _causal_adaptive_baseline(
        src, BASELINE_TAU_WINDOWS, BASELINE_WARMUP_WINDOWS
    )
    rel = (src - mean_prev) / (scale_prev + 1e-2)
    rel = np.clip(np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0), -REL_CLIP, REL_CLIP)

    # The baseline is not yet informative during warm-up; emit the neutral value
    # rather than a spurious deviation. Reported as a protocol detail.
    warm = min(int(BASELINE_WARMUP_WINDOWS), t_len)
    rel[:warm] = 0.0
    return np.concatenate([abs_feats, rel.astype(np.float32)], axis=-1).astype(np.float32)
