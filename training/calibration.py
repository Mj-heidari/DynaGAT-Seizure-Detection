"""
Prior correction and causal online score normalisation.

Two distinct distribution shifts separate a trained seizure detector from a
deployed one, and an earlier iteration corrected neither.

1. *Sampling prior shift.* Training epochs are built from a resampled clip pool
   in which ictal windows are ~50x more frequent than in continuous EEG. A model
   fitted on that pool estimates p(ictal | x, sampled), not p(ictal | x). The
   two differ by a constant in logit space, so the correction is exact:

       logit_true = logit_sampled - logit(pi_sampled) + logit(pi_true)

   This alone moves the usable decision threshold from ~0.96 back to a normal
   range and restores the meaning of the probability.

2. *Patient prior shift.* Even after (1), the mean output level differs between
   patients because their interictal EEG differs. A threshold chosen on
   validation patients therefore does not transfer. We remove this with a
   strictly causal, per-recording adaptive baseline of the logit itself: the
   score is how far the current logit sits above that recording's own recent
   past, in robust units. No labels, no future samples, no test-set statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    ADAPTIVE_MIX,
    ADAPTIVE_NORM,
    ADAPTIVE_TAU_WINDOWS,
    ADAPTIVE_WARMUP_WINDOWS,
)

__all__ = ["prior_correction_offset", "causal_adaptive_z", "OnlineScorer"]


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-9), 1.0 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def prior_correction_offset(sampled_prior: float, true_prior: float) -> float:
    """Additive logit offset that maps sampled-prior logits to true-prior logits."""
    return _logit(true_prior) - _logit(sampled_prior)


def causal_adaptive_z(
    values: np.ndarray,
    tau: float = ADAPTIVE_TAU_WINDOWS,
    warmup: int = ADAPTIVE_WARMUP_WINDOWS,
) -> np.ndarray:
    """
    Strictly causal robust z-score of a 1-D time series.

    The location and scale at index t use only indices < t, with a time-varying
    update rate 1/(t+1) that settles to 1/tau. Identical machinery to the
    feature-level baseline, applied here to the detector output.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float64)
    m = float(v[0])
    d = 0.0
    inv_tau = 1.0 / float(tau)
    for t in range(n):
        out[t] = (v[t] - m) / (d + 1e-2)
        a = max(1.0 / (t + 1.0), inv_tau)
        d = (1.0 - a) * d + a * abs(v[t] - m)
        m = (1.0 - a) * m + a * v[t]
    warm = min(int(warmup), n)
    out[:warm] = 0.0
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0).astype(
        np.float32
    )


@dataclass
class OnlineScorer:
    """
    Converts raw model logits into the score the alarm logic thresholds.

    Parameters
    ----------
    prior_offset : additive logit offset from :func:`prior_correction_offset`.
    logit_mean, logit_std : standardisation constants estimated on the
        *validation* patients only, used to put the absolute term on the same
        scale as the adaptive term.
    mix : weight of the adaptive term, 0 = absolute only, 1 = adaptive only.
    """

    prior_offset: float = 0.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mix: float = ADAPTIVE_MIX
    tau: float = ADAPTIVE_TAU_WINDOWS
    warmup: int = ADAPTIVE_WARMUP_WINDOWS
    enabled: bool = ADAPTIVE_NORM

    def corrected_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) + float(self.prior_offset)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        z = self.corrected_logits(logits)
        return (1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))).astype(np.float32)

    def score_recording(self, logits: np.ndarray) -> np.ndarray:
        """Causal detector score for the ordered logits of ONE recording."""
        z = self.corrected_logits(logits)
        absolute = (z - self.logit_mean) / max(float(self.logit_std), 1e-6)
        if not self.enabled or self.mix <= 0.0:
            return absolute.astype(np.float32)
        adaptive = causal_adaptive_z(z, tau=self.tau, warmup=self.warmup)
        return ((1.0 - self.mix) * absolute + self.mix * adaptive).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "prior_offset": float(self.prior_offset),
            "logit_mean": float(self.logit_mean),
            "logit_std": float(self.logit_std),
            "mix": float(self.mix),
            "tau": float(self.tau),
            "warmup": int(self.warmup),
            "enabled": bool(self.enabled),
        }
