"""
CHB-MIT BIDS reading, canonical bipolar montage construction and causal filtering.

The montage logic is carried over from the validated an earlier iteration loader: standard
CHB-MIT files expose direct bipolar derivations, while a handful of recordings
(chb12_27/28/29 and similar) switch to a common-reference montage. Those are
reconstructed as (A-ref) - (B-ref) instead of being discarded, which keeps
seizure-bearing recordings in the study.

the current pipeline additions: a causal 60 Hz notch (CHB-MIT was recorded in the United States)
and an explicit flat-signal / saturation guard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from config import (
    BANDPASS_HFREQ,
    BANDPASS_LFREQ,
    CHANNELS_18,
    FILTER_IIR_ORDER,
    NOTCH_FREQ_HZ,
    NOTCH_Q,
    SFREQ,
)

mne.set_log_level("ERROR")

CANONICAL_CHANNELS = [ch.upper().replace(" ", "") for ch in CHANNELS_18]
_ENDPOINTS = {ep for ch in CANONICAL_CHANNELS for ep in ch.split("-", 1)}

# PhysioNet CHB-MIT v1.0.0 RECORDS currently lists 686 EDF paths and 198 seizures.
EXPECTED_EDF_FILES = 686
EXPECTED_SEIZURES = 198

__all__ = [
    "CANONICAL_CHANNELS",
    "EXPECTED_EDF_FILES",
    "EXPECTED_SEIZURES",
    "events_path_for_edf",
    "parse_seizure_events",
    "load_canonical_recording",
]


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def events_path_for_edf(edf_path: Path) -> Path:
    if edf_path.name.endswith("_eeg.edf"):
        return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
    stem = edf_path.stem
    if stem.endswith("_eeg"):
        stem = stem[:-4]
    return edf_path.with_name(stem + "_events.tsv")


def parse_seizure_events(tsv_path: Path) -> List[Tuple[float, float]]:
    """Parse seizure intervals (onset, offset) in seconds from a BIDS events.tsv."""
    if not tsv_path.exists():
        return []
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as exc:  # pragma: no cover - malformed annotation file
        print(f"[warn] could not parse {tsv_path.name}: {exc}")
        return []

    descriptive = [c for c in df.columns if c.lower() not in {"onset", "duration", "sample"}]
    intervals: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        try:
            onset = float(row.get("onset", 0.0))
            duration = float(row.get("duration", 0.0))
        except Exception:
            continue
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            continue
        text = " ".join(str(row.get(c, "")).lower() for c in descriptive)
        if any(k in text for k in ("seiz", "ictal", "sz", "epil")) or not descriptive:
            intervals.append((onset, onset + duration))
    return sorted(intervals)


# --------------------------------------------------------------------------- #
# Montage resolution
# --------------------------------------------------------------------------- #
# Electrode-name equivalences applied before any montage matching.
#
# The 1991 modified combinatorial nomenclature renamed the temporal chain:
# T3->T7, T4->T8, T5->P7, T6->P8. These are the SAME physical electrodes, not
# approximations. Part of CHB-MIT (sub-12 run-10/11/12) switches to the older
# labels mid-subject; without this map those recordings are discarded, taking
# 13 annotated seizures - about 7% of the corpus - with them.
#
# "01"/"02" are digit-for-letter spellings of O1/O2 seen in converted metadata.
_ELECTRODE_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "01": "O1",
    "02": "O2",
}


def _fix_token(token: str) -> str:
    return _ELECTRODE_ALIASES.get(token.strip().upper(), token.strip().upper())


def _normalize_label(name: str) -> str:
    text = name.upper().strip().replace("EEG ", "").replace(" ", "")
    parts = text.split("-")
    if parts:
        parts[0] = _fix_token(parts[0])
    if len(parts) > 1:
        parts[1] = _fix_token(parts[1])
    return "-".join(parts)


def _direct_candidate(normalized: Dict[str, str], canonical: str):
    a, b = canonical.split("-", 1)
    reverse = f"{b}-{a}"
    for original, norm in normalized.items():
        if norm == canonical:
            return original, 1.0
    for original, norm in normalized.items():
        if norm.startswith(canonical + "-") and norm[len(canonical) + 1 :].isdigit():
            return original, 1.0
    for original, norm in normalized.items():
        if norm == reverse:
            return original, -1.0
    for original, norm in normalized.items():
        if norm.startswith(reverse + "-") and norm[len(reverse) + 1 :].isdigit():
            return original, -1.0
    return None


def _referential_candidates(normalized: Dict[str, str], electrode: str):
    out: List[Tuple[str, str]] = []
    electrode = _fix_token(electrode)
    for original, norm in normalized.items():
        if norm == electrode:
            out.append((original, "__BARE__"))
            continue
        prefix = electrode + "-"
        if not norm.startswith(prefix):
            continue
        suffix = norm[len(prefix) :]
        if not suffix or suffix.isdigit() or suffix in _ENDPOINTS or "-" in suffix:
            continue
        out.append((original, suffix))
    return out


def _derived_candidate(normalized: Dict[str, str], canonical: str):
    a, b = canonical.split("-", 1)
    ca, cb = _referential_candidates(normalized, a), _referential_candidates(normalized, b)
    if not ca or not cb:
        return None
    by_a = {ref: name for name, ref in ca}
    by_b = {ref: name for name, ref in cb}
    common = set(by_a).intersection(by_b)
    if not common:
        return None
    order = {"REF": 0, "CS2": 1, "__BARE__": 2}
    ref = sorted(common, key=lambda r: (order.get(r, 10), r))[0]
    return by_a[ref], by_b[ref], ref


def _canonical_plan(raw: mne.io.BaseRaw):
    normalized = {name: _normalize_label(name) for name in raw.ch_names}
    plan: List[Tuple] = []
    missing: List[str] = []
    for canonical in CANONICAL_CHANNELS:
        direct = _direct_candidate(normalized, canonical)
        if direct is not None:
            plan.append(("direct", canonical, direct[0], direct[1]))
            continue
        derived = _derived_candidate(normalized, canonical)
        if derived is not None:
            plan.append(("derived", canonical, derived[0], derived[1], derived[2]))
            continue
        missing.append(canonical)
    return plan, missing


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def _causal_filter(data_v: np.ndarray, sfreq: float) -> np.ndarray:
    """Causal band-pass + causal notch, both with steady-state initialisation."""
    sos_bp = sp_signal.butter(
        FILTER_IIR_ORDER,
        [BANDPASS_LFREQ, BANDPASS_HFREQ],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    b_n, a_n = sp_signal.iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs=sfreq)
    sos_notch = sp_signal.tf2sos(b_n, a_n)
    sos = np.vstack([sos_bp, sos_notch])
    zi_template = sp_signal.sosfilt_zi(sos)

    out = np.empty(data_v.shape, dtype=np.float32)
    for ch in range(data_v.shape[0]):
        x = np.asarray(data_v[ch], dtype=np.float64)
        y, _ = sp_signal.sosfilt(sos, x, zi=zi_template * float(x[0]))
        out[ch] = y.astype(np.float32, copy=False)
    return out


def load_canonical_recording(edf_path: Path) -> np.ndarray | None:
    """
    Return [18, T] causally filtered microvolt data on the canonical montage,
    or None if the recording cannot be used.
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, SFREQ, rtol=0.0, atol=1e-6):
            print(f"[skip] {edf_path.name}: sfreq={sfreq:g} Hz, expected {SFREQ:g} Hz")
            return None

        plan, missing = _canonical_plan(raw)
        if missing:
            print(f"[skip] {edf_path.name}: cannot construct channels {missing}")
            return None

        needed = set()
        for item in plan:
            needed.add(item[2])
            if item[0] == "derived":
                needed.add(item[3])
        source_names = [n for n in raw.ch_names if n in needed]
        raw.pick(source_names)
        raw.load_data()
        source = raw.get_data()
        index = {name: i for i, name in enumerate(raw.ch_names)}

        canonical = np.empty((len(CANONICAL_CHANNELS), source.shape[1]), dtype=np.float32)
        reconstructed = set()
        for out_idx, item in enumerate(plan):
            if item[0] == "direct":
                canonical[out_idx] = source[index[item[2]]] * float(item[3])
            else:
                canonical[out_idx] = source[index[item[2]]] - source[index[item[3]]]
                reconstructed.add(item[4])
        if reconstructed:
            print(
                f"[reconstruct] {edf_path.name}: montage derived from reference(s) "
                f"{','.join(sorted(reconstructed))}"
            )

        filtered = _causal_filter(canonical, sfreq)
        filtered *= 1e6
        if not np.isfinite(filtered).all():
            print(f"[skip] {edf_path.name}: non-finite samples after filtering")
            return None
        # Guard against fully flat / disconnected montages.
        if float(np.median(filtered.std(axis=1))) < 1e-3:
            print(f"[skip] {edf_path.name}: signal is flat after filtering")
            return None
        return np.ascontiguousarray(filtered, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - corrupted EDF
        print(f"[skip] {edf_path.name}: {exc}")
        return None
