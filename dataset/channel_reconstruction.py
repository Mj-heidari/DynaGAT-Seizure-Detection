from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import bids_loader as bl


# The current PhysioNet RECORDS file contains 686 paths: the often-quoted
# 664-file count predates the 22 EDF files of chb24. The seizure total is 198.
EXPECTED_CURRENT_RECORDS = 686
EXPECTED_SEIZURES = 198

# Electrode names needed to synthesize the canonical bipolar montage.
_ENDPOINTS = {
    endpoint
    for channel in bl.CANONICAL_CHANNELS
    for endpoint in channel.split("-", 1)
}


def _fix_electrode_token(token: str) -> str:
    token = token.strip().upper()
    # One CHB-MIT summary prints O1/O2 in a form that can appear as 01/02 in
    # converted metadata. Treat those as electrode spelling variants only.
    if token == "01":
        return "O1"
    if token == "02":
        return "O2"
    return token


def _normalize_label(name: str) -> str:
    text = name.upper().strip().replace("EEG ", "").replace(" ", "")
    parts = text.split("-")
    if parts:
        parts[0] = _fix_electrode_token(parts[0])
    if len(parts) > 1:
        parts[1] = _fix_electrode_token(parts[1])
    return "-".join(parts)


def _direct_candidate(
    normalized: Dict[str, str], canonical: str
) -> Tuple[str, float] | None:
    """Find direct bipolar channel, including duplicate suffixes and reversal."""
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


def _referential_candidates(
    normalized: Dict[str, str], electrode: str
) -> List[Tuple[str, str]]:
    """Return (original_name, common_reference_key) for one electrode."""
    out: List[Tuple[str, str]] = []
    electrode = _fix_electrode_token(electrode)

    for original, norm in normalized.items():
        if norm == electrode:
            out.append((original, "__BARE__"))
            continue

        prefix = electrode + "-"
        if not norm.startswith(prefix):
            continue
        suffix = norm[len(prefix) :]
        # Do not misinterpret an existing bipolar derivation (F7-T7) or a
        # duplicate suffix (T8-P8-0) as a common-reference source channel.
        if not suffix or suffix.isdigit() or suffix in _ENDPOINTS:
            continue
        if "-" in suffix:
            continue
        out.append((original, suffix))
    return out


def _derived_candidate(
    normalized: Dict[str, str], canonical: str
) -> Tuple[str, str, str] | None:
    """Find two channels sharing a reference so A-ref - B-ref = A-B."""
    a, b = canonical.split("-", 1)
    a_candidates = _referential_candidates(normalized, a)
    b_candidates = _referential_candidates(normalized, b)
    if not a_candidates or not b_candidates:
        return None

    by_ref_a = {ref: name for name, ref in a_candidates}
    by_ref_b = {ref: name for name, ref in b_candidates}
    common = set(by_ref_a).intersection(by_ref_b)
    if not common:
        return None

    def rank(ref: str) -> Tuple[int, str]:
        preferred = {"REF": 0, "CS2": 1, "__BARE__": 2}
        return preferred.get(ref, 10), ref

    ref = sorted(common, key=rank)[0]
    return by_ref_a[ref], by_ref_b[ref], ref


def _canonical_plan(raw: mne.io.BaseRaw) -> Tuple[List[Tuple], List[str]]:
    normalized = {name: _normalize_label(name) for name in raw.ch_names}
    plan: List[Tuple] = []
    missing: List[str] = []

    for canonical in bl.CANONICAL_CHANNELS:
        direct = _direct_candidate(normalized, canonical)
        if direct is not None:
            source, sign = direct
            plan.append(("direct", canonical, source, sign))
            continue

        derived = _derived_candidate(normalized, canonical)
        if derived is not None:
            source_a, source_b, ref = derived
            plan.append(("derived", canonical, source_a, source_b, ref))
            continue

        missing.append(canonical)

    return plan, missing


def clean_raw_with_reconstruction(edf_path: Path) -> np.ndarray | None:
    """
    Load the canonical 18 bipolar derivations.

    Standard CHB-MIT files use direct bipolar channels. For recordings such as
    chb12_27/28/29, which switch to common-reference or monopolar montages,
    synthesize A-B as (A-ref) - (B-ref). This preserves the exact physical
    bipolar derivation instead of discarding seizure-containing recordings.
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, bl.SFREQ, rtol=0.0, atol=1e-6):
            print(
                f"[skip] {edf_path.name}: unexpected sfreq={sfreq:g} Hz; "
                f"expected native CHB-MIT {bl.SFREQ:g} Hz"
            )
            return None

        plan, missing = _canonical_plan(raw)
        if missing:
            print(f"[skip] {edf_path.name}: cannot construct canonical channels {missing}")
            return None

        source_names: List[str] = []
        for item in plan:
            if item[0] == "direct":
                source_names.append(item[2])
            else:
                source_names.extend([item[2], item[3]])
        # Stable de-duplication keeps the EDF's channel order deterministic.
        needed = set(source_names)
        source_names = [name for name in raw.ch_names if name in needed]

        raw.pick(source_names)
        raw.load_data()
        source_data = raw.get_data()
        index = {name: i for i, name in enumerate(raw.ch_names)}

        canonical_v = np.empty(
            (len(bl.CANONICAL_CHANNELS), source_data.shape[1]), dtype=np.float32
        )
        reconstructed_refs = set()
        for out_idx, item in enumerate(plan):
            if item[0] == "direct":
                _, _, source, sign = item
                canonical_v[out_idx] = source_data[index[source]] * float(sign)
            else:
                _, _, source_a, source_b, ref = item
                canonical_v[out_idx] = (
                    source_data[index[source_a]] - source_data[index[source_b]]
                )
                reconstructed_refs.add(ref)

        if reconstructed_refs:
            refs = ",".join(sorted(reconstructed_refs))
            print(
                f"[reconstruct] {edf_path.name}: canonical bipolar montage "
                f"derived from common reference(s) {refs}"
            )

        filtered_uv = bl._causal_bandpass(canonical_v, sfreq)
        filtered_uv *= 1e6
        if not np.isfinite(filtered_uv).all():
            print(f"[skip] {edf_path.name}: non-finite samples after filtering")
            return None
        return np.ascontiguousarray(filtered_uv, dtype=np.float32)
    except Exception as exc:
        print(f"[skip] {edf_path.name}: {exc}")
        return None


def install_robust_loader() -> None:
    """Install montage reconstruction into the existing preprocessing pipeline."""
    bl.clean_raw = clean_raw_with_reconstruction
    bl.EXPECTED_CHBMIT_EDF_FILES = EXPECTED_CURRENT_RECORDS
    bl.EXPECTED_CHBMIT_SEIZURES = EXPECTED_SEIZURES


def _subject_sort_key(subject: str) -> Tuple[int, str]:
    try:
        return int(subject.split("-")[-1]), subject
    except Exception:
        return 10_000, subject


def _build_subject(sub_dir: Path) -> Dict | None:
    edf_files = sorted(sub_dir.rglob("*.edf"))
    if not edf_files:
        print(f"[skip] {sub_dir.name}: no EDF files found")
        return None

    recordings: List[Dict] = []
    subject_sum = torch.zeros(bl.NODE_FEATURE_DIM, dtype=torch.float64)
    subject_sumsq = torch.zeros(bl.NODE_FEATURE_DIM, dtype=torch.float64)
    subject_count = 0
    total_windows = 0
    positive_windows = 0
    total_seizures = 0
    skipped_recordings = 0
    event_files_found = 0

    pbar = tqdm(edf_files, desc=f"{sub_dir.name}", ncols=100)
    for edf_path in pbar:
        tsv_path = bl._events_path_for_edf(edf_path)
        if tsv_path.exists():
            event_files_found += 1
        intervals = bl.parse_seizure_events(tsv_path)
        raw_data = clean_raw_with_reconstruction(edf_path)
        if raw_data is None:
            skipped_recordings += 1
            continue

        try:
            rec = bl.extract_recording(raw_data, intervals)
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
        return None

    cache_path = bl.PROCESSED_DATA_DIR / f"{sub_dir.name}_temporal_graphs.pt"
    payload = {
        "cache_version": bl.CACHE_VERSION,
        "preprocessing_tag": bl.PREPROCESSING_TAG,
        "node_feature_dim": bl.NODE_FEATURE_DIM,
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
        "sampling_rate_hz": float(bl.SFREQ),
        "signal_unit": "microvolt",
    }
    bl._atomic_torch_save(payload, cache_path)

    duration_hours = sum(float(r["duration_sec"]) for r in recordings) / 3600.0
    row = {
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
        "cache_version": bl.CACHE_VERSION,
        "feature_dim": bl.NODE_FEATURE_DIM,
    }
    print(
        f"[+] {sub_dir.name}: {len(recordings)}/{len(edf_files)} recordings | "
        f"{total_windows:,} windows | {positive_windows:,} positive | "
        f"{total_seizures} seizures -> {cache_path.name}"
    )
    return row


def rebuild_selected_subjects(
    subjects: Sequence[str], overwrite: bool = False
) -> None:
    """Rebuild only requested subjects and merge their rows into the manifest."""
    if not bl.BIDS_ROOT.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {bl.BIDS_ROOT}")

    requested = list(dict.fromkeys(str(s) for s in subjects))
    rows: List[Dict] = []
    for subject in requested:
        sub_dir = bl.BIDS_ROOT / subject
        if not sub_dir.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
        cache_path = bl.PROCESSED_DATA_DIR / f"{subject}_temporal_graphs.pt"
        if cache_path.exists() and not overwrite:
            print(f"[skip] {subject}: cache exists; use --overwrite to rebuild it")
            continue
        row = _build_subject(sub_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        return

    manifest_path = bl.PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[~manifest["subject"].astype(str).isin([r["subject"] for r in rows])]
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
    else:
        manifest = pd.DataFrame(rows)

    order = sorted(range(len(manifest)), key=lambda i: _subject_sort_key(str(manifest.iloc[i]["subject"])))
    manifest = manifest.iloc[order].reset_index(drop=True)
    bl._atomic_dataframe_csv(manifest, manifest_path)
    print(f"\n[+] Preprocessing manifest updated: {manifest_path}")

    total_edf = int(manifest["edf_files"].sum())
    total_valid = int(manifest["valid_recordings"].sum())
    total_seiz = int(manifest["seizures"].sum())
    total_hours = float(manifest["recording_hours"].sum())
    print(
        f"[*] Dataset QA: EDF={total_edf}, valid={total_valid}, "
        f"seizures={total_seiz}, hours={total_hours:.2f}"
    )

    if len(manifest) >= 23:
        if total_edf != EXPECTED_CURRENT_RECORDS:
            print(
                f"[warn] Current PhysioNet RECORDS contains {EXPECTED_CURRENT_RECORDS} EDF paths; "
                f"manifest contains {total_edf}."
            )
        if total_seiz != EXPECTED_SEIZURES:
            print(
                f"[warn] CHB-MIT contains {EXPECTED_SEIZURES} seizures; "
                f"manifest contains {total_seiz}."
            )
        if int(manifest["skipped_recordings"].sum()) != 0:
            print(
                f"[warn] {int(manifest['skipped_recordings'].sum())} recordings remain skipped."
            )
