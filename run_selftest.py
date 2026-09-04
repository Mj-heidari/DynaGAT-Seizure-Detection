"""
Self-test for DynaGAT.

Runs entirely on synthetic data, so it can be executed before the CHB-MIT cache
exists. It checks the properties that silently break seizure-detection
pipelines: temporal causality, correctness of the Granger estimator, causality
of the online normalisation, exactness of the prior correction, and the event
metric definitions. Finally it builds a miniature synthetic cohort and runs a
complete fold end to end.

    python run_selftest.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from dataset.causal_graph import (
    build_causal_topk,
    granger_causality_batch,
    granger_reference_naive,
)
from dataset.features import apply_causal_baseline, extract_absolute_features
from evaluation.events import (
    evaluate_events,
    generate_alarms,
    match_events,
    window_end_times,
    window_metrics,
)
from evaluation.operating_point import select_operating_point
from models.dynagat import DynaGAT
from training.calibration import OnlineScorer, causal_adaptive_z, prior_correction_offset

PASS, FAIL = "  [ok]  ", "  [FAIL]"
_failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print((PASS if condition else FAIL) + name + (f"   {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


# --------------------------------------------------------------------------- #
def test_granger() -> None:
    print("\n1. Directed Granger-causal graph")
    torch.manual_seed(0)
    w = torch.randn(3, 6, 512)
    fast = granger_causality_batch(w, order=4, ridge=1e-3)
    slow = granger_reference_naive(w, order=4, ridge=1e-3)
    err = float((fast - slow).abs().max())
    check("fast estimator matches the explicit reference", err < 1e-3, f"max |diff| = {err:.2e}")

    torch.manual_seed(1)
    n, length = 6, 1024
    e = torch.randn(1, n, length)
    x = torch.zeros(1, n, length)
    for t in range(3, length):
        x[0, 0, t] = 0.5 * x[0, 0, t - 1] + e[0, 0, t]
        x[0, 1, t] = 0.3 * x[0, 1, t - 1] + 0.8 * x[0, 0, t - 2] + e[0, 1, t]
        x[0, 2, t] = 0.5 * x[0, 2, t - 1] + e[0, 2, t]
        x[0, 3, t] = 0.2 * x[0, 3, t - 1] + 0.9 * x[0, 2, t - 1] + e[0, 3, t]
        x[0, 4, t] = 0.4 * x[0, 4, t - 1] + e[0, 4, t]
        x[0, 5, t] = 0.4 * x[0, 5, t - 1] + e[0, 5, t]
    gc = granger_causality_batch(x, order=6, ridge=1e-4)[0]
    true_edges = float(min(gc[1, 0], gc[3, 2]))
    others = gc.clone()
    others[1, 0] = others[3, 2] = 0.0
    spurious = float(others.max())
    check(
        "recovers the injected directed edges 0->1 and 2->3",
        true_edges > 10 * max(spurious, 1e-6),
        f"true >= {true_edges:.3f} vs strongest spurious {spurious:.3f}",
    )
    check(
        "estimator is asymmetric (direction is not an artefact)",
        gc[1, 0] > 20 * gc[0, 1] and gc[3, 2] > 20 * gc[2, 3],
        f"0->1 {gc[1,0]:.3f} vs 1->0 {gc[0,1]:.3f}",
    )

    i_dst, i_w, o_dst, o_w = build_causal_topk(gc.unsqueeze(0), k=C.TOP_K_CAUSAL)
    check(
        "top-k normalisation is scale free",
        bool(
            torch.allclose(
                build_causal_topk(gc.unsqueeze(0) * 7.3, k=C.TOP_K_CAUSAL)[1],
                i_w,
                atol=1e-4,
            )
        ),
    )
    check("parent of node 1 is node 0", int(i_dst[0, 1, 0]) == 0)
    check("child of node 0 is node 1", int(o_dst[0, 0, 0]) == 1)


def test_features() -> None:
    print("\n2. Node features")
    torch.manual_seed(0)
    w = torch.randn(32, 18, C.WINDOW_SAMPLES) * 40.0
    f = extract_absolute_features(w)
    check("absolute block shape", tuple(f.shape) == (32, 18, C.ABS_FEATURE_DIM), str(tuple(f.shape)))
    check("absolute block finite", bool(torch.isfinite(f).all()))

    a = np.random.randn(1200, 18, C.ABS_FEATURE_DIM).astype(np.float32)
    a[800:840, :, 7] += 5.0
    full = apply_causal_baseline(a)
    check("full feature dim", full.shape[-1] == C.NODE_FEATURE_DIM, str(full.shape))

    # Strict causality: perturbing the future must not change the past.
    b = a.copy()
    b[900:] += 17.0
    fa, fb = apply_causal_baseline(a), apply_causal_baseline(b)
    check(
        "causal baseline uses no future window",
        float(np.abs(fa[:900] - fb[:900]).max()) < 1e-5,
        f"max drift {np.abs(fa[:900]-fb[:900]).max():.2e}",
    )
    rel = full[:, 0, C.ABS_FEATURE_DIM]
    check(
        "relative block responds to a power surge",
        rel[805:838].mean() > 3.0 and abs(rel[200:790].mean()) < 0.5,
        f"quiet {rel[200:790].mean():.2f} -> surge {rel[805:838].mean():.2f}",
    )


def test_montage() -> None:
    print("\n2b. Montage resolution")
    try:
        from dataset.io_edf import (
            CANONICAL_CHANNELS,
            _canonical_plan,
            _direct_candidate,
            _derived_candidate,
            _normalize_label,
        )
    except ImportError as exc:
        check("montage helpers importable", False, f"{exc} (install mne)")
        return

    def resolvable(labels):
        norm = {n: _normalize_label(n) for n in labels}
        return [
            c for c in CANONICAL_CHANNELS
            if _direct_candidate(norm, c) is None and _derived_candidate(norm, c) is None
        ]

    standard = list(CANONICAL_CHANNELS)
    check("standard bipolar montage resolves", not resolvable(standard))
    check(
        "standard labels are not rewritten by the alias map",
        all(_normalize_label(c) == c for c in standard),
    )

    # Pre-1991 nomenclature: T3/T4/T5/T6 are the same electrodes as T7/T8/P7/P8.
    # CHB-MIT sub-12 run-10/11/12 use it and carry 13 annotated seizures.
    legacy = [
        "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
        "Fz-Cz", "Cz-Pz", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp2-F8", "F8-T4",
        "T4-T6", "T6-O2",
    ]
    unresolved = resolvable(legacy)
    check(
        "pre-1991 T3/T4/T5/T6 montage resolves (sub-12 run-10/11/12)",
        not unresolved,
        f"unresolved: {unresolved}" if unresolved else "all 18 derivations recovered",
    )

    norm_rev = {"T7-F7": _normalize_label("T7-F7")}
    got = _direct_candidate(norm_rev, "F7-T7")
    check("reversed polarity is detected and signed", got is not None and got[1] == -1.0)

    norm_dup = {"T8-P8-0": _normalize_label("T8-P8-0")}
    check("duplicate channel suffix is tolerated", _direct_candidate(norm_dup, "T8-P8") is not None)

    common_ref = {f"{e}-REF": f"{e}-REF" for c in CANONICAL_CHANNELS for e in c.split("-", 1)}
    check(
        "common-reference montage can be synthesised",
        not resolvable(list(common_ref)),
    )


def test_model_causality() -> None:
    print("\n3. Model temporal causality")
    torch.manual_seed(0)
    model = DynaGAT().eval()
    b, t, n, k = 2, C.SEQUENCE_LENGTH, 18, C.TOP_K_CAUSAL
    x = torch.randn(b, t, n, C.NODE_FEATURE_DIM)
    ind = torch.randint(0, n, (b, t, n, k))
    inw = torch.rand(b, t, n, k)
    od = torch.randint(0, n, (b, t, n, k))
    ow = torch.rand(b, t, n, k)
    vm = torch.ones(b, t, dtype=torch.bool)
    with torch.no_grad():
        base = model(x, ind, inw, od, ow, vm)
        cut = t // 2
        x2 = x.clone()
        x2[:, cut:] = torch.randn_like(x2[:, cut:]) * 5.0
        pert = model(x2, ind, inw, od, ow, vm)
    drift = float((base[:, :cut] - pert[:, :cut]).abs().max())
    changed = float((base[:, cut:] - pert[:, cut:]).abs().max())
    check(
        "logits at t are unaffected by inputs after t",
        drift < 1e-4,
        f"max drift {drift:.2e} (future logits moved by {changed:.2f})",
    )
    check("future inputs do change future logits", changed > 1e-2)

    params = sum(p.numel() for p in model.parameters())
    check("parameter count is reportable", params > 0, f"{params:,} parameters")

    with torch.no_grad():
        out = model(x, ind, inw, od, ow, vm)
    check("no NaNs in forward pass", bool(torch.isfinite(out).all()))
    vm2 = vm.clone()
    vm2[:, -5:] = False
    with torch.no_grad():
        out2 = model(x, ind, inw, od, ow, vm2)
    check("padded clips stay finite", bool(torch.isfinite(out2).all()))


def test_calibration() -> None:
    print("\n4. Calibration and online scoring")
    off = prior_correction_offset(0.12, 0.0025)
    manual = np.log(0.0025 / 0.9975) - np.log(0.12 / 0.88)
    check("prior correction is exact", abs(off - manual) < 1e-9, f"offset {off:.4f}")

    v = np.random.randn(3000)
    v[1500:1540] += 6.0
    z = causal_adaptive_z(v)
    w = v.copy()
    w[1600:] += 30.0
    check(
        "online normalisation uses no future sample",
        float(np.abs(z[:1600] - causal_adaptive_z(w)[:1600]).max()) < 1e-9,
    )
    check(
        "online normalisation flags the burst",
        z[1505:1535].mean() > 3.0 and abs(z[300:1400].mean()) < 0.5,
        f"quiet {z[300:1400].mean():.2f} -> burst {z[1505:1535].mean():.2f}",
    )
    sc = OnlineScorer(prior_offset=off, logit_mean=0.0, logit_std=1.0, mix=0.5)
    p = sc.probabilities(np.array([0.0]))
    check("probabilities land in the deployment prior range", 0.0 < p[0] < 0.02, f"p={p[0]:.4f}")


def test_events() -> None:
    print("\n5. Event metrics")
    s = np.zeros(200)
    s[20:26] = 1.0
    s[60:66] = 1.0
    a = generate_alarms(s, 0.5, k=3, m=4, refractory_sec=30.0)
    check("k-of-m fires on the 3rd supra-threshold window", list(a) == [22, 62], str(list(a)))

    s2 = np.zeros(200)
    s2[20:60] = 1.0
    a2 = generate_alarms(s2, 0.5, k=3, m=4, refractory_sec=30.0)
    check("refractory period suppresses repeats", len(a2) == 2, f"alarms {list(a2)}")

    times = window_end_times(200)[a]
    det, fa, lat = match_events(times, [(20.0, 30.0)])
    check("in-window alarm detects the seizure", det == 1 and lat[0] == 6.0, f"latency {lat}")
    check("out-of-window alarm is a false alarm", fa == 1)

    det2, fa2, _ = match_events(np.array([19.0]), [(20.0, 30.0)], early_tolerance=0.0)
    check("no pre-onset credit under the primary protocol", det2 == 0 and fa2 == 1)
    det3, _, _ = match_events(np.array([19.0]), [(20.0, 30.0)], early_tolerance=10.0)
    check("secondary protocol grants 10 s pre-onset tolerance", det3 == 1)

    y = np.zeros(10000, dtype=int)
    y[100:150] = 1
    sc = np.random.rand(10000)
    sc[100:150] += 1.0
    wm = window_metrics(y, sc, np.clip(sc / 2, 0, 1))
    check("window metrics computed", wm["auroc"] > 0.9 and np.isfinite(wm["auprc"]),
          f"AUROC {wm['auroc']:.3f} AUPRC {wm['auprc']:.3f}")


# --------------------------------------------------------------------------- #
def _synthetic_cache(subject: str, n_recordings: int, n_windows: int, seizures_per_rec: int):
    rng = np.random.default_rng(abs(hash(subject)) % (2**31))
    recordings = []
    f_sum = torch.zeros(C.NODE_FEATURE_DIM, dtype=torch.float64)
    f_sq = torch.zeros(C.NODE_FEATURE_DIM, dtype=torch.float64)
    f_n = 0
    total = pos = seiz = 0
    for r in range(n_recordings):
        x = rng.normal(0, 1, (n_windows, 18, C.NODE_FEATURE_DIM)).astype(np.float32)
        labels = np.zeros(n_windows, dtype=np.uint8)
        bw = np.ones(n_windows, dtype=np.float32)
        intervals = []
        for s in range(seizures_per_rec):
            onset_w = 120 + s * 260 + int(rng.integers(0, 40))
            dur_w = 40
            if onset_w + dur_w + 10 >= n_windows:
                break
            labels[onset_w : onset_w + dur_w] = 1
            bw[max(0, onset_w - 8) : onset_w + dur_w] = C.BOUNDARY_WEIGHT_MAX
            # inject a learnable, localised signature
            x[onset_w : onset_w + dur_w, :6, 7] += 3.0
            x[onset_w : onset_w + dur_w, :, C.ABS_FEATURE_DIM] += 2.5
            intervals.append(
                (float(onset_w * C.WINDOW_STRIDE_SEC), float((onset_w + dur_w) * C.WINDOW_STRIDE_SEC))
            )
        xt = torch.from_numpy(x)
        recordings.append(
            {
                "x": xt.to(torch.float16),
                "in_dst": torch.randint(0, 18, (n_windows, 18, C.TOP_K_CAUSAL), dtype=torch.uint8),
                "in_weight": torch.rand(n_windows, 18, C.TOP_K_CAUSAL).to(torch.float16),
                "out_dst": torch.randint(0, 18, (n_windows, 18, C.TOP_K_CAUSAL), dtype=torch.uint8),
                "out_weight": torch.rand(n_windows, 18, C.TOP_K_CAUSAL).to(torch.float16),
                "labels": torch.from_numpy(labels),
                "boundary_weights": torch.from_numpy(bw).to(torch.float16),
                "n_windows": n_windows,
                "duration_sec": float(n_windows * C.WINDOW_STRIDE_SEC + C.WINDOW_SEC),
                "seizure_intervals": intervals,
                "positive_windows": int(labels.sum()),
                "file_name": f"{subject}_{r}.edf",
                "recording_id": f"{subject}/{subject}_{r}.edf",
            }
        )
        f_sum += xt.double().sum(dim=(0, 1))
        f_sq += xt.double().square().sum(dim=(0, 1))
        f_n += n_windows * 18
        total += n_windows
        pos += int(labels.sum())
        seiz += len(intervals)
    return {
        "cache_version": C.CACHE_VERSION,
        "preprocessing_tag": C.PREPROCESSING_TAG,
        "node_feature_dim": C.NODE_FEATURE_DIM,
        "top_k_causal": C.TOP_K_CAUSAL,
        "subject": subject,
        "recordings": recordings,
        "feature_sum": f_sum,
        "feature_sumsq": f_sq,
        "feature_count": f_n,
        "total_windows": total,
        "positive_windows": pos,
        "total_seizures": seiz,
        "valid_recordings": len(recordings),
        "skipped_recordings": 0,
        "event_files_found": len(recordings),
        "sampling_rate_hz": C.SFREQ,
        "window_sec": C.WINDOW_SEC,
        "stride_sec": C.WINDOW_STRIDE_SEC,
    }


def test_end_to_end() -> None:
    print("\n6. End-to-end fold on a synthetic cohort")
    from training.trainer import FoldConfig, make_folds, run_fold

    tmp = Path(tempfile.mkdtemp(prefix="dynagat_selftest_"))
    cache_dir, out_dir = tmp / "cache", tmp / "results"
    cache_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    try:
        subs = [f"sub-{i:02d}" for i in range(1, 9)]
        for s in subs:
            torch.save(_synthetic_cache(s, 2, 1200, 3), cache_dir / f"{s}_v4.pt")
        folds = make_folds(subs, n_val=3)
        check("fold construction leaks nothing", all(
            f["test"] not in f["train"] and f["test"] not in f["validation"]
            and not set(f["train"]) & set(f["validation"])
            for f in folds
        ))
        cfg = FoldConfig(epochs=3, batch_size=16, num_workers=0, amp=False, tag="selftest")
        t0 = time.perf_counter()
        summary = run_fold(folds[0], cfg, cache_dir=cache_dir, results_dir=out_dir)
        dt = time.perf_counter() - t0
        check("fold ran to completion", "event_sensitivity" in summary, f"{dt:.0f}s")
        check(
            "detector learns the injected signature",
            summary["auroc"] > 0.75,
            f"test AUROC {summary['auroc']:.3f}, AUPRC {summary['auprc']:.3f}, "
            f"sens {summary['event_sensitivity']:.2f}",
        )
        expected = prior_correction_offset(
            summary["train_sampled_prior"], summary["train_true_prior"]
        )
        check(
            "prior correction offset matches the sampling ratio",
            abs(summary["prior_offset"] - expected) < 1e-6,
            f"offset {summary['prior_offset']:.3f} (synthetic priors are close, so a "
            f"small offset is correct; on CHB-MIT it is about -4)",
        )
        check(
            "operating point is admissible under the FA cap",
            bool(summary["op_admissible"]),
            f"threshold {summary['threshold']:.3f}, k={summary['persistence_k']}/"
            f"m={summary['persistence_m']}",
        )
        produced = sorted(p.name for p in out_dir.iterdir())
        check("all fold artifacts written", len(produced) >= 4, ", ".join(produced))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 78)
    print("DynaGAT self-test")
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"signature: {C.experiment_signature()}")
    print("=" * 78)

    test_granger()
    test_features()
    test_montage()
    test_model_causality()
    test_calibration()
    test_events()
    test_end_to_end()

    print("\n" + "=" * 78)
    if _failures:
        print(f"FAILED {len(_failures)} check(s):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
