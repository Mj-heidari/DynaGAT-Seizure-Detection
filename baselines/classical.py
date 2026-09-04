"""
Classical machine-learning baseline under the identical LOPO protocol.

Gradient-boosted trees on channel-aggregated window features, with the same
prior correction, the same causal online normalisation and the same
validation-selected operating point as the deep model. Reported alongside
DynaGAT so the comparison isolates the model rather than the protocol.

    python -m baselines.classical                 # all folds
    python -m baselines.classical --folds 1 2 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    NEGATIVE_TO_IMPORTANT_RATIO,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SECONDARY_EARLY_TOLERANCE_SEC,
    VALIDATION_FA_PER_HOUR_CAP,
    experiment_signature,
)
from dataset.sequence_dataset import compute_fold_normalization, load_cache
from evaluation.events import evaluate_events, window_metrics
from evaluation.operating_point import select_operating_point
from training.calibration import OnlineScorer, prior_correction_offset
from training.trainer import list_subjects, make_folds

CONTEXT_WINDOWS = 15


def aggregate(x: np.ndarray) -> np.ndarray:
    """
    [T, 18, F] -> [T, 4F] channel aggregation plus trailing-context deltas.

    Mean/std/max across channels captures 'is something happening anywhere',
    and the causal delta against a 15 s trailing mean gives the tree model the
    same kind of short-horizon context the temporal stack has.
    """
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mx = x.max(axis=1)
    cs = np.cumsum(np.concatenate([np.zeros((1, mean.shape[1]), mean.dtype), mean]), axis=0)
    idx = np.arange(mean.shape[0])
    lo = np.maximum(0, idx - CONTEXT_WINDOWS)
    trailing = (cs[idx + 1] - cs[lo]) / np.maximum(1, (idx + 1 - lo))[:, None]
    return np.concatenate([mean, std, mx, mean - trailing], axis=1).astype(np.float32)


def _patient_matrix(cache: Dict, mean: np.ndarray, std: np.ndarray):
    feats, labels, meta = [], [], []
    for rec in cache["recordings"]:
        x = rec["x"].numpy().astype(np.float32)
        x = np.clip((x - mean) / std, -6.0, 6.0)
        f = aggregate(x)
        feats.append(f)
        labels.append(rec["labels"].numpy().astype(np.int64))
        meta.append(
            {
                "recording_id": str(rec["recording_id"]),
                "n": f.shape[0],
                "duration_sec": float(rec["duration_sec"]),
                "seizure_intervals": [tuple(map(float, s)) for s in rec.get("seizure_intervals", [])],
            }
        )
    return np.concatenate(feats), np.concatenate(labels), meta


def _split(values: np.ndarray, meta: Sequence[Dict]) -> Dict[str, np.ndarray]:
    out, off = {}, 0
    for m in meta:
        out[m["recording_id"]] = values[off : off + m["n"]]
        off += m["n"]
    return out


def run_fold(fold: Dict, cache_dir: Path) -> Dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED + fold["fold"])
    train_caches = [load_cache(cache_dir / f"{s}_v4.pt") for s in fold["train"]]
    m_t, s_t = compute_fold_normalization(train_caches)
    mean, std = m_t.numpy(), s_t.numpy().clip(1e-6)

    xs, ys = [], []
    for cache in train_caches:
        x, y, _ = _patient_matrix(cache, mean, std)
        pos = np.nonzero(y == 1)[0]
        neg = np.nonzero(y == 0)[0]
        take = min(neg.size, max(2048, NEGATIVE_TO_IMPORTANT_RATIO * 40 * max(1, pos.size)))
        neg = rng.choice(neg, size=take, replace=False)
        sel = np.concatenate([pos, neg])
        xs.append(x[sel])
        ys.append(y[sel])
        del x, y
    xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    sampled_prior = float(ytr.mean())
    true_prior = float(
        sum(int(c["positive_windows"]) for c in train_caches)
        / max(1, sum(int(c["total_windows"]) for c in train_caches))
    )
    del xs, ys, train_caches

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_SEED,
    )
    clf.fit(xtr, ytr)
    del xtr, ytr

    def predict(subjects: Sequence[str]) -> Dict[str, Dict[str, Dict]]:
        out: Dict[str, Dict[str, Dict]] = {}
        for sub in subjects:
            cache = load_cache(cache_dir / f"{sub}_v4.pt")
            x, y, meta = _patient_matrix(cache, mean, std)
            logit = clf.decision_function(x).astype(np.float32)
            by_rec_l, by_rec_y = _split(logit, meta), _split(y, meta)
            out[sub] = {
                m["recording_id"]: {
                    "logit": by_rec_l[m["recording_id"]],
                    "labels": by_rec_y[m["recording_id"]],
                    "duration_sec": m["duration_sec"],
                    "seizure_intervals": m["seizure_intervals"],
                }
                for m in meta
            }
            del cache, x, y
        return out

    val_pred = predict(fold["validation"])
    offset = prior_correction_offset(sampled_prior, true_prior)
    all_val = np.concatenate(
        [r["logit"] for recs in val_pred.values() for r in recs.values()]
    ) + offset
    scorer = OnlineScorer(
        prior_offset=offset,
        logit_mean=float(all_val.mean()),
        logit_std=float(all_val.std() + 1e-6),
    )

    def scored(pred):
        return {
            p: {
                rid: {
                    "score": scorer.score_recording(r["logit"]),
                    "prob": scorer.probabilities(r["logit"]),
                    "labels": r["labels"],
                    "duration_sec": r["duration_sec"],
                    "seizure_intervals": r["seizure_intervals"],
                }
                for rid, r in recs.items()
            }
            for p, recs in pred.items()
        }

    val_s = scored(val_pred)
    op, _ = select_operating_point(val_s, fa_cap=VALIDATION_FA_PER_HOUR_CAP)
    test_s = scored(predict([fold["test"]]))
    recs = {rid: r for rr in test_s.values() for rid, r in rr.items()}
    y = np.concatenate([r["labels"] for r in recs.values()])
    sc = np.concatenate([r["score"] for r in recs.values()])
    pr = np.concatenate([r["prob"] for r in recs.values()])
    wm = window_metrics(y, sc, pr)
    ev = evaluate_events(recs, op.threshold, op.k, op.m)
    ev10 = evaluate_events(recs, op.threshold, op.k, op.m,
                           early_tolerance=SECONDARY_EARLY_TOLERANCE_SEC)
    elapsed = time.perf_counter() - t0
    print(
        f"[gbm fold {fold['fold']:02d}] {fold['test']}: sens {ev.sensitivity:.3f} "
        f"({ev.detected_seizures}/{ev.gt_seizures}) FA/h {ev.fa_per_hour:.3f} "
        f"AUROC {wm['auroc']:.4f} AUPRC {wm['auprc']:.4f}  [{elapsed/60:.1f} min]"
    )
    return {
        "fold": fold["fold"], "tag": "baseline_gbm", "test_patient": fold["test"],
        "validation_patients": "+".join(fold["validation"]),
        "signature": experiment_signature(),
        "evaluation_role": "development" if fold["fold"] == 1 else "primary",
        "train_true_prior": true_prior, "train_sampled_prior": sampled_prior,
        "prior_offset": offset, "threshold": op.threshold,
        "persistence_k": op.k, "persistence_m": op.m, "op_admissible": op.admissible,
        "auroc": wm["auroc"], "auprc": wm["auprc"], "ece": wm.get("ece", np.nan),
        "brier": wm.get("brier", np.nan),
        "test_positive_fraction": wm["positive_fraction"], "n_test_windows": wm["n_windows"],
        **ev.as_dict(), **ev10.as_dict(prefix="tol10_"),
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir) if args.cache_dir else PROCESSED_DATA_DIR
    subjects = list_subjects(cache_dir)
    folds = make_folds(subjects)
    if args.folds:
        folds = [f for f in folds if f["fold"] in set(args.folds)]

    out = RESULTS_DIR / "baseline_gbm_lopo_summary.csv"
    rows: List[Dict] = []
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev.get("signature", "") == experiment_signature()]
        rows = prev.to_dict("records")
    done = {int(r["fold"]) for r in rows}
    for fold in folds:
        if fold["fold"] in done:
            continue
        rows.append(run_fold(fold, cache_dir))
        pd.DataFrame(rows).sort_values("fold").to_csv(out, index=False)
    print(f"[+] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
