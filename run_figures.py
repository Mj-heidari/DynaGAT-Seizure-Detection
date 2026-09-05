"""
Publication figure and table generator for DynaGAT.

    python run_figures.py                    # everything available
    python run_figures.py --only causal_matrix
    python run_figures.py --list

Writes vector PDF (for LaTeX) and 600 dpi PNG (for previews) into
paper_figures/, CSV + LaTeX tables into paper_tables/, and a ready-to-paste
figures.tex with captions.

Every figure degrades gracefully: if its input is missing it is skipped with a
note rather than aborting the run, so this is safe to call at any point.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import config as C
from paperviz.style import (
    DIV, INK, SEQ, SERIES, STATUS, W1, W15, W2,
    annotate_cells, apply_style, diverging_norm, save,
)

FIGDIR = C.PAPER_FIGURES_DIR
TABDIR = C.PAPER_TABLES_DIR
RESDIR = C.RESULTS_DIR

ARM_LABELS = {
    "dynagat": "DynaGAT (full)",
    "abl_no_causal": "w/o causal view",
    "abl_no_static": "w/o anatomical view",
    "abl_causal_in_only": "causal in-edges only",
    "abl_causal_out_only": "causal out-edges only",
    "abl_no_adaptive": "w/o online adaptation",
    "abl_no_prior": "w/o prior correction",
    "abl_adaptive_only": "adaptive score only",
    "base_no_graph": "no graph (temporal only)",
    "baseline_gbm": "Gradient-boosted trees",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_arms() -> Dict[str, pd.DataFrame]:
    arms: Dict[str, pd.DataFrame] = {}
    for path in sorted(RESDIR.glob("*_lopo_summary.csv")):
        tag = path.name.replace("_lopo_summary.csv", "")
        df = pd.read_csv(path)
        if not df.empty:
            arms[tag] = df.sort_values("fold").reset_index(drop=True)
    return arms


def primary(df: pd.DataFrame) -> pd.DataFrame:
    if "evaluation_role" in df.columns:
        out = df[df.evaluation_role == "primary"]
        if not out.empty:
            return out.copy()
    return df.copy()


def manifest() -> Optional[pd.DataFrame]:
    p = C.PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    m["mean_seizure_sec"] = m.positive_windows / m.seizures.clip(lower=1)
    return m


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(v, reps=C.BOOTSTRAP_REPLICATES, seed=C.BOOTSTRAP_SEED):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(seed)
    m = v[r.integers(0, v.size, (reps, v.size))].mean(1)
    return v.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def load_predictions(tag: str = "dynagat"):
    """Pool window-level scores, probabilities and labels over folds."""
    out = []
    for f in sorted(RESDIR.glob(f"{tag}_fold_*_test_predictions.npz")):
        fold = int(f.name.split("_fold_")[1][:2])
        d = np.load(f)
        for key in d.files:
            if not key.endswith("::labels"):
                continue
            base = key[: -len("::labels")]
            if f"{base}::score" not in d.files:
                continue
            out.append({
                "fold": fold,
                "recording": base,
                "labels": d[key].astype(np.int64),
                "score": d[f"{base}::score"].astype(np.float64),
                "prob": d[f"{base}::prob"].astype(np.float64) if f"{base}::prob" in d.files else None,
            })
    return out


# --------------------------------------------------------------------------- #
# 1. Patient x metric heatmap
# --------------------------------------------------------------------------- #
def fig_patient_metric_heatmap(arms, **_):
    if "dynagat" not in arms:
        return None
    p = primary(arms["dynagat"]).copy()
    m = manifest()
    if m is not None:
        p = p.merge(m[["subject", "mean_seizure_sec"]],
                    left_on="test_patient", right_on="subject", how="left")

    # (column, label, higher_is_better)
    spec = [
        ("event_sensitivity", "Sensitivity", True),
        ("event_precision", "Precision", True),
        ("event_f1", "Event F1", True),
        ("fa_per_hour", "FA / h", False),
        ("auroc", "AUROC", True),
        ("auprc", "AUPRC", True),
        ("median_latency_sec", "Latency (s)", False),
        ("ece", "ECE", False),
    ]
    spec = [s for s in spec if s[0] in p.columns]
    p = p.sort_values("event_sensitivity", ascending=False)
    raw = p[[s[0] for s in spec]].to_numpy(dtype=float)

    # Colour by within-column min-max, ORIENTED so dark always means better.
    # Raw values are printed in the cells, so the normalisation only drives
    # colour and cannot mislead about magnitude.
    norm_m = np.zeros_like(raw)
    for j, (_, _, hib) in enumerate(spec):
        col = raw[:, j]
        finite = np.isfinite(col)
        if finite.sum() == 0:
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        z = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        norm_m[:, j] = z if hib else 1.0 - z

    n = len(p)
    fig, ax = plt.subplots(figsize=(W15, 0.22 * n + 1.25))
    cmap = SEQ.copy()
    cmap.set_bad(INK["grid"])          # missing values must look missing,
    im = ax.imshow(np.ma.masked_invalid(norm_m),   # not like a zero
                   cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fmts = {"fa_per_hour": "{:.2f}", "median_latency_sec": "{:.0f}", "ece": "{:.3f}"}
    for j, (k, _, _) in enumerate(spec):
        for i in range(n):
            v = raw[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.2,
                        color=INK["muted"], style="italic")
                continue
            lum = SEQ(norm_m[i, j])
            dark = (0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]) < 0.5
            ax.text(j, i, fmts.get(k, "{:.2f}").format(v), ha="center",
                    va="center", fontsize=5.6,
                    color="#ffffff" if dark else INK["primary"])

    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([s[1] for s in spec], rotation=35, ha="right")
    labels = [t.replace("sub-", "P") for t in p.test_patient]
    if "mean_seizure_sec" in p.columns:
        labels = [f"{a}  ({b:.0f}s)" for a, b in zip(labels, p.mean_seizure_sec)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, len(spec), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=INK["surface"], linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015, shrink=0.35,
                      aspect=12, ticks=[0, 1])
    cb.ax.set_yticklabels(["worst", "best"], fontsize=6)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    ax.set_title("Per-patient performance, leave-one-patient-out\n"
                 "(cells show raw values; shading is within-column, dark = better)",
                 fontsize=8.5, pad=8)
    save(fig, "fig_patient_metric_heatmap", FIGDIR)
    return "fig_patient_metric_heatmap"


# --------------------------------------------------------------------------- #
# 2. Directed Granger-causality matrices  (the method figure)
# --------------------------------------------------------------------------- #
def _accumulate_causal(cache, interictal_stride: int = 25):
    n = C.NUM_NODES
    ict = np.zeros((n, n)); ict_n = 0
    itc = np.zeros((n, n)); itc_n = 0
    rows = np.repeat(np.arange(n), C.TOP_K_CAUSAL)
    for rec in cache["recordings"]:
        lab = rec["labels"].numpy().astype(bool)
        dst = rec["in_dst"].numpy()
        w = rec["in_weight"].numpy().astype(np.float32)
        idx_i = np.nonzero(lab)[0]
        idx_o = np.nonzero(~lab)[0][::interictal_stride]
        for idx, acc, key in ((idx_i, "i", None), (idx_o, "o", None)):
            if idx.size == 0:
                continue
            d = dst[idx].reshape(idx.size, -1)
            ww = w[idx].reshape(idx.size, -1)
            mat = np.zeros((n, n))
            np.add.at(mat, (np.tile(rows, idx.size), d.ravel()), ww.ravel())
            if acc == "i":
                ict += mat; ict_n += idx.size
            else:
                itc += mat; itc_n += idx.size
    return (ict / max(1, ict_n), ict_n), (itc / max(1, itc_n), itc_n)


def fig_causal_matrix(max_subjects: int = 23, **_):
    import torch
    caches = sorted(C.PROCESSED_DATA_DIR.glob("sub-*_v4.pt"))[:max_subjects]
    if not caches:
        return None
    ict_list, itc_list = [], []
    for path in caches:
        try:
            cache = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            continue
        (a, an), (b, bn) = _accumulate_causal(cache)
        if an > 0 and bn > 0:
            ict_list.append(a); itc_list.append(b)
        del cache
    if not ict_list:
        return None

    # Equal weight per patient, so a long-recording patient cannot dominate.
    ict = np.mean(ict_list, axis=0)
    itc = np.mean(itc_list, axis=0)
    diff = ict - itc
    ch = [c.replace("-", "\u2013") for c in C.CHANNELS_18]

    n_pat = len(ict_list)
    # Explicit grid: three matrix panels, each colourbar in its own column.
    # Letting matplotlib squeeze a shared colourbar between subplots is what
    # made it overlap the third panel's tick labels.
    fig = plt.figure(figsize=(W2, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05],
                          wspace=0.42, left=0.075, right=0.955,
                          top=0.80, bottom=0.20)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    cax0 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])
    cax1 = fig.add_subplot(gs[4])

    vmax = max(ict.max(), itc.max())
    for ax, mat, title in ((ax0, itc, "Interictal"), (ax1, ict, "Ictal")):
        im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=8)
    nrm = diverging_norm(diff)
    im2 = ax2.imshow(diff, cmap=DIV, norm=nrm, aspect="equal")
    ax2.set_title("Ictal \u2212 interictal", fontsize=8)

    for k, ax in enumerate((ax0, ax1, ax2)):
        ax.set_xticks(range(18)); ax.set_yticks(range(18))
        ax.set_xticklabels(ch, rotation=90, fontsize=4.4)
        # Only the leftmost panel carries the channel names: all three share
        # the same order, and repeating them collides with the colourbars.
        ax.set_yticklabels(ch if k == 0 else [], fontsize=4.4)
        ax.tick_params(length=1.2, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 0:
            ax.set_ylabel("receiver", fontsize=6.5)
        ax.set_xlabel("driver", fontsize=6.5, labelpad=1)

    for cax, im_, lab in ((cax0, im, "mean strength"), (cax1, im2, "change")):
        cb = fig.colorbar(im_, cax=cax)
        cb.set_label(lab, fontsize=6, labelpad=2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5, length=1.5)

    fig.suptitle("Directed Granger-causal connectivity, mean over "
                 f"{n_pat} patients", fontsize=8.5, y=0.955)
    save(fig, "fig_causal_matrix", FIGDIR)

    np.savetxt(TABDIR / "causal_matrix_ictal.csv", ict, delimiter=",",
               header=",".join(C.CHANNELS_18), comments="")
    np.savetxt(TABDIR / "causal_matrix_interictal.csv", itc, delimiter=",",
               header=",".join(C.CHANNELS_18), comments="")
    np.savetxt(TABDIR / "causal_matrix_difference.csv", diff, delimiter=",",
               header=",".join(C.CHANNELS_18), comments="")

    # Asymmetry: how much of the ictal change is directional rather than shared?
    asym = np.abs(diff - diff.T).sum() / (np.abs(diff).sum() * 2 + 1e-12)
    print(f"    directional asymmetry index of the ictal change: {asym:.3f} "
          f"(0 = perfectly symmetric, 1 = purely one-way)")
    return "fig_causal_matrix"


# --------------------------------------------------------------------------- #
# 3. Forest plot
# --------------------------------------------------------------------------- #
def fig_forest(arms, **_):
    if "dynagat" not in arms:
        return None
    p = primary(arms["dynagat"]).sort_values("event_sensitivity")
    lo, hi = zip(*[wilson(int(r.detected_seizures), int(r.gt_seizures))
                   for r in p.itertuples()])
    s = p.event_sensitivity.to_numpy()
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(W1, 0.21 * len(p) + 1.1))
    ax.grid(axis="x", zorder=0)
    ax.errorbar(s, y, xerr=[s - np.array(lo), np.array(hi) - s], fmt="o",
                ms=4, lw=1, capsize=1.8, color=SERIES[0],
                ecolor=INK["axis"], zorder=3)
    mean = float(np.mean(s))
    ax.axvline(mean, color=SERIES[1], lw=1.2, zorder=2)
    ax.text(mean, len(p) - 0.3, f" mean {mean:.2f}", color=SERIES[1],
            fontsize=6.5, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t.replace('sub-','P')}  {int(a)}/{int(b)}"
                        for t, a, b in zip(p.test_patient, p.detected_seizures,
                                           p.gt_seizures)], fontsize=6)
    ax.set_xlabel("Event sensitivity (95% Wilson CI)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.8, len(p) - 0.2)
    save(fig, "fig_forest_sensitivity", FIGDIR)
    return "fig_forest_sensitivity"


# --------------------------------------------------------------------------- #
# 4. Operating-point transfer
# --------------------------------------------------------------------------- #
def fig_operating_transfer(arms, **_):
    if "dynagat" not in arms:
        return None
    p = primary(arms["dynagat"])
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    ax.grid(zorder=0)
    over = p.fa_per_hour > C.VALIDATION_FA_PER_HOUR_CAP
    ax.scatter(p.fa_per_hour[~over], p.event_sensitivity[~over], s=26,
               c=SERIES[0], edgecolor=INK["surface"], linewidth=0.6,
               zorder=3, label="within FA cap")
    ax.scatter(p.fa_per_hour[over], p.event_sensitivity[over], s=26,
               c=STATUS["critical"], marker="s",
               edgecolor=INK["surface"], linewidth=0.6, zorder=3,
               label="exceeds FA cap")
    for r in p.itertuples():
        ax.annotate(str(r.test_patient).replace("sub-", ""),
                    (r.fa_per_hour, r.event_sensitivity), fontsize=5.2,
                    color=INK["secondary"], xytext=(3, 2.5),
                    textcoords="offset points")
    ax.axvline(C.VALIDATION_FA_PER_HOUR_CAP, color=INK["muted"], lw=1, zorder=2)
    ax.text(C.VALIDATION_FA_PER_HOUR_CAP, 1.04,
            f" cap {C.VALIDATION_FA_PER_HOUR_CAP:g}/h", fontsize=6,
            color=INK["secondary"], va="bottom")
    ax.set_xlabel("False alarms per hour (held-out patient)")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "fig_operating_point_transfer", FIGDIR)
    return "fig_operating_point_transfer"


# --------------------------------------------------------------------------- #
# 5. Seizure duration vs sensitivity  (the diagnostic finding)
# --------------------------------------------------------------------------- #
def fig_duration_vs_sensitivity(arms, **_):
    m = manifest()
    if "dynagat" not in arms or m is None:
        return None
    p = primary(arms["dynagat"]).merge(
        m[["subject", "mean_seizure_sec"]], left_on="test_patient",
        right_on="subject", how="left")
    if p.mean_seizure_sec.isna().all():
        return None
    fig, ax = plt.subplots(figsize=(W15, 2.8))
    ax.grid(zorder=0)
    sizes = 12 + 3.2 * p.gt_seizures
    sc = ax.scatter(p.mean_seizure_sec, p.event_sensitivity, s=sizes,
                    c=p.auroc, cmap=SEQ, vmin=0.45, vmax=1.0,
                    edgecolor=INK["surface"], linewidth=0.7, zorder=3)
    # Greedy de-collision: alternate the label side when two points are close
    # in both axes, so no annotation is written on top of another.
    placed = []
    for r in p.sort_values("mean_seizure_sec").itertuples():
        x, yv = float(r.mean_seizure_sec), float(r.event_sensitivity)
        dx, dy, ha = 4, 3, "left"
        for px, py, pdy in placed:
            if abs(np.log10(x) - np.log10(px)) < 0.055 and abs(yv - py) < 0.07:
                dy = -9 if pdy > 0 else 8
                dx = -4 if dy > 0 else 4
                ha = "right" if dx < 0 else "left"
        placed.append((x, yv, dy))
        ax.annotate(str(r.test_patient).replace("sub-", "P"), (x, yv),
                    fontsize=5.4, color=INK["secondary"], ha=ha,
                    xytext=(dx, dy), textcoords="offset points")
    kmax = int(p.persistence_m.max()) if "persistence_m" in p.columns else 10
    ax.axvspan(0, kmax, color=STATUS["critical"], alpha=0.07, zorder=1)
    ax.text(kmax * 1.15, 1.10, "shorter than the persistence window",
            fontsize=6, color=STATUS["critical"], ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Mean seizure duration (s, log scale)")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(-0.05, 1.15)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("window AUROC", fontsize=7)
    cb.outline.set_visible(False)
    rho = p.mean_seizure_sec.corr(p.event_sensitivity, method="spearman")
    ax.set_title("Detection needs both a usable ranking and enough windows"
                 f"   (Spearman $\\rho$ = {rho:.2f})", fontsize=8)
    save(fig, "fig_duration_vs_sensitivity", FIGDIR)
    return "fig_duration_vs_sensitivity"


# --------------------------------------------------------------------------- #
# 6. Pooled ROC / PR
# --------------------------------------------------------------------------- #
def fig_roc_pr(**_):
    preds = load_predictions()
    if not preds:
        return None
    y = np.concatenate([p["labels"] for p in preds])
    s = np.concatenate([p["score"] for p in preds])
    order = np.argsort(-s)
    ys = y[order].astype(float)
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    tpr, fpr = tp / max(1, tp[-1]), fp / max(1, fp[-1])
    prec, rec = tp / np.maximum(tp + fp, 1), tp / max(1, tp[-1])

    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.6))
    axes[0].grid(zorder=0)
    axes[0].plot(fpr, tpr, color=SERIES[0], zorder=3)
    axes[0].plot([0, 1], [0, 1], color=INK["axis"], lw=0.8, zorder=2)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"Pooled window ROC (AUC = {auc:.3f})", fontsize=8)

    axes[1].grid(zorder=0)
    axes[1].plot(rec, prec, color=SERIES[1], zorder=3)
    base = float(y.mean())
    axes[1].axhline(base, color=INK["muted"], lw=1, zorder=2)
    axes[1].text(0.02, base * 1.15, f"chance {base:.4f}", fontsize=6,
                 color=INK["secondary"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Pooled window precision\u2013recall", fontsize=8)
    save(fig, "fig_pooled_roc_pr", FIGDIR)
    return "fig_pooled_roc_pr"


# --------------------------------------------------------------------------- #
# 7. Ablation heatmap + bars
# --------------------------------------------------------------------------- #
def fig_ablation(arms, **_):
    if len(arms) < 2:
        return None
    spec = [("event_sensitivity", "Sensitivity", True),
            ("fa_per_hour", "FA / h", False),
            ("event_f1", "Event F1", True),
            ("auroc", "AUROC", True),
            ("auprc", "AUPRC", True)]
    order = [t for t in ARM_LABELS if t in arms] + [t for t in arms if t not in ARM_LABELS]
    rows, labels = [], []
    for tag in order:
        p = primary(arms[tag])
        rows.append([p[k].mean() if k in p.columns else np.nan for k, _, _ in spec])
        labels.append(ARM_LABELS.get(tag, tag))
    raw = np.array(rows, dtype=float)

    rel = np.zeros_like(raw)
    for j, (_, _, hib) in enumerate(spec):
        col = raw[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        z = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        rel[:, j] = z if hib else 1.0 - z

    fig, ax = plt.subplots(figsize=(W15, 0.3 * len(labels) + 1.4))
    ax.imshow(rel, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    fmts = {"fa_per_hour": "{:.3f}", "auroc": "{:.3f}", "auprc": "{:.3f}"}
    for j, (k, _, _) in enumerate(spec):
        for i in range(len(labels)):
            v = raw[i, j]
            txt = "--" if not np.isfinite(v) else fmts.get(k, "{:.3f}").format(v)
            c = SEQ(rel[i, j])
            dark = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) < 0.5
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                    color="#ffffff" if dark else INK["primary"])
    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([s[1] for s in spec], rotation=25, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(spec), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=INK["surface"], linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Ablation and baseline comparison (mean over primary folds;\n"
                 "shading within column, dark = better)", fontsize=8.5, pad=8)
    save(fig, "fig_ablation_heatmap", FIGDIR)
    return "fig_ablation_heatmap"


# --------------------------------------------------------------------------- #
# 8. Pooled window-level confusion matrix
# --------------------------------------------------------------------------- #
def fig_confusion(arms, **_):
    preds = load_predictions()
    if not preds or "dynagat" not in arms:
        return None
    thr = {int(r.fold): float(r.threshold) for r in arms["dynagat"].itertuples()}
    dev = C.DEVELOPMENT_FOLD
    tp = fp = tn = fn = 0
    for rec in preds:
        if rec["fold"] == dev or rec["fold"] not in thr:
            continue
        pred = rec["score"] >= thr[rec["fold"]]
        y = rec["labels"].astype(bool)
        tp += int((pred & y).sum()); fp += int((pred & ~y).sum())
        tn += int((~pred & ~y).sum()); fn += int((~pred & y).sum())
    if (tp + fn) == 0:
        return None
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    rown = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(W1, 2.9))
    ax.imshow(rown, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            c = SEQ(rown[i, j])
            dark = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) < 0.5
            col = "#ffffff" if dark else INK["primary"]
            ax.text(j, i - 0.16, f"{names[i][j]}  {rown[i,j]*100:.1f}%",
                    ha="center", va="center", fontsize=8, color=col)
            ax.text(j, i + 0.16, f"{int(cm[i,j]):,}", ha="center", va="center",
                    fontsize=6.5, color=col)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted\ninterictal", "predicted\nictal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["interictal", "ictal"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Pooled window-level confusion\n"
                 "(row-normalised, counts below)", fontsize=8)
    save(fig, "fig_confusion_matrix", FIGDIR)
    pd.DataFrame(cm, index=["interictal", "ictal"],
                 columns=["pred interictal", "pred ictal"]).to_csv(
        TABDIR / "confusion_matrix.csv")
    return "fig_confusion_matrix"


# --------------------------------------------------------------------------- #
# 9. Validation -> test transfer
# --------------------------------------------------------------------------- #
def fig_val_test_transfer(arms, **_):
    if "dynagat" not in arms:
        return None
    p = primary(arms["dynagat"])
    if "val_event_sensitivity" not in p.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.9))
    pairs = [("val_event_sensitivity", "event_sensitivity", "Event sensitivity", None),
             ("val_fa_per_hour", "fa_per_hour", "False alarms / h",
              C.VALIDATION_FA_PER_HOUR_CAP)]
    for ax, (vk, tk, title, cap) in zip(axes, pairs):
        q = p.sort_values(tk)
        y = np.arange(len(q))
        ax.grid(axis="x", zorder=0)
        ax.hlines(y, q[vk], q[tk], color=INK["axis"], lw=1, zorder=2)
        ax.scatter(q[vk], y, s=18, c=SERIES[2], zorder=3, label="validation")
        ax.scatter(q[tk], y, s=18, c=SERIES[0], zorder=3, label="held-out test")
        if cap is not None:
            ax.axvline(cap, color=STATUS["critical"], lw=1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace("sub-", "P") for t in q.test_patient], fontsize=5.6)
        ax.set_xlabel(title)
        ax.legend(loc="lower right", fontsize=6)
    fig.suptitle("Operating point chosen on validation patients, applied unchanged "
                 "to the held-out patient", fontsize=8.5, y=1.02)
    save(fig, "fig_val_test_transfer", FIGDIR)
    return "fig_val_test_transfer"


# --------------------------------------------------------------------------- #
# 10. Convergence
# --------------------------------------------------------------------------- #
def fig_convergence(**_):
    files = sorted(RESDIR.glob("dynagat_fold_*_history.csv"))
    if not files:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.5))
    for ax in axes:
        ax.grid(zorder=0)
    for f in files:
        h = pd.read_csv(f)
        axes[0].plot(h.epoch, h.train_loss, lw=0.8, alpha=0.5,
                     color=SERIES[0], zorder=3)
        v = h.dropna(subset=["quick_val_auprc"])
        axes[1].plot(v.epoch, v.quick_val_auprc, lw=0.8, alpha=0.5,
                     marker="o", ms=2.2, color=SERIES[1], zorder=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation AUPRC")
    fig.suptitle(f"Convergence across {len(files)} leave-one-patient-out folds",
                 fontsize=8.5, y=1.02)
    save(fig, "fig_convergence", FIGDIR)
    return "fig_convergence"


# --------------------------------------------------------------------------- #
# 11. Score separation
# --------------------------------------------------------------------------- #
def fig_score_distribution(**_):
    preds = load_predictions()
    if not preds:
        return None
    y = np.concatenate([p["labels"] for p in preds])
    s = np.concatenate([p["score"] for p in preds])
    lo, hi = np.percentile(s, [0.05, 99.99])
    bins = np.linspace(lo, hi, 90)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.grid(zorder=0)
    ax.hist(s[y == 0], bins=bins, density=True, color=INK["muted"],
            alpha=0.75, label="interictal", zorder=3)
    ax.hist(s[y == 1], bins=bins, density=True, color=SERIES[1],
            alpha=0.8, label="ictal", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Causal detector score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6.5)
    ax.set_title("Score separation on held-out patients", fontsize=8)
    save(fig, "fig_score_distribution", FIGDIR)
    return "fig_score_distribution"


# --------------------------------------------------------------------------- #
# 12. Reliability
# --------------------------------------------------------------------------- #
def fig_calibration(**_):
    preds = [p for p in load_predictions() if p["prob"] is not None]
    if not preds:
        return None
    y = np.concatenate([p["labels"] for p in preds]).astype(float)
    q = np.concatenate([p["prob"] for p in preds])
    edges = np.geomspace(max(q.min(), 1e-6), min(q.max(), 1.0), 13)
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (q >= a) & (q < b)
        if sel.sum() < 50:
            continue
        xs.append(q[sel].mean()); ys.append(y[sel].mean()); ns.append(int(sel.sum()))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.grid(zorder=0)
    lim = [min(min(xs), min(ys)) * 0.5, 1.0]
    ax.plot(lim, lim, color=INK["axis"], lw=0.9, zorder=2)
    ax.scatter(xs, ys, s=np.clip(np.array(ns) / 3000, 8, 90), c=SERIES[0],
               edgecolor=INK["surface"], linewidth=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability after prior correction\n(marker area $\\propto$ windows in bin)",
                 fontsize=8)
    save(fig, "fig_calibration", FIGDIR)
    return "fig_calibration"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def write_tables(arms) -> List[str]:
    made = []
    if "dynagat" in arms:
        p = primary(arms["dynagat"]).copy()
        m = manifest()
        if m is not None:
            p = p.merge(m[["subject", "seizures", "recording_hours", "mean_seizure_sec"]],
                        left_on="test_patient", right_on="subject", how="left")
        cols = {"test_patient": "Patient", "gt_seizures": "Seizures",
                "mean_seizure_sec": "Mean dur. (s)", "detected_seizures": "Detected",
                "event_sensitivity": "Sens.", "fa_per_hour": "FA/h",
                "median_latency_sec": "Latency (s)", "auroc": "AUROC", "auprc": "AUPRC"}
        have = [c for c in cols if c in p.columns]
        t = p[have].rename(columns=cols)
        t.to_csv(TABDIR / "patient_level_results.csv", index=False)
        (TABDIR / "patient_level_results.tex").write_text(_latex(
            t, "Patient-level leave-one-patient-out performance.",
            "tab:patient_level"), encoding="utf-8")
        made.append("patient_level_results")

    rows = []
    for tag, df in arms.items():
        p = primary(df)
        r = {"Method": ARM_LABELS.get(tag, tag), "Folds": len(p)}
        for k, lab in [("event_sensitivity", "Sensitivity"), ("fa_per_hour", "FA/h"),
                       ("event_f1", "Event F1"), ("auroc", "AUROC"), ("auprc", "AUPRC"),
                       ("median_latency_sec", "Latency (s)")]:
            if k not in p.columns:
                continue
            mean, lo, hi = bootstrap_ci(p[k])
            r[lab] = (f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
                      if k in ("event_sensitivity", "fa_per_hour")
                      else f"{mean:.3f}")
        if {"detected_seizures", "gt_seizures"} <= set(p.columns):
            det, gt = int(p.detected_seizures.sum()), int(p.gt_seizures.sum())
            r["Pooled sens."] = f"{det}/{gt} = {det/gt:.3f}" if gt else "--"
        rows.append(r)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(TABDIR / "main_comparison.csv", index=False)
        (TABDIR / "main_comparison.tex").write_text(_latex(
            t, "Patient-independent performance on CHB-MIT. Means over the primary "
               "folds with patient-level bootstrap 95\\% confidence intervals.",
            "tab:main"), encoding="utf-8")
        made.append("main_comparison")

    m = manifest()
    if m is not None:
        t = m[["subject", "valid_recordings", "recording_hours", "seizures",
               "mean_seizure_sec", "windows", "positive_fraction"]].rename(columns={
            "subject": "Patient", "valid_recordings": "Recordings",
            "recording_hours": "Hours", "seizures": "Seizures",
            "mean_seizure_sec": "Mean dur. (s)", "windows": "Windows",
            "positive_fraction": "Ictal frac."})
        tot = pd.DataFrame([{
            "Patient": "Total", "Recordings": int(m.valid_recordings.sum()),
            "Hours": float(m.recording_hours.sum()), "Seizures": int(m.seizures.sum()),
            "Mean dur. (s)": float(m.positive_windows.sum() / m.seizures.sum()),
            "Windows": int(m.windows.sum()),
            "Ictal frac.": float(m.positive_windows.sum() / m.windows.sum())}])
        t = pd.concat([t, tot], ignore_index=True)
        t.to_csv(TABDIR / "dataset_summary.csv", index=False)
        (TABDIR / "dataset_summary.tex").write_text(_latex(
            t, "CHB-MIT cohort after preprocessing.", "tab:dataset", "%.4g"),
            encoding="utf-8")
        made.append("dataset_summary")
    return made


def _latex(df, caption, label, fmt="%.3f") -> str:
    body = df.to_latex(index=False, escape=True, na_rep="--",
                       float_format=lambda v: fmt % v,
                       column_format="l" + "r" * (df.shape[1] - 1))
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n")


CAPTIONS = {
    "fig_causal_matrix":
        "Directed Granger-causal connectivity averaged over all 23 patients, with "
        "each patient contributing equally. Rows index the receiving derivation and "
        "columns the driving derivation, so the matrix is read as driver "
        "$\\rightarrow$ receiver and its asymmetry is meaningful. The right panel "
        "shows the ictal minus interictal change on a diverging scale with a "
        "symmetric limit, so the neutral colour is exactly zero.",
    "fig_patient_metric_heatmap":
        "Per-patient performance under leave-one-patient-out evaluation. Cells give "
        "raw values; shading is normalised within each column and oriented so that "
        "darker is always better. Patient labels carry the mean seizure duration.",
    "fig_forest_sensitivity":
        "Per-patient event sensitivity with 95\\% Wilson intervals, ordered by "
        "sensitivity. Labels give detected/total seizures.",
    "fig_operating_point_transfer":
        "Transfer of the validation-selected operating point to the held-out "
        "patient. Square markers exceed the validation false-alarm cap.",
    "fig_duration_vs_sensitivity":
        "Event sensitivity against mean seizure duration, coloured by window-level "
        "AUROC and sized by seizure count. Detection requires both a usable ranking "
        "and enough supra-threshold windows to satisfy the persistence rule.",
    "fig_pooled_roc_pr":
        "Pooled window-level ROC and precision-recall curves over the primary folds. "
        "Precision uses a logarithmic axis because the ictal prior is 1:299.",
    "fig_ablation_heatmap":
        "Ablation and baseline comparison. Cells give raw means over the primary "
        "folds; shading is within-column with darker meaning better.",
    "fig_confusion_matrix":
        "Pooled window-level confusion matrix at the validation-selected threshold, "
        "row-normalised with absolute counts beneath each rate.",
    "fig_val_test_transfer":
        "Validation and held-out values of the two quantities that define the "
        "operating point, paired per patient.",
    "fig_convergence":
        "Training loss and validation AUPRC across leave-one-patient-out folds.",
    "fig_score_distribution":
        "Distribution of the causal detector score on held-out patients.",
    "fig_calibration":
        "Reliability of the prior-corrected probabilities on held-out patients.",
}


def write_figures_tex(made: List[str]) -> None:
    widths = {"fig_causal_matrix": 1.0, "fig_duration_vs_sensitivity": 1.0,
              "fig_patient_metric_heatmap": 0.85, "fig_ablation_heatmap": 0.9,
              "fig_val_test_transfer": 1.0, "fig_pooled_roc_pr": 1.0,
              "fig_convergence": 1.0}
    out = ["% Auto-generated by run_figures.py",
           "% Requires \\usepackage{graphicx}", ""]
    for name in made:
        w = widths.get(name, 0.62)
        env = "figure*" if w >= 0.95 else "figure"
        out += [f"\\begin{{{env}}}[t]", "  \\centering",
                f"  \\includegraphics[width={w}\\linewidth]{{paper_figures/{name}.pdf}}",
                f"  \\caption{{{CAPTIONS.get(name, name)}}}",
                f"  \\label{{fig:{name.replace('fig_', '')}}}",
                f"\\end{{{env}}}", ""]
    (TABDIR / "figures.tex").write_text("\n".join(out), encoding="utf-8")


FIGURES = {
    "causal_matrix": fig_causal_matrix,
    "patient_metric_heatmap": fig_patient_metric_heatmap,
    "forest": fig_forest,
    "operating_transfer": fig_operating_transfer,
    "duration_vs_sensitivity": fig_duration_vs_sensitivity,
    "roc_pr": fig_roc_pr,
    "ablation": fig_ablation,
    "confusion": fig_confusion,
    "val_test_transfer": fig_val_test_transfer,
    "convergence": fig_convergence,
    "score_distribution": fig_score_distribution,
    "calibration": fig_calibration,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(FIGURES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k in sorted(FIGURES):
            print(" ", k)
        return 0

    apply_style()
    arms = load_arms()
    if not arms:
        print(f"[warn] no *_lopo_summary.csv in {RESDIR}; only cache-based figures will run")
    else:
        print(f"[*] arms: {', '.join(arms)}")

    made: List[str] = []
    wanted = args.only or list(FIGURES)
    for key in wanted:
        fn = FIGURES[key]
        try:
            name = fn(arms=arms)
        except Exception as exc:
            print(f"  [fail] {key}: {exc}")
            traceback.print_exc(limit=1)
            continue
        if name:
            made.append(name)
            print(f"  [ok]   {name}")
        else:
            print(f"  [skip] {key} (input not available yet)")

    tables = write_tables(arms) if arms else []
    for t in tables:
        print(f"  [ok]   table {t}")
    if made:
        write_figures_tex(made)
        print(f"  [ok]   {TABDIR / 'figures.tex'}")

    print(f"\n[+] {len(made)} figure(s) -> {FIGDIR}")
    print(f"[+] {len(tables)} table(s) -> {TABDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
