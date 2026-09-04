"""
Export publication artifacts from completed LOPO runs.

    python run_export.py

Reads every results/*_lopo_summary.csv, then writes:

  paper_results/  per-patient and pooled statistics with bootstrap intervals
  paper_tables/   the same tables as CSV and LaTeX
  paper_figures/  vector PDF + 600 dpi PNG figures

Fold 1 is the development fold and is excluded from the primary statistics; it
is retained separately so the protocol is auditable.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

PRIMARY_METRICS = [
    ("event_sensitivity", "Event sensitivity"),
    ("fa_per_hour", "False alarms / h"),
    ("event_precision", "Event precision"),
    ("event_f1", "Event F1"),
    ("auroc", "Window AUROC"),
    ("auprc", "Window AUPRC"),
    ("median_latency_sec", "Median latency (s)"),
    ("ece", "Expected calibration error"),
]

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


def bootstrap_ci(values: np.ndarray, reps: int = C.BOOTSTRAP_REPLICATES,
                 seed: int = C.BOOTSTRAP_SEED, alpha: float = 0.05):
    """Patient-level percentile bootstrap of the mean."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(reps, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def load_arms(results_dir: Path) -> Dict[str, pd.DataFrame]:
    arms: Dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("*_lopo_summary.csv")):
        tag = path.name.replace("_lopo_summary.csv", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "signature" in df.columns:
            sig = C.experiment_signature()
            keep = df[df["signature"] == sig]
            if keep.empty:
                print(f"[warn] {path.name}: no rows match the current signature; skipping")
                continue
            df = keep
        arms[tag] = df.sort_values("fold").reset_index(drop=True)
    return arms


def to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.3f") -> str:
    body = df.to_latex(index=False, escape=True, float_format=lambda v: float_fmt % v,
                       na_rep="--", column_format="l" + "r" * (df.shape[1] - 1))
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n"
    )


# --------------------------------------------------------------------------- #
def export_per_patient(arms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    main = arms.get("dynagat")
    if main is None:
        return pd.DataFrame()
    cols = [
        "fold", "test_patient", "evaluation_role", "gt_seizures", "detected_seizures",
        "event_sensitivity", "false_alarms", "interictal_hours", "fa_per_hour",
        "event_precision", "event_f1", "median_latency_sec", "auroc", "auprc", "ece",
        "threshold", "persistence_k", "persistence_m", "op_admissible",
        "tol10_event_sensitivity",
    ]
    have = [c for c in cols if c in main.columns]
    table = main[have].copy()
    lo, hi = zip(*[
        wilson(int(r.detected_seizures), int(r.gt_seizures))
        for r in table.itertuples()
    ]) if {"detected_seizures", "gt_seizures"}.issubset(table.columns) else ([], [])
    if len(lo):
        table["sens_ci_low"] = lo
        table["sens_ci_high"] = hi
    table.to_csv(C.PAPER_RESULTS_DIR / "all_per_patient_results.csv", index=False)
    primary = table[table["evaluation_role"] == "primary"]
    primary.to_csv(C.PAPER_RESULTS_DIR / "primary_per_patient_results.csv", index=False)

    show = primary[[
        c for c in ["test_patient", "gt_seizures", "detected_seizures", "event_sensitivity",
                    "fa_per_hour", "median_latency_sec", "auroc", "auprc"] if c in primary.columns
    ]].rename(columns={
        "test_patient": "Patient", "gt_seizures": "Seizures", "detected_seizures": "Detected",
        "event_sensitivity": "Sens.", "fa_per_hour": "FA/h",
        "median_latency_sec": "Latency (s)", "auroc": "AUROC", "auprc": "AUPRC",
    })
    show.to_csv(C.PAPER_TABLES_DIR / "patient_level_results.csv", index=False)
    (C.PAPER_TABLES_DIR / "patient_level_results.tex").write_text(
        to_latex(show, "Patient-level leave-one-patient-out performance of DynaGAT "
                       "on the primary folds.", "tab:patient_level"),
        encoding="utf-8",
    )
    return table


def export_summary(arms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict] = []
    for tag, df in arms.items():
        primary = df[df.get("evaluation_role", "primary") == "primary"]
        if primary.empty:
            primary = df
        row: Dict = {"arm": ARM_LABELS.get(tag, tag), "tag": tag, "n_folds": len(primary)}
        for key, _ in PRIMARY_METRICS:
            if key not in primary.columns:
                continue
            mean, lo, hi = bootstrap_ci(primary[key].to_numpy())
            row[key] = mean
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
            row[f"{key}_median"] = float(np.nanmedian(primary[key].to_numpy()))
        if {"detected_seizures", "gt_seizures"}.issubset(primary.columns):
            det, gt = int(primary["detected_seizures"].sum()), int(primary["gt_seizures"].sum())
            row["pooled_detected"] = det
            row["pooled_seizures"] = gt
            row["pooled_sensitivity"] = det / gt if gt else float("nan")
            row["pooled_sens_ci_low"], row["pooled_sens_ci_high"] = wilson(det, gt)
        if {"false_alarms", "interictal_hours"}.issubset(primary.columns):
            fa, hrs = int(primary["false_alarms"].sum()), float(primary["interictal_hours"].sum())
            row["pooled_false_alarms"] = fa
            row["pooled_interictal_hours"] = hrs
            row["pooled_fa_per_hour"] = fa / hrs if hrs > 0 else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = [t for t in ARM_LABELS if t in set(summary["tag"])] + [
        t for t in summary["tag"] if t not in ARM_LABELS
    ]
    summary["__o"] = summary["tag"].map({t: i for i, t in enumerate(order)})
    summary = summary.sort_values("__o").drop(columns="__o")
    summary.to_csv(C.PAPER_RESULTS_DIR / "arm_summary.csv", index=False)

    def fmt(row, key):
        if key not in row or not np.isfinite(row[key]):
            return "--"
        lo, hi = row.get(f"{key}_ci_low", np.nan), row.get(f"{key}_ci_high", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            return f"{row[key]:.3f} [{lo:.3f}, {hi:.3f}]"
        return f"{row[key]:.3f}"

    main_tbl = pd.DataFrame(
        [
            {
                "Method": r["arm"],
                "Sensitivity [95\\% CI]": fmt(r, "event_sensitivity"),
                "FA/h [95\\% CI]": fmt(r, "fa_per_hour"),
                "AUROC": f"{r.get('auroc', float('nan')):.3f}",
                "AUPRC": f"{r.get('auprc', float('nan')):.3f}",
                "Latency (s)": f"{r.get('median_latency_sec', float('nan')):.1f}",
            }
            for _, r in summary.iterrows()
        ]
    )
    main_tbl.to_csv(C.PAPER_TABLES_DIR / "main_comparison.csv", index=False)
    (C.PAPER_TABLES_DIR / "main_comparison.tex").write_text(
        to_latex(main_tbl,
                 "Patient-independent performance on CHB-MIT. Means over the primary "
                 "leave-one-patient-out folds with patient-level bootstrap 95\\% "
                 "confidence intervals.", "tab:main"),
        encoding="utf-8",
    )
    return summary


def export_dataset_table() -> pd.DataFrame:
    path = C.PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    m = pd.read_csv(path)
    show = m[["subject", "valid_recordings", "recording_hours", "seizures",
              "windows", "positive_fraction"]].rename(columns={
        "subject": "Patient", "valid_recordings": "Recordings",
        "recording_hours": "Hours", "seizures": "Seizures",
        "windows": "Windows", "positive_fraction": "Ictal fraction",
    })
    total = pd.DataFrame([{
        "Patient": "Total", "Recordings": int(m["valid_recordings"].sum()),
        "Hours": float(m["recording_hours"].sum()), "Seizures": int(m["seizures"].sum()),
        "Windows": int(m["windows"].sum()),
        "Ictal fraction": float(m["positive_windows"].sum() / max(1, m["windows"].sum())),
    }])
    show = pd.concat([show, total], ignore_index=True)
    show.to_csv(C.PAPER_TABLES_DIR / "dataset_summary.csv", index=False)
    (C.PAPER_TABLES_DIR / "dataset_summary.tex").write_text(
        to_latex(show, "CHB-MIT cohort after preprocessing.", "tab:dataset", "%.4g"),
        encoding="utf-8",
    )
    return show


# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.PAPER_FIGURES_DIR / f"{name}.pdf")
    fig.savefig(C.PAPER_FIGURES_DIR / f"{name}.png", dpi=600)
    plt.close(fig)


def export_figures(arms: Dict[str, pd.DataFrame], summary: pd.DataFrame) -> List[str]:
    if not HAVE_MPL:
        print("[warn] matplotlib not installed; skipping figures")
        return []
    plt.rcParams.update({
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
    })
    made: List[str] = []
    main = arms.get("dynagat")

    if main is not None and "event_sensitivity" in main.columns:
        p = main[main["evaluation_role"] == "primary"].sort_values("event_sensitivity")
        fig, ax = plt.subplots(figsize=(5.2, 0.24 * max(6, len(p)) + 1.2))
        y = np.arange(len(p))
        lo = [wilson(int(r.detected_seizures), int(r.gt_seizures))[0] for r in p.itertuples()]
        hi = [wilson(int(r.detected_seizures), int(r.gt_seizures))[1] for r in p.itertuples()]
        s = p["event_sensitivity"].to_numpy()
        ax.errorbar(s, y, xerr=[s - np.array(lo), np.array(hi) - s], fmt="o",
                    ms=4, lw=1, capsize=2, color="#2b6cb0")
        ax.axvline(float(np.mean(s)), color="#c53030", ls="--", lw=1,
                   label=f"mean {np.mean(s):.2f}")
        ax.set_yticks(y)
        ax.set_yticklabels(p["test_patient"])
        ax.set_xlabel("Event sensitivity (95% Wilson CI)")
        ax.set_xlim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Patient-level sensitivity, leave-one-patient-out")
        _save(fig, "fig_patient_sensitivity_forest"); made.append("fig_patient_sensitivity_forest")

        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.scatter(p["fa_per_hour"], p["event_sensitivity"], s=28, c="#2b6cb0",
                   edgecolor="white", zorder=3)
        for r in p.itertuples():
            ax.annotate(str(r.test_patient).replace("sub-", ""),
                        (r.fa_per_hour, r.event_sensitivity),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axvline(C.VALIDATION_FA_PER_HOUR_CAP, color="#c53030", ls="--", lw=1,
                   label=f"validation cap {C.VALIDATION_FA_PER_HOUR_CAP:g}/h")
        ax.set_xlabel("False alarms per hour (held-out patient)")
        ax.set_ylabel("Event sensitivity")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Operating-point transfer to unseen patients")
        _save(fig, "fig_operating_point_transfer"); made.append("fig_operating_point_transfer")

    if not summary.empty and "event_sensitivity" in summary.columns:
        s = summary.dropna(subset=["event_sensitivity"])
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.28 * len(s) + 1.8), sharey=True)
        y = np.arange(len(s))
        axes[0].barh(y, s["event_sensitivity"], color="#2b6cb0",
                     xerr=[s["event_sensitivity"] - s["event_sensitivity_ci_low"],
                           s["event_sensitivity_ci_high"] - s["event_sensitivity"]],
                     capsize=2, error_kw=dict(lw=0.8))
        axes[0].set_xlabel("Event sensitivity"); axes[0].set_xlim(0, 1)
        axes[0].set_yticks(y); axes[0].set_yticklabels(s["arm"])
        axes[1].barh(y, s["fa_per_hour"], color="#c53030")
        axes[1].axvline(C.VALIDATION_FA_PER_HOUR_CAP, color="k", ls="--", lw=1)
        axes[1].set_xlabel("False alarms per hour")
        fig.suptitle("Ablation and baseline comparison", y=0.99)
        _save(fig, "fig_ablation_comparison"); made.append("fig_ablation_comparison")

    hist_files = sorted(C.RESULTS_DIR.glob("dynagat_fold_*_history.csv"))
    if hist_files:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
        for f in hist_files:
            h = pd.read_csv(f)
            axes[0].plot(h["epoch"], h["train_loss"], lw=0.9, alpha=0.55)
            v = h.dropna(subset=["quick_val_auprc"])
            axes[1].plot(v["epoch"], v["quick_val_auprc"], lw=0.9, alpha=0.55, marker="o", ms=2.5)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation AUPRC")
        fig.suptitle(f"Convergence across {len(hist_files)} leave-one-patient-out folds", y=0.99)
        _save(fig, "fig_convergence"); made.append("fig_convergence")

    npz = sorted(C.RESULTS_DIR.glob("dynagat_fold_*_test_predictions.npz"))
    if npz:
        ys, ss = [], []
        for f in npz:
            d = np.load(f)
            for key in d.files:
                if key.endswith("::labels"):
                    base = key[: -len("::labels")]
                    if f"{base}::score" in d.files:
                        ys.append(d[key]); ss.append(d[f"{base}::score"])
        if ys:
            y = np.concatenate(ys); s = np.concatenate(ss)
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
            order = np.argsort(-s)
            yo = y[order]
            tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
            axes[0].plot(fp / max(1, fp[-1]), tp / max(1, tp[-1]), lw=1.4, color="#2b6cb0")
            axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color="gray")
            axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
            axes[0].set_title("Pooled window-level ROC")
            prec = tp / np.maximum(tp + fp, 1); rec = tp / max(1, tp[-1])
            axes[1].plot(rec, prec, lw=1.4, color="#c53030")
            axes[1].axhline(y.mean(), ls="--", lw=0.8, color="gray",
                            label=f"chance {y.mean():.4f}")
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].set_yscale("log"); axes[1].legend(frameon=False, fontsize=8)
            axes[1].set_title("Pooled window-level precision-recall")
            _save(fig, "fig_pooled_roc_pr"); made.append("fig_pooled_roc_pr")

            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            bins = np.linspace(s.min(), s.max(), 80)
            ax.hist(s[y == 0], bins=bins, density=True, alpha=0.6, label="interictal", color="#718096")
            ax.hist(s[y == 1], bins=bins, density=True, alpha=0.7, label="ictal", color="#c53030")
            ax.set_yscale("log"); ax.set_xlabel("Causal detector score"); ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.set_title("Score separation on held-out patients")
            _save(fig, "fig_score_distribution"); made.append("fig_score_distribution")
    return made


def main() -> int:
    arms = load_arms(C.RESULTS_DIR)
    if not arms:
        print(f"[error] no *_lopo_summary.csv found in {C.RESULTS_DIR}. Run run_lopo.py first.")
        return 1
    print(f"[*] arms found: {', '.join(arms)}")
    per_patient = export_per_patient(arms)
    summary = export_summary(arms)
    dataset = export_dataset_table()
    figures = export_figures(arms, summary)

    if not summary.empty:
        print("\n" + "=" * 78)
        cols = [c for c in ["arm", "n_folds", "event_sensitivity", "fa_per_hour",
                            "auroc", "auprc", "median_latency_sec"] if c in summary.columns]
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("=" * 78)

    (C.PAPER_RESULTS_DIR / "environment.json").write_text(
        json.dumps(
            {
                "signature": C.experiment_signature(),
                "preprocessing_tag": C.PREPROCESSING_TAG,
                "python": sys.version,
                "platform": platform.platform(),
                "arms": list(arms),
                "n_primary_folds": int(len(per_patient[per_patient["evaluation_role"] == "primary"]))
                if not per_patient.empty else 0,
                "figures": figures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[+] tables  -> {C.PAPER_TABLES_DIR}")
    print(f"[+] results -> {C.PAPER_RESULTS_DIR}")
    print(f"[+] figures -> {C.PAPER_FIGURES_DIR} ({len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
