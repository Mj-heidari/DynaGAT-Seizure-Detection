# Design notes

These notes record why the pipeline is built the way it is. Each choice below
was made in response to a measured failure of a simpler alternative, which was
evaluated on a single development fold (test `sub-01`, validation `sub-02..05`)
before the full protocol was run.

## The baseline that motivated the design

An earlier iteration — undirected broadband phase-synchrony graph, no prior
correction, no online adaptation, pooled operating-point selection — produced:

| metric | validation | held-out test |
|---|---|---|
| AUROC | 0.953 | 0.917 |
| AUPRC | 0.451 | **0.048** |
| event sensitivity | 0.737 | **0.364** |
| event precision | 0.107 | 0.089 |
| FA/h | 0.437 | 0.560 |
| selected threshold | 0.960 | applied unchanged |

A 9× drop in AUPRC between validation and test, with AUROC almost intact, is the
signature of a **ranking-is-fine / calibration-is-broken** failure, not of a model that
cannot see seizures. Six causes, in order of impact.

### 1. Training and evaluation used different class priors (largest single cause)

`sequence_dataset.py` built each epoch from all ictal clips plus a 6:1 sample of
interictal clips, giving a training prior near 12 % ictal. Evaluation ran over every
window, prior ≈ 0.25 %. The network therefore estimated `p(ictal | x, sampled)`, which
differs from `p(ictal | x)` by a constant of about 4 logits. Nothing corrected it, so the
usable threshold drifted to 0.96–0.99, precision-recall collapsed, and the operating point
sat on the near-vertical part of the score distribution where a small shift changes
everything.

**the current pipeline:** the sampling prior is measured every epoch and undone analytically
(`training/calibration.py`). The self-test checks the algebra exactly.

### 2. The "dynamic view" carried almost no information

The functional graph came from a wPLI computed on the **broadband 0.5–45 Hz analytic
signal** of a 2 s window. wPLI is only interpretable narrowband; on broadband data the
Hilbert phase has no consistent meaning, and at 0.5 Hz a 2 s window holds one cycle.
Top-4 hard sparsification then rebuilt the graph independently per window, so the edges
flickered. The paper's central "dual-view" claim rested on a view that was close to noise.

**the current pipeline:** the functional view is a *directed* Granger-causality graph estimated from a
well-posed least-squares problem on a 4 s window, normalised per window so it is
scale-free, and read separately along in-edges and out-edges. The estimator is verified
against an explicit reference implementation and against synthetic VAR data with known
directed structure.

### 3. No adaptation to the held-out patient

Feature normalisation used training-patient statistics only. CHB-MIT patients differ
enormously in amplitude and spectral profile, so a held-out patient was scored under a
shifted input distribution and its output level sat somewhere the validation threshold
did not anticipate.

**the current pipeline:** 8 of the 34 features are causal trailing-baseline deviations computed within each
recording, and the detector output itself is converted to a causal per-recording adaptive
score. Neither uses labels or future samples.

### 4. The operating point was selected on pooled totals

A pooled false-alarm cap is dominated by whichever validation patient contributes the most
interictal hours. The selected point met the cap on average while being far too permissive
for a typical patient.

**the current pipeline:** a candidate is admissible only if the *median* validation patient meets the cap and
the mean stays within 1.5× of it; among admissible candidates the mean per-patient
sensitivity is maximised.

### 5. Focal loss stacked on top of resampling

`FOCAL_ALPHA=0.75, gamma=2` was applied to an already 6:1-resampled pool, compounding the
imbalance correction and pushing the model further into an over-confident regime.

**the current pipeline:** mild positive weighting plus label smoothing, and a multiple-instance term on the
clip maximum that optimises the quantity actually scored — whether *some* window inside a
seizure fires — rather than per-window recall.

### 6. Only one of 23 folds ever ran

Every reported number came from one patient with 11 seizures. There was no cohort estimate
and no confidence interval.

**the current pipeline:** `run_lopo.py` writes each fold as it completes and resumes safely, so a full sweep
survives interruption. Fold 1 remains the development fold and is excluded from primary
statistics.

---

## Secondary changes

* 2 s → 4 s windows: an 18-channel MVAR of order 6 needs the samples.
* A causal 60 Hz notch was added (CHB-MIT was recorded in the United States).
* 20 → 34 node features, including permutation entropy, Teager energy, intra-window
  non-stationarity and cross-channel spatial contrast.
* Strict k-consecutive persistence → k-of-m, which tolerates a single dropped window
  inside a genuine event without admitting isolated noise spikes.
* `torch-geometric` removed. For an 18-node graph, gathered fixed-degree neighbourhood
  attention is faster and lighter than scatter-based message passing.
* Model capacity roughly doubled (0.45 M → 1.06 M parameters) but VRAM use fell, because
  the dense graph construction was the bottleneck rather than the parameter count.

## What to report in the paper

The `no_prior` and `no_adaptive` ablation arms reproduce the earlier iteration's failure modes inside the current pipeline's
codebase. Reporting them turns "our earlier version did not work" into a quantified
methodological finding: how much of patient-independent seizure-detection performance is
representation, and how much is the decision layer. That contrast is a stronger
contribution than the architecture alone.
