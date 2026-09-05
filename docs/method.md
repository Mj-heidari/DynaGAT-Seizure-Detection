# Method


**Signal.** 18 canonical bipolar derivations, 256 Hz, causal 4th-order Butterworth
band-pass 0.5–45 Hz plus a causal 60 Hz notch. 4 s windows at 1 s stride,
timestamped at the window **end**.

**Node features (34 per channel).**
26 absolute descriptors — 7 log relative band powers, log total power, three Hjorth
parameters, line length, RMS, zero-crossing rate, Teager energy, spectral entropy /
centroid / 90 % edge / flatness, kurtosis, skewness, order-3 permutation entropy, two
intra-window non-stationarity ratios and two cross-channel spatial-contrast scores —
plus 8 **causal trailing-baseline** descriptors. The second block expresses how far the
current value sits above that recording's own recent past, in robust units, using a
time-varying causal estimator that starts as an expanding-window mean and settles to a
5 min exponential baseline. This is the main mechanism that lets one model serve
patients it has never seen.

**Dual view.**
*Anatomical view* — a fixed montage-adjacency graph with intra-chain, chain-to-chain and
inter-hemispheric homologous edges.
*Causal view* — a per-window **directed Granger-causality graph**. For each window we
estimate `GC[i,j] = log(var(x_i | own past) / var(x_i | own past + x_j past))` for all
306 ordered channel pairs, keep the 5 strongest incoming and 5 strongest outgoing edges
per channel, and normalise each window by its own mean edge strength so the graph is
invariant to per-patient signal-to-noise. Because seizure spread is directional, the
model reads the in-edges and out-edges through separate GATv2 stacks.

The two views are fused by a learned per-feature gate, pooled by attention over channels,
and passed to a strictly causal temporal stack (dilated causal TCN with dilations
1/2/4/8, then a 2-layer causal Transformer over 32 s of context). ~1.06 M parameters.

**Decision layer.** Training epochs oversample ictal clips; the resulting sampling prior is
recorded and undone analytically (`logit_true = logit_sampled − logit(π_sampled) + logit(π_true)`),
then the logit is converted to a causal, per-recording adaptive score. The alarm threshold
and k-of-m persistence are chosen on the *per-patient distribution* of the validation
patients under a 0.5 FA/h cap, not on their pooled totals.

Every operation is causal in time. `run_selftest.py` verifies numerically that a logit at
window *t* is bit-identical when all inputs after *t* are replaced with noise.


## Ablation arms

| `--ablation` | Question it answers |
|---|---|
| `full` | the complete model |
| `no_causal` | does the directed Granger view add anything over anatomy alone? |
| `no_static` | is the anatomical prior still needed? |
| `causal_in_only` / `causal_out_only` | does edge direction matter? |
| `no_adaptive` | how much of the transfer comes from online adaptation? |
| `no_prior` | how much does the sampling-prior correction contribute? |
| `adaptive_only` | is the absolute score term still contributing? |
| `no_graph` | graph-free control: same features, same temporal stack, no message passing |

`no_prior` and `no_adaptive` isolate the decision layer from the representation
and belong in the paper.
