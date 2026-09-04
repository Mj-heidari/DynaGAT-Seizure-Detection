"""
Batched directed Granger-causal graph estimation on the GPU.

This module is the methodological core of DynaGAT. For every analysis
window it estimates a *directed* connectivity matrix

    GC[i, j] = log( var(residual of x_i | own past)
                    / var(residual of x_i | own past + x_j past) )   >= 0

i.e. the pairwise (bivariate) Granger causality from channel j to channel i.
Unlike the broadband wPLI used in an earlier iteration, this quantity is directed, has a clear
statistical meaning, and is estimated from a well-posed least-squares problem.

Efficiency
----------
A naive implementation refits N*(N-1) autoregressions per window. Instead we
build one Gram matrix per window over all lagged regressors and the targets,

    M = [ Z ; Y ]  with  Z in R^{N p x (L-p)},  Y in R^{N x (L-p)}
    G = M M^T      in R^{(N p + N) x (N p + N)}

and obtain every reduced/full residual variance from sub-blocks of G via a
Schur complement. All N^2 pairs of all windows in a chunk are then solved as
one batched Cholesky of p x p systems. This is ~40x cheaper than refitting and
is numerically identical up to the ridge term (verified in tests/).
"""
from __future__ import annotations

import torch

__all__ = ["granger_causality_batch", "granger_reference_naive", "build_causal_topk"]


def _build_gram(wins: torch.Tensor, order: int) -> torch.Tensor:
    """
    Gram matrix of [lagged regressors ; targets].

    Parameters
    ----------
    wins : [B, N, L] float32, already z-scored per (window, channel).
    order : MVAR lag order p.

    Returns
    -------
    G : [B, N*p + N, N*p + N]
    """
    b, n, length = wins.shape
    p = int(order)
    eff = length - p
    if eff <= (n * p + n):
        raise ValueError(
            f"Window too short for order {p}: {eff} samples for {n * p + n} regressors"
        )

    # Lagged design. rows are ordered (channel-major, lag-minor):
    #   row (c * p + (lag - 1)) holds x_c[t - lag] for t = p .. L-1
    lag_rows = torch.empty((b, n, p, eff), dtype=wins.dtype, device=wins.device)
    for lag in range(1, p + 1):
        lag_rows[:, :, lag - 1, :] = wins[:, :, p - lag : length - lag]
    z = lag_rows.reshape(b, n * p, eff)
    y = wins[:, :, p:]                                   # [B, N, eff]
    m = torch.cat([z, y], dim=1)                         # [B, Np+N, eff]
    return m @ m.transpose(1, 2)


def granger_causality_batch(
    wins: torch.Tensor,
    order: int = 6,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """
    Pairwise directed Granger causality for a batch of multichannel windows.

    Parameters
    ----------
    wins : [B, N, L] float tensor of windowed EEG (any scaling; standardised
           internally per window and channel).
    order : MVAR lag order.
    ridge : Tikhonov coefficient, relative to the effective sample size.

    Returns
    -------
    gc : [B, N, N] float tensor, gc[b, i, j] = strength of j -> i, zero diagonal.
    """
    if wins.ndim != 3:
        raise ValueError(f"expected [B, N, L], got {tuple(wins.shape)}")
    b, n, length = wins.shape
    p = int(order)
    eff = length - p

    x = wins.float()
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

    g = _build_gram(x, p)
    np_ = n * p
    lam = float(ridge) * float(eff)
    eye_p = torch.eye(p, device=g.device, dtype=g.dtype)

    # Block views ------------------------------------------------------------
    # gb[b, i, j] = G[rows of channel i's lags, cols of channel j's lags]
    gb = g[:, :np_, :np_].reshape(b, n, p, n, p).permute(0, 1, 3, 2, 4).contiguous()
    # gy[b, i, l, k] = G[channel i lag l, target k]
    gy = g[:, :np_, np_:].reshape(b, n, p, n)
    yy = torch.diagonal(g[:, np_:, np_:], dim1=1, dim2=2)                # [B, N]

    idx = torch.arange(n, device=g.device)
    a_mat = gb[:, idx, idx] + lam * eye_p                                # [B, N, p]
    # a_vec[b, i, l] = G[channel i lag l, target i]
    a_vec = torch.diagonal(gy, dim1=1, dim2=3).permute(0, 2, 1).contiguous()

    l_a, info_a = torch.linalg.cholesky_ex(a_mat)
    if int(info_a.max()) != 0:
        a_mat = a_mat + (10.0 * lam) * eye_p
        l_a = torch.linalg.cholesky(a_mat)

    ainv_a = torch.cholesky_solve(a_vec.unsqueeze(-1), l_a).squeeze(-1)   # [B, N, p]
    sig_red = (yy - (a_vec * ainv_a).sum(-1)).clamp_min(1e-8)            # [B, N]

    # Full models ------------------------------------------------------------
    l_a_e = l_a.unsqueeze(2).expand(b, n, n, p, p)
    ainv_b = torch.cholesky_solve(gb, l_a_e)                             # A_i^-1 G[R_i, R_j]
    schur = (
        a_mat.unsqueeze(1).expand(b, n, n, p, p)                          # C_j = A_j
        - torch.einsum("bijlm,bijln->bijmn", gb, ainv_b)
    )
    schur = schur + lam * eye_p

    # c_j = G[R_j, y_i]  ->  cmat[b, i, j, m]
    cmat = gy.permute(0, 3, 1, 2).contiguous()
    u = cmat - torch.einsum("bijlm,bil->bijm", gb, ainv_a)               # [B, N, N, p]

    l_s, info_s = torch.linalg.cholesky_ex(schur)
    if int(info_s.max()) != 0:
        schur = schur + (10.0 * lam) * eye_p
        l_s = torch.linalg.cholesky(schur)

    quad = (u * torch.cholesky_solve(u.unsqueeze(-1), l_s).squeeze(-1)).sum(-1)
    sig_full = (sig_red.unsqueeze(2) - quad).clamp_min(1e-8)

    gc = torch.log(sig_red.unsqueeze(2) / sig_full).clamp_min(0.0)
    gc[:, idx, idx] = 0.0
    return torch.nan_to_num(gc, nan=0.0, posinf=0.0, neginf=0.0)


def granger_reference_naive(
    wins: torch.Tensor, order: int = 6, ridge: float = 1e-3
) -> torch.Tensor:
    """Slow, explicit reference implementation used only to validate the fast path."""
    b, n, length = wins.shape
    p = int(order)
    eff = length - p
    x = wins.float()
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    lam = float(ridge) * float(eff)

    lags = torch.empty((b, n, p, eff), dtype=x.dtype, device=x.device)
    for lag in range(1, p + 1):
        lags[:, :, lag - 1, :] = x[:, :, p - lag : length - lag]

    out = torch.zeros(b, n, n, dtype=x.dtype, device=x.device)
    for bi in range(b):
        for i in range(n):
            y = x[bi, i, p:]
            zi = lags[bi, i].T                                   # [eff, p]
            sr = _ls_resid(zi, y, lam)
            for j in range(n):
                if i == j:
                    continue
                zij = torch.cat([zi, lags[bi, j].T], dim=1)      # [eff, 2p]
                sf = _ls_resid(zij, y, lam)
                out[bi, i, j] = torch.log(
                    sr.clamp_min(1e-8) / sf.clamp_min(1e-8)
                ).clamp_min(0.0)
    return out


def _ls_resid(design: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    g = design.T @ design + lam * torch.eye(
        design.shape[1], device=design.device, dtype=design.dtype
    )
    rhs = design.T @ y
    beta = torch.linalg.solve(g, rhs)
    return (y @ y) - rhs @ beta


def build_causal_topk(gc: torch.Tensor, k: int = 5):
    """
    Scale-free normalisation + top-k sparsification of a directed GC matrix.

    Normalising each window by its own mean off-diagonal strength makes the
    graph invariant to per-patient signal-to-noise, which is what allows the
    same model to transfer across patients.

    Returns
    -------
    in_dst    : [B, N, K] int64 - the K strongest Granger *parents* of each node
    in_weight : [B, N, K] float
    out_dst   : [B, N, K] int64 - the K strongest Granger *children* of each node
    out_weight: [B, N, K] float
    """
    b, n, _ = gc.shape
    idx = torch.arange(n, device=gc.device)
    off = gc.clone()
    off[:, idx, idx] = 0.0
    denom = off.sum(dim=(1, 2), keepdim=True) / float(n * (n - 1))
    norm = torch.log1p(off / denom.clamp_min(1e-6))
    norm[:, idx, idx] = 0.0

    in_weight, in_dst = torch.topk(norm, k=k, dim=2)               # parents of i
    out_weight, out_dst = torch.topk(norm.transpose(1, 2), k=k, dim=2)  # children of i
    return in_dst, in_weight, out_dst, out_weight
