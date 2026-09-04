"""
DynaGAT: causal dual-view graph attention network for online seizure detection.

Two spatial views over the 18-channel bipolar montage:

  * anatomical view - a fixed, undirected montage-adjacency graph;
  * causal view     - a per-window *directed* Granger-causality graph, entered
                      twice: once along the incoming edges (a node attends to
                      its Granger parents) and once along the outgoing edges
                      (a node attends to its Granger children). Seizure spread
                      is directional, so in-flow and out-flow carry different
                      information and are encoded separately.

The two views are fused by a learned per-feature gate, pooled by attention over
channels, and passed to a strictly causal temporal stack (dilated causal TCN +
causal Transformer). Every operation is causal in time: the logit at window t
uses only windows <= t, so the reported latencies are achievable online.

The implementation uses gathered fixed-degree neighbourhoods rather than a
dense N x N attention matrix, which is what keeps a 32-window x 64-clip batch
inside 6 GB of VRAM.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DROPOUT,
    GAT_HEADS,
    GAT_LAYERS,
    GRAPH_HIDDEN,
    GRAPH_OUT,
    NODE_EMBED,
    NODE_FEATURE_DIM,
    NUM_NODES,
    STATIC_EDGE_INDEX,
    TCN_DILATIONS,
    TCN_HIDDEN,
    TOP_K_CAUSAL,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)

__all__ = ["DynaGAT", "build_static_neighbourhood"]


# --------------------------------------------------------------------------- #
# Static neighbourhood, padded to a fixed degree
# --------------------------------------------------------------------------- #
def build_static_neighbourhood() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (idx [N, Ks], mask [N, Ks]) for the anatomical graph with self-loops."""
    nbrs = [{n} for n in range(NUM_NODES)]
    for u, v in STATIC_EDGE_INDEX:
        nbrs[u].add(v)
        nbrs[v].add(u)
    k = max(len(s) for s in nbrs)
    idx = torch.zeros(NUM_NODES, k, dtype=torch.long)
    mask = torch.zeros(NUM_NODES, k, dtype=torch.bool)
    for n, s in enumerate(nbrs):
        ordered = sorted(s)
        idx[n, : len(ordered)] = torch.tensor(ordered, dtype=torch.long)
        idx[n, len(ordered) :] = n          # pad with self, masked out anyway
        mask[n, : len(ordered)] = True
    return idx, mask


# --------------------------------------------------------------------------- #
# Gather-based GATv2
# --------------------------------------------------------------------------- #
class GatheredGATv2(nn.Module):
    """
    GATv2 attention over fixed-size gathered neighbourhoods.

    Scores follow Brody et al. (2022): e_ij = a^T LeakyReLU(W_l x_i + W_r x_j),
    i.e. the nonlinearity precedes the attention vector, which makes the
    attention *dynamic* rather than a fixed per-node ranking.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        edge_dim: int = 0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % heads:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.dim = out_dim // heads
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_dim = edge_dim
        if edge_dim:
            self.lin_e = nn.Linear(edge_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, 1, 1, heads, self.dim))
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: torch.Tensor,               # [G, N, Din]
        nbr_idx: torch.Tensor,         # [G, N, K] long
        nbr_mask: torch.Tensor,        # [G, N, K] bool
        edge_attr: torch.Tensor | None = None,   # [G, N, K, edge_dim]
    ) -> torch.Tensor:
        g, n, _ = x.shape
        k = nbr_idx.shape[-1]
        h_l = self.lin_l(x)                                  # [G, N, C]
        h_r = self.lin_r(x)                                  # [G, N, C]

        flat_idx = nbr_idx.reshape(g, n * k, 1).expand(-1, -1, self.out_dim)
        h_nbr = torch.gather(h_r, 1, flat_idx).view(g, n, k, self.out_dim)

        score_in = h_l.unsqueeze(2) + h_nbr                  # [G, N, K, C]
        if edge_attr is not None and self.edge_dim:
            score_in = score_in + self.lin_e(edge_attr)
        score_in = F.leaky_relu(score_in, self.negative_slope)
        e = (score_in.view(g, n, k, self.heads, self.dim) * self.att).sum(-1)  # [G,N,K,H]

        e = e.masked_fill(~nbr_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h_nbr.view(g, n, k, self.heads, self.dim) * alpha.unsqueeze(-1)
        return msg.sum(dim=2).reshape(g, n, self.out_dim)


class GraphView(nn.Module):
    """A stack of residual GATv2 layers over one view."""

    def __init__(self, in_dim: int, hidden: int, layers: int, heads: int,
                 dropout: float, edge_dim: int = 0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        for _ in range(layers):
            self.layers.append(GatheredGATv2(hidden, hidden, heads, dropout, edge_dim))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout

    def forward(self, x, nbr_idx, nbr_mask, edge_attr=None):
        h = self.proj(x)
        for layer, norm in zip(self.layers, self.norms):
            z = layer(h, nbr_idx, nbr_mask, edge_attr)
            h = norm(F.elu(z) + h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(nodes).squeeze(-1), dim=1).unsqueeze(-1)
        return (nodes * w).sum(dim=1)


# --------------------------------------------------------------------------- #
# Temporal stack
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels * 2, 3, dilation)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                      # [B, C, T]
        h = F.glu(self.conv1(x), dim=1)
        h = self.dropout(h)
        h = self.conv2(h)
        y = self.norm((x + h).transpose(1, 2)).transpose(1, 2)
        return y


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class DynaGAT(nn.Module):
    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        node_embed: int = NODE_EMBED,
        graph_hidden: int = GRAPH_HIDDEN,
        graph_out: int = GRAPH_OUT,
        tcn_hidden: int = TCN_HIDDEN,
        heads: int = GAT_HEADS,
        gat_layers: int = GAT_LAYERS,
        dropout: float = DROPOUT,
        use_static: bool = True,
        use_causal: bool = True,
        causal_direction: str = "both",       # 'in' | 'out' | 'both'
        graph_mode: str = "graph",            # 'graph' | 'none'
    ) -> None:
        super().__init__()
        self.graph_mode = graph_mode
        if graph_mode == "none":
            # Graph-free control arm: identical feature front end, identical
            # temporal stack, no message passing between channels. Isolates the
            # contribution of the dual-view graph attention itself.
            use_static, use_causal = False, False
        elif not (use_static or use_causal):
            raise ValueError("at least one view must be enabled")
        if causal_direction not in {"in", "out", "both"}:
            raise ValueError("causal_direction must be 'in', 'out' or 'both'")

        self.use_static = use_static
        self.use_causal = use_causal
        self.causal_direction = causal_direction
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(in_dim)
        self.embed = nn.Sequential(nn.Linear(in_dim, node_embed), nn.GELU())

        s_idx, s_mask = build_static_neighbourhood()
        self.register_buffer("static_idx", s_idx, persistent=False)
        self.register_buffer("static_mask", s_mask, persistent=False)

        if graph_mode == "none":
            self.mlp_view = nn.Sequential(
                nn.Linear(node_embed, graph_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(graph_hidden, graph_hidden),
            )
        if use_static:
            self.static_view = GraphView(node_embed, graph_hidden, gat_layers, heads, dropout)
        if use_causal:
            n_dir = 2 if causal_direction == "both" else 1
            self.causal_in = (
                GraphView(node_embed, graph_hidden, gat_layers, heads, dropout, edge_dim=1)
                if causal_direction in {"in", "both"} else None
            )
            self.causal_out = (
                GraphView(node_embed, graph_hidden, gat_layers, heads, dropout, edge_dim=1)
                if causal_direction in {"out", "both"} else None
            )
            self.causal_merge = nn.Linear(graph_hidden * n_dir, graph_hidden)

        n_views = int(use_static) + int(use_causal)
        if n_views == 2:
            self.fusion_gate = nn.Sequential(
                nn.Linear(graph_hidden * 2, graph_hidden),
                nn.GELU(),
                nn.Linear(graph_hidden, graph_hidden),
                nn.Sigmoid(),
            )
        self.node_norm = nn.LayerNorm(graph_hidden)
        self.pool = AttentionPool(graph_hidden)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden * 2, graph_out), nn.GELU(), nn.LayerNorm(graph_out)
        )

        self.tcn_in = nn.Conv1d(graph_out, tcn_hidden, 1)
        self.tcn = nn.ModuleList(
            [CausalTCNBlock(tcn_hidden, d, dropout) for d in TCN_DILATIONS]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=tcn_hidden,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=tcn_hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.pos = nn.Parameter(torch.zeros(1, 512, tcn_hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.temporal_norm = nn.LayerNorm(tcn_hidden)

        self.head_norm = nn.LayerNorm(tcn_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(tcn_hidden * 2, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1)
        )
        self.onset_head = nn.Linear(tcn_hidden * 2, 1)

    # ---------------------------------------------------------------- graphs #
    def _encode_graphs(self, x, in_dst, in_w, out_dst, out_w):
        g, n, _ = x.shape
        h0 = self.embed(self.input_norm(x))

        if self.graph_mode == "none":
            fused = self.node_norm(self.mlp_view(h0))
            pooled = self.pool(fused)
            return self.graph_proj(torch.cat([pooled, fused.mean(dim=1)], dim=-1)), \
                torch.ones(g, device=x.device)

        parts = []
        if self.use_static:
            s_idx = self.static_idx.unsqueeze(0).expand(g, -1, -1)
            s_mask = self.static_mask.unsqueeze(0).expand(g, -1, -1)
            parts.append(("static", self.static_view(h0, s_idx, s_mask)))

        if self.use_causal:
            ones = None
            pieces = []
            if self.causal_in is not None:
                m = torch.ones_like(in_dst, dtype=torch.bool)
                pieces.append(self.causal_in(h0, in_dst, m, in_w.unsqueeze(-1)))
            if self.causal_out is not None:
                m = torch.ones_like(out_dst, dtype=torch.bool)
                pieces.append(self.causal_out(h0, out_dst, m, out_w.unsqueeze(-1)))
            parts.append(("causal", self.causal_merge(torch.cat(pieces, dim=-1))))

        if len(parts) == 2:
            hs, hc = parts[0][1], parts[1][1]
            gate = self.fusion_gate(torch.cat([hs, hc], dim=-1))
            fused = gate * hs + (1.0 - gate) * hc
        else:
            fused = parts[0][1]
            gate = torch.ones_like(fused)

        fused = self.node_norm(fused)
        pooled = self.pool(fused)
        mean = fused.mean(dim=1)
        return self.graph_proj(torch.cat([pooled, mean], dim=-1)), gate.mean(dim=(1, 2))

    # ---------------------------------------------------------------- forward #
    def forward(
        self,
        x: torch.Tensor,               # [B, T, N, F]
        in_dst: torch.Tensor,          # [B, T, N, K]
        in_weight: torch.Tensor,
        out_dst: torch.Tensor,
        out_weight: torch.Tensor,
        valid_mask: torch.Tensor | None = None,   # [B, T]
        return_aux: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"expected x [B,T,N,F], got {tuple(x.shape)}")
        b, t, n, _ = x.shape
        if n != NUM_NODES:
            raise ValueError(f"expected {NUM_NODES} nodes, got {n}")
        if t > self.pos.shape[1]:
            raise ValueError("sequence longer than positional embedding capacity")

        g = b * t
        emb, gate = self._encode_graphs(
            x.reshape(g, n, -1),
            in_dst.reshape(g, n, TOP_K_CAUSAL),
            in_weight.reshape(g, n, TOP_K_CAUSAL),
            out_dst.reshape(g, n, TOP_K_CAUSAL),
            out_weight.reshape(g, n, TOP_K_CAUSAL),
        )
        seq = emb.view(b, t, -1)

        h = self.tcn_in(seq.transpose(1, 2))
        for block in self.tcn:
            h = block(h)
        h = h.transpose(1, 2) + self.pos[:, :t, :]

        causal_mask = torch.triu(
            torch.ones((t, t), device=h.device, dtype=torch.bool), diagonal=1
        )
        kpm = None
        if valid_mask is not None:
            kpm = ~valid_mask.to(device=h.device, dtype=torch.bool)
            # A row that is fully padded would produce NaNs in softmax; keep the
            # diagonal alive for those rows and mask their loss instead.
            kpm = kpm & ~torch.eye(t, device=h.device, dtype=torch.bool).any(0, keepdim=True)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=kpm)
        h = self.temporal_norm(h)

        prev = torch.zeros_like(h)
        prev[:, 1:, :] = h[:, :-1, :]
        joint = self.head_norm(torch.cat([h, h - prev], dim=-1))
        logits = self.classifier(joint).squeeze(-1)
        if return_aux:
            onset_logits = self.onset_head(joint).squeeze(-1)
            return logits, onset_logits, gate.view(b, t)
        return logits
