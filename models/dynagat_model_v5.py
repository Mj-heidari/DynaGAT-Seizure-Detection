from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from config import (
    DROPOUT,
    GAT_HEADS,
    GRAPH_HIDDEN,
    NODE_FEATURE_DIM,
    NUM_NODES,
    TCN_HIDDEN,
    TOP_K_DYNAMIC,
    get_static_edge_tensor,
)


class ResidualDualViewGATv2Encoder(nn.Module):
    """Residual static/dynamic GATv2 with attention pooling and feature-wise fusion."""

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        hidden_dim: int = GRAPH_HIDDEN,
        out_dim: int = GRAPH_HIDDEN,
        heads: int = GAT_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0 or out_dim % heads != 0:
            raise ValueError("hidden_dim and out_dim must be divisible by heads")

        self.dropout = float(dropout)
        self.num_nodes = NUM_NODES
        self.top_k = TOP_K_DYNAMIC

        def static_layer(in_dim: int, out_total: int) -> GATv2Conv:
            return GATv2Conv(
                in_dim,
                out_total // heads,
                heads=heads,
                dropout=dropout,
            )

        def dynamic_layer(in_dim: int, out_total: int) -> GATv2Conv:
            return GATv2Conv(
                in_dim,
                out_total // heads,
                heads=heads,
                edge_dim=1,
                dropout=dropout,
            )

        self.static_1 = static_layer(in_channels, hidden_dim)
        self.static_2 = static_layer(hidden_dim, out_dim)
        self.static_3 = static_layer(out_dim, out_dim)
        self.dynamic_1 = dynamic_layer(in_channels, hidden_dim)
        self.dynamic_2 = dynamic_layer(hidden_dim, out_dim)
        self.dynamic_3 = dynamic_layer(out_dim, out_dim)

        self.static_norm1 = nn.LayerNorm(hidden_dim)
        self.static_norm2 = nn.LayerNorm(out_dim)
        self.static_norm3 = nn.LayerNorm(out_dim)
        self.dynamic_norm1 = nn.LayerNorm(hidden_dim)
        self.dynamic_norm2 = nn.LayerNorm(out_dim)
        self.dynamic_norm3 = nn.LayerNorm(out_dim)

        self.static_pool_score = nn.Sequential(
            nn.Linear(out_dim, max(16, out_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, out_dim // 2), 1),
        )
        self.dynamic_pool_score = nn.Sequential(
            nn.Linear(out_dim, max(16, out_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, out_dim // 2), 1),
        )

        # Feature-wise gating is more expressive than one scalar gate per window,
        # while still keeping static and functional branches explicitly interpretable.
        self.fusion_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(out_dim)

        self.register_buffer("static_edge_index", get_static_edge_tensor(), persistent=False)
        dynamic_src = torch.arange(NUM_NODES).view(NUM_NODES, 1).repeat(1, TOP_K_DYNAMIC)
        self.register_buffer("dynamic_src", dynamic_src.reshape(-1), persistent=False)

    def _repeat_static_edges(self, graph_count: int, device: torch.device) -> torch.Tensor:
        base = self.static_edge_index.to(device)
        edge_count = base.shape[1]
        offsets = torch.arange(graph_count, device=device, dtype=torch.long) * self.num_nodes
        return (
            base.unsqueeze(0) + offsets.view(graph_count, 1, 1)
        ).permute(1, 0, 2).reshape(2, graph_count * edge_count)

    def _build_dynamic_edges(
        self,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_count = dynamic_dst.shape[0]
        offsets = torch.arange(
            graph_count, device=dynamic_dst.device, dtype=torch.long
        ).view(graph_count, 1) * self.num_nodes
        src = self.dynamic_src.to(dynamic_dst.device).view(1, -1).expand(graph_count, -1)
        dst = dynamic_dst.reshape(graph_count, -1)
        src = src + offsets
        dst = dst + offsets
        return (
            torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0),
            dynamic_weight.reshape(-1, 1),
        )

    @staticmethod
    def _attention_pool(nodes: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        weights = torch.softmax(scorer(nodes).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(nodes * weights, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_count = x.shape[0]
        x_flat = x.reshape(graph_count * self.num_nodes, -1)
        static_ei = self._repeat_static_edges(graph_count, x.device)
        dynamic_ei, dynamic_ea = self._build_dynamic_edges(dynamic_dst, dynamic_weight)

        hs1 = self.static_norm1(F.elu(self.static_1(x_flat, static_ei)))
        hs1 = F.dropout(hs1, p=self.dropout, training=self.training)
        hs2 = self.static_norm2(F.elu(self.static_2(hs1, static_ei)) + hs1)
        hs2 = F.dropout(hs2, p=self.dropout, training=self.training)
        hs3 = self.static_norm3(F.elu(self.static_3(hs2, static_ei)) + hs2)

        hd1 = self.dynamic_norm1(
            F.elu(self.dynamic_1(x_flat, dynamic_ei, edge_attr=dynamic_ea))
        )
        hd1 = F.dropout(hd1, p=self.dropout, training=self.training)
        hd2 = self.dynamic_norm2(
            F.elu(self.dynamic_2(hd1, dynamic_ei, edge_attr=dynamic_ea)) + hd1
        )
        hd2 = F.dropout(hd2, p=self.dropout, training=self.training)
        hd3 = self.dynamic_norm3(
            F.elu(self.dynamic_3(hd2, dynamic_ei, edge_attr=dynamic_ea)) + hd2
        )

        hs_nodes = hs3.view(graph_count, self.num_nodes, -1)
        hd_nodes = hd3.view(graph_count, self.num_nodes, -1)
        gs = self._attention_pool(hs_nodes, self.static_pool_score)
        gd = self._attention_pool(hd_nodes, self.dynamic_pool_score)

        gate = self.fusion_gate(torch.cat([gs, gd], dim=-1))
        fused = gate * gs + (1.0 - gate) * gd
        fused = self.out_norm(F.dropout(fused, p=self.dropout, training=self.training))
        return fused, gate


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_pad = int(kernel_size) - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class ResidualMultiScaleTemporalEncoder(nn.Module):
    """Causal multi-scale encoder with gated fusion and an explicit residual path."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        branch = max(hidden // 3, 8)
        self.short = CausalConv1d(channels, branch, kernel_size=3)
        self.medium = CausalConv1d(channels, branch, kernel_size=5)
        self.long = CausalConv1d(channels, branch, kernel_size=7)
        self.fusion = nn.Conv1d(branch * 3, hidden * 2, kernel_size=1)
        self.residual = (
            nn.Identity() if channels == hidden else nn.Conv1d(channels, hidden, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.gelu(self.short(x))
        b = F.gelu(self.medium(x))
        c = F.gelu(self.long(x))
        fused = F.glu(self.fusion(torch.cat([a, b, c], dim=1)), dim=1)
        y = self.dropout(fused) + self.residual(x)
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class DynaGATOnsetModelV5(nn.Module):
    """Causal v5 detector with residual graph learning and onset-delta classification."""

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        graph_hidden: int = GRAPH_HIDDEN,
        tcn_hidden: int = TCN_HIDDEN,
        heads: int = GAT_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.graph_encoder = ResidualDualViewGATv2Encoder(
            in_channels=in_channels,
            hidden_dim=graph_hidden,
            out_dim=graph_hidden,
            heads=heads,
            dropout=dropout,
        )
        self.tcn = ResidualMultiScaleTemporalEncoder(graph_hidden, tcn_hidden, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_hidden,
            nhead=4,
            dim_feedforward=tcn_hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 256, tcn_hidden))
        nn.init.normal_(self.temporal_pos_embedding, std=0.02)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.temporal_dropout = nn.Dropout(dropout)
        self.temporal_norm = nn.LayerNorm(tcn_hidden)

        # The delta term is strictly backward-looking: h_t - h_(t-1). It gives the
        # classifier an explicit transition feature without accessing future windows.
        self.head_norm = nn.LayerNorm(tcn_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(tcn_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_gate: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,18,F], got {tuple(x.shape)}")
        batch_size, time_steps, nodes, _ = x.shape
        if nodes != NUM_NODES:
            raise ValueError(f"Expected {NUM_NODES} nodes, got {nodes}")
        if time_steps > self.temporal_pos_embedding.size(1):
            raise ValueError("Sequence length exceeds positional embedding capacity")

        graph_count = batch_size * time_steps
        graph_embedding, gate = self.graph_encoder(
            x.reshape(graph_count, NUM_NODES, -1),
            dynamic_dst.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC),
            dynamic_weight.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC),
        )
        sequence = graph_embedding.view(batch_size, time_steps, -1)
        temporal = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        temporal = temporal + self.temporal_pos_embedding[:, :time_steps, :]

        causal_mask = torch.triu(
            torch.ones((time_steps, time_steps), device=temporal.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, time_steps):
                raise ValueError(
                    f"valid_mask must be {(batch_size, time_steps)}, got {tuple(valid_mask.shape)}"
                )
            key_padding_mask = ~valid_mask.to(device=temporal.device, dtype=torch.bool)

        temporal = self.temporal_transformer(
            temporal,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        temporal = self.temporal_norm(self.temporal_dropout(temporal))

        previous = torch.zeros_like(temporal)
        previous[:, 1:, :] = temporal[:, :-1, :]
        delta = temporal - previous
        head_input = self.head_norm(torch.cat([temporal, delta], dim=-1))
        logits = self.classifier(head_input).squeeze(-1)

        if return_gate:
            # Mean feature-wise gate retains a compact interpretable per-window score.
            return logits, gate.mean(dim=-1).view(batch_size, time_steps)
        return logits
