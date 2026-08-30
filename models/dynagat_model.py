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


class DualViewGATv2Encoder(nn.Module):
    """Static montage GATv2 + edge-conditioned functional GATv2 + learned gate."""

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

        self.static_1 = GATv2Conv(
            in_channels, hidden_dim // heads, heads=heads, dropout=dropout
        )
        self.static_2 = GATv2Conv(
            hidden_dim, out_dim // heads, heads=heads, dropout=dropout
        )
        self.static_3 = GATv2Conv(
            out_dim, out_dim // heads, heads=heads, dropout=dropout
        )

        self.dynamic_1 = GATv2Conv(
            in_channels,
            hidden_dim // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
        )
        self.dynamic_2 = GATv2Conv(
            hidden_dim,
            out_dim // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
        )
        self.dynamic_3 = GATv2Conv(
            out_dim,
            out_dim // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
        )

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
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
            base.unsqueeze(0)
            + offsets.view(graph_count, 1, 1)
        ).permute(1, 0, 2).reshape(2, graph_count * edge_count)

    def _build_dynamic_edges(
        self,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # dynamic_dst / weight: [G, 18, K]
        graph_count = dynamic_dst.shape[0]
        offsets = torch.arange(
            graph_count, device=dynamic_dst.device, dtype=torch.long
        ).view(graph_count, 1) * self.num_nodes

        src = self.dynamic_src.to(dynamic_dst.device).view(1, -1).expand(graph_count, -1)
        dst = dynamic_dst.reshape(graph_count, -1)
        src = src + offsets
        dst = dst + offsets
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        edge_attr = dynamic_weight.reshape(-1, 1)
        return edge_index, edge_attr

    def forward(
        self,
        x: torch.Tensor,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x              : [G, 18, F]
        dynamic_dst    : [G, 18, K]
        dynamic_weight : [G, 18, K]

        Returns
        -------
        fused: [G, D]
        gate : [G, 1]   (1 -> static, 0 -> dynamic)
        """
        graph_count = x.shape[0]
        x_flat = x.reshape(graph_count * self.num_nodes, -1)

        static_ei = self._repeat_static_edges(graph_count, x.device)
        dynamic_ei, dynamic_ea = self._build_dynamic_edges(dynamic_dst, dynamic_weight)

        hs = F.elu(self.static_1(x_flat, static_ei))
        hs = F.dropout(hs, p=self.dropout, training=self.training)
        hs = self.static_2(hs, static_ei)
        hs = F.elu(self.static_3(hs, static_ei))

        hd = F.elu(self.dynamic_1(x_flat, dynamic_ei, edge_attr=dynamic_ea))
        hd = F.dropout(hd, p=self.dropout, training=self.training)
        hd = self.dynamic_2(hd, dynamic_ei, edge_attr=dynamic_ea)
        hd = F.elu(self.dynamic_3(hd, dynamic_ei, edge_attr=dynamic_ea))

        gs = hs.view(graph_count, self.num_nodes, -1).mean(dim=1)
        gd = hd.view(graph_count, self.num_nodes, -1).mean(dim=1)

        gate = self.gate(torch.cat([gs, gd], dim=-1))
        fused = gate * gs + (1.0 - gate) * gd
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        fused = self.out_norm(fused)
        return fused, gate


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_relu(self.net(x) + self.residual(x))



class MultiScaleTemporalEncoder(nn.Module):
    """Multi-resolution temporal feature extractor for EEG evolution."""
    def __init__(self, channels: int, hidden: int, dropout: float):
        super().__init__()
        branch = max(hidden // 3, 8)
        self.short = nn.Conv1d(channels, branch, kernel_size=3, padding=1)
        self.medium = nn.Conv1d(channels, branch, kernel_size=5, padding=2)
        self.long = nn.Conv1d(channels, branch, kernel_size=7, padding=3)
        self.fusion = nn.Sequential(
            nn.Conv1d(branch * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        a = F.gelu(self.short(x))
        b = F.gelu(self.medium(x))
        c = F.gelu(self.long(x))
        y = self.fusion(torch.cat([a, b, c], dim=1))
        return self.norm(y.transpose(1, 2)).transpose(1, 2)

class DynaGATOnsetModel(nn.Module):
    """
    Real temporal architecture:
        [B,T,18,F]
          -> dual-view GATv2 independently for each time window
          -> [B,T,D]
          -> causal TCN over T
          -> one onset logit per window [B,T]
    """

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        graph_hidden: int = GRAPH_HIDDEN,
        tcn_hidden: int = TCN_HIDDEN,
        heads: int = GAT_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.graph_encoder = DualViewGATv2Encoder(
            in_channels=in_channels,
            hidden_dim=graph_hidden,
            out_dim=graph_hidden,
            heads=heads,
            dropout=dropout,
        )
        self.tcn = MultiScaleTemporalEncoder(
            graph_hidden, tcn_hidden, dropout
        )

        # Temporal attention refinement: learns seizure evolution across adjacent EEG windows
        # after graph reasoning. This reduces isolated false alarms.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_hidden,
            nhead=4,
            dim_feedforward=tcn_hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Final temporal reasoning block:
        # learn short-term seizure evolution with trainable positional context.
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 256, tcn_hidden))
        nn.init.normal_(self.temporal_pos_embedding, std=0.02)

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.temporal_norm = nn.LayerNorm(tcn_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(tcn_hidden, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        dynamic_dst: torch.Tensor,
        dynamic_weight: torch.Tensor,
        return_gate: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,18,F], got {tuple(x.shape)}")

        batch_size, time_steps, nodes, _ = x.shape
        if nodes != NUM_NODES:
            raise ValueError(f"Expected {NUM_NODES} nodes, got {nodes}")

        graph_count = batch_size * time_steps
        x_graphs = x.reshape(graph_count, NUM_NODES, -1)
        dst_graphs = dynamic_dst.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC)
        weight_graphs = dynamic_weight.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC)

        graph_embedding, gate = self.graph_encoder(x_graphs, dst_graphs, weight_graphs)
        sequence = graph_embedding.view(batch_size, time_steps, -1)

        temporal = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)

        # Add temporal position information before attention.
        steps = temporal.size(1)
        if steps <= self.temporal_pos_embedding.size(1):
            temporal = temporal + self.temporal_pos_embedding[:, :steps, :]

        temporal = self.temporal_transformer(temporal)
        temporal = self.temporal_dropout(temporal)
        temporal = self.temporal_norm(temporal)
        logits = self.classifier(temporal).squeeze(-1)

        if return_gate:
            return logits, gate.view(batch_size, time_steps)
        return logits


# Backward-compatible class name for old imports.
DynaGAT_Onset_Model = DynaGATOnsetModel
