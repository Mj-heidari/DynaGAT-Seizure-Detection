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
    """Static montage GATv2 + functional GATv2 + learned fusion gate."""

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
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        edge_attr = dynamic_weight.reshape(-1, 1)
        return edge_index, edge_attr

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


class CausalConv1d(nn.Module):
    """1-D convolution that never reads future time steps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_pad = int(kernel_size) - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class MultiScaleTemporalEncoder(nn.Module):
    """Causal short/medium/long temporal feature extractor."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        branch = max(hidden // 3, 8)
        self.short = CausalConv1d(channels, branch, kernel_size=3)
        self.medium = CausalConv1d(channels, branch, kernel_size=5)
        self.long = CausalConv1d(channels, branch, kernel_size=7)
        self.fusion = nn.Sequential(
            nn.Conv1d(branch * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.gelu(self.short(x))
        b = F.gelu(self.medium(x))
        c = F.gelu(self.long(x))
        y = self.fusion(torch.cat([a, b, c], dim=1))
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class DynaGATOnsetModel(nn.Module):
    """
    Patient-independent causal seizure-onset detector.

    Input [B,T,18,F]
      -> dual-view GATv2 per EEG window
      -> causal multi-scale temporal encoder
      -> causal Transformer refinement
      -> one onset logit per window [B,T]

    No output at time t can attend to a future EEG window t+1...T.
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
        self.tcn = MultiScaleTemporalEncoder(graph_hidden, tcn_hidden, dropout)

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
        valid_mask: torch.Tensor | None = None,
        return_gate: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,18,F], got {tuple(x.shape)}")

        batch_size, time_steps, nodes, _ = x.shape
        if nodes != NUM_NODES:
            raise ValueError(f"Expected {NUM_NODES} nodes, got {nodes}")
        if time_steps > self.temporal_pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {time_steps} exceeds positional capacity "
                f"{self.temporal_pos_embedding.size(1)}"
            )

        graph_count = batch_size * time_steps
        x_graphs = x.reshape(graph_count, NUM_NODES, -1)
        dst_graphs = dynamic_dst.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC)
        weight_graphs = dynamic_weight.reshape(graph_count, NUM_NODES, TOP_K_DYNAMIC)

        graph_embedding, gate = self.graph_encoder(x_graphs, dst_graphs, weight_graphs)
        sequence = graph_embedding.view(batch_size, time_steps, -1)

        temporal = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        temporal = temporal + self.temporal_pos_embedding[:, :time_steps, :]

        causal_mask = torch.triu(
            torch.ones(
                (time_steps, time_steps),
                device=temporal.device,
                dtype=torch.bool,
            ),
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
        temporal = self.temporal_dropout(temporal)
        temporal = self.temporal_norm(temporal)
        logits = self.classifier(temporal).squeeze(-1)

        if return_gate:
            return logits, gate.view(batch_size, time_steps)
        return logits


# Backward-compatible class name for older imports/checkpoints.
DynaGAT_Onset_Model = DynaGATOnsetModel
