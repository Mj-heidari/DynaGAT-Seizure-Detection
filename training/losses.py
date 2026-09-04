"""
Training objective for DynaGAT.

Three terms:

  * a boundary-weighted, label-smoothed BCE on every valid window;
  * a multiple-instance ("bag") term that requires the maximum score inside an
    ictal clip to be high and the maximum score inside a purely interictal clip
    to be low. Detection is scored per *event*, not per window, so optimising
    the clip maximum matches the evaluation metric far better than window BCE
    alone. This is the main reason the current pipeline's event sensitivity is not bounded by
    per-window recall;
  * an auxiliary onset term on the peri-onset windows, which sharpens the
    detection latency.

an earlier iteration stacked focal loss on top of an already resampled pool, which compounded the
imbalance correction and drove the model into an over-confident regime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DynaGATLoss"]


class DynaGATLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 3.0,
        label_smoothing: float = 0.02,
        bag_weight: float = 0.3,
        onset_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.smooth = float(label_smoothing)
        self.bag_weight = float(bag_weight)
        self.onset_weight = float(onset_weight)

    def forward(
        self,
        logits: torch.Tensor,             # [B, T]
        onset_logits: torch.Tensor,       # [B, T]
        targets: torch.Tensor,            # [B, T] in {0,1}
        boundary_weights: torch.Tensor,   # [B, T]
        valid_mask: torch.Tensor,         # [B, T] bool
    ):
        tgt = targets.to(logits.dtype)
        mask = valid_mask.to(logits.dtype)
        smooth_tgt = tgt * (1.0 - self.smooth) + 0.5 * self.smooth

        pw = torch.where(tgt > 0.5, torch.as_tensor(self.pos_weight, device=logits.device,
                                                    dtype=logits.dtype),
                         torch.ones_like(tgt))
        bce = F.binary_cross_entropy_with_logits(logits, smooth_tgt, reduction="none")
        window_loss = (bce * pw * boundary_weights * mask).sum() / mask.sum().clamp_min(1.0)

        # --- multiple-instance term ---------------------------------------- #
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~valid_mask, neg_inf)
        clip_max = masked_logits.max(dim=1).values                     # [B]
        clip_label = ((tgt * mask).sum(dim=1) > 0).to(logits.dtype)    # [B]
        has_valid = mask.sum(dim=1) > 0
        bag = F.binary_cross_entropy_with_logits(
            clip_max[has_valid], clip_label[has_valid], reduction="mean"
        ) if bool(has_valid.any()) else logits.sum() * 0.0

        # --- onset auxiliary ------------------------------------------------ #
        prev = torch.zeros_like(tgt)
        prev[:, 1:] = tgt[:, :-1]
        onset_target = ((tgt > 0.5) & (prev < 0.5)).to(logits.dtype)
        onset_mask = mask * (boundary_weights > 1.0).to(logits.dtype)
        denom = onset_mask.sum().clamp_min(1.0)
        onset = (
            F.binary_cross_entropy_with_logits(onset_logits, onset_target, reduction="none")
            * onset_mask
        ).sum() / denom

        total = window_loss + self.bag_weight * bag + self.onset_weight * onset
        return total, {
            "window": float(window_loss.detach()),
            "bag": float(bag.detach()),
            "onset": float(onset.detach()),
        }
