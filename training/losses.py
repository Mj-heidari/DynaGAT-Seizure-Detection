from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareFocalLoss(nn.Module):
    """Focal BCE multiplied by onset-boundary weights and an explicit valid mask."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        boundary_weights: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        boundary_weights = boundary_weights.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal = alpha_t * (1.0 - p_t).clamp_min(1e-8).pow(self.gamma)
        loss = bce * focal * boundary_weights

        if valid_mask is not None:
            mask = valid_mask.to(dtype=loss.dtype)
            return (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss.mean()
