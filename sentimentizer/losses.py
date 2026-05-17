"""Loss functions for sentiment classification.

Separates loss implementations from trainer.py for easier unit testing
and cleaner separation of concerns.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with focal modulation for hard-example mining.

    Down-weights easy (high-confidence) predictions, focusing training
    on hard cases like ambiguous neutral reviews.

    Ref: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        weight: Optional class weights tensor of shape (num_classes,).
            Passed to CrossEntropyLoss for class balancing.
        gamma: Focusing parameter. Higher gamma reduces loss contribution
            from easy examples. Default 2.0.
        label_smoothing: Label smoothing factor in [0, 1). Default 0.0.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal cross-entropy loss.

        Args:
            logits: Raw logits of shape (B, num_classes).
            targets: Ground truth class indices of shape (B,).

        Returns:
            Scalar loss value.
        """
        ce_loss = self.ce(logits, targets)  # (B,)
        p_t = torch.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()
