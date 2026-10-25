#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for highly imbalanced data.

    This implements the standard focal loss:

        FL = - alpha * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p      if y = 1
        p_t = 1 - p  if y = 0

    Args:
        alpha: Balancing factor between positive and negative samples.
        gamma: Focusing parameter.
        reduction: "mean", "sum" or "none".
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw logits of shape (B, 1) or (B,).
            targets: Binary labels of the same shape, 0 or 1.

        Returns:
            Scalar loss if reduction != "none", otherwise per-element loss.
        """
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = -weight * (1.0 - pt).pow(self.gamma) * torch.log(pt + 1e-12)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
