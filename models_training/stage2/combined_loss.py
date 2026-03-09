#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 Segmentation Loss Functions
-----------------------------------
Contains:
    - dice_loss: Multi-class Dice loss.
    - compute_combined_loss: Weighted sum of Cross-Entropy and Dice loss.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Computes multi-class Dice loss (one-vs-all).

    Args:
        logits: (B, C, L) Raw output from model.
        targets: (B, L) Integer class labels.
        epsilon: Smoothing factor to avoid division by zero.

    Returns:
        Scalar Dice loss (1 - Dice).
    """
    num_classes = logits.size(1)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)  # (B, C, L)

    # One-hot encode targets
    target_onehot = F.one_hot(targets, num_classes=num_classes)  # (B, L, C)
    target_onehot = target_onehot.permute(0, 2, 1).float()       # (B, C, L)

    # Calculate intersection and union over batch and length dims
    dims = (0, 2)
    intersection = torch.sum(probs * target_onehot, dims)
    union = torch.sum(probs + target_onehot, dims)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    loss = 1.0 - dice

    return loss.mean()

def compute_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    dice_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes a combined Cross-Entropy and Dice loss.

    Args:
        logits: (B, C, L) Model outputs.
        targets: (B, L) Ground truth labels.
        class_weights: Optional tensor of shape (C,) for CE loss.
        dice_weight: Weighting factor for Dice loss.

    Returns:
        total_loss: The combined loss tensor.
        components: Dictionary containing individual 'ce' and 'dice' values.
    """
    # Cross-Entropy Loss
    ce = F.cross_entropy(logits, targets, weight=class_weights)

    # Dice Loss
    if dice_weight > 0.0:
        dl = dice_loss(logits, targets)
        total = ce + (dice_weight * dl)
        return total, {"ce": float(ce.item()), "dice": float(dl.item())}
    
    return ce, {"ce": float(ce.item()), "dice": 0.0}