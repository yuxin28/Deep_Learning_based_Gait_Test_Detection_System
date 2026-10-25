#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 Utility Functions
-------------------------
Contains helper functions for computing confusion matrices and extraction 
classification metrics (Accuracy, Macro-F1, Weighted-F1).
"""

import numpy as np
import torch
from typing import Tuple

def update_confmat(cm: np.ndarray, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> None:
    """
    Updates the confusion matrix in-place using efficient numpy indexing.
    
    Args:
        cm: (num_classes, num_classes) numpy array accumulator.
        y_true: Ground truth labels (Tensor).
        y_pred: Predicted labels (Tensor).
        num_classes: Number of classes.
    """
    # Flatten inputs
    yt = y_true.reshape(-1).cpu().numpy()
    yp = y_pred.reshape(-1).cpu().numpy()
    
    # Calculate indices for the confusion matrix
    idx = yt * num_classes + yp
    
    # Count occurrences using bincount for speed
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    
    # Reshape and accumulate
    cm += counts.reshape(num_classes, num_classes)

def metrics_from_cm(cm: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Accuracy, Macro-F1, and Weighted-F1 from a confusion matrix.

    Args:
        cm: (num_classes, num_classes) Confusion Matrix.

    Returns:
        (accuracy, macro_f1, weighted_f1)
    """
    cm = cm.astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    total = support.sum()

    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # Calculate per-class metrics, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    # Aggregate metrics
    macro_f1 = float(f1.mean())
    
    # Avoid division by zero for weighted F1 and Accuracy
    safe_total = max(total, 1e-9)
    weighted_f1 = float((f1 * (support / safe_total)).sum())
    acc = float(tp.sum() / safe_total)

    return acc, macro_f1, weighted_f1