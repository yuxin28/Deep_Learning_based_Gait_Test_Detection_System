#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Stage 1.
Matches Stage 2 structure.
"""
import numpy as np
import torch

def update_confmat(cm: np.ndarray,
                   y_true: torch.Tensor,
                   y_pred: torch.Tensor,
                   num_classes: int = 2) -> None:
    """
    Update confusion matrix in-place.
    y_true, y_pred: (B,) or (B, 1) integer labels in [0, 1]
    """
    yt = y_true.reshape(-1).cpu().numpy()
    yp = y_pred.reshape(-1).cpu().numpy()
    idx = yt * num_classes + yp
    # Handle potential boundary errors safely
    idx = idx.astype(int)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    cm += counts.reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray):
    """
    Compute Precision, Recall, and Weighted F1.
    Returns full arrays for Precision and Recall to allow detailed diagnosis.
    """
    cm = cm.astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    total = support.sum()

    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    # Weighted F1 (overall performance considering imbalance)
    weighted_f1 = float((f1 * (support / max(total, 1e-9))).sum())

    # Return raw arrays for P and R so we can print Class 1 specifically
    return precision, recall, weighted_f1