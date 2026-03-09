import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


class GaitBinaryDetector(nn.Module):
    """Binary gait test detector on top of a backbone.

    The backbone must return a global feature vector of dimension `feat_dim`
    for each input segment.

    Args:
        backbone:  Feature extractor module.
        feat_dim:  Dimensionality of backbone output.
        hidden_dim: Hidden size in the MLP classification head.

    Input:
        x: (B, C, T) or whatever the backbone expects.

    Output:
        logits: (B, 1) raw binary logits.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)

        if isinstance(feat, dict):
            # For TCN_BiLSTM_Backbone, we take the global vector
            feat = feat["global"]

        # If backbone returns (B, D, 1), squeeze last dimension
        if feat.dim() == 3 and feat.size(-1) == 1:
            feat = feat.squeeze(-1)

        logits = self.cls_head(feat)  # (B, 1)
        return logits
