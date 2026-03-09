#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Model Definition for Stage 2 Segmentation
-------------------------------------------------
Configurable UNet supporting:
- Depthwise separable convolutions (standard)
- Optional sensor-channel fusion
- Optional attention gates in skip connections
- Optional GRU bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. Optional Modules (Fusion, Attention)
# =========================================================================

class SensorChannelFusion(nn.Module):
    """
    Fuses multiple sensor channels before the encoder.
    """
    def __init__(self, in_channels, out_channels, groups=4, mid_channels=64, use_bn=True):
        super().__init__()
        assert in_channels % groups == 0
        
        def bn(c):
            return nn.BatchNorm1d(c) if use_bn else nn.Identity()

        # Path 1: Pointwise
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not use_bn),
            bn(out_channels),
            nn.GELU(),
        )

        # Path 2: Grouped -> Pointwise
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=not use_bn),
            bn(mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=not use_bn),
            bn(out_channels),
            nn.GELU(),
        )

        # Fusion
        self.finalfusion = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=1, bias=not use_bn),
            bn(out_channels),
        )

        # Residual shortcut
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not use_bn),
            bn(out_channels),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        f1 = self.path1(x)
        f2 = self.path2(x)
        fused = self.finalfusion(torch.cat([f1, f2], dim=1))
        return self.activation(fused + self.input_proj(x))

class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.relu(g1 + x1)
        psi = self.psi(alpha)
        return x * psi

# =========================================================================
# 2. Convolutional Blocks
# =========================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DoubleConv(nn.Module):
    """
    Two Depthwise Separable Convs with BN, ReLU, and Dropout.
    Structure matches pre-trained checkpoints.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_rate),
            
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_rate),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling path: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout_rate),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling path: ConvTranspose -> (Attention) -> Concat -> DoubleConv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )

        if use_attention:
            self.attn_gate = AttentionGate(
                F_g=in_channels // 2,
                F_l=in_channels // 2,
                F_int=in_channels // 4,
            )

        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch due to odd input dims
        if x1.size(-1) != x2.size(-1):
            x1 = F.interpolate(x1, size=x2.size(-1), mode="linear", align_corners=False)

        if self.use_attention:
            x2 = self.attn_gate(x1, x2)

        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# =========================================================================
# 3. Main Network: GaitSegUNet
# =========================================================================

class GaitSegUNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        base_filter=64,
        dropout_rate=0.1,
        use_attention=False,
        use_recurrent_bottleneck=False,
        use_fusion=False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_recurrent_bottleneck = use_recurrent_bottleneck
        self.use_fusion = use_fusion

        f = base_filter

        # Optional Fusion
        if self.use_fusion:
            self.fusion = SensorChannelFusion(n_channels, 64)
            n_channels = 64

        # Encoder
        self.inc = DoubleConv(n_channels, f, dropout_rate)
        self.down1 = Down(f, f * 2, dropout_rate)
        self.down2 = Down(f * 2, f * 4, dropout_rate)
        self.down3 = Down(f * 4, f * 8, dropout_rate)
        self.down4 = Down(f * 8, f * 16, dropout_rate)

        # Optional GRU Bottleneck
        if self.use_recurrent_bottleneck:
            bottleneck_channels = f * 16
            self.recurrent_layer = nn.GRU(
                input_size=bottleneck_channels,
                hidden_size=bottleneck_channels // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.recurrent_adapter = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=1)

        # Decoder
        self.up1 = Up(f * 16, f * 8, dropout_rate, use_attention)
        self.up2 = Up(f * 8, f * 4, dropout_rate, use_attention)
        self.up3 = Up(f * 4, f * 2, dropout_rate, use_attention)
        self.up4 = Up(f * 2, f, dropout_rate, use_attention)

        self.outc = OutConv(f, n_classes)

    def forward(self, x):
        if self.use_fusion:
            x = self.fusion(x)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        if self.use_recurrent_bottleneck:
            seq = x5.permute(0, 2, 1)
            seq, _ = self.recurrent_layer(seq)
            seq = seq.permute(0, 2, 1)
            x5 = self.recurrent_adapter(seq) + x5

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)