#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Backbone Model
----------------------
Provides two TCN-based architectures with distinct internal structures to match 
specific pre-trained weight keys:

1. TCNBackbone: Uses ModuleLists and skip connections.
2. TCN_BiLSTM_Backbone: Uses Sequential containers and BiLSTM heads.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Shared Blocks
# ==========================================

def _norm1d(norm: str, num_channels: int) -> nn.Module:
    if norm == "BN":
        return nn.BatchNorm1d(num_channels)
    if norm == "IN":
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm == "LN":
        return nn.GroupNorm(1, num_channels)
    if norm == "None" or norm is None:
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm}")

class CausalConv1d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int = 1, 
        bias: bool = True, 
        causal: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            pad = (self.dilation * (self.kernel_size - 1), 0)
        else:
            total = self.dilation * (self.kernel_size - 1)
            left = total // 2
            right = total - left
            pad = (left, right)
        x = F.pad(x, pad)
        return self.conv(x)

class TemporalBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int, 
        dropout: float = 0.1, 
        norm: str = "BN", 
        causal: bool = True
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, causal=causal)
        self.norm1 = _norm1d(norm, out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, causal=causal)
        self.norm2 = _norm1d(norm, out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) 
            if in_channels != out_channels else nn.Identity()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        return self.activation(out + residual)

class Projector(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.0, out_dim: int = 128):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.pool(x).squeeze(-1)
        v = self.head(v)
        return self.drop(v)

class SensorChannelFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.proj(x)))

# ==========================================
# Architecture 1: ModuleList TCN + Skip
# ==========================================

class DilatedTCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int,
        dilations: List[int],
        num_stacks: int,
        dropout: float,
        norm: str,
        causal: bool,
        use_skip: bool,
        skip_dim: Optional[int] = None
    ):
        super().__init__()
        self.use_skip = use_skip
        
        if not channels:
            raise ValueError("Channels list cannot be empty.")
            
        last_block_ch = channels[-1]
        self.skip_dim = (skip_dim if (use_skip and skip_dim is not None) else last_block_ch) if use_skip else None

        blocks = []
        skips = []
        in_ch = in_channels
        
        for _ in range(num_stacks):
            for i, d in enumerate(dilations):
                out_ch = channels[min(i, len(channels)-1)]
                block = TemporalBlock(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    kernel_size=kernel_size, 
                    dilation=d, 
                    dropout=dropout, 
                    norm=norm, 
                    causal=causal
                )
                blocks.append(block)
                if use_skip:
                    skips.append(nn.Conv1d(out_ch, self.skip_dim, kernel_size=1))
                in_ch = out_ch
        
        self.blocks = nn.ModuleList(blocks)
        self.skip_projs = nn.ModuleList(skips) if use_skip else None

        if use_skip:
            self.merge = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv1d(self.skip_dim, last_block_ch, kernel_size=1),
                _norm1d(norm, last_block_ch),
                nn.ReLU(inplace=True),
            )
            
        self.out_channels_ = last_block_ch

    @property
    def out_channels(self) -> int:
        return self.out_channels_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_feat = None
        k = 0
        for block in self.blocks:
            x = block(x)
            if self.use_skip:
                proj = self.skip_projs[k](x)
                if skip_feat is None:
                    skip_feat = proj
                else:
                    skip_feat = skip_feat + proj
                k += 1
        
        if self.use_skip:
            x = self.merge(skip_feat)
        return x

class TCNBackbone(nn.Module):
    """
    Standard TCN Backbone with optional channel fusion and skip connections.
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int] = (64, 64, 128, 128),
        kernel_size: int = 8,
        dilations: List[int] = (1, 2, 4, 8, 16, 32),
        num_stacks: int = 1,
        dropout: float = 0.1,
        norm: str = "BN",
        causal: bool = True,
        use_skip: bool = True,
        fusion: bool = False,
        proj_dim: int = 128
    ):
        super().__init__()
        
        if fusion:
            self.fusion = SensorChannelFusion(in_channels, 64)
            in_channels = 64
        else:
            self.fusion = nn.Identity()

        self.tcn = DilatedTCNBackbone(
            in_channels=in_channels,
            channels=list(channels),
            kernel_size=kernel_size,
            dilations=list(dilations),
            num_stacks=num_stacks,
            dropout=dropout,
            norm=norm,
            causal=causal,
            use_skip=use_skip,
            skip_dim=None
        )
        
        self.projector = Projector(
            in_ch=self.tcn.out_channels, 
            out_dim=proj_dim
        )
        self.out_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)
        feat = self.tcn(x)
        vec = self.projector(feat)
        return vec

# ==========================================
# Architecture 2: Sequential TCN + BiLSTM
# ==========================================

class SequentialTCNWrapper(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        channels: List[int], 
        kernel_size: int, 
        dilations: List[int], 
        num_stacks: int, 
        dropout: float, 
        norm: str, 
        causal: bool
    ):
        super().__init__()
        blocks = []
        in_ch = in_channels
        
        for _ in range(num_stacks):
            for i, d in enumerate(dilations):
                out_ch = channels[min(i, len(channels) - 1)]
                blocks.append(TemporalBlock(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    kernel_size=kernel_size, 
                    dilation=d, 
                    dropout=dropout, 
                    norm=norm, 
                    causal=causal
                ))
                in_ch = out_ch
        
        self.tcn = nn.Sequential(*blocks)
        self.tcn_out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)

class BiLSTM_Wrapper(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        bidirectional: bool, 
        proj_dim: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.projector = nn.Linear(lstm_out_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        lstm_out, _ = self.lstm(x)
        feat = lstm_out.mean(dim=1)
        return self.projector(feat)

class TCN_BiLSTM_Backbone(nn.Module):
    """
    Hybrid Backbone: Sequential TCN -> BiLSTM -> Projection.
    """
    def __init__(
        self,
        in_channels: int,
        tcn_channels: Tuple[int] = (64, 64, 128),
        tcn_kernel_size: int = 15,
        tcn_dilations: Tuple[int] = (8, 16, 32),
        tcn_stacks: int = 1,
        tcn_dropout: float = 0.1,
        tcn_norm: str = "BN",
        tcn_causal: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.1, # kept for API compat, though not used in wrapped LSTM init above
        lstm_bidirectional: bool = True,
        lstm_proj_out: int = 128
    ):
        super().__init__()
        
        self.tcn = SequentialTCNWrapper(
            in_channels=in_channels,
            channels=list(tcn_channels),
            kernel_size=tcn_kernel_size,
            dilations=list(tcn_dilations),
            num_stacks=tcn_stacks,
            dropout=tcn_dropout,
            norm=tcn_norm,
            causal=tcn_causal
        )

        self.bilstm = BiLSTM_Wrapper(
            input_size=self.tcn.tcn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            proj_dim=lstm_proj_out
        )
        
        self.out_dim = lstm_proj_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x)           # (B, C, T)
        feat = feat.permute(0, 2, 1) # (B, T, C) for LSTM
        vec = self.bilstm(feat)      # (B, proj_dim)
        return vec