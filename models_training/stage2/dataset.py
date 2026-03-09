#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 Segmentation Dataset
----------------------------
Contains:
1. PDGaitAugmentation: Lightweight, physiologically-safe IMU augmentations.
2. UNetDataset: Loads (12, T) IMU data and (T,) frame-level labels from .npy files.
"""

import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

# =========================================================================
# 1. IMU Augmentation
# =========================================================================

class PDGaitAugmentation:
    """
    Applies random augmentations suitable for 1D IMU sensor data:
    - Gaussian Noise
    - Amplitude Scaling
    - Baseline Drift
    - Random Masking
    """
    def __init__(
        self,
        aug_prob: float = 0.2,
        noise_std_range: Tuple[float, float] = (0.001, 0.005),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        baseline_range: Tuple[float, float] = (-0.01, 0.01),
        mask_prob: float = 0.1,
        mask_ratio_range: Tuple[float, float] = (0.02, 0.05),
    ):
        self.aug_prob = aug_prob
        self.noise_std_range = noise_std_range
        self.scale_range = scale_range
        self.baseline_range = baseline_range
        self.mask_prob = mask_prob
        self.mask_ratio_range = mask_ratio_range

        self.methods = [
            ("sensor_noise", 0.35),
            ("amplitude_scaling", 0.35),
            ("baseline_drift", 0.20),
            ("random_masking", 0.10),
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, T) Normalized IMU tensor.
        Returns:
            Augmented tensor.
        """
        if np.random.rand() > self.aug_prob:
            return x

        x_aug = x.clone()

        # Decide whether to apply 1 or 2 augmentations
        num_ops = 1 if np.random.rand() < 0.4 else 2
        
        chosen_methods = np.random.choice(
            [m[0] for m in self.methods], 
            size=num_ops, 
            replace=False, 
            p=[m[1] for m in self.methods]
        )

        for method in chosen_methods:
            if method == "sensor_noise":
                x_aug = self._sensor_noise(x_aug)
            elif method == "amplitude_scaling":
                x_aug = self._amplitude_scaling(x_aug)
            elif method == "baseline_drift":
                x_aug = self._baseline_drift(x_aug)
            elif method == "random_masking":
                x_aug = self._random_masking(x_aug)

        # Safety check: avoid destructively altering signal statistics
        if not self._validate(x, x_aug):
            return x
        
        return x_aug

    def _sensor_noise(self, x):
        std = np.random.uniform(*self.noise_std_range)
        return x + torch.randn_like(x) * std

    def _amplitude_scaling(self, x):
        scale = np.random.uniform(*self.scale_range)
        return x * float(scale)

    def _baseline_drift(self, x):
        bias = np.random.uniform(*self.baseline_range)
        return x + float(bias)

    def _random_masking(self, x):
        if np.random.rand() > self.mask_prob:
            return x
        
        C, T = x.shape
        mask_len = int(T * np.random.uniform(*self.mask_ratio_range))
        if mask_len < 2:
            return x
            
        start = np.random.randint(0, T - mask_len)
        # Mask 1 to 4 channels randomly
        num_channels = np.random.randint(1, min(4, C) + 1)
        chs = np.random.choice(C, num_channels, replace=False)
        
        out = x.clone()
        out[chs, start:start + mask_len] = 0.0
        return out

    def _validate(self, orig, aug):
        """Ensures augmentation didn't destroy signal properties."""
        std_o = torch.std(orig, dim=-1) + 1e-8
        std_a = torch.std(aug, dim=-1)
        
        ratio = std_a / std_o
        if torch.any(ratio < 0.5) or torch.any(ratio > 1.5):
            return False

        mean_diff = torch.abs(torch.mean(aug, dim=-1) - torch.mean(orig, dim=-1))
        if torch.any(mean_diff > 0.2 * std_o):
            return False
            
        return True


# =========================================================================
# 2. UNet Dataset
# =========================================================================

class UNetDataset(Dataset):
    """
    Loads IMU data for segmentation tasks.

    Input: .npy files of shape (N, T, 14)
           - Channels 0-11: Sensor Data
           - Channel 13:    Frame-level Labels (0..4)

    Returns:
           x: (12, T) Normalized Float Tensor
           y: (T,)    Long Tensor (Labels)
    """

    def __init__(
        self,
        npy_paths: List[str],
        norm_params: Dict[str, np.ndarray],
        frequency: float = 102.4,
        use_augmentation: bool = True,
        aug_config: Optional[dict] = None,
    ):
        self.npy_paths = npy_paths
        
        # Load normalization params
        self.norm_mean = torch.tensor(norm_params["mean"], dtype=torch.float32)
        self.norm_std = torch.tensor(norm_params["std"], dtype=torch.float32)

        # Setup augmentation
        self.use_augmentation = use_augmentation
        self.aug = PDGaitAugmentation(**(aug_config or {})) if use_augmentation else None

        # Indexing: Map global index -> (file_idx, sample_idx)
        self.file_index_map = []
        for f_id, path in enumerate(npy_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
                
            # Read only header to get shape
            with open(path, "rb") as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                num_samples = shape[0]
                
            for i in range(num_samples):
                self.file_index_map.append((f_id, i))

    def __len__(self):
        return len(self.file_index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, sample_id = self.file_index_map[idx]
        path = self.npy_paths[file_id]
        
        # Load specific sample using mmap
        # (N, T, 14) -> (T, 14)
        arr = np.load(path, mmap_mode="r")
        sample = arr[sample_id]

        # 1. Process IMU Data (12 channels)
        imu = sample[:, :12].astype(np.float32)
        x = torch.from_numpy(imu).transpose(0, 1)  # (T, 12) -> (12, T)

        # Normalize
        x = (x - self.norm_mean[:, None]) / (self.norm_std[:, None] + 1e-8)

        # Augment (Training only)
        if self.aug is not None:
            x = self.aug(x)

        # 2. Process Labels (Channel 13)
        labels = sample[:, 13].astype(np.int64)
        y = torch.from_numpy(labels)  # (T,)

        return x, y