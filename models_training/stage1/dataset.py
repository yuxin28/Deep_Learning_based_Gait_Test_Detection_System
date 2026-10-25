#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

class GaitSensorDataset(Dataset):
    """
    Binary dataset for gait test detection from .npy files.

    Input (x): (12, T) Normalized IMU data.
    Label (y): (1,)    Binary label (1 if gait test in decision window, else 0).
    """

    def __init__(
        self,
        npy_paths: List[str],
        norm_params: Dict[str, torch.Tensor],
        sampling_freq: float = 102.4,
        segment_duration: int = 90,
        decision_duration: int = 5,
    ):
        self.npy_paths = npy_paths
        self.sampling_freq = sampling_freq
        self.segment_duration = segment_duration
        self.segment_length = int(sampling_freq * segment_duration)
        self.decision_duration = decision_duration
        self.decision_length = int(sampling_freq * decision_duration)

        # Normalization parameters (12 channels)
        self.norm_mean = torch.tensor(norm_params["mean"], dtype=torch.float32)
        self.norm_std = torch.tensor(norm_params["std"], dtype=torch.float32)

        if self.norm_mean.shape != (12,) or self.norm_std.shape != (12,):
            raise ValueError("Normalization params must have shape (12,) for 12 sensor channels.")

        self._validate_files()
        self._compute_file_indices()

        logging.info(f"GaitSensorDataset: {len(self)} samples from {len(npy_paths)} files.")

    def _validate_files(self) -> None:
        missing_files = [path for path in self.npy_paths if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")

    def _compute_file_indices(self) -> None:
        """Pre-compute mapping from global index -> (file_idx, sample_idx) by reading headers."""
        self.file_indices: List[Tuple[int, int]] = []

        for file_idx, npy_path in enumerate(self.npy_paths):
            try:
                with open(npy_path, "rb") as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)
                    sample_count = shape[0]

                for sample_idx in range(sample_count):
                    self.file_indices.append((file_idx, sample_idx))
            except Exception as e:
                logging.error(f"Error reading header of {npy_path}: {e}")
                raise

    def __len__(self) -> int:
        return len(self.file_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, sample_idx = self.file_indices[idx]
        return self._load_sample(file_idx, sample_idx)

    def _load_sample(self, file_idx: int, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path = self.npy_paths[file_idx]
        
        # mmap_mode="r" avoids loading the entire file into memory
        data = np.load(npy_path, mmap_mode="r")
        sample = data[sample_idx]  # shape: (T, 14)

        # 1. Process Sensor Data (Columns 0-11)
        sensor_data = sample[:, :12]
        x = torch.from_numpy(sensor_data.T.copy()).float()  # (12, T)
        x = (x - self.norm_mean.unsqueeze(1)) / self.norm_std.unsqueeze(1)

        # 2. Process Label (Column 13 contains gait types)
        gait_types = sample[:, 13]
        y = self._generate_target(gait_types)

        return x, y

    def _generate_target(self, gait_types: np.ndarray) -> torch.Tensor:
        """
        Determines if a valid gait test exists within the center decision window.
        """
        midpoint = (self.segment_length // 2) - 1 if self.segment_length % 2 == 0 else self.segment_length // 2
        start_decision = max(0, midpoint - self.decision_length // 2)
        end_decision = min(self.segment_length, start_decision + self.decision_length)

        # If any gait activity (>0) is detected in the window, label as positive (1.0)
        if gait_types[start_decision:end_decision].sum() > 0:
            return torch.tensor([1.0])
        return torch.tensor([0.0])