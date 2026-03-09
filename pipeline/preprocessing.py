#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gait Detection Preprocessing
----------------------------
Handles the extraction of valid gait segments from raw IMU signals.
1. Checks signal quality/columns.
2. Uses `gaitmap` (Ullrich Gait Sequence Detection) to filter non-gait windows.
3. Generates sliding windows for downstream model processing.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Required columns for the pipeline
SENSOR_COLS = [
    'right_sensor_acc_x', 'right_sensor_acc_y', 'right_sensor_acc_z',
    'right_sensor_gyr_x', 'right_sensor_gyr_y', 'right_sensor_gyr_z',
    'left_sensor_acc_x', 'left_sensor_acc_y', 'left_sensor_acc_z',
    'left_sensor_gyr_x', 'left_sensor_gyr_y', 'left_sensor_gyr_z'
]

def detect_gait_sequences(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Detects gait sequences in IMU sensor data using the Ullrich Gait Sequence Detection.
    
    Args:
        signal (pd.DataFrame): DataFrame containing accelerometer and gyroscope data.
    
    Returns:
        pd.DataFrame: A DataFrame containing 'start', 'end', and 'duration_s' of detected sequences.
    """
    # Lazy import to avoid dependency issues if gaitmap isn't installed for simple UI tasks
    try:
        from gaitmap.utils.coordinate_conversion import convert_to_fbf
        from gaitmap.gait_detection import UllrichGaitSequenceDetection
    except ImportError:
        raise ImportError("The 'gaitmap' library is required for this preprocessing step.")

    # Validate columns
    if not all(col in signal.columns for col in SENSOR_COLS):
        missing = set(SENSOR_COLS) - set(signal.columns)
        raise ValueError(f"Signal is missing columns: {missing}")
    
    # Prepare data structure for gaitmap (MultiIndex)
    signal_copy = signal[SENSOR_COLS].copy()
    rename_map = {
        'right_sensor_acc_x': ('right_sensor', 'acc_x'),
        'right_sensor_acc_y': ('right_sensor', 'acc_y'),
        'right_sensor_acc_z': ('right_sensor', 'acc_z'),
        'right_sensor_gyr_x': ('right_sensor', 'gyr_x'),
        'right_sensor_gyr_y': ('right_sensor', 'gyr_y'),
        'right_sensor_gyr_z': ('right_sensor', 'gyr_z'),
        'left_sensor_acc_x':  ('left_sensor', 'acc_x'),
        'left_sensor_acc_y':  ('left_sensor', 'acc_y'),
        'left_sensor_acc_z':  ('left_sensor', 'acc_z'),
        'left_sensor_gyr_x':  ('left_sensor', 'gyr_x'),
        'left_sensor_gyr_y':  ('left_sensor', 'gyr_y'),
        'left_sensor_gyr_z':  ('left_sensor', 'gyr_z'),
    }
    signal_copy.columns = pd.MultiIndex.from_tuples([rename_map[c] for c in signal_copy.columns])

    # Check signal length (assuming 102.4 Hz)
    if len(signal_copy) < 103:  # approx 1 second
        return pd.DataFrame(columns=['start', 'end', 'duration_s'])
    
    # Run Detection
    bf_data = convert_to_fbf(signal_copy, left_like="left_", right_like="right_")
    gsd = UllrichGaitSequenceDetection()
    gsd.detect(data=bf_data["left_sensor"], sampling_rate_hz=102.4)
    
    return gsd.gait_sequences_

def extract_sliding_windows(
    raw_signal: pd.DataFrame, 
    sampling_rate: float, 
    sliding_window_s: int, 
    stride_s: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts fixed-length data segments containing detected gait activity.
    
    Args:
        raw_signal: Raw sensor dataframe.
        sampling_rate: Hz (e.g., 102.4).
        sliding_window_s: Window length in seconds.
        stride_s: Step size in seconds.
        
    Returns:
        (segments, start_timestamps) or None.
    """
    segments: List[np.ndarray] = []
    timestamps_start: List[int] = []

    window_len = int(sliding_window_s * sampling_rate)
    stride_len = int(stride_s * sampling_rate)

    if len(raw_signal) < window_len:
        print(f"[Warning] Signal length {len(raw_signal)} < Window {window_len}.")
        return None
    
    indices = raw_signal.index.to_numpy()
    
    # Create sliding windows efficiently
    # shape: (num_windows, window_len)
    idx_windows = np.lib.stride_tricks.sliding_window_view(indices, window_shape=window_len)[::stride_len]
    
    for window_idxs in idx_windows:
        segment = raw_signal.loc[window_idxs]
        
        if segment.empty or segment.isna().any().any():
            continue

        # Check if the segment contains valid gait
        # Optimization: We check for ANY gait sequence inside this 90s window
        gait_seqs = detect_gait_sequences(segment)
        if gait_seqs.empty:
            continue
            
        # Ensure exact shape
        seg_arr = segment.to_numpy(dtype=np.float32)
        if seg_arr.shape[0] != window_len:
            continue
        
        segments.append(seg_arr)
        timestamps_start.append(window_idxs[0])
    
    if not segments:
        return None
    
    return np.array(segments, dtype=np.float32), np.array(timestamps_start, dtype=np.int64)

def process_single_file(
    signal: pd.DataFrame,
    frequency: float = 102.4,
    sliding_window_s: int = 90,
    stride_s: int = 30,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Main preprocessing entry point for a single file.
    
    Returns:
        segments: (N, WindowLen, Channels)
        timestamps: (N,)
    """
    if signal is None or signal.empty:
        return None, None

    # Column Validation & Selection
    if not all(col in signal.columns for col in SENSOR_COLS):
        print(f"[Error] Signal missing required sensor columns.")
        return None, None

    signal_filtered = signal[SENSOR_COLS].copy()

    # Extract Segments
    result = extract_sliding_windows(signal_filtered, frequency, sliding_window_s, stride_s)
    
    if result is None:
        return None, None
        
    return result