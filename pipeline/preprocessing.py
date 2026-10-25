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
from typing import Any, Dict, Optional, Tuple, List

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Required columns for the pipeline
SENSOR_COLS = [
    'right_sensor_acc_x', 'right_sensor_acc_y', 'right_sensor_acc_z',
    'right_sensor_gyr_x', 'right_sensor_gyr_y', 'right_sensor_gyr_z',
    'left_sensor_acc_x', 'left_sensor_acc_y', 'left_sensor_acc_z',
    'left_sensor_gyr_x', 'left_sensor_gyr_y', 'left_sensor_gyr_z'
]

FS = 102.4

# Validation-selected Ullrich preprocessing configuration used in the final
# pipeline. These defaults can still be overridden via `ullrich_config`.
DEFAULT_ULLRICH_CONFIG: Dict[str, Any] = {
    "sensor_scope": "both",
    "merge_gait_sequences_from_sensors": True,
    "sensor_channel_config": "gyr_ml",
    "peak_prominence": 12.0,
    "active_signal_threshold": None,
    "additional_margin_s": 0.0,
}


def _build_body_frame_signal(signal: pd.DataFrame):
    """Prepare the body-frame signal representation expected by gaitmap."""
    from gaitmap.utils.coordinate_conversion import convert_to_fbf

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
    return convert_to_fbf(signal_copy, left_like="left_", right_like="right_")


def _merge_gait_sequence_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge gait sequence intervals from multiple sensors into one sorted, merged DataFrame."""
    intervals: List[Tuple[int, int]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            if end > start:
                intervals.append((start, end))

    if not intervals:
        return pd.DataFrame(columns=["gs_id", "start", "end"])

    intervals.sort()
    merged: List[List[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    merged_df = pd.DataFrame(merged, columns=["start", "end"])
    return merged_df.reset_index().rename(columns={"index": "gs_id"})


def detect_gait_sequences(
    signal: pd.DataFrame,
    ullrich_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Detects gait sequences in IMU sensor data using the Ullrich Gait Sequence Detection.
    
    Args:
        signal (pd.DataFrame): DataFrame containing accelerometer and gyroscope data.
        ullrich_config: Optional override for Ullrich parameters. Supports the
            original gaitmap constructor arguments plus a local `sensor_scope`
            option with values `left`, `right`, or `both`. When omitted, the
            final validation-selected default uses both sensors, merges the
            resulting gait sequences, and runs `gyr_ml` with
            `peak_prominence=12`.
    
    Returns:
        pd.DataFrame: A DataFrame containing gait sequence boundaries in sample positions.
    """
    # Lazy import to avoid dependency issues if gaitmap isn't installed for simple UI tasks
    try:
        from gaitmap.gait_detection import UllrichGaitSequenceDetection
    except ImportError:
        raise ImportError("The 'gaitmap' library is required for this preprocessing step.")

    # Validate columns
    if not all(col in signal.columns for col in SENSOR_COLS):
        missing = set(SENSOR_COLS) - set(signal.columns)
        raise ValueError(f"Signal is missing columns: {missing}")
    
    # Check signal length (assuming 102.4 Hz)
    if len(signal) < 103:  # approx 1 second
        return pd.DataFrame(columns=["gs_id", "start", "end"])
    
    # Run Detection
    config = dict(DEFAULT_ULLRICH_CONFIG)
    config.update(dict(ullrich_config or {}))
    sensor_scope = str(config.pop("sensor_scope", "left")).lower()
    if sensor_scope not in {"left", "right", "both"}:
        raise ValueError(f"Unsupported Ullrich sensor_scope '{sensor_scope}'.")

    bf_data = _build_body_frame_signal(signal)
    if sensor_scope == "left":
        data = bf_data["left_sensor"]
    elif sensor_scope == "right":
        data = bf_data["right_sensor"]
    else:
        data = bf_data

    gsd = UllrichGaitSequenceDetection(**config)
    gsd.detect(data=data, sampling_rate_hz=FS)

    gait_sequences = gsd.gait_sequences_
    if isinstance(gait_sequences, dict):
        gait_sequences = _merge_gait_sequence_frames(list(gait_sequences.values()))
    elif gait_sequences is None or gait_sequences.empty:
        gait_sequences = pd.DataFrame(columns=["gs_id", "start", "end"])
    else:
        if "gs_id" not in gait_sequences.columns:
            gait_sequences = gait_sequences.reset_index().rename(columns={"index": "gs_id"})
        gait_sequences = gait_sequences[["gs_id", "start", "end"]].copy()

    return gait_sequences


def convert_gait_sequences_to_intervals(
    signal: pd.DataFrame,
    gait_sequences: pd.DataFrame,
    sampling_rate_hz: float = FS,
) -> List[Tuple[int, int]]:
    """
    Convert gaitmap start/end sample indices to timestamp intervals in nanoseconds.

    The returned intervals are absolute nanosecond timestamps aligned to the signal index.
    """
    if gait_sequences is None or gait_sequences.empty:
        return []

    timestamps = signal.index.to_numpy(dtype=np.int64, copy=False)
    if timestamps.size == 0:
        return []

    sample_period_ns = int(round(1e9 / sampling_rate_hz))
    intervals: List[Tuple[int, int]] = []

    for _, row in gait_sequences.iterrows():
        start_idx = int(row["start"])
        end_idx = int(row["end"])
        start_idx = min(max(start_idx, 0), len(timestamps) - 1)
        end_idx = min(max(end_idx, 1), len(timestamps))

        start_ns = int(timestamps[start_idx])
        if end_idx < len(timestamps):
            end_ns = int(timestamps[end_idx])
        else:
            end_ns = int(timestamps[-1]) + sample_period_ns

        if end_ns > start_ns:
            intervals.append((start_ns, end_ns))

    intervals.sort()
    return intervals


def _build_window_keep_mask(
    start_positions: np.ndarray,
    window_len: int,
    gait_sequences: pd.DataFrame,
) -> np.ndarray:
    """
    Mark windows that overlap at least one detected gait sequence.

    `gaitmap` returns start/end in sample positions relative to the input signal.
    A window [start, end) is kept if it overlaps any gait interval [gs_start, gs_end).
    """
    keep_mask = np.zeros(len(start_positions), dtype=bool)
    if gait_sequences is None or gait_sequences.empty:
        return keep_mask

    end_positions = start_positions + window_len
    gait_intervals = gait_sequences[["start", "end"]].to_numpy(dtype=np.int64, copy=False)

    for gs_start, gs_end in gait_intervals:
        keep_mask |= (start_positions < gs_end) & (end_positions > gs_start)

    return keep_mask


def build_window_filter_metadata(
    raw_signal: pd.DataFrame,
    sampling_rate: float,
    sliding_window_s: int,
    stride_s: float,
    ullrich_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute which preprocessing windows survive Ullrich gait filtering and NaN removal.

    Returns lightweight metadata suitable for analysis/grid search without
    materializing the actual window tensors.
    """
    if not all(col in raw_signal.columns for col in SENSOR_COLS):
        missing = set(SENSOR_COLS) - set(raw_signal.columns)
        raise ValueError(f"Signal is missing columns: {missing}")

    signal_filtered = raw_signal[SENSOR_COLS].copy()
    window_len = int(sliding_window_s * sampling_rate)
    stride_len = int(stride_s * sampling_rate)

    if len(signal_filtered) < window_len:
        return None

    gait_seqs = detect_gait_sequences(signal_filtered, ullrich_config=ullrich_config)
    num_windows = ((len(signal_filtered) - window_len) // stride_len) + 1
    start_positions = np.arange(num_windows, dtype=np.int64) * stride_len

    if num_windows <= 0:
        return None

    signal_values = signal_filtered.to_numpy(dtype=np.float32, copy=False)
    row_has_nan = np.isnan(signal_values).any(axis=1).astype(np.int64)
    nan_prefix = np.concatenate(([0], np.cumsum(row_has_nan)))
    valid_window_mask = np.array(
        [nan_prefix[start + window_len] - nan_prefix[start] == 0 for start in start_positions],
        dtype=bool,
    )

    ullrich_keep_mask = _build_window_keep_mask(start_positions, window_len, gait_seqs)
    keep_mask = ullrich_keep_mask & valid_window_mask

    return {
        "window_len": window_len,
        "stride_len": stride_len,
        "start_positions": start_positions,
        "valid_window_mask": valid_window_mask,
        "ullrich_keep_mask": ullrich_keep_mask,
        "keep_mask": keep_mask,
        "gait_sequences": gait_seqs,
    }

def extract_sliding_windows(
    raw_signal: pd.DataFrame, 
    sampling_rate: float, 
    sliding_window_s: int, 
    stride_s: float,
    ullrich_config: Optional[Dict[str, Any]] = None,
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

    metadata = build_window_filter_metadata(
        raw_signal,
        sampling_rate=sampling_rate,
        sliding_window_s=sliding_window_s,
        stride_s=stride_s,
        ullrich_config=ullrich_config,
    )
    if metadata is None:
        window_len = int(sliding_window_s * sampling_rate)
        print(f"[Warning] Signal length {len(raw_signal)} < Window {window_len}.")
        return None

    keep_mask = metadata["keep_mask"]
    if not keep_mask.any():
        return None

    signal_values = raw_signal.to_numpy(dtype=np.float32, copy=False)
    timestamps = raw_signal.index.to_numpy(dtype=np.int64, copy=False)
    window_len = metadata["window_len"]

    for start_pos in metadata["start_positions"][keep_mask]:
        end_pos = start_pos + window_len

        seg_arr = signal_values[start_pos:end_pos]
        if seg_arr.shape[0] != window_len:
            continue

        segments.append(seg_arr)
        timestamps_start.append(int(timestamps[start_pos]))
    
    if not segments:
        return None
    
    return np.stack(segments).astype(np.float32, copy=False), np.array(timestamps_start, dtype=np.int64)

def process_single_file(
    signal: pd.DataFrame,
    frequency: float = 102.4,
    sliding_window_s: int = 90,
    stride_s: int = 30,
    ullrich_config: Optional[Dict[str, Any]] = None,
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
    result = extract_sliding_windows(
        signal_filtered,
        frequency,
        sliding_window_s,
        stride_s,
        ullrich_config=ullrich_config,
    )
    
    if result is None:
        return None, None
        
    return result
