#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gait Detection Pipeline
-----------------------
Orchestrates the end-to-end analysis:
1. Preprocessing (Windowing)
2. Stage 1: Binary Gait Detection (TCN/BiLSTM)
3. Continuity Rule (Bridging small gaps in detection)
4. Stage 2: Gait Segmentation (UNet)
5. Post-processing (Event extraction from probabilities)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union

# -------------------------------------------------------------------------
# 1. Project Root Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # pipeline/pipeline.py -> Project_Root 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -------------------------------------------------------------------------
# 2. Pipeline Imports
# -------------------------------------------------------------------------
from pipeline.preprocessing import process_single_file
from pipeline.stage1 import detect_gait_sequences
from pipeline.continuity_rule import apply_continuity_rule
from pipeline.stage2 import segment_gait_test
from pipeline.postprocessing import run_postprocessing
from pipeline.plot_detected_events import plot_detected_events

# -------------------------------------------------------------------------
# 3. Main Detection Function
# -------------------------------------------------------------------------

def detect_gait_test(
    signal_path: Union[str, Path], 
    axis_name: str = "right_sensor_gyr_y", 
    stage1_backbone: str = "tcn", 
    stage2_model: str = "unet_bigru", 
    plot_signal: bool = True, 
    use_author_weights_stage1: bool = True, 
    use_author_weights_stage2: bool = True
) -> Optional[List[Dict]]:
    """
    Runs the full gait detection pipeline on a single H5 signal file.

    Args:
        signal_path: Path to the .h5 file containing IMU data.
        axis_name: Axis name to use for plotting (if enabled).
        stage1_backbone: Backbone type for Stage 1 ('tcn' or 'tcn_bilstm').
        stage2_model: Model type for Stage 2 ('unet_att_gru' or 'unet_bigru').
        plot_signal: Whether to visualize the results.
        use_author_weights_stage1: Use pre-trained author weights for Stage 1.
        use_author_weights_stage2: Use pre-trained author weights for Stage 2.

    Returns:
        List[Dict]: A list of detected events (class_id, start, end), or None if failed.
    """
    
    # 1. Load Signal
    if not os.path.exists(signal_path):
        print(f"[Error] File not found: {signal_path}")
        return None
    
    try: 
        signal = pd.read_hdf(signal_path)
    except Exception as e:
        print(f"[Error] Failed to read HDF file: {e}")
        return None

    # 2. Preprocessing (Windowing)
    wins, ts = process_single_file(signal)
    if wins is None:
        print("[Info] No valid windows generated during preprocessing.")
        return None

    # 3. Stage 1: Binary Detection
    mask = detect_gait_sequences(
        wins, 
        backbone_type=stage1_backbone, 
        use_author_weights=use_author_weights_stage1
    )
    
    # If no gait detected, exit early
    if len(mask) == 0 or not np.any(mask):
        return None

    # 4. Continuity Rule
    new_mask = apply_continuity_rule(ts, mask)
    
    # Filter windows based on mask
    stage1_result = wins[new_mask]
    new_ts = ts[new_mask]

    # 5. Stage 2: Segmentation
    raw_result = segment_gait_test(
        stage1_result, 
        model_type=stage2_model, 
        use_author_weights=use_author_weights_stage2
    )

    # 6. Post-processing (Probabilities -> Events)
    all_events = run_postprocessing(
        window_probs=raw_result,
        window_start_times=new_ts
    )

    if not all_events:
        return None

    # 7. Visualization
    if plot_signal:
        plot_detected_events(signal, all_events, axis_name=axis_name)

    return all_events

# -------------------------------------------------------------------------
# 4. Main Entry Point (Example Usage)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example Configuration
    SAMPLE_FILE = ROOT / "pipeline" / "samples" / "PAT162_week_8_2024_10_28.h5"
    
    detect_gait_test(
        signal_path=SAMPLE_FILE, 
        axis_name="right_sensor_gyr_y", 
        stage1_backbone="tcn", 
        stage2_model="unet_bigru", 
        use_author_weights_stage1=True, 
        use_author_weights_stage2=True
    )