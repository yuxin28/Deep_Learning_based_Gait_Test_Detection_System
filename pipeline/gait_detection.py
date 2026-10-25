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
from typing import Any, Callable, Dict, List, Optional, Union

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
from pipeline.preprocessing import (
    process_single_file,
)
from pipeline.stage1 import detect_gait_sequences as detect_stage1_gait_sequences
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
    use_author_weights_stage2: bool = True,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    continuity_config: Optional[Dict[str, Any]] = None,
    postprocessing_config: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
    signal: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
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
        preprocessing_config: Optional config override for preprocessing/windowing,
            including `sliding_window_s`, `stride_s`, `sampling_rate_hz`, and
            nested `ullrich_config`.
        continuity_config: Optional config override for the continuity rule.
        postprocessing_config: Optional config override for the post-processing stage.
        return_debug: If True, also return debug artifacts from continuity and post-processing.
        signal: Optional already-loaded signal; when given, `signal_path` is not read again.
        progress_callback: Optional function called with a stage name ("preprocessing",
            "stage1", "continuity", "stage2", "postprocessing") before each stage starts.
            An exception raised by the callback aborts the pipeline (used for cancelling).

    Returns:
        List[Dict]: A list of detected events (class_id, start, end), or None if failed.
        If `return_debug=True`, returns `{"events": ..., "debug": ...}` instead.
    """

    def debug_payload(
        *,
        status: str,
        events: Optional[List[Dict]] = None,
        stage1_probs: Optional[np.ndarray] = None,
        forwarded_stage1_probs: Optional[np.ndarray] = None,
        continuity_debug: Optional[Dict[str, Any]] = None,
        postprocessing_debug: Optional[Dict[str, Any]] = None,
        raw_window_probs: Optional[np.ndarray] = None,
        raw_window_start_times: Optional[np.ndarray] = None,
        candidate_window_start_times: Optional[np.ndarray] = None,
        forwarded_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Standardize debug payloads, including cacheable post-processing inputs."""
        return {
            "events": [] if events is None else events,
            "debug": {
                "status": status,
                "stage1_probs": stage1_probs,
                "forwarded_stage1_probs": forwarded_stage1_probs,
                "candidate_window_start_times": candidate_window_start_times,
                "forwarded_mask": forwarded_mask,
                "continuity_debug": continuity_debug,
                "postprocessing_debug": postprocessing_debug,
                "postprocessing_inputs": {
                    "window_probs": raw_window_probs,
                    "window_start_times": raw_window_start_times,
                },
            },
        }
    
    def report(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(stage)

    # 1. Load Signal
    if signal is None:
        if not os.path.exists(signal_path):
            print(f"[Error] File not found: {signal_path}")
            return debug_payload(status="missing_signal_path") if return_debug else None

        try:
            signal = pd.read_hdf(signal_path)
        except Exception as e:
            print(f"[Error] Failed to read HDF file: {e}")
            return debug_payload(status="failed_to_read_signal") if return_debug else None

    # 2. Preprocessing (Windowing)
    report("preprocessing")
    preprocessing_cfg = dict(preprocessing_config or {})
    ullrich_config = preprocessing_cfg.get("ullrich_config")
    wins, ts = process_single_file(
        signal,
        frequency=float(preprocessing_cfg.get("sampling_rate_hz", 102.4)),
        sliding_window_s=int(preprocessing_cfg.get("sliding_window_s", 90)),
        stride_s=float(preprocessing_cfg.get("stride_s", 30)),
        ullrich_config=ullrich_config,
    )
    if wins is None:
        print("[Info] No valid windows generated during preprocessing.")
        return debug_payload(status="no_valid_windows") if return_debug else None

    # 3. Stage 1: Binary Detection
    report("stage1")
    if return_debug:
        mask, stage1_probs = detect_stage1_gait_sequences(
            wins,
            backbone_type=stage1_backbone,
            use_author_weights=use_author_weights_stage1,
            return_probs=True,
        )
    else:
        mask = detect_stage1_gait_sequences(
            wins,
            backbone_type=stage1_backbone,
            use_author_weights=use_author_weights_stage1,
        )
        stage1_probs = None
    
    # If no gait detected, exit early
    if len(mask) == 0 or not np.any(mask):
        return debug_payload(
            status="no_stage1_positive_windows",
            stage1_probs=stage1_probs,
            candidate_window_start_times=ts,
        ) if return_debug else None

    # 4. Continuity Rule
    report("continuity")
    if return_debug:
        new_mask, continuity_debug = apply_continuity_rule(
            ts,
            mask,
            config=continuity_config,
            return_debug=True,
        )
    else:
        new_mask = apply_continuity_rule(
            ts,
            mask,
            config=continuity_config,
        )
        continuity_debug = None
    
    # Filter windows based on mask
    stage1_result = wins[new_mask]
    new_ts = ts[new_mask]
    forwarded_stage1_probs = stage1_probs[new_mask] if stage1_probs is not None else None

    if stage1_result.size == 0:
        return debug_payload(
            status="no_windows_after_continuity",
            stage1_probs=stage1_probs,
            forwarded_stage1_probs=forwarded_stage1_probs,
            candidate_window_start_times=ts,
            forwarded_mask=new_mask,
            continuity_debug=continuity_debug,
        ) if return_debug else None

    # 5. Stage 2: Segmentation
    report("stage2")
    raw_result = segment_gait_test(
        stage1_result, 
        model_type=stage2_model, 
        use_author_weights=use_author_weights_stage2
    )

    # 6. Post-processing (Probabilities -> Events)
    report("postprocessing")
    if return_debug:
        all_events, postprocessing_debug = run_postprocessing(
            window_probs=raw_result,
            window_start_times=new_ts,
            postprocessing_config=postprocessing_config,
            return_debug=True,
        )
    else:
        all_events = run_postprocessing(
            window_probs=raw_result,
            window_start_times=new_ts,
            postprocessing_config=postprocessing_config,
        )
        postprocessing_debug = None

    if not all_events:
        return debug_payload(
            status="no_events_after_postprocessing",
            events=[],
            stage1_probs=stage1_probs,
            forwarded_stage1_probs=forwarded_stage1_probs,
            candidate_window_start_times=ts,
            forwarded_mask=new_mask,
            continuity_debug=continuity_debug,
            postprocessing_debug=postprocessing_debug,
            raw_window_probs=raw_result,
            raw_window_start_times=new_ts,
        ) if return_debug else None

    # 7. Visualization
    if plot_signal:
        plot_detected_events(signal, all_events, axis_name=axis_name)

    if return_debug:
        return debug_payload(
            status="ok",
            events=all_events,
            stage1_probs=stage1_probs,
            forwarded_stage1_probs=forwarded_stage1_probs,
            candidate_window_start_times=ts,
            forwarded_mask=new_mask,
            continuity_debug=continuity_debug,
            postprocessing_debug=postprocessing_debug,
            raw_window_probs=raw_result,
            raw_window_start_times=new_ts,
        )

    return all_events

# -------------------------------------------------------------------------
# 4. Main Entry Point (Example Usage)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example Configuration
    SAMPLE_FILE = ROOT / "pipeline" / "samples" / "sample.h5"
    
    detect_gait_test(
        signal_path=SAMPLE_FILE, 
        axis_name="right_sensor_gyr_y", 
        stage1_backbone="tcn", 
        stage2_model="unet_bigru", 
        use_author_weights_stage1=True, 
        use_author_weights_stage2=True
    )
