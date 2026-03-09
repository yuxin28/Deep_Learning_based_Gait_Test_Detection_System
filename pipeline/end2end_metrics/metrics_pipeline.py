#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-End Metrics Pipeline
---------------------------
Runs the full gait detection and segmentation pipeline on test datasets 
(Between-Subject and Within-Subject) and computes evaluation metrics.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------------
# 1. Project Root Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # pipeline/end2end_metrics/metrics_pipeline.py -> Project_Root 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.gait_detection import detect_gait_test
from pipeline.end2end_metrics.utils import (
    load_json, 
    save_json, 
    initialize_metrics_dict, 
    process_targets, 
    evaluate_event_detection, 
    aggregate_and_report
)

# -------------------------------------------------------------------------
# Processing Logic
# -------------------------------------------------------------------------

def process_dataset(
    paths_dict: dict, 
    metrics_storage: dict, 
    description: str, 
    use_author_weights_stage1: bool = True, 
    use_author_weights_stage2: bool = True
) -> dict:
    """
    Runs detection and evaluation for a specific dataset (dict of paths).
    """
    for patid_date, paths in tqdm(paths_dict.items(), desc=description):
        
        # 1. Extract Patient ID (Format: "PATID_Date_...")
        patid = patid_date.split("_")[0]
        
        # 2. Load Ground Truth (CSV) and Signal (H5)
        # Assuming paths lists have at least one element
        csv_path = paths["csv_files"][0]
        h5_path = paths["h5_files"][0]
        
        try:
            targets = pd.read_csv(csv_path)
            targets_processed = process_targets(targets)
        except Exception as e:
            print(f"[Error] Skipping {patid_date}: {e}")
            continue

        # 3. Run End-to-End Prediction Pipeline
        predictions = detect_gait_test(
            signal_path=h5_path, 
            plot_signal=False,
            use_author_weights_stage1=use_author_weights_stage1,
            use_author_weights_stage2=use_author_weights_stage2
        )

        # 4. Evaluate: Compare Prediction vs Ground Truth
        # evaluate_event_detection safely handles empty/None predictions
        metrics = evaluate_event_detection(predictions, targets_processed)

        # 5. Update Metrics Storage
        if patid in metrics_storage:
            for class_id in range(1, 5):
                if class_id in metrics:
                    metrics_storage[patid][class_id]["TP"] += metrics[class_id]["TP"]
                    metrics_storage[patid][class_id]["FP"] += metrics[class_id]["FP"]
                    metrics_storage[patid][class_id]["FN"] += metrics[class_id]["FN"]
        else:
            print(f"[Warning] Patient {patid} found in paths but not in initialized metrics keys.")

    return metrics_storage


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- Configuration ---
    INFO_DIR = ROOT / "pipeline" / "end2end_metrics" / "test_dataset_info"
    RESULTS_DIR = ROOT / "pipeline" / "end2end_metrics" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    USE_AUTHOR_W1 = True
    USE_AUTHOR_W2 = True

    print("Loading dataset configurations...")

    # Load Patient Lists (Keys)
    pat_between_list = list(load_json(INFO_DIR / "test_between.json").keys())
    pat_within_list  = list(load_json(INFO_DIR / "test_within.json").keys())

    # Load File Paths
    pat_between_paths = load_json(INFO_DIR / "test_between_paths.json")
    pat_within_paths  = load_json(INFO_DIR / "test_within_paths.json")

    # Initialize Containers
    pat_between_metrics = initialize_metrics_dict(pat_between_list)
    pat_within_metrics  = initialize_metrics_dict(pat_within_list)

    # --- Process Datasets ---

    # 1. Between-Subject
    process_dataset(
        paths_dict=pat_between_paths, 
        metrics_storage=pat_between_metrics, 
        description="Between-Subject",
        use_author_weights_stage1=USE_AUTHOR_W1,
        use_author_weights_stage2=USE_AUTHOR_W2
    )

    # 2. Within-Subject
    process_dataset(
        paths_dict=pat_within_paths, 
        metrics_storage=pat_within_metrics, 
        description="Within-Subject",
        use_author_weights_stage1=USE_AUTHOR_W1,
        use_author_weights_stage2=USE_AUTHOR_W2
    )

    # --- Aggregate & Report ---
    
    print("\nAggregating Results...")
    
    between_results = aggregate_and_report(
        metrics_data=pat_between_metrics, 
        prefix="Between-Subject",
    )

    within_results = aggregate_and_report(
        metrics_data=pat_within_metrics, 
        prefix="Within-Subject",
    )

    # --- Save Output ---
    
    print(f"\nSaving results to: {RESULTS_DIR}")
    save_json(pat_between_metrics, RESULTS_DIR / "between_metrics.json")
    save_json(pat_within_metrics, RESULTS_DIR / "within_metrics.json")
    save_json(between_results, RESULTS_DIR / "between_results.json")
    save_json(within_results, RESULTS_DIR / "within_results.json")
    
    print("Done.")