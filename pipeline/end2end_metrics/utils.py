#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Utility Functions
--------------------------
Helper functions for:
- Loading/Saving JSONs.
- Processing target events from CSV.
- Computing IoU (Intersection over Union).
- Evaluating event detection (TP/FP/FN matching).
- Aggregating and reporting metrics.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union, Dict
from collections import defaultdict

# -------------------------------------------------------------------------
# 1. Project Root Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # pipeline/end2end_metrics/utils.py -> Project_Root 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -------------------------------------------------------------------------
# I/O Helpers
# -------------------------------------------------------------------------

def load_json(file_path: Path):
    """Loads a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data: dict, file_path: Path):
    """Saves a dictionary as a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def initialize_metrics_dict(patient_ids: list) -> dict:
    """Initializes a nested dictionary structure for storing metrics per patient/class."""
    return {
        patid: {
            class_id: {"TP": 0, "FP": 0, "FN": 0} 
            for class_id in range(1, 5)
        } 
        for patid in patient_ids
    }

# -------------------------------------------------------------------------
# Event Processing & Matching
# -------------------------------------------------------------------------

def process_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans target events: removes 'TUG' and maps string types to integer class IDs.
    """
    if targets is None or targets.empty:
        return pd.DataFrame(columns=['class_id', 'time_stamp_start', 'time_stamp_end'])
    
    targets = targets[targets["type"] != "TUG"].copy()
    
    class_map = {
        "preferred_walk": 1,
        "fast_walk": 2,
        "slow_walk": 3,
        "2min_walk": 4
    }
    targets["class_id"] = targets["type"].map(class_map)
    return targets

def compute_iou(pred_interval, true_interval):
    """Computes Intersection over Union (IoU) for two time intervals."""
    start_pred, end_pred = pred_interval
    start_true, end_true = true_interval

    intersection_start = max(start_pred, start_true)
    intersection_end = min(end_pred, end_true)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(start_pred, start_true)
    union_end = max(end_pred, end_true)
    union = union_end - union_start

    return intersection / union if union > 0 else 0.0

def evaluate_event_detection(
    detected_events: Union[List[dict], pd.DataFrame], 
    target_events: pd.DataFrame, 
    iou_threshold: float = 0.5
) -> dict:
    """
    Matches detected events to ground truth using Greedy IoU matching.
    Returns TP, FP, FN counts per class.
    """
    
    # Standardize input
    if isinstance(detected_events, list):
        detected_events = pd.DataFrame(detected_events)
        
    # --- Edge Case: No Detections ---
    if detected_events is None or detected_events.empty:
        metrics = {}
        unique_classes = target_events['class_id'].unique() if not target_events.empty else []
        for class_id in unique_classes:
            fn = len(target_events[target_events["class_id"] == class_id])
            metrics[class_id] = {"TP": 0, "FP": 0, "FN": fn}
        return metrics

    # --- Edge Case: No Targets ---
    if target_events is None or target_events.empty:
        metrics = {}
        for class_id in detected_events['class_id'].unique():
            fp = len(detected_events[detected_events["class_id"] == class_id])
            metrics[class_id] = {"TP": 0, "FP": fp, "FN": 0}
        return metrics    

    # --- Matching Logic ---
    detected_events = detected_events.copy()
    target_events = target_events.copy()
    
    detected_events["is_matched"] = False
    target_events["is_matched"] = False
    
    for idx_gt, gt_row in target_events.iterrows():
        class_id = gt_row["class_id"]
        gt_start, gt_end = gt_row["time_stamp_start"], gt_row["time_stamp_end"]
        
        # Candidate detections: same class, not yet matched
        candidates = detected_events[
            (detected_events["class_id"] == class_id) & 
            (detected_events["is_matched"] == False)
        ]
        
        for idx_det, det_row in candidates.iterrows():
            det_start, det_end = det_row["timestamp_start_ns"], det_row["timestamp_end_ns"]
            
            if compute_iou((gt_start, gt_end), (det_start, det_end)) > iou_threshold:
                detected_events.at[idx_det, "is_matched"] = True
                target_events.at[idx_gt, "is_matched"] = True
                break 
    
    # --- Compile Metrics ---
    metrics = {}
    all_classes = sorted(target_events['class_id'].unique()) 
    
    for class_id in all_classes:
        tp = len(detected_events[(detected_events["class_id"] == class_id) & (detected_events["is_matched"] == True)])
        fp = len(detected_events[(detected_events["class_id"] == class_id) & (detected_events["is_matched"] == False)])
        fn = len(target_events[(target_events["class_id"] == class_id) & (target_events["is_matched"] == False)])
        
        metrics[class_id] = {"TP": tp, "FP": fp, "FN": fn}
        
    return metrics

# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------

def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    """Calculates Precision, Recall, and F1-Score from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}

def aggregate_and_report(metrics_data: Dict, prefix: str) -> Dict:
    """
    Aggregates metrics across all patients and calculates final scores per class.
    """
    aggregated = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for pat_id, classes_data in metrics_data.items():
        for class_id, counts in classes_data.items():
            aggregated[class_id]["TP"] += counts["TP"]
            aggregated[class_id]["FP"] += counts["FP"]
            aggregated[class_id]["FN"] += counts["FN"]

    final_results = {}
    for class_id, counts in sorted(aggregated.items()):
        metrics = calculate_metrics(counts["TP"], counts["FP"], counts["FN"])
        print(f"{prefix} Class {class_id} Metrics: {metrics}")
        final_results[class_id] = metrics

    return final_results