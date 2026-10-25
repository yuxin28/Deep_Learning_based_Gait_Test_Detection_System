#!/usr/bin/env python3
"""
Evaluate the final pipeline on an external home-recording dataset with 4x10MWT batteries.

Protocol of this dataset: three batteries per day, each battery is a 4x10MWT at
preferred, fast and slow speed (one annotation row per speed). There is no 2MWT.

Expected layout (one session per date folder, at any depth below the root):

    <dataset-root>/.../YYYY-MM-DD/<recording>.h5
    <dataset-root>/.../YYYY-MM-DD/<targets>.csv     # columns: type, time_stamp_start, time_stamp_end (ns)

The pipeline runs unchanged (final configuration: TCN -> U-Net+BiGRU ->
competition-aware decoding -> island300), so it can still output 2MWT events.
In the main results those are counted as false positives and reported separately.

Metrics (main results):
  1. Event level: same-class greedy matching with IoU > 0.5 (as in the thesis),
     per-class and macro P/R/F1 over preferred/fast/slow, plus 2MWT false alarms.
  2. Localization: IoU and absolute start/end error of matched events.
  3. Confusion: for every GT event, the class of the best-overlapping prediction.
  4. Battery recovery: on complete GT batteries (each speed exactly once), share of
     subtests recovered with same class and IoU >= 0.3; >= 2/3 and 3/3 success rates.
  5. Independent battery detection: predictions and GT are grouped separately
     (gap <= 600 s), then matched one-to-one by positive overlap (as in the thesis).

Diagnostics:
  A. Relaxed event matching (IoU >= 0.3, IoU >= 0.1, any overlap) and same-class
     coverage of each GT event, to separate boundary errors from real misses.
  B. Control without predicted 2MWT events (all metrics recomputed).
  C. Per-day pipeline stages: recording length, NaN share, candidate windows,
     Stage 1 positives, forwarded windows, events, and the stage at which each
     GT event was lost. Stage 1 window-level P/R/F1 before and after the continuity
     rule (window positive when its central 30 s overlaps a GT event), and the
     size of the continuity groups.
  D. Stage 2 probabilities per GT event (smoothed, as seen by the decoder): highest
     probability of the event's own class, and which class dominates the interval.
  E. Post-processing variants recomputed from cached Stage 2 probabilities
     (island filter off, lower ON/OFF thresholds), without running the models again.
  F. Class-agnostic detection (did any event, of any class including 2MWT, occur at
     the test?): event level by any temporal overlap, and time level by the overlap
     of the union of GT intervals and the union of predicted intervals. Reported for
     the main result, the no-2MWT control and every post-processing variant.
  G. Whole battery as the target for every post-processing variant: 600 s grouping,
     one-to-one matching at any overlap, IoU >= 0.3 and IoU >= 0.5 (all and complete
     GT batteries), battery IoU and boundary errors, time level on battery spans, and
     the predicted batteries without a GT match (candidates for missing annotations).

Usage:
    python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root /path/to/dataset
    # ablation: forward only Stage 1 positive windows (continuity rule off):
    python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root ... --no-continuity --output-dir output/<new dir>
    # re-evaluate saved predictions without running the models again:
    python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root ... --output-dir <previous run> --reuse-predictions
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.gait_detection import detect_gait_test
from pipeline.postprocessing import (
    CLASS_ON_THRESHOLDS,
    DEFAULT_ISLAND_NEIGHBORHOOD_SEC,
    FS,
    SMOOTH_S,
    reconstruct_continuous_sequence,
    run_postprocessing,
    smooth_probabilities,
)
from pipeline.preprocessing import SENSOR_COLS
from pipeline.continuity_rule import DEFAULT_CONTINUITY_RULE_CONFIG
from pipeline.end2end_metrics.utils import compute_iou, evaluate_event_detection, process_targets
from pipeline.end2end_metrics.battery_detection_10min import (
    BATTERY_GROUP_GAP_SEC,
    group_events_into_batteries,
    match_batteries,
    normalize_session_events,
    summarize_match_result,
)

# ----------------------------------------------------------------------------
# Change this to your dataset location (or pass --dataset-root).
# ----------------------------------------------------------------------------
DATASET_ROOT = Path("/path/to/4x10mwt_dataset")

TARGET_CLASSES = [1, 2, 3]
CLASS_NAMES = {1: "preferred", 2: "fast", 3: "slow", 4: "2MWT"}
EVENT_IOU_THRESHOLD = 0.5
RECOVERY_IOU_THRESHOLD = 0.3
# (label, threshold, strict): strict means IoU > threshold, otherwise IoU >= threshold
RELAXED_MATCHING = [
    ("iou_gt_0.5", 0.5, True),
    ("iou_ge_0.3", 0.3, False),
    ("iou_ge_0.1", 0.1, False),
    ("any_overlap", 0.0, True),
]
WINDOW_SEC = 90.0  # pipeline default window length
COMPLETE_COUNTS = Counter({1: 1, 2: 1, 3: 1})
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# With a 0 s gap every window is its own continuity group, so only Stage 1
# positive windows are forwarded to Stage 2 (continuity rule effectively off).
NO_CONTINUITY_CONFIG = {"gaps_duration_ns": 0}

# Post-processing variants (overrides of the final config); 2MWT thresholds stay unchanged.
POSTPROCESSING_VARIANTS = {
    "final": None,
    "no_island": {"island_filter": {"enabled": False}},
    "thr_0.30": {"class_on_thresholds": {1: 0.30, 2: 0.30, 3: 0.30}, "class_off_thresholds": {1: 0.21, 2: 0.21, 3: 0.21}},
    "thr_0.20": {"class_on_thresholds": {1: 0.20, 2: 0.20, 3: 0.20}, "class_off_thresholds": {1: 0.14, 2: 0.14, 3: 0.14}},
    "no_island+thr_0.30": {"island_filter": {"enabled": False},
                           "class_on_thresholds": {1: 0.30, 2: 0.30, 3: 0.30}, "class_off_thresholds": {1: 0.21, 2: 0.21, 3: 0.21}},
}


# ----------------------------------------------------------------------------
# Dataset discovery, loading and inference
# ----------------------------------------------------------------------------
def discover_sessions(dataset_root: Path) -> list[dict]:
    sessions = []
    for folder in sorted(p for p in dataset_root.rglob("*") if p.is_dir() and DATE_DIR.match(p.name)):
        rel = folder.relative_to(dataset_root)
        sessions.append({
            "session_id": "__".join(rel.parts),
            "subject": "__".join(rel.parts[:-1]) or "all",
            "date": folder.name,
            "h5": [str(p) for p in sorted(folder.glob("*.h5"))],
            "csv": [str(p) for p in sorted(folder.glob("*.csv"))],
        })
    return sessions


def load_targets(csv_path: str) -> pd.DataFrame:
    targets = process_targets(pd.read_csv(csv_path))
    unknown = sorted(set(targets.loc[targets["class_id"].isna(), "type"].astype(str))) if not targets.empty else []
    targets = targets.dropna(subset=["class_id"]).copy()
    targets["class_id"] = targets["class_id"].astype(int)
    targets["time_stamp_start"] = targets["time_stamp_start"].astype("int64")
    targets["time_stamp_end"] = targets["time_stamp_end"].astype("int64")
    targets = targets[targets["time_stamp_end"] > targets["time_stamp_start"]]
    targets = targets.sort_values("time_stamp_start").reset_index(drop=True)
    targets.attrs["unknown_types"] = unknown
    return targets


def _int_list(values) -> list[int]:
    return [] if values is None else [int(v) for v in np.asarray(values).ravel()]


def run_pipeline(h5_path: str, continuity_config: Optional[dict] = None) -> tuple[list[dict], str, dict, Optional[dict]]:
    signal = pd.read_hdf(h5_path)
    output = detect_gait_test(signal_path=h5_path, signal=signal, plot_signal=False, return_debug=True,
                              continuity_config=continuity_config)
    debug = output.get("debug", {})
    status = debug.get("status", "unknown")
    events = [
        {
            "class_id": int(e["class_id"]),
            "timestamp_start_ns": int(e["timestamp_start_ns"]),
            "timestamp_end_ns": int(e["timestamp_end_ns"]),
            "score": float(e.get("score", 0.0)),
        }
        for e in output.get("events", [])
    ]

    candidates = _int_list(debug.get("candidate_window_start_times"))
    probs = debug.get("stage1_probs")
    probs = [] if probs is None else np.asarray(probs).ravel().tolist()
    forwarded_mask = debug.get("forwarded_mask")
    sensor_cols = [c for c in SENSOR_COLS if c in signal.columns]
    diagnostics = {
        "recording_hours": (int(signal.index[-1]) - int(signal.index[0])) / 3.6e12 if len(signal.index) else 0.0,
        "nan_row_fraction": float(signal[sensor_cols].isna().any(axis=1).mean()) if sensor_cols else None,
        "missing_sensor_columns": [c for c in SENSOR_COLS if c not in signal.columns],
        "candidate_windows": candidates,
        "stage1_positive_windows": [t for t, p in zip(candidates, probs) if p >= 0.5],
        "forwarded_windows": [] if forwarded_mask is None else
                             [t for t, keep in zip(candidates, np.asarray(forwarded_mask).ravel()) if keep],
    }
    inputs = debug.get("postprocessing_inputs") or {}
    stage2 = None
    if inputs.get("window_probs") is not None and len(inputs["window_probs"]):
        stage2 = {"window_probs": np.asarray(inputs["window_probs"], dtype=np.float32),
                  "window_start_times": np.asarray(inputs["window_start_times"], dtype=np.int64)}
    return events, status, diagnostics, stage2


def load_sessions(sessions: list[dict], output_dir: Path, reuse_predictions: bool,
                  continuity_config: Optional[dict] = None) -> tuple[list[dict], list[dict]]:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir = output_dir / "stage2_probs"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    loaded, failures = [], []
    for i, session in enumerate(sessions, 1):
        sid = session["session_id"]
        print(f"[{i}/{len(sessions)}] {sid}", flush=True)
        if len(session["h5"]) != 1 or len(session["csv"]) != 1:
            failures.append({"session_id": sid, "error": f"expected 1 h5 and 1 csv, found {len(session['h5'])} h5 and {len(session['csv'])} csv"})
            continue
        try:
            targets = load_targets(session["csv"][0])
            cache = pred_dir / f"{sid}.json"
            stage2_path = stage2_dir / f"{sid}.npz"
            if reuse_predictions and cache.exists():
                cached = json.loads(cache.read_text())
                predictions, status, diagnostics = cached["events"], cached["status"], cached.get("diagnostics")
            else:
                predictions, status, diagnostics, stage2 = run_pipeline(session["h5"][0], continuity_config)
                cache.write_text(json.dumps({"session_id": sid, "status": status, "events": predictions,
                                             "diagnostics": diagnostics}))
                if stage2 is not None:
                    np.savez_compressed(stage2_path, **stage2)
                elif stage2_path.exists():
                    stage2_path.unlink()
        except Exception as exc:
            failures.append({"session_id": sid, "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            continue
        if targets.attrs.get("unknown_types"):
            print(f"    ignored unknown target types: {targets.attrs['unknown_types']}")
        loaded.append({**session, "targets": targets, "predictions": predictions,
                       "status": status, "diagnostics": diagnostics,
                       "stage2_path": str(stage2_path) if stage2_path.exists() else None})
    return loaded, failures


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def match_events(predictions: list[dict], targets: pd.DataFrame, threshold: float = EVENT_IOU_THRESHOLD,
                 strict: bool = True) -> list[tuple[int, int, float]]:
    """Same greedy rule as utils.evaluate_event_detection; returns (gt_idx, pred_idx, iou)."""
    used = set()
    pairs = []
    for gt_idx, gt in targets.iterrows():
        for pred_idx, pred in enumerate(predictions):
            if pred_idx in used or pred["class_id"] != gt["class_id"]:
                continue
            iou = compute_iou((gt["time_stamp_start"], gt["time_stamp_end"]),
                              (pred["timestamp_start_ns"], pred["timestamp_end_ns"]))
            if (iou > threshold) if strict else (iou >= threshold):
                used.add(pred_idx)
                pairs.append((gt_idx, pred_idx, iou))
                break
    return pairs


def overlap_ns(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def covered_fraction(start: int, end: int, intervals: list[tuple[int, int]]) -> float:
    """Share of [start, end] covered by the union of intervals."""
    clipped = sorted((max(s, start), min(e, end)) for s, e in intervals if e > start and s < end)
    covered, cursor = 0, start
    for s, e in clipped:
        s = max(s, cursor)
        if e > s:
            covered += e - s
            cursor = e
    return covered / (end - start)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Overlap length of two interval unions (inputs merged first)."""
    a, b = merge_intervals(a), merge_intervals(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def class_agnostic_metrics(data: list[dict]) -> dict:
    """Detection regardless of class: any predicted event (incl. 2MWT) overlapping a GT event counts."""
    totals, per_day = Counter(), []
    for s in data:
        gt = [(int(r["time_stamp_start"]), int(r["time_stamp_end"])) for _, r in s["targets"].iterrows()]
        pred = [(p["timestamp_start_ns"], p["timestamp_end_ns"]) for p in s["predictions"]]
        row = {
            "session_id": s["session_id"], "date": s["date"],
            "gt_events": len(gt),
            "gt_detected": sum(any(overlap_ns(a, b, x, y) > 0 for x, y in pred) for a, b in gt),
            "gt_covered_ge_50": sum(intersection_length([(a, b)], pred) >= 0.5 * (b - a) for a, b in gt),
            "predicted_events": len(pred),
            "predicted_overlapping_gt": sum(any(overlap_ns(x, y, a, b) > 0 for a, b in gt) for x, y in pred),
            "gt_time_ns": sum(e - s_ for s_, e in merge_intervals(gt)),
            "predicted_time_ns": sum(e - s_ for s_, e in merge_intervals(pred)),
            "overlap_time_ns": intersection_length(gt, pred),
        }
        per_day.append(row)
        totals.update({k: v for k, v in row.items() if k not in ("session_id", "date")})

    def ratio(a, b):
        return totals[a] / totals[b] if totals[b] else None

    def f1(p, r):
        return 2 * p * r / (p + r) if p is not None and r is not None and p + r else 0.0

    recall, precision = ratio("gt_detected", "gt_events"), ratio("predicted_overlapping_gt", "predicted_events")
    t_recall, t_precision = ratio("overlap_time_ns", "gt_time_ns"), ratio("overlap_time_ns", "predicted_time_ns")
    return {
        "event_level": {"matching": "any temporal overlap, any predicted class",
                        "gt_events": totals["gt_events"], "gt_detected": totals["gt_detected"],
                        "predicted_events": totals["predicted_events"],
                        "predicted_overlapping_gt": totals["predicted_overlapping_gt"],
                        "recall": recall, "precision": precision, "f1": f1(precision, recall),
                        "share_gt_covered_ge_50": ratio("gt_covered_ge_50", "gt_events")},
        "time_level": {"gt_time_min": totals["gt_time_ns"] / 6e10, "predicted_time_min": totals["predicted_time_ns"] / 6e10,
                       "overlap_time_min": totals["overlap_time_ns"] / 6e10,
                       "recall": t_recall, "precision": t_precision, "f1": f1(t_precision, t_recall)},
        "per_day": per_day,
    }


def best_overlap_class(gt, predictions: list[dict]) -> str:
    best, best_ov = None, 0
    for pred in predictions:
        ov = overlap_ns(gt["time_stamp_start"], gt["time_stamp_end"], pred["timestamp_start_ns"], pred["timestamp_end_ns"])
        if ov > best_ov:
            best, best_ov = pred, ov
    return CLASS_NAMES[best["class_id"]] if best else "none"


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


def mean_sd(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "sd": None, "median": None, "n": 0}
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0, "median": median(values), "n": len(values)}


def counts_from_pairs(data: list[dict], threshold: float, strict: bool) -> dict:
    counts = {c: Counter() for c in [1, 2, 3, 4]}
    for s in data:
        pairs = match_events(s["predictions"], s["targets"], threshold, strict)
        matched_gt = {g for g, _, _ in pairs}
        matched_pred = {p for _, p, _ in pairs}
        for idx, pred in enumerate(s["predictions"]):
            counts[pred["class_id"]]["TP" if idx in matched_pred else "FP"] += 1
        for idx, gt in s["targets"].iterrows():
            if idx not in matched_gt:
                counts[int(gt["class_id"])]["FN"] += 1
    return counts


def event_summary(counts: dict) -> dict:
    per_class = {CLASS_NAMES[c]: prf(counts[c]["TP"], counts[c]["FP"], counts[c]["FN"]) for c in TARGET_CLASSES}
    return {
        "per_class": per_class,
        **{f"macro_{k}": mean(per_class[CLASS_NAMES[c]][k] for c in TARGET_CLASSES) for k in ("precision", "recall", "f1")},
    }


def battery_recovery(gt_battery: dict, predictions: list[dict]) -> int:
    """Number of GT subtests in the battery matched by a same-class prediction with IoU >= 0.3."""
    used = set()
    detected = 0
    for member in gt_battery["member_events"]:
        best_idx, best_iou = None, RECOVERY_IOU_THRESHOLD
        for idx, pred in enumerate(predictions):
            if idx in used or pred["class_id"] != member["class_id"]:
                continue
            iou = compute_iou((member["timestamp_start_ns"], member["timestamp_end_ns"]),
                              (pred["timestamp_start_ns"], pred["timestamp_end_ns"]))
            if iou >= best_iou:
                best_idx, best_iou = idx, iou
        if best_idx is not None:
            used.add(best_idx)
            detected += 1
    return detected


# ----------------------------------------------------------------------------
# Metrics for one variant (main, or without predicted 2MWT)
# ----------------------------------------------------------------------------
def compute_metrics(data: list[dict], tables: Optional[dict] = None) -> dict:
    counts = {c: Counter() for c in [1, 2, 3, 4]}
    subject_counts = defaultdict(lambda: {c: Counter() for c in [1, 2, 3, 4]})
    matched_rows, confusion, false_2mwt = [], Counter(), []
    coverage = defaultdict(list)
    split_events = 0
    all_pred_batteries, all_gt_batteries = [], []
    recovery = Counter()

    for s in data:
        sid, targets, predictions = s["session_id"], s["targets"], s["predictions"]
        per_class = evaluate_event_detection(predictions, targets, iou_threshold=EVENT_IOU_THRESHOLD)
        pairs = match_events(predictions, targets)
        assert sum(v.get("TP", 0) for v in per_class.values()) == len(pairs), "event matcher mismatch"
        for cid in [1, 2, 3, 4]:
            c = per_class.get(cid, {"TP": 0, "FP": 0, "FN": 0})
            counts[cid].update(c)
            subject_counts[s["subject"]][cid].update(c)

        for gt_idx, pred_idx, iou in pairs:
            gt, pred = targets.loc[gt_idx], predictions[pred_idx]
            matched_rows.append({
                "session_id": sid, "class_id": int(gt["class_id"]), "iou": iou,
                "start_error_sec": (pred["timestamp_start_ns"] - gt["time_stamp_start"]) / 1e9,
                "end_error_sec": (pred["timestamp_end_ns"] - gt["time_stamp_end"]) / 1e9,
            })
        for _, gt in targets.iterrows():
            cid = int(gt["class_id"])
            confusion[(CLASS_NAMES[cid], best_overlap_class(gt, predictions))] += 1
            same = [(p["timestamp_start_ns"], p["timestamp_end_ns"]) for p in predictions if p["class_id"] == cid]
            frac = covered_fraction(int(gt["time_stamp_start"]), int(gt["time_stamp_end"]), same)
            coverage[CLASS_NAMES[cid]].append(frac)
            split_events += sum(overlap_ns(gt["time_stamp_start"], gt["time_stamp_end"], a, b) > 0 for a, b in same) >= 2
        for pred in predictions:
            if pred["class_id"] == 4:
                overlapped = [CLASS_NAMES[int(g["class_id"])] for _, g in targets.iterrows()
                              if overlap_ns(g["time_stamp_start"], g["time_stamp_end"], pred["timestamp_start_ns"], pred["timestamp_end_ns"]) > 0]
                false_2mwt.append({"session_id": sid, "start_ns": pred["timestamp_start_ns"],
                                   "duration_sec": (pred["timestamp_end_ns"] - pred["timestamp_start_ns"]) / 1e9,
                                   "overlaps_gt": "+".join(overlapped) or "none"})

        gt_events = [{"class_id": int(r["class_id"]), "timestamp_start_ns": int(r["time_stamp_start"]),
                      "timestamp_end_ns": int(r["time_stamp_end"])} for _, r in targets.iterrows()]
        pred_b = group_events_into_batteries(normalize_session_events("external", sid, predictions, "prediction"), "prediction")
        gt_b = group_events_into_batteries(normalize_session_events("external", sid, gt_events, "ground_truth"), "ground_truth")
        for b in gt_b:
            b["is_complete_standard_battery"] = Counter(b["class_id_sequence"]) == COMPLETE_COUNTS
            if b["is_complete_standard_battery"]:
                found = battery_recovery(b, predictions)
                recovery["batteries"] += 1
                recovery["subtests_detected"] += found
                recovery["structured_success"] += found >= 2
                recovery["strict_success"] += found == 3
        all_pred_batteries += pred_b
        all_gt_batteries += gt_b
        if tables is not None:
            tables["sessions"].append({
                "session_id": sid, "subject": s["subject"], "date": s["date"], "status": s["status"],
                "gt_events": len(targets), "predicted_events": len(predictions), "matched_events": len(pairs),
                "predicted_2mwt": sum(p["class_id"] == 4 for p in predictions),
                "gt_batteries": len(gt_b), "gt_complete_batteries": sum(b["is_complete_standard_battery"] for b in gt_b),
                "predicted_batteries": len(pred_b),
            })

    event = event_summary(counts)
    tp_all = sum(counts[c]["TP"] for c in TARGET_CLASSES)
    fp_all = sum(counts[c]["FP"] for c in [1, 2, 3, 4])
    fn_all = sum(counts[c]["FN"] for c in TARGET_CLASSES)
    subject_macro = {subject: mean(prf(sc[c]["TP"], sc[c]["FP"], sc[c]["FN"])["f1"] for c in TARGET_CLASSES)
                     for subject, sc in subject_counts.items()}
    event.update({
        "matching": f"same class, IoU > {EVENT_IOU_THRESHOLD}, greedy one-to-one",
        "micro_including_2mwt_fp": prf(tp_all, fp_all, fn_all),
        "predicted_2mwt_false_alarms": {
            "count": len(false_2mwt),
            "total_duration_min": sum(r["duration_sec"] for r in false_2mwt) / 60,
            "overlapping_gt_class": dict(Counter(r["overlaps_gt"] for r in false_2mwt)),
        },
        "subject_wise_macro_f1": {"per_subject": subject_macro, **mean_sd(list(subject_macro.values()))},
    })

    localization = {}
    for name, cids in [("overall", TARGET_CLASSES)] + [(CLASS_NAMES[c], [c]) for c in TARGET_CLASSES]:
        rows = [r for r in matched_rows if r["class_id"] in cids]
        localization[name] = {
            "iou": mean_sd([r["iou"] for r in rows]),
            "abs_start_error_sec": mean_sd([abs(r["start_error_sec"]) for r in rows]),
            "abs_end_error_sec": mean_sd([abs(r["end_error_sec"]) for r in rows]),
        }

    relaxed = {}
    for label, threshold, strict in RELAXED_MATCHING:
        relaxed[label] = event_summary(counts_from_pairs(data, threshold, strict))
    all_cov = [v for vals in coverage.values() for v in vals]
    coverage_summary = {
        name: {"mean": mean(vals) if vals else None,
               "share_ge_0.5": sum(v >= 0.5 for v in vals) / len(vals) if vals else None,
               "share_ge_0.8": sum(v >= 0.8 for v in vals) / len(vals) if vals else None,
               "share_zero": sum(v == 0 for v in vals) / len(vals) if vals else None,
               "n": len(vals)}
        for name, vals in [("overall", all_cov)] + [(CLASS_NAMES[c], coverage[CLASS_NAMES[c]]) for c in TARGET_CLASSES]
    }

    battery_detection = {}
    for name, gt_subset in [("all_gt_batteries", all_gt_batteries),
                            ("complete_gt_batteries", [b for b in all_gt_batteries if b["is_complete_standard_battery"]])]:
        battery_detection[name] = summarize_match_result(match_batteries(all_pred_batteries, gt_subset))
        battery_detection[name]["num_gt_batteries"] = len(gt_subset)
    battery_detection["num_predicted_batteries"] = len(all_pred_batteries)

    if tables is not None:
        tables["matched"] += matched_rows
        tables["false_2mwt"] += false_2mwt

    n_bat = recovery["batteries"]
    return {
        "class_agnostic_detection": class_agnostic_metrics(data),
        "event_level": event,
        "event_level_relaxed_matching": relaxed,
        "same_class_coverage_of_gt_events": {**coverage_summary,
                                             "gt_events_split_into_2plus_same_class_predictions": split_events},
        "localization_matched_events": localization,
        "confusion_gt_vs_best_overlapping_prediction": {f"{g} -> {p}": n for (g, p), n in sorted(confusion.items())},
        "battery_recovery_complete_gt": {
            "matching": f"same class, IoU >= {RECOVERY_IOU_THRESHOLD}",
            "complete_gt_batteries": n_bat,
            "battery_score": recovery["subtests_detected"] / (3 * n_bat) if n_bat else None,
            "structured_success_rate_ge_2_of_3": recovery["structured_success"] / n_bat if n_bat else None,
            "strict_success_rate_3_of_3": recovery["strict_success"] / n_bat if n_bat else None,
        },
        "independent_battery_detection": {
            "grouping": f"adjacent events with gap <= {BATTERY_GROUP_GAP_SEC:.0f} s, predictions and GT grouped separately",
            "matching": "positive temporal overlap, one-to-one, highest battery IoU first",
            **battery_detection,
        },
    }


# ----------------------------------------------------------------------------
# Stage diagnostics
# ----------------------------------------------------------------------------
def stage_diagnostics(data: list[dict]) -> dict:
    window_ns = int(WINDOW_SEC * 1e9)
    days, lost_total, per_gt = [], Counter(), []
    missing = [s["session_id"] for s in data if not s.get("diagnostics")]
    for s in data:
        diag = s.get("diagnostics")
        if not diag:
            continue
        stages = {name: [(t, t + window_ns) for t in diag[key]] for name, key in
                  [("candidate", "candidate_windows"), ("stage1", "stage1_positive_windows"), ("forwarded", "forwarded_windows")]}
        preds = [(p["timestamp_start_ns"], p["timestamp_end_ns"]) for p in s["predictions"]]
        matched = {g for g, _, _ in match_events(s["predictions"], s["targets"])}
        lost = Counter()
        for idx, gt in s["targets"].iterrows():
            a, b = int(gt["time_stamp_start"]), int(gt["time_stamp_end"])
            reached = {name: any(overlap_ns(a, b, x, y) > 0 for x, y in iv) for name, iv in stages.items()}
            if not reached["candidate"]:
                stage = "no_candidate_window"
            elif not reached["stage1"]:
                stage = "stage1_negative"
            elif not reached["forwarded"]:
                stage = "not_forwarded"
            elif not any(overlap_ns(a, b, x, y) > 0 for x, y in preds):
                stage = "no_event_after_decoding"
            elif idx not in matched:
                stage = "event_but_not_matched"
            else:
                stage = "matched"
            lost[stage] += 1
            per_gt.append({"session_id": s["session_id"], "gt_index": int(idx), "stage": stage})
        lost_total.update(lost)
        days.append({
            "session_id": s["session_id"], "date": s["date"], "status": s["status"],
            "recording_hours": diag["recording_hours"], "nan_row_fraction": diag["nan_row_fraction"],
            "missing_sensor_columns": diag["missing_sensor_columns"],
            "candidate_windows": len(diag["candidate_windows"]),
            "stage1_positive_windows": len(diag["stage1_positive_windows"]),
            "forwarded_windows": len(diag["forwarded_windows"]),
            "predicted_events": len(s["predictions"]), "gt_events": len(s["targets"]),
            **{f"gt_{k}": lost.get(k, 0) for k in ["no_candidate_window", "stage1_negative", "not_forwarded",
                                                    "no_event_after_decoding", "event_but_not_matched", "matched"]},
        })
    return {
        "note": "Each GT event is assigned to the first stage at which no window/event overlaps it.",
        "sessions_without_diagnostics": missing,
        "gt_event_funnel": dict(lost_total),
        "per_day": days,
        "per_gt": per_gt,
    }


def stage1_window_metrics(data: list[dict]) -> dict:
    """Window-level Stage 1 metrics over candidate windows, before and after the continuity rule.

    A window is labelled positive when its central third (30-60 s of a 90 s window, as in
    training) overlaps any GT event.
    """
    window_ns = int(WINDOW_SEC * 1e9)
    center = (window_ns // 3, 2 * window_ns // 3)
    counts = {"stage1_raw": Counter(), "after_continuity": Counter()}
    for s in data:
        diag = s.get("diagnostics")
        if not diag:
            continue
        gt = [(int(g["time_stamp_start"]), int(g["time_stamp_end"])) for _, g in s["targets"].iterrows()]
        predicted = {"stage1_raw": set(diag["stage1_positive_windows"]),
                     "after_continuity": set(diag["forwarded_windows"])}
        for t in diag["candidate_windows"]:
            label = any(overlap_ns(t + center[0], t + center[1], a, b) > 0 for a, b in gt)
            for name, positive in predicted.items():
                key = ("TP" if label else "FP") if t in positive else ("FN" if label else "TN")
                counts[name][key] += 1
    return {
        "label": "positive if the central 30 s of the 90 s window overlaps any GT event",
        "windows": sum(counts["stage1_raw"].values()),
        **{name: {**prf(c["TP"], c["FP"], c["FN"]), "TN": c["TN"]} for name, c in counts.items()},
    }


def continuity_group_summary(data: list[dict], gap_ns: int) -> dict:
    """Continuity groups over candidate windows (new group when adjacent starts are > gap_ns apart)."""
    window_ns = int(WINDOW_SEC * 1e9)
    groups = []
    for s in data:
        diag = s.get("diagnostics")
        if not diag or not diag["candidate_windows"]:
            continue
        positive = set(diag["stage1_positive_windows"])
        session_groups = [[diag["candidate_windows"][0]]]
        for t in diag["candidate_windows"][1:]:
            if t - session_groups[-1][-1] > gap_ns:
                session_groups.append([t])
            else:
                session_groups[-1].append(t)
        groups += [{"windows": g, "positive": sum(t in positive for t in g)} for g in session_groups]
    forwarded = [g for g in groups if g["positive"] > 0]
    sizes = [len(g["windows"]) for g in forwarded]
    largest = max(forwarded, key=lambda g: len(g["windows"]), default=None)
    return {
        "gap_sec": gap_ns / 1e9,
        "groups": len(groups),
        "forwarded_groups": len(forwarded),
        "forwarded_windows": sum(sizes),
        "stage1_positive_windows_in_forwarded_groups": sum(g["positive"] for g in forwarded),
        "forwarded_group_size": mean_sd([float(n) for n in sizes]),
        "largest_group_windows": len(largest["windows"]) if largest else 0,
        "largest_group_hours": ((largest["windows"][-1] - largest["windows"][0] + window_ns) / 3.6e12) if largest else 0.0,
    }


def battery_as_target(data: list[dict]) -> dict:
    """Whole batteries as the target: 600 s grouping, one-to-one matching at several overlap criteria."""
    pred_b, gt_b = [], []
    for s in data:
        gt_events = [{"class_id": int(r["class_id"]), "timestamp_start_ns": int(r["time_stamp_start"]),
                      "timestamp_end_ns": int(r["time_stamp_end"])} for _, r in s["targets"].iterrows()]
        pred_b += group_events_into_batteries(normalize_session_events("external", s["session_id"], s["predictions"], "prediction"), "prediction")
        gt_b += group_events_into_batteries(normalize_session_events("external", s["session_id"], gt_events, "ground_truth"), "ground_truth")
    for b in gt_b:
        b["is_complete_standard_battery"] = Counter(b["class_id_sequence"]) == COMPLETE_COUNTS

    matching = {}
    for subset, gts in [("all", gt_b), ("complete", [b for b in gt_b if b["is_complete_standard_battery"]])]:
        for label, min_iou in [("any_overlap", None), ("iou_ge_0.3", 0.3), ("iou_ge_0.5", 0.5)]:
            m = summarize_match_result(match_batteries(pred_b, gts, min_iou=min_iou))
            matching[f"{subset}/{label}"] = {"gt_batteries": len(gts), **m}

    sessions = {s["session_id"] for s in data}
    spans = {sid: ([(b["battery_start_ns"], b["battery_end_ns"]) for b in gt_b if b["session_id"] == sid],
                   [(b["battery_start_ns"], b["battery_end_ns"]) for b in pred_b if b["session_id"] == sid]) for sid in sessions}
    gt_time = sum(e - s_ for g, _ in spans.values() for s_, e in merge_intervals(g))
    pred_time = sum(e - s_ for _, p in spans.values() for s_, e in merge_intervals(p))
    overlap = sum(intersection_length(g, p) for g, p in spans.values())
    t_recall, t_precision = (overlap / gt_time if gt_time else None), (overlap / pred_time if pred_time else None)

    unmatched = match_batteries(pred_b, gt_b)["unmatched_predictions"]
    return {
        "grouping": f"adjacent events with gap <= {BATTERY_GROUP_GAP_SEC:.0f} s, predictions and GT grouped separately",
        "matching": matching,
        "time_level_battery_spans": {
            "recall": t_recall, "precision": t_precision,
            "f1": 2 * t_precision * t_recall / (t_precision + t_recall) if t_precision and t_recall else 0.0,
        },
        "unmatched_predicted_batteries": [
            {"session_id": b["session_id"], "start_ns": b["battery_start_ns"],
             "duration_sec": b["battery_duration_sec"], "class_sequence": " -> ".join(b["class_sequence"])}
            for b in unmatched],
    }


# ----------------------------------------------------------------------------
# Stage 2 probabilities and post-processing variants
# ----------------------------------------------------------------------------
def load_stage2(session: dict) -> Optional[dict]:
    if not session.get("stage2_path"):
        return None
    with np.load(session["stage2_path"]) as npz:
        return {"window_probs": npz["window_probs"], "window_start_times": npz["window_start_times"]}


def stage2_probability_analysis(data: list[dict], per_gt: list[dict]) -> dict:
    """Per GT event: smoothed Stage 2 probabilities inside the event, as seen by the decoder."""
    stage_of = {(r["session_id"], r["gt_index"]): r["stage"] for r in per_gt}
    rows, missing = [], []
    for s in data:
        stage2 = load_stage2(s)
        if stage2 is None:
            missing.append(s["session_id"])
        else:
            probs, min_ts, _ = reconstruct_continuous_sequence(stage2["window_probs"], stage2["window_start_times"])
            scores = smooth_probabilities(probs, SMOOTH_S)
        for idx, gt in s["targets"].iterrows():
            cid = int(gt["class_id"])
            row = {"session_id": s["session_id"], "date": s["date"], "class": CLASS_NAMES[cid],
                   "stage": stage_of.get((s["session_id"], int(idx)), "unknown")}
            if stage2 is None:
                rows.append({**row, "own_class_max": 0.0, "own_class_mean": 0.0, "dominant_class": "no Stage 2 input",
                             "dominant_mean": None, "background_mean": None, "stage2_frames": 0})
                continue
            a = max(0, int(round((int(gt["time_stamp_start"]) - min_ts) / 1e9 * FS)))
            b = min(len(scores), int(round((int(gt["time_stamp_end"]) - min_ts) / 1e9 * FS)))
            seg = scores[a:b] if b > a else np.zeros((0, scores.shape[1]), dtype=np.float32)
            if len(seg) == 0:
                rows.append({**row, "own_class_max": 0.0, "own_class_mean": 0.0, "dominant_class": "outside Stage 2 input",
                             "dominant_mean": None, "background_mean": None, "stage2_frames": 0})
                continue
            means = seg.mean(axis=0)
            dominant = int(np.argmax(means))
            rows.append({**row, "own_class_max": float(seg[:, cid].max()), "own_class_mean": float(means[cid]),
                         "dominant_class": "background" if dominant == 0 else CLASS_NAMES[dominant],
                         "dominant_mean": float(means[dominant]), "background_mean": float(means[0]),
                         "stage2_frames": int(len(seg))})

    on = CLASS_ON_THRESHOLDS[1]
    summary = {}
    for stage in ["matched", "event_but_not_matched", "no_event_after_decoding", "all"]:
        sel = [r for r in rows if stage == "all" or r["stage"] == stage]
        if not sel:
            continue
        summary[stage] = {
            "n": len(sel),
            "own_class_max_median": float(np.median([r["own_class_max"] for r in sel])),
            f"share_own_max_ge_{on:.2f}": sum(r["own_class_max"] >= on for r in sel) / len(sel),
            "share_own_max_0.20_to_on": sum(0.20 <= r["own_class_max"] < on for r in sel) / len(sel),
            "share_own_max_lt_0.20": sum(r["own_class_max"] < 0.20 for r in sel) / len(sel),
            "dominant_class": dict(Counter(r["dominant_class"] for r in sel)),
        }
    return {"on_threshold": on, "sessions_without_stage2_cache": missing, "summary": summary, "per_gt": rows}


def postprocessing_variants(data: list[dict], names: list[str]) -> dict:
    """Recompute events from cached Stage 2 probabilities for each post-processing variant."""
    missing = [s["session_id"] for s in data if s.get("stage2_path") is None and s["predictions"]]
    if missing:
        return {"skipped": f"no Stage 2 cache for sessions with events: {missing}; rerun without --reuse-predictions"}
    results, reproduced = {}, None
    stage2_cache = {}
    for name in names:
        override = POSTPROCESSING_VARIANTS[name]
        print(f"  post-processing variant {name}", flush=True)
        variant_data = []
        for s in data:
            if s["session_id"] not in stage2_cache:
                stage2_cache[s["session_id"]] = load_stage2(s)
            stage2 = stage2_cache[s["session_id"]]
            events = [] if stage2 is None else run_postprocessing(
                stage2["window_probs"], stage2["window_start_times"], postprocessing_config=override)
            events = [{"class_id": int(e["class_id"]), "timestamp_start_ns": int(e["timestamp_start_ns"]),
                       "timestamp_end_ns": int(e["timestamp_end_ns"]), "score": float(e["score"])} for e in events]
            variant_data.append({**s, "predictions": events})
        if name == "final":
            same = sum([(e["class_id"], e["timestamp_start_ns"], e["timestamp_end_ns"]) for e in v["predictions"]]
                       == [(e["class_id"], e["timestamp_start_ns"], e["timestamp_end_ns"]) for e in s["predictions"]]
                       for v, s in zip(variant_data, data))
            reproduced = f"{same}/{len(data)} sessions identical to the pipeline output"
        m = compute_metrics(variant_data)
        no_event = sum(
            not any(overlap_ns(g["time_stamp_start"], g["time_stamp_end"], p["timestamp_start_ns"], p["timestamp_end_ns"]) > 0
                    for p in v["predictions"])
            for v in variant_data for _, g in v["targets"].iterrows())
        results[name] = {
            "override": override,
            "events": sum(len(v["predictions"]) for v in variant_data),
            "predicted_2mwt": m["event_level"]["predicted_2mwt_false_alarms"]["count"],
            "gt_events_without_any_overlapping_event": no_event,
            "event_macro_f1_iou_gt_0.5": m["event_level"]["macro_f1"],
            "event_macro_f1_any_overlap": m["event_level_relaxed_matching"]["any_overlap"]["macro_f1"],
            "event_per_class_f1": {k: v["f1"] for k, v in m["event_level"]["per_class"].items()},
            "battery_score": m["battery_recovery_complete_gt"]["battery_score"],
            "battery_detection_all_gt": {k: m["independent_battery_detection"]["all_gt_batteries"][k]
                                         for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
            "class_agnostic_detection": {level: {k: v for k, v in m["class_agnostic_detection"][level].items()}
                                         for level in ("event_level", "time_level")},
            "class_agnostic_per_day": m["class_agnostic_detection"]["per_day"],
            "battery_as_target": battery_as_target(variant_data),
        }
    return {"reproduction_check": reproduced, "variants": results}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_summary(result: dict, path: Path) -> str:
    main, ctrl, cov = result["main"], result["without_predicted_2mwt"], result["coverage"]
    ev, rec, bd = main["event_level"], main["battery_recovery_complete_gt"], main["independent_battery_detection"]
    lines = [
        "# 4x10MWT external dataset: evaluation summary", "",
        f"Pipeline: {result['pipeline']}", "",
        f"Sessions: {cov['sessions_evaluated']} evaluated, {cov['sessions_failed']} failed (of {cov['sessions_found']} found).", "",
        "## Event level (IoU > 0.5)", "",
        "| Class | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in ev["per_class"].items():
        lines.append(f"| {name} | {m['TP']} | {m['FP']} | {m['FN']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} |")
    fa, sw = ev["predicted_2mwt_false_alarms"], ev["subject_wise_macro_f1"]
    lines += [
        f"| **Macro (3 classes)** | | | | {fmt(ev['macro_precision'])} | {fmt(ev['macro_recall'])} | **{fmt(ev['macro_f1'])}** |", "",
        f"Predicted 2MWT (all false positives): {fa['count']} events, {fa['total_duration_min']:.1f} min; overlapping GT: {fa['overlapping_gt_class']}.",
        f"Micro P/R/F1 with 2MWT counted as FP: {fmt(ev['micro_including_2mwt_fp']['precision'])} / {fmt(ev['micro_including_2mwt_fp']['recall'])} / {fmt(ev['micro_including_2mwt_fp']['f1'])}.",
    ]
    if sw["n"] > 1:
        lines.append(f"Subject-wise macro F1: {fmt(sw['mean'])} ± {fmt(sw['sd'])} (n={sw['n']}).")
    ca = main["class_agnostic_detection"]
    ce, ct = ca["event_level"], ca["time_level"]
    lines += ["", "## Class-agnostic detection (any predicted class, any overlap)", "",
              "| Level | Recall | Precision | F1 | Details |", "|---|---:|---:|---:|---|",
              f"| Event | {fmt(ce['recall'])} | {fmt(ce['precision'])} | {fmt(ce['f1'])} | "
              f"{ce['gt_detected']}/{ce['gt_events']} GT detected, {ce['predicted_overlapping_gt']}/{ce['predicted_events']} predictions on a GT event, "
              f"{fmt(ce['share_gt_covered_ge_50'])} of GT >= 50% covered |",
              f"| Time | {fmt(ct['recall'])} | {fmt(ct['precision'])} | {fmt(ct['f1'])} | "
              f"GT {ct['gt_time_min']:.0f} min, predicted {ct['predicted_time_min']:.0f} min, overlap {ct['overlap_time_min']:.0f} min |"]
    lines += ["", "## Localization of matched events", "", "| | IoU mean | abs start error mean (s) | abs end error mean (s) | n |", "|---|---:|---:|---:|---:|"]
    for name, loc in main["localization_matched_events"].items():
        lines.append(f"| {name} | {fmt(loc['iou']['mean'])} | {fmt(loc['abs_start_error_sec']['mean'], 1)} | {fmt(loc['abs_end_error_sec']['mean'], 1)} | {loc['iou']['n']} |")
    lines += ["", "## GT class vs best-overlapping prediction", "", "| GT -> predicted | Count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in main["confusion_gt_vs_best_overlapping_prediction"].items()]
    lines += [
        "", "## Battery recovery on complete GT batteries (IoU >= 0.3)", "",
        f"Complete GT batteries: {rec['complete_gt_batteries']}. Battery score {fmt(rec['battery_score'])}, "
        f">= 2/3 recovered {fmt(rec['structured_success_rate_ge_2_of_3'])}, 3/3 recovered {fmt(rec['strict_success_rate_3_of_3'])}.",
        "", "## Independent battery detection (600 s grouping)", "",
        "| GT set | GT batteries | TP | FP | FN | Precision | Recall | F1 | mean battery IoU |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_gt_batteries", "complete_gt_batteries"):
        m = bd[name]
        lines.append(f"| {name} | {m['num_gt_batteries']} | {m['tp']} | {m['fp']} | {m['fn']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['boundary_metrics']['mean_battery_iou'])} |")

    # Diagnostics A: relaxed matching and coverage
    lines += ["", "# Diagnostics", "", "## A. Relaxed event matching (same class)", "",
              "| Matching | Preferred F1 | Fast F1 | Slow F1 | Macro P | Macro R | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, m in main["event_level_relaxed_matching"].items():
        pc = m["per_class"]
        lines.append(f"| {label} | {fmt(pc['preferred']['f1'])} | {fmt(pc['fast']['f1'])} | {fmt(pc['slow']['f1'])} | "
                     f"{fmt(m['macro_precision'])} | {fmt(m['macro_recall'])} | {fmt(m['macro_f1'])} |")
    cv = main["same_class_coverage_of_gt_events"]
    lines += ["", "Share of each GT event covered by same-class predictions:", "",
              "| | mean coverage | >= 50% covered | >= 80% covered | not covered | n |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["overall", "preferred", "fast", "slow"]:
        c = cv[name]
        lines.append(f"| {name} | {fmt(c['mean'])} | {fmt(c['share_ge_0.5'])} | {fmt(c['share_ge_0.8'])} | {fmt(c['share_zero'])} | {c['n']} |")
    lines.append(f"\nGT events split into 2+ same-class predictions: {cv['gt_events_split_into_2plus_same_class_predictions']}.")

    # Diagnostics B: without predicted 2MWT
    cev, crec, cbd = ctrl["event_level"], ctrl["battery_recovery_complete_gt"], ctrl["independent_battery_detection"]
    lines += ["", "## B. Control: predicted 2MWT events removed", "",
              "| Metric | Main | Without predicted 2MWT |", "|---|---:|---:|",
              f"| Event macro F1 (IoU > 0.5) | {fmt(ev['macro_f1'])} | {fmt(cev['macro_f1'])} |",
              f"| Event micro F1 (2MWT as FP) | {fmt(ev['micro_including_2mwt_fp']['f1'])} | {fmt(cev['micro_including_2mwt_fp']['f1'])} |",
              f"| Event macro F1 (any overlap) | {fmt(main['event_level_relaxed_matching']['any_overlap']['macro_f1'])} | {fmt(ctrl['event_level_relaxed_matching']['any_overlap']['macro_f1'])} |",
              f"| Battery score | {fmt(rec['battery_score'])} | {fmt(crec['battery_score'])} |"]
    for name in ("all_gt_batteries", "complete_gt_batteries"):
        a, b = bd[name], cbd[name]
        lines.append(f"| Battery detection P / R / F1 ({name}) | {fmt(a['precision'])} / {fmt(a['recall'])} / {fmt(a['f1'])} | "
                     f"{fmt(b['precision'])} / {fmt(b['recall'])} / {fmt(b['f1'])} |")
    lines.append(f"| Predicted batteries | {bd['num_predicted_batteries']} | {cbd['num_predicted_batteries']} |")
    cca = ctrl["class_agnostic_detection"]
    lines.append(f"| Class-agnostic event P / R / F1 | {fmt(ce['precision'])} / {fmt(ce['recall'])} / {fmt(ce['f1'])} | "
                 f"{fmt(cca['event_level']['precision'])} / {fmt(cca['event_level']['recall'])} / {fmt(cca['event_level']['f1'])} |")
    lines.append(f"| Class-agnostic time P / R / F1 | {fmt(ct['precision'])} / {fmt(ct['recall'])} / {fmt(ct['f1'])} | "
                 f"{fmt(cca['time_level']['precision'])} / {fmt(cca['time_level']['recall'])} / {fmt(cca['time_level']['f1'])} |")
    lines.append("\nRemoving 2MWT uses knowledge that this dataset has no 2MWT; report it only as a sensitivity analysis.")

    # Diagnostics C: stages
    st = result["stage_diagnostics"]
    lines += ["", "## C. Pipeline stages per day", ""]
    if st["sessions_without_diagnostics"]:
        lines.append(f"No stage information for {len(st['sessions_without_diagnostics'])} sessions "
                     "(predictions reused from a run without diagnostics; rerun without --reuse-predictions).\n")
    if st["per_day"]:
        f = st["gt_event_funnel"]
        total = sum(f.values())
        lines.append("Where GT events were lost (first stage with no overlapping window/event):\n")
        lines += ["| Stage | GT events | Share |", "|---|---:|---:|"]
        for key, label in [("no_candidate_window", "No candidate window (preprocessing / Ullrich)"),
                           ("stage1_negative", "Candidate windows, all Stage 1 negative"),
                           ("not_forwarded", "Stage 1 positive but not forwarded"),
                           ("no_event_after_decoding", "Forwarded, but no event after decoding"),
                           ("event_but_not_matched", "Overlapping event, but not matched (class or IoU)"),
                           ("matched", "Matched (same class, IoU > 0.5)")]:
            lines.append(f"| {label} | {f.get(key, 0)} | {fmt(f.get(key, 0) / total if total else None)} |")
        lines += ["", "| Date | Status | Hours | NaN rows | Candidates | Stage 1 + | Forwarded | Events | GT | Lost: no cand. | Lost: Stage 1 | Lost: decoding | Not matched | Matched |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for d in st["per_day"]:
            lines.append(
                f"| {d['date']} | {d['status']} | {d['recording_hours']:.1f} | {fmt(d['nan_row_fraction'] * 100 if d['nan_row_fraction'] is not None else None, 1)}% | "
                f"{d['candidate_windows']} | {d['stage1_positive_windows']} | {d['forwarded_windows']} | {d['predicted_events']} | {d['gt_events']} | "
                f"{d['gt_no_candidate_window']} | {d['gt_stage1_negative'] + d['gt_not_forwarded']} | {d['gt_no_event_after_decoding']} | "
                f"{d['gt_event_but_not_matched']} | {d['gt_matched']} |")
        missing_cols = {d["date"]: d["missing_sensor_columns"] for d in st["per_day"] if d["missing_sensor_columns"]}
        if missing_cols:
            lines.append(f"\nMissing sensor columns: {missing_cols}")

        sw = result["stage1_windows"]
        lines += ["", f"Stage 1 window level ({sw['windows']} candidate windows; {sw['label']}):", "",
                  "| | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, label in [("stage1_raw", "Stage 1 raw output (p >= 0.5)"), ("after_continuity", "After continuity (forwarded)")]:
            m = sw[key]
            lines.append(f"| {label} | {m['TP']} | {m['FP']} | {m['FN']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} |")
        cg = result["continuity_groups"]
        lines.append(f"\nContinuity groups (new group when adjacent candidate windows are > {cg['gap_sec']:.0f} s apart): "
                     f"{cg['forwarded_groups']} of {cg['groups']} groups forwarded, {cg['forwarded_windows']} windows "
                     f"({cg['stage1_positive_windows_in_forwarded_groups']} Stage 1 positive); median group size "
                     f"{fmt(cg['forwarded_group_size']['median'], 0)} windows, largest {cg['largest_group_windows']} windows "
                     f"({cg['largest_group_hours']:.1f} h).")

    # Diagnostics D: Stage 2 probabilities per GT event
    sp = result["stage2_probabilities"]
    on = sp["on_threshold"]
    lines += ["", "## D. Stage 2 probability inside each GT event (smoothed, as seen by the decoder)", ""]
    if sp["sessions_without_stage2_cache"]:
        lines.append(f"No Stage 2 cache for: {', '.join(sp['sessions_without_stage2_cache'])} "
                     "(no windows reached Stage 2, or predictions were reused from a run without the cache).\n")
    labels = {"matched": "Matched", "event_but_not_matched": "Event, not matched",
              "no_event_after_decoding": "No event after decoding", "all": "All GT events"}
    lines += [f"| GT events | n | median own-class max | own max >= {on:.2f} | 0.20 to {on:.2f} | < 0.20 | dominant class over the event |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for key, label in labels.items():
        if key in sp["summary"]:
            m = sp["summary"][key]
            dom = ", ".join(f"{k} {v}" for k, v in sorted(m["dominant_class"].items(), key=lambda kv: -kv[1]))
            lines.append(f"| {label} | {m['n']} | {fmt(m['own_class_max_median'])} | {fmt(m[f'share_own_max_ge_{on:.2f}'])} | "
                         f"{fmt(m['share_own_max_0.20_to_on'])} | {fmt(m['share_own_max_lt_0.20'])} | {dom} |")
    lines.append(f"\nOwn max >= {on:.2f} for a GT event without any event means Stage 2 did see it and the event was lost in "
                 "decoding, cleanup or the island filter; a low own max means Stage 2 itself did not recognise it.")

    # Diagnostics E: post-processing variants
    pv = result["postprocessing_variants"]
    lines += ["", "## E. Post-processing variants (recomputed from cached Stage 2 probabilities)", ""]
    if "skipped" in pv:
        lines.append(pv["skipped"])
    else:
        lines.append(f"Reproduction check: {pv['reproduction_check']}.\n")
        lines += ["| Variant | Events | Pred. 2MWT | GT without any event | Macro F1 (IoU > 0.5) | Macro F1 (any overlap) | Battery score | Battery P / R / F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for name, v in pv["variants"].items():
            b = v["battery_detection_all_gt"]
            lines.append(f"| {name} | {v['events']} | {v['predicted_2mwt']} | {v['gt_events_without_any_overlapping_event']} | "
                         f"{fmt(v['event_macro_f1_iou_gt_0.5'])} | {fmt(v['event_macro_f1_any_overlap'])} | {fmt(v['battery_score'])} | "
                         f"{fmt(b['precision'])} / {fmt(b['recall'])} / {fmt(b['f1'])} |")
        lines += ["", "Class-agnostic detection per variant (any predicted class, any overlap):", "",
                  "| Variant | GT detected | Event R | Event P | Event F1 | GT >= 50% covered | Time R | Time P | Time F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, v in pv["variants"].items():
            e, t = v["class_agnostic_detection"]["event_level"], v["class_agnostic_detection"]["time_level"]
            lines.append(f"| {name} | {e['gt_detected']}/{e['gt_events']} | {fmt(e['recall'])} | {fmt(e['precision'])} | {fmt(e['f1'])} | "
                         f"{fmt(e['share_gt_covered_ge_50'])} | {fmt(t['recall'])} | {fmt(t['precision'])} | {fmt(t['f1'])} |")
        lines.append("\nthr_X lowers the ON threshold of preferred/fast/slow to X (OFF = 0.7 X); 2MWT thresholds are unchanged. "
                     "These variants are tuned on this external dataset and are diagnostics only.")

        # Diagnostics F: whole batteries as the target
        lines += ["", "## G. Whole battery as the target (600 s grouping)", "",
                  "| Variant | GT batteries | Matching | TP | FP | FN | Precision | Recall | F1 | IoU mean / median | abs start / end error (s) |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for name, v in pv["variants"].items():
            for key, m in v["battery_as_target"]["matching"].items():
                subset, label = key.split("/")
                bm = m["boundary_metrics"]
                lines.append(f"| {name} | {subset} ({m['gt_batteries']}) | {label} | {m['tp']} | {m['fp']} | {m['fn']} | "
                             f"{fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                             f"{fmt(bm['mean_battery_iou'])} / {fmt(bm['median_battery_iou'])} | "
                             f"{fmt(bm['mean_absolute_start_error_sec'], 1)} / {fmt(bm['mean_absolute_end_error_sec'], 1)} |")
        lines += ["", "Time level on battery spans:", "", "| Variant | Recall | Precision | F1 |", "|---|---:|---:|---:|"]
        for name, v in pv["variants"].items():
            t = v["battery_as_target"]["time_level_battery_spans"]
            lines.append(f"| {name} | {fmt(t['recall'])} | {fmt(t['precision'])} | {fmt(t['f1'])} |")
        lines += ["", "Predicted batteries without a matching GT battery (any overlap):", "",
                  "| Variant | Session | Duration (s) | Predicted content |", "|---|---|---:|---|"]
        for name, v in pv["variants"].items():
            for u in v["battery_as_target"]["unmatched_predicted_batteries"]:
                lines.append(f"| {name} | {u['session_id']} | {u['duration_sec']:.0f} | {u['class_sequence']} |")

    if cov["failures"]:
        lines += ["", "## Failed sessions", ""] + [f"- {f['session_id']}: {f['error']}" for f in cov["failures"]]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output" / f"external_4x10mwt_{datetime.now().strftime('%Y-%m-%d')}")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Use predictions saved in <output-dir>/predictions instead of running the models.")
    parser.add_argument("--variants", nargs="+", default=list(POSTPROCESSING_VARIANTS),
                        choices=list(POSTPROCESSING_VARIANTS),
                        help="Post-processing variants to recompute from cached Stage 2 probabilities.")
    parser.add_argument("--no-continuity", action="store_true",
                        help="Ablation: forward only Stage 1 positive windows to Stage 2. Use a new --output-dir.")
    args = parser.parse_args()
    continuity_config = NO_CONTINUITY_CONFIG if args.no_continuity else None
    pipeline_text = ("final configuration (tcn, unet_bigru, bundled weights, competition-aware, "
                     f"island {DEFAULT_ISLAND_NEIGHBORHOOD_SEC:.0f} s)")
    if args.no_continuity:
        pipeline_text += " WITH CONTINUITY RULE OFF (ablation, not the thesis configuration)"

    if not args.dataset_root.is_dir():
        sys.exit(f"Dataset root not found: {args.dataset_root} (edit DATASET_ROOT or pass --dataset-root)")
    sessions = discover_sessions(args.dataset_root)
    if not sessions:
        sys.exit(f"No YYYY-MM-DD folders found below {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root), "sessions_found": len(sessions),
        "reuse_predictions": args.reuse_predictions, "git_commit": commit,
        "continuity_rule": "off (only Stage 1 positive windows forwarded)" if args.no_continuity else "on (thesis rule)",
        "pipeline": pipeline_text,
    }, indent=2))

    data, failures = load_sessions(sessions, args.output_dir, args.reuse_predictions, continuity_config)
    tables = {"sessions": [], "matched": [], "false_2mwt": []}
    result = {
        "pipeline": pipeline_text,
        "coverage": {"sessions_found": len(sessions), "sessions_evaluated": len(data),
                     "sessions_failed": len(failures), "failures": failures},
        "main": compute_metrics(data, tables),
        "without_predicted_2mwt": compute_metrics(
            [{**s, "predictions": [p for p in s["predictions"] if p["class_id"] != 4]} for s in data]),
        "stage_diagnostics": stage_diagnostics(data),
        "stage1_windows": stage1_window_metrics(data),
        "continuity_groups": continuity_group_summary(
            data, int((continuity_config or DEFAULT_CONTINUITY_RULE_CONFIG)["gaps_duration_ns"])),
    }
    print("Stage 2 probability analysis...", flush=True)
    result["stage2_probabilities"] = stage2_probability_analysis(data, result["stage_diagnostics"]["per_gt"])
    print("Post-processing variants...", flush=True)
    result["postprocessing_variants"] = postprocessing_variants(data, args.variants)

    out = args.output_dir
    pd.DataFrame(result["stage2_probabilities"].pop("per_gt")).to_csv(out / "gt_stage2_probabilities.csv", index=False)
    per_day = pd.DataFrame(result["main"]["class_agnostic_detection"]["per_day"])
    for name, v in result["postprocessing_variants"].get("variants", {}).items():
        variant_day = pd.DataFrame(v.pop("class_agnostic_per_day")).set_index("session_id")
        per_day[f"gt_detected_{name}"] = per_day["session_id"].map(variant_day["gt_detected"])
        per_day[f"predicted_events_{name}"] = per_day["session_id"].map(variant_day["predicted_events"])
    per_day.to_csv(out / "class_agnostic_per_day.csv", index=False)
    result["stage_diagnostics"].pop("per_gt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    pd.DataFrame(tables["sessions"]).to_csv(out / "sessions.csv", index=False)
    pd.DataFrame(tables["matched"]).to_csv(out / "matched_event_pairs.csv", index=False)
    pd.DataFrame(tables["false_2mwt"]).to_csv(out / "predicted_2mwt_false_alarms.csv", index=False)
    pd.DataFrame(result["stage_diagnostics"]["per_day"]).to_csv(out / "stage_diagnostics_per_day.csv", index=False)
    pd.DataFrame([{"session_id": s["session_id"], "class": CLASS_NAMES[int(g["class_id"])],
                   "start_ns": int(g["time_stamp_start"]), "end_ns": int(g["time_stamp_end"]),
                   "duration_sec": (g["time_stamp_end"] - g["time_stamp_start"]) / 1e9}
                  for s in data for _, g in s["targets"].iterrows()]).to_csv(out / "target_events.csv", index=False)
    pd.DataFrame([{"session_id": s["session_id"], "class": CLASS_NAMES.get(p["class_id"], str(p["class_id"])),
                   "start_ns": p["timestamp_start_ns"], "end_ns": p["timestamp_end_ns"],
                   "duration_sec": (p["timestamp_end_ns"] - p["timestamp_start_ns"]) / 1e9, "score": p["score"]}
                  for s in data for p in s["predictions"]]).to_csv(out / "predicted_events.csv", index=False)
    print()
    print(write_summary(result, out / "METRICS_SUMMARY.md"))
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
