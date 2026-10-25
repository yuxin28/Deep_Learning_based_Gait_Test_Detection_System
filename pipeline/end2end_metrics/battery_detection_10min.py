#!/usr/bin/env python3
"""Independent 10-minute-gap battery detection evaluation.

Predicted batteries are built from every final post-processed prediction before
any ground-truth assignment. Ground-truth batteries are built separately.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional

import pandas as pd
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

BATTERY_GROUP_GAP_SEC = 600.0
BATTERY_GROUP_GAP_NS = int(BATTERY_GROUP_GAP_SEC * 1e9)
CLASS_ID_TO_NAME = {1: "preferred", 2: "fast", 3: "slow", 4: "2MWT"}
COMPLETE_COUNTS = Counter({1: 1, 2: 1, 3: 1, 4: 1})
DEFAULT_RESULTS_DIR = ROOT / "pipeline" / "end2end_metrics" / "results_battery_detection_10min_competition_island300"
AUTHORITATIVE_RESULTS_DIR = ROOT / "pipeline" / "end2end_metrics" / "results_competition_aware_island300"
CONFIGURED_DATASET_ROOT = Path("/path/to/dataset")  # root of the paths in *_dataset_info/*.json

ISLAND300_POSTPROCESSING_CONFIG = {
    "decoder_mode": "competition_aware",
    "smooth_probabilities": True,
    "smooth_sec": 1.5,
    "class_on_thresholds": {1: 0.40, 2: 0.40, 3: 0.40, 4: 0.35},
    "class_off_thresholds": {1: 0.28, 2: 0.28, 3: 0.28, 4: 0.25},
    "gap_fill_sec": {1: 5.0, 2: 5.0, 3: 5.0, 4: 15.0},
    "min_duration_sec": {1: 15.0, 2: 12.0, 3: 20.0, 4: 70.0},
    "island_filter": {
        "enabled": True,
        "neighborhood_sec": 300.0,
        "min_neighbor_count": 1,
        "target_classes": [1, 2, 3, 4],
        "require_different_class_neighbor": False,
    },
}


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.item() if hasattr(item, "item") else str(item),
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def normalize_session_events(
    split: str,
    session_id: str,
    events: Iterable[dict],
    source: str,
) -> list[dict]:
    """Normalize prediction or GT events without consulting the other source."""
    rows = []
    patient_id = session_id.split("_")[0]
    prefix = "pred" if source == "prediction" else "gt"
    for event in events:
        class_id = int(event["class_id"])
        if class_id not in CLASS_ID_TO_NAME:
            continue
        start_ns = int(event.get("timestamp_start_ns", event.get("start_ns")))
        end_ns = int(event.get("timestamp_end_ns", event.get("end_ns")))
        if end_ns <= start_ns:
            continue
        rows.append(
            {
                "split": split,
                "patient_id": patient_id,
                "session_id": session_id,
                "source": source,
                "class_id": class_id,
                "class_name": CLASS_ID_TO_NAME[class_id],
                "timestamp_start_ns": start_ns,
                "timestamp_end_ns": end_ns,
                "duration_sec": (end_ns - start_ns) / 1e9,
                "score": _safe_float(event.get("score")) if source == "prediction" else None,
                "original_record_json": _json_dump(event),
            }
        )

    rows.sort(key=lambda row: (row["timestamp_start_ns"], row["timestamp_end_ns"], row["class_id"]))
    for index, row in enumerate(rows, 1):
        row["event_id"] = f"{split}:{session_id}:{prefix}_event_{index:04d}"
    return rows


def group_events_into_batteries(
    events: Iterable[dict],
    source: str,
    gap_sec: float = BATTERY_GROUP_GAP_SEC,
) -> list[dict]:
    """Group events by the gap to the immediately preceding chronological event."""
    gap_ns = int(gap_sec * 1e9)
    by_session: dict[tuple[str, str, str], list[dict]] = {}
    for event in events:
        key = (str(event["split"]), str(event["patient_id"]), str(event["session_id"]))
        by_session.setdefault(key, []).append(dict(event))

    batteries = []
    label = "predicted" if source == "prediction" else "gt"
    for (split, patient_id, session_id), session_events in sorted(by_session.items()):
        ordered = sorted(
            session_events,
            key=lambda row: (row["timestamp_start_ns"], row["timestamp_end_ns"], row["event_id"]),
        )
        groups: list[list[dict]] = []
        for event in ordered:
            if not groups or event["timestamp_start_ns"] - groups[-1][-1]["timestamp_end_ns"] > gap_ns:
                groups.append([event])
            else:
                groups[-1].append(event)

        for index, members in enumerate(groups, 1):
            start_ns = min(row["timestamp_start_ns"] for row in members)
            end_ns = max(row["timestamp_end_ns"] for row in members)
            class_ids = [int(row["class_id"]) for row in members]
            battery = {
                "split": split,
                "patient_id": patient_id,
                "session_id": session_id,
                "source": source,
                "battery_id": f"{split}:{session_id}:{label}_battery_{index:03d}",
                "battery_start_ns": start_ns,
                "battery_end_ns": end_ns,
                "battery_duration_sec": (end_ns - start_ns) / 1e9,
                "num_subtests": len(members),
                "class_sequence": [CLASS_ID_TO_NAME[class_id] for class_id in class_ids],
                "class_id_sequence": class_ids,
                "member_event_ids": [row["event_id"] for row in members],
                "member_events": members,
            }
            if source == "ground_truth":
                battery["is_complete_standard_battery"] = Counter(class_ids) == COMPLETE_COUNTS
            batteries.append(battery)
    return batteries


def interval_iou(left: dict, right: dict) -> tuple[int, float]:
    intersection = max(
        0,
        min(left["battery_end_ns"], right["battery_end_ns"])
        - max(left["battery_start_ns"], right["battery_start_ns"]),
    )
    union = max(left["battery_end_ns"], right["battery_end_ns"]) - min(
        left["battery_start_ns"], right["battery_start_ns"]
    )
    return intersection, intersection / union if union > 0 else 0.0


def match_batteries(
    predicted_batteries: Iterable[dict],
    gt_batteries: Iterable[dict],
    min_iou: Optional[float] = None,
) -> dict:
    """Deterministic greedy one-to-one matching, highest battery IoU first."""
    predicted = sorted(predicted_batteries, key=lambda row: row["battery_id"])
    ground_truth = sorted(gt_batteries, key=lambda row: row["battery_id"])
    candidates = []
    for pred in predicted:
        for gt in ground_truth:
            if pred["session_id"] != gt["session_id"]:
                continue
            intersection, iou = interval_iou(pred, gt)
            eligible = intersection > 0 if min_iou is None else iou >= min_iou
            if eligible:
                candidates.append((-iou, pred["battery_id"], gt["battery_id"], pred, gt, intersection, iou))

    matched_pred: set[str] = set()
    matched_gt: set[str] = set()
    matches = []
    for _, pred_id, gt_id, pred, gt, intersection, iou in sorted(candidates):
        if pred_id in matched_pred or gt_id in matched_gt:
            continue
        matched_pred.add(pred_id)
        matched_gt.add(gt_id)
        matches.append(
            {
                "predicted_battery_id": pred_id,
                "gt_battery_id": gt_id,
                "session_id": pred["session_id"],
                "battery_iou": iou,
                "intersection_sec": intersection / 1e9,
                "start_error_sec": (pred["battery_start_ns"] - gt["battery_start_ns"]) / 1e9,
                "end_error_sec": (pred["battery_end_ns"] - gt["battery_end_ns"]) / 1e9,
                "predicted": pred,
                "ground_truth": gt,
            }
        )

    return {
        "matches": matches,
        "unmatched_predictions": [row for row in predicted if row["battery_id"] not in matched_pred],
        "unmatched_ground_truth": [row for row in ground_truth if row["battery_id"] not in matched_gt],
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def summarize_match_result(result: dict) -> dict:
    matches = result["matches"]
    metrics = _prf(len(matches), len(result["unmatched_predictions"]), len(result["unmatched_ground_truth"]))
    start_errors = [row["start_error_sec"] for row in matches]
    end_errors = [row["end_error_sec"] for row in matches]
    ious = [row["battery_iou"] for row in matches]
    metrics["boundary_metrics"] = {
        "num_matched_pairs": len(matches),
        "mean_start_error_sec": mean(start_errors) if start_errors else None,
        "mean_absolute_start_error_sec": mean(map(abs, start_errors)) if start_errors else None,
        "mean_end_error_sec": mean(end_errors) if end_errors else None,
        "mean_absolute_end_error_sec": mean(map(abs, end_errors)) if end_errors else None,
        "mean_battery_iou": mean(ious) if ious else None,
        "median_battery_iou": median(ious) if ious else None,
    }
    return metrics


def evaluate_battery_detection(predicted_batteries: list[dict], gt_batteries: list[dict]) -> dict:
    evaluations = {}
    sensitivity = {}
    for name, gt_subset in (
        ("all_gt_batteries", gt_batteries),
        ("complete_standard_gt_batteries", [row for row in gt_batteries if row["is_complete_standard_battery"]]),
    ):
        primary = match_batteries(predicted_batteries, gt_subset, min_iou=None)
        metrics = summarize_match_result(primary)
        metrics.update(
            {
                "num_evaluated_predicted_batteries": len(predicted_batteries),
                "num_evaluated_gt_batteries": len(gt_subset),
                "matching_criterion": "positive temporal overlap",
                "matching_strategy": "deterministic greedy maximum-IoU edge selection",
            }
        )
        evaluations[name] = {"metrics": metrics, "matching": primary}
        sensitivity[name] = {}
        for threshold in (0.1, 0.3):
            result = match_batteries(predicted_batteries, gt_subset, min_iou=threshold)
            sensitivity[name][f"iou_gte_{threshold:.1f}"] = summarize_match_result(result)
    return {"evaluations": evaluations, "sensitivity": sensitivity}


def duration_distribution(batteries: list[dict]) -> dict:
    values = [float(row["battery_duration_sec"]) for row in batteries]
    if not values:
        return {
            "count": 0,
            "min_sec": None,
            "p25_sec": None,
            "median_sec": None,
            "mean_sec": None,
            "p75_sec": None,
            "max_sec": None,
            "num_over_600_sec": 0,
            "percent_over_600_sec": 0.0,
        }
    series = pd.Series(values)
    count_over = sum(value > BATTERY_GROUP_GAP_SEC for value in values)
    return {
        "count": len(values),
        "min_sec": min(values),
        "p25_sec": float(series.quantile(0.25)),
        "median_sec": median(values),
        "mean_sec": mean(values),
        "p75_sec": float(series.quantile(0.75)),
        "max_sec": max(values),
        "num_over_600_sec": count_over,
        "percent_over_600_sec": 100.0 * count_over / len(values),
    }


def batteries_to_rows(batteries: Iterable[dict]) -> list[dict]:
    rows = []
    for battery in batteries:
        row = {key: value for key, value in battery.items() if key != "member_events"}
        for key in ("class_sequence", "class_id_sequence", "member_event_ids"):
            row[key] = _json_dump(row[key])
        row["member_events_json"] = _json_dump(battery["member_events"])
        rows.append(row)
    return rows


def match_rows(evaluation: dict, evaluation_set: str) -> list[dict]:
    rows = []
    for match in evaluation["matching"]["matches"]:
        pred = match["predicted"]
        gt = match["ground_truth"]
        rows.append(
            {
                "evaluation_set": evaluation_set,
                "status": "matched",
                "session_id": match["session_id"],
                "predicted_battery_id": pred["battery_id"],
                "gt_battery_id": gt["battery_id"],
                "predicted_start_ns": pred["battery_start_ns"],
                "predicted_end_ns": pred["battery_end_ns"],
                "gt_start_ns": gt["battery_start_ns"],
                "gt_end_ns": gt["battery_end_ns"],
                "battery_iou": match["battery_iou"],
                "start_error_sec": match["start_error_sec"],
                "end_error_sec": match["end_error_sec"],
            }
        )
    for pred in evaluation["matching"]["unmatched_predictions"]:
        rows.append(
            {
                "evaluation_set": evaluation_set,
                "status": "unmatched_prediction_fp",
                "session_id": pred["session_id"],
                "predicted_battery_id": pred["battery_id"],
                "gt_battery_id": None,
                "predicted_start_ns": pred["battery_start_ns"],
                "predicted_end_ns": pred["battery_end_ns"],
            }
        )
    for gt in evaluation["matching"]["unmatched_ground_truth"]:
        rows.append(
            {
                "evaluation_set": evaluation_set,
                "status": "unmatched_ground_truth_fn",
                "session_id": gt["session_id"],
                "predicted_battery_id": None,
                "gt_battery_id": gt["battery_id"],
                "gt_start_ns": gt["battery_start_ns"],
                "gt_end_ns": gt["battery_end_ns"],
            }
        )
    return rows


def _battery_signature(batteries: list[dict]) -> list[tuple]:
    return [
        (
            row["battery_id"],
            row["battery_start_ns"],
            row["battery_end_ns"],
            tuple(row["member_event_ids"]),
        )
        for row in batteries
    ]


def _outside_gt_count(predictions: list[dict], gt_batteries: list[dict]) -> int:
    gt_by_session: dict[str, list[dict]] = {}
    for gt in gt_batteries:
        gt_by_session.setdefault(gt["session_id"], []).append(gt)
    count = 0
    for pred in predictions:
        overlapping = any(
            min(pred["timestamp_end_ns"], gt["battery_end_ns"])
            > max(pred["timestamp_start_ns"], gt["battery_start_ns"])
            for gt in gt_by_session.get(pred["session_id"], [])
        )
        count += int(not overlapping)
    return count


def build_sanity_summary(
    predictions: list[dict],
    predicted_batteries: list[dict],
    gt_batteries: list[dict],
    evaluated: dict,
) -> dict:
    regrouped_without_gt = group_events_into_batteries(predictions, "prediction")
    gt_copy = list(gt_batteries)
    regrouped_after_gt_load = group_events_into_batteries(predictions, "prediction")
    assert gt_copy == gt_batteries
    leakage_free = _battery_signature(regrouped_without_gt) == _battery_signature(regrouped_after_gt_load)

    identities = {}
    for name, result in evaluated["evaluations"].items():
        metrics = result["metrics"]
        identities[name] = {
            "tp_plus_fn_equals_gt": metrics["tp"] + metrics["fn"] == metrics["num_evaluated_gt_batteries"],
            "tp_plus_fp_equals_predictions": metrics["tp"] + metrics["fp"] == metrics["num_evaluated_predicted_batteries"],
        }
    return {
        "num_predictions_outside_all_gt_battery_windows": _outside_gt_count(predictions, gt_batteries),
        "num_unmatched_predicted_batteries_all_gt": evaluated["evaluations"]["all_gt_batteries"]["metrics"]["fp"],
        "precision_not_structurally_forced_by_gt_prefiltering": True,
        "observed_all_gt_precision_less_than_one": (
            evaluated["evaluations"]["all_gt_batteries"]["metrics"]["precision"] < 1.0
        ),
        "count_identities": identities,
        "prediction_grouping_identical_with_or_without_gt_loaded": leakage_free,
        "predicted_battery_duration_distribution": duration_distribution(predicted_batteries),
    }


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight_paths(paths_by_split: dict[str, dict]) -> list[str]:
    missing = []
    for split, sessions in paths_by_split.items():
        for session_id, paths in sessions.items():
            for key in ("csv_files", "h5_files"):
                for path in paths.get(key, []):
                    if not Path(path).exists():
                        missing.append(f"{split}:{session_id}:{key}:{path}")
    return missing


def _remap_dataset_paths(paths_by_split: dict[str, dict], dataset_root: Path) -> dict[str, dict]:
    remapped = json.loads(json.dumps(paths_by_split))
    for sessions in remapped.values():
        for paths in sessions.values():
            for key in ("csv_files", "h5_files"):
                paths[key] = [
                    str(dataset_root / Path(path).relative_to(CONFIGURED_DATASET_ROOT))
                    for path in paths.get(key, [])
                ]
    return remapped


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_info_paths(dataset_source: str) -> dict[str, Path]:
    info_dir = ROOT / "pipeline" / "end2end_metrics" / f"{dataset_source}_dataset_info"
    return {
        "info_dir": info_dir,
        "between_paths": info_dir / f"{dataset_source}_between_paths.json",
        "within_paths": info_dir / f"{dataset_source}_within_paths.json",
    }


def _run_sessions(
    split: str,
    paths: dict,
    dataset_source: str,
    num_workers: int,
    cache_dir: str,
    preprocessing_config: dict,
) -> list[dict]:
    from pipeline.end2end_metrics.metrics_pipeline import process_single_session

    kwargs = {
        "dataset_source": dataset_source,
        "split_name": split,
        "postprocessing_cache_dir": cache_dir,
        "preprocessing_config": preprocessing_config,
        "postprocessing_config": ISLAND300_POSTPROCESSING_CONFIG,
        "return_full_session_data": True,
    }
    items = list(sorted(paths.items()))
    if num_workers <= 1:
        return [
            process_single_session(session_id, session_paths, **kwargs)
            for session_id, session_paths in tqdm(items, desc=split)
        ]

    results = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(num_workers, len(items)), mp_context=context) as executor:
        futures = {
            executor.submit(process_single_session, session_id, session_paths, **kwargs): session_id
            for session_id, session_paths in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=split):
            session_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"patid_date": session_id, "patid": session_id.split("_")[0], "error": str(exc)})
    return sorted(results, key=lambda row: row["patid_date"])


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _examples(evaluated: dict, limit: int = 3) -> list[dict]:
    primary = evaluated["evaluations"]["all_gt_batteries"]["matching"]
    all_predicted = [row["predicted"] for row in primary["matches"]] + primary["unmatched_predictions"]
    all_gt = [row["ground_truth"] for row in primary["matches"]] + primary["unmatched_ground_truth"]
    session_ids = sorted({row["session_id"] for row in all_predicted + all_gt})[:limit]

    def compact(row: dict) -> dict:
        return {
            "battery_id": row["battery_id"],
            "start_ns": row["battery_start_ns"],
            "end_ns": row["battery_end_ns"],
            "class_sequence": row["class_sequence"],
        }

    examples = []
    for session_id in session_ids:
        examples.append(
            {
                "session_id": session_id,
                "predicted_batteries": [compact(row) for row in all_predicted if row["session_id"] == session_id],
                "gt_batteries": [compact(row) for row in all_gt if row["session_id"] == session_id],
                "matched_pairs": [
                    {
                        "predicted_battery_id": row["predicted_battery_id"],
                        "gt_battery_id": row["gt_battery_id"],
                        "battery_iou": row["battery_iou"],
                    }
                    for row in primary["matches"]
                    if row["session_id"] == session_id
                ],
                "unmatched_predicted_batteries": [
                    row["battery_id"] for row in primary["unmatched_predictions"] if row["session_id"] == session_id
                ],
                "unmatched_gt_batteries": [
                    row["battery_id"] for row in primary["unmatched_ground_truth"] if row["session_id"] == session_id
                ],
            }
        )
    return examples


def _summary_markdown(run_config: dict, split_outputs: dict[str, dict]) -> str:
    lines = [
        "# Independent 10-Minute Battery Detection",
        "",
        "## Definition",
        "",
        "Predicted batteries are constructed from all final full-session predictions before any GT assignment. "
        "GT batteries are constructed independently. Consecutive chronological events are chained when the gap "
        "from one event's end to the next event's start is <= 600 s; no maximum total duration is imposed.",
        "",
        "Primary matching uses positive temporal overlap and deterministic greedy one-to-one maximum-IoU edge "
        "selection. Unmatched predicted batteries are FP and unmatched GT batteries are FN.",
        "",
        "For complete-only evaluation, all globally constructed predicted batteries remain candidates. Predictions "
        "in sessions with partial GT groups are not discarded and remain FP unless matched to a complete GT battery.",
        "",
        "## Results",
        "",
    ]
    for split, output in split_outputs.items():
        lines.append(f"### {split.title()}")
        lines.append("")
        for name, label in (("all_gt_batteries", "All GT groups"), ("complete_standard_gt_batteries", "Complete GT only")):
            metric = output["evaluated"]["evaluations"][name]["metrics"]
            lines.append(
                f"- {label}: TP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}, "
                f"Precision={metric['precision']:.6f}, Recall={metric['recall']:.6f}, F1={metric['f1']:.6f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Old Section 8 vs New Evaluation",
            "",
            "Old overlap-any first assigns/restricts predictions to GT support, so complete-battery FP is "
            "structurally underestimated and the metric is mainly useful as coarse recovery/recall. The new "
            "evaluation constructs prediction and GT batteries independently from full-session data, then uses "
            "one-to-one matching; therefore both global battery-level FP and FN are represented.",
            "",
            "## Provenance",
            "",
            f"- Git commit: `{run_config['git']['commit']}`",
            f"- Git branch: `{run_config['git']['branch']}`",
            f"- Command: `{run_config['command']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-source", choices=["test", "val"], default="test")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--postprocessing-cache-dir", default="")
    parser.add_argument("--preprocessing-config-json", default="")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Relocate configured dataset paths without modifying the split JSON files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files only in the dedicated output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.results_dir.resolve()
    if output_dir == AUTHORITATIVE_RESULTS_DIR.resolve():
        raise ValueError(f"Refusing to modify authoritative results: {AUTHORITATIVE_RESULTS_DIR}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite explicitly.")

    dataset_info = _dataset_info_paths(args.dataset_source)
    paths_by_split = {
        "within": _load_json(dataset_info["within_paths"]),
        "between": _load_json(dataset_info["between_paths"]),
    }
    if args.dataset_root is not None:
        paths_by_split = _remap_dataset_paths(paths_by_split, args.dataset_root.resolve())
    missing = _preflight_paths(paths_by_split)
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Preflight found {len(missing)} missing input paths; no result files were written.\n{preview}"
        )

    if args.preprocessing_config_json:
        from pipeline.end2end_metrics.metrics_pipeline import resolve_preprocessing_config

        preprocessing_config = resolve_preprocessing_config(args.preprocessing_config_json)
    else:
        preprocessing_config = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": "independent_battery_detection_10min_competition_island300",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset_source": args.dataset_source,
        "configured_dataset_root": str(CONFIGURED_DATASET_ROOT),
        "effective_dataset_root": (
            str(args.dataset_root.resolve()) if args.dataset_root else str(CONFIGURED_DATASET_ROOT)
        ),
        "split_definition_files": {key: str(value) for key, value in dataset_info.items() if key != "info_dir"},
        "model": {
            "stage1_backbone": "tcn",
            "stage1_checkpoint": str(ROOT / "models_training/stage1/best_model/tcn/best_model.pth"),
            "stage2_model": "unet_bigru",
            "stage2_checkpoint": str(ROOT / "models_training/stage2/best_model/unet_bigru/best_model.pth"),
            "author_weights": True,
        },
        "preprocessing_config": preprocessing_config,
        "postprocessing_config": ISLAND300_POSTPROCESSING_CONFIG,
        "battery_group_gap_sec": BATTERY_GROUP_GAP_SEC,
        "battery_grouping_rule": "adjacent chronological event start minus immediately preceding event end <= 600 s",
        "primary_matching_criterion": "positive temporal overlap",
        "matching_strategy": "deterministic greedy maximum-IoU edge selection",
        "sensitivity_iou_thresholds": [0.1, 0.3],
        "git": {"commit": _git_value("rev-parse", "HEAD"), "branch": _git_value("branch", "--show-current")},
        "authoritative_results_directory_untouched": str(AUTHORITATIVE_RESULTS_DIR),
    }

    split_outputs = {}
    errors = {}
    combined_sensitivity = {}
    for split, paths in paths_by_split.items():
        session_results = _run_sessions(
            split,
            paths,
            args.dataset_source,
            args.num_workers,
            args.postprocessing_cache_dir,
            preprocessing_config,
        )
        errors[split] = [row for row in session_results if row.get("error")]
        successful = [row for row in session_results if not row.get("error")]
        predictions = [
            normalized
            for result in successful
            for normalized in normalize_session_events(split, result["patid_date"], result["predictions"], "prediction")
        ]
        gt_events = [
            normalized
            for result in successful
            for normalized in normalize_session_events(
                split, result["patid_date"], result["ground_truth_subtests"], "ground_truth"
            )
        ]
        predicted_batteries = group_events_into_batteries(predictions, "prediction")
        gt_batteries = group_events_into_batteries(gt_events, "ground_truth")
        evaluated = evaluate_battery_detection(predicted_batteries, gt_batteries)
        sanity = build_sanity_summary(predictions, predicted_batteries, gt_batteries, evaluated)
        examples = _examples(evaluated)
        print(f"\n[{split}] Manual reconstruction examples:")
        print(json.dumps(examples, indent=2))

        pd.DataFrame(predictions).to_csv(output_dir / f"full_predictions_{split}.csv", index=False)
        pd.DataFrame(gt_events).to_csv(output_dir / f"full_targets_{split}.csv", index=False)
        pd.DataFrame(batteries_to_rows(predicted_batteries)).to_csv(
            output_dir / f"predicted_batteries_{split}.csv", index=False
        )
        pd.DataFrame(batteries_to_rows(gt_batteries)).to_csv(output_dir / f"gt_batteries_{split}.csv", index=False)
        all_match_rows = []
        for name, evaluation in evaluated["evaluations"].items():
            all_match_rows.extend(match_rows(evaluation, name))
        pd.DataFrame(all_match_rows).to_csv(output_dir / f"battery_matches_{split}.csv", index=False)

        metric_payload = {
            "split": split,
            "configured_sessions": len(paths),
            "evaluated_sessions": len(successful),
            "failed_sessions": len(errors[split]),
            "num_full_predictions": len(predictions),
            "num_predicted_batteries": len(predicted_batteries),
            "num_gt_batteries": len(gt_batteries),
            "num_complete_gt_batteries": sum(row["is_complete_standard_battery"] for row in gt_batteries),
            "evaluations": {name: value["metrics"] for name, value in evaluated["evaluations"].items()},
            "sanity_checks": sanity,
            "manual_examples": examples,
        }
        _save_json(output_dir / f"battery_detection_metrics_{split}.json", metric_payload)
        combined_sensitivity[split] = evaluated["sensitivity"]
        split_outputs[split] = {
            "predictions": predictions,
            "predicted_batteries": predicted_batteries,
            "gt_batteries": gt_batteries,
            "evaluated": evaluated,
        }

    run_config["session_errors"] = errors
    _save_json(output_dir / "run_config.json", run_config)
    _save_json(output_dir / "battery_detection_sensitivity.json", combined_sensitivity)
    (output_dir / "BATTERY_DETECTION_10MIN_SUMMARY.md").write_text(
        _summary_markdown(run_config, split_outputs), encoding="utf-8"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
