#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Battery-level evaluation for standardized gait test batteries.

The evaluation follows two rules:
1. Split one day of annotations into temporally continuous target blocks.
2. For every block, always compute performed_event_score.
   Only blocks that contain one full standard battery
   (preferred, fast, slow, 2MWT; exactly one each) contribute to battery_score.

Success summaries are tracked at three levels:
- continuous-block success: at least 2/3 of GT subtests detected in any block
- protocol-threshold success: at least 3/4 detected for complete batteries
- protocol-complete success: 4/4 detected for complete batteries
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CLASS_ID_TO_NAME = {
    1: "preferred",
    2: "fast",
    3: "slow",
    4: "2MWT",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}
SUBTEST_NAMES = ("preferred", "fast", "slow", "2MWT")
BATTERY_CLASS_IDS = tuple(CLASS_ID_TO_NAME.keys())

IOU_THRESHOLD = 0.3
COVERAGE_THRESHOLD = 0.5
SUCCESS_RATIO_NUMERATOR = 2
SUCCESS_RATIO_DENOMINATOR = 3


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """Compute binary precision/recall/F1 from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_mean_sd(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute mean and sample SD for a sequence of floats."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "sd": None,
        }

    mean = float(sum(values) / len(values))
    if len(values) == 1:
        sd = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        sd = float(variance ** 0.5)

    return {
        "count": len(values),
        "mean": mean,
        "sd": sd,
    }


def _as_records(items: Optional[Iterable[Dict]]) -> List[Dict]:
    """Convert supported inputs to a plain list of dictionaries."""
    if items is None:
        return []

    if hasattr(items, "to_dict"):
        return list(items.to_dict("records"))

    return list(items)


def _intersection_length(
    interval_a: Sequence[int],
    interval_b: Sequence[int],
) -> int:
    """Compute the overlap duration between two intervals."""
    start_a, end_a = interval_a
    start_b, end_b = interval_b
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _has_temporal_overlap(
    interval_a: Sequence[int],
    interval_b: Sequence[int],
) -> bool:
    """Check for strictly positive temporal overlap."""
    return _intersection_length(interval_a, interval_b) > 0


def compute_iou(
    pred_interval: Sequence[int],
    true_interval: Sequence[int],
) -> float:
    """Compute the intersection-over-union between two time intervals."""
    pred_start, pred_end = pred_interval
    true_start, true_end = true_interval

    intersection = _intersection_length(pred_interval, true_interval)

    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(0, union_end - union_start)

    return float(intersection / union) if union > 0 else 0.0


def compute_coverage(
    pred_interval: Sequence[int],
    true_interval: Sequence[int],
) -> float:
    """Compute how much of the ground-truth interval is covered by the prediction."""
    true_start, true_end = true_interval
    intersection = _intersection_length(pred_interval, true_interval)

    true_duration = max(0, true_end - true_start)
    return float(intersection / true_duration) if true_duration > 0 else 0.0


def _read_required_key(item: Dict, keys: Sequence[str]) -> int:
    """Read the first available key from a list of alternatives."""
    for key in keys:
        if key in item and item[key] is not None:
            return int(item[key])
    raise KeyError(f"Missing required key. Tried: {keys}")


def _read_optional_score(item: Dict) -> float:
    """Read a score if present; otherwise fall back to zero."""
    if "score" not in item or item["score"] is None:
        return 0.0
    return float(item["score"])


def _normalize_ground_truth(subtests: Iterable[Dict]) -> List[Dict]:
    """Convert GT items to a consistent internal representation and sort by start time."""
    normalized = []
    for subtest in _as_records(subtests):
        class_id = int(subtest["class_id"])
        if class_id not in CLASS_ID_TO_NAME:
            continue

        start_ns = _read_required_key(
            subtest,
            ("start_ns", "time_stamp_start", "timestamp_start_ns"),
        )
        end_ns = _read_required_key(
            subtest,
            ("end_ns", "time_stamp_end", "timestamp_end_ns"),
        )

        if end_ns < start_ns:
            raise ValueError(f"Ground-truth interval has end < start: {subtest}")

        normalized.append(
            {
                "class_id": class_id,
                "subtest_name": CLASS_ID_TO_NAME[class_id],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "raw": {
                    "class_id": class_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            }
        )

    return sorted(
        normalized,
        key=lambda item: (item["start_ns"], item["end_ns"], item["class_id"]),
    )


def _normalize_predictions(predicted_events: Iterable[Dict]) -> List[Dict]:
    """Convert prediction items to a consistent internal representation."""
    normalized = []
    for index, event in enumerate(_as_records(predicted_events)):
        class_id = int(event["class_id"])
        if class_id not in CLASS_ID_TO_NAME:
            continue

        start_ns = _read_required_key(
            event,
            ("timestamp_start_ns", "start_ns", "time_stamp_start"),
        )
        end_ns = _read_required_key(
            event,
            ("timestamp_end_ns", "end_ns", "time_stamp_end"),
        )

        if end_ns < start_ns:
            raise ValueError(f"Prediction interval has end < start: {event}")

        score = _read_optional_score(event)
        normalized.append(
            {
                "prediction_index": index,
                "class_id": class_id,
                "subtest_name": CLASS_ID_TO_NAME[class_id],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "score": score,
                "raw": {
                    "class_id": class_id,
                    "timestamp_start_ns": start_ns,
                    "timestamp_end_ns": end_ns,
                    "score": score,
                },
            }
        )

    return sorted(
        normalized,
        key=lambda item: (item["start_ns"], item["end_ns"], item["class_id"]),
    )


def _match_metric_name(class_id: int) -> str:
    """Select the matching metric for a given class."""
    return "coverage" if class_id == CLASS_NAME_TO_ID["2MWT"] else "iou"


def _match_threshold(class_id: int) -> float:
    """Select the validity threshold for a given class."""
    return COVERAGE_THRESHOLD if class_id == CLASS_NAME_TO_ID["2MWT"] else IOU_THRESHOLD


def _match_value(prediction: Dict, ground_truth: Dict) -> float:
    """Compute the class-specific overlap value used for matching."""
    pred_interval = (prediction["start_ns"], prediction["end_ns"])
    gt_interval = (ground_truth["start_ns"], ground_truth["end_ns"])

    if ground_truth["class_id"] == CLASS_NAME_TO_ID["2MWT"]:
        return compute_coverage(pred_interval, gt_interval)
    return compute_iou(pred_interval, gt_interval)


def _build_detected_subtests(matches: List[Dict]) -> Dict[str, bool]:
    """Collapse per-event matches into one bool per standard subtest type."""
    detected_subtests = {name: False for name in SUBTEST_NAMES}
    for match in matches:
        if match["detected"]:
            detected_subtests[match["subtest_name"]] = True
    return detected_subtests


def _extract_subtest_iou_stats(matches: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute matched-IoU statistics overall and per subtest class."""
    iou_values_by_name = {name: [] for name in SUBTEST_NAMES}
    for match in matches:
        iou_value = _read_match_iou_value(match)
        if not match.get("detected") or iou_value is None:
            continue
        iou_values_by_name[match["subtest_name"]].append(float(iou_value))

    per_subtest = {
        name: _compute_mean_sd(values)
        for name, values in iou_values_by_name.items()
    }
    overall_values = [
        value
        for values in iou_values_by_name.values()
        for value in values
    ]

    return {
        "overall": _compute_mean_sd(overall_values),
        "per_subtest": per_subtest,
    }


def _read_match_iou_value(match: Dict) -> Optional[float]:
    """Read a stored IoU value or recompute it from match payloads."""
    if match.get("iou_value") is not None:
        return float(match["iou_value"])

    if not match.get("detected") or match.get("matched_prediction") is None:
        return None

    pred = match["matched_prediction"]
    gt = match["ground_truth"]
    pred_start = int(pred.get("timestamp_start_ns", pred.get("start_ns")))
    pred_end = int(pred.get("timestamp_end_ns", pred.get("end_ns")))
    gt_start = int(gt["start_ns"])
    gt_end = int(gt["end_ns"])
    return compute_iou((pred_start, pred_end), (gt_start, gt_end))


def _prediction_interval(prediction: Dict) -> Tuple[int, int]:
    """Read a prediction interval from any supported result schema."""
    return (
        _read_required_key(
            prediction,
            ("timestamp_start_ns", "start_ns", "time_stamp_start"),
        ),
        _read_required_key(
            prediction,
            ("timestamp_end_ns", "end_ns", "time_stamp_end"),
        ),
    )


def _ground_truth_interval(ground_truth: Dict) -> Tuple[int, int]:
    """Read a ground-truth interval from any supported result schema."""
    return (
        _read_required_key(
            ground_truth,
            ("start_ns", "time_stamp_start", "timestamp_start_ns"),
        ),
        _read_required_key(
            ground_truth,
            ("end_ns", "time_stamp_end", "timestamp_end_ns"),
        ),
    )


def _compute_overlap_any_fields(
    predicted_events: Iterable[Dict],
    ground_truth_block: Iterable[Dict],
) -> Dict:
    """
    Compute class-agnostic temporal overlap recovery fields.

    A GT subtest is counted as touched if any gait-test prediction overlaps it
    with strictly positive duration, regardless of predicted class.
    """
    predictions = _as_records(predicted_events)
    ground_truth = _as_records(ground_truth_block)
    overlap_count = 0

    prediction_intervals = [
        _prediction_interval(prediction)
        for prediction in predictions
        if int(prediction.get("class_id", -1)) in CLASS_ID_TO_NAME
    ]

    for gt_item in ground_truth:
        gt_interval = _ground_truth_interval(gt_item)
        if any(
            _has_temporal_overlap(pred_interval, gt_interval)
            for pred_interval in prediction_intervals
        ):
            overlap_count += 1

    return {
        "overlap_any_success": overlap_count > 0,
        "overlap_any_overlap_count": overlap_count,
        "overlap_any_has_assigned_predictions": len(prediction_intervals) > 0,
    }


def _read_overlap_any_fields(battery_result: Dict) -> Dict:
    """Read stored overlap-any fields or recompute them from block payloads."""
    if (
        "overlap_any_success" in battery_result
        and "overlap_any_overlap_count" in battery_result
        and "overlap_any_has_assigned_predictions" in battery_result
    ):
        return {
            "overlap_any_success": bool(battery_result["overlap_any_success"]),
            "overlap_any_overlap_count": int(battery_result["overlap_any_overlap_count"] or 0),
            "overlap_any_has_assigned_predictions": bool(
                battery_result["overlap_any_has_assigned_predictions"]
            ),
        }

    return _compute_overlap_any_fields(
        predicted_events=battery_result.get("assigned_predictions", []),
        ground_truth_block=battery_result.get("ground_truth_subtests", []),
    )


def _meets_success_ratio(detected_count: int, target_count: int) -> bool:
    """Check whether detected_count / target_count is at least 2/3."""
    if target_count <= 0:
        return False
    return detected_count * SUCCESS_RATIO_DENOMINATOR >= target_count * SUCCESS_RATIO_NUMERATOR


def _build_battery_summary(detected_subtests: Dict[str, bool]) -> Dict:
    """Compute battery-level fields for a complete standard battery target."""
    detected_count = sum(bool(flag) for flag in detected_subtests.values())
    protocol_threshold_success = _meets_success_ratio(detected_count, 4)

    return {
        "battery_detected_count": detected_count,
        "battery_target_count": 4,
        "battery_score": detected_count / 4.0,
        "protocol_threshold_success": protocol_threshold_success,
        "protocol_complete_success": detected_count == 4,
        # Backward-compatible field names kept for downstream code and result consumers.
        "structured_success": protocol_threshold_success,
        "strict_success": detected_count == 4,
    }


def _classify_target_block(ground_truth_block: List[Dict]) -> Tuple[str, bool]:
    """Classify one continuous GT block as complete or non-compliant."""
    class_counts = Counter(item["class_id"] for item in ground_truth_block)
    is_complete_battery = (
        len(ground_truth_block) == 4
        and all(class_counts.get(class_id, 0) == 1 for class_id in BATTERY_CLASS_IDS)
    )

    if is_complete_battery:
        return "complete_battery", True

    if any(count > 1 for count in class_counts.values()):
        return "repeated_subtests", False

    return "partial_battery", False


def match_subtests(
    predicted_events: Iterable[Dict],
    ground_truth_subtests: Iterable[Dict],
) -> Dict:
    """
    Match GT subtests to predictions with class-specific rules.

    Each prediction can be matched at most once. For a GT subtest, the best valid
    prediction of the same class is selected using:
    1. higher IoU / coverage
    2. higher prediction score
    3. lower original prediction index for deterministic tie-breaking
    """
    predictions = _normalize_predictions(predicted_events)
    ground_truth = _normalize_ground_truth(ground_truth_subtests)

    available_predictions = {}
    for prediction in predictions:
        available_predictions.setdefault(prediction["class_id"], []).append(prediction)

    used_prediction_indices = set()
    matches = []

    for gt_subtest in ground_truth:
        class_id = gt_subtest["class_id"]
        metric_name = _match_metric_name(class_id)
        threshold = _match_threshold(class_id)

        best_prediction: Optional[Dict] = None
        best_key: Optional[Tuple[float, float, int]] = None
        best_value = 0.0

        for prediction in available_predictions.get(class_id, []):
            prediction_index = prediction["prediction_index"]
            if prediction_index in used_prediction_indices:
                continue

            value = _match_value(prediction, gt_subtest)
            if value < threshold:
                continue

            key = (value, prediction["score"], -prediction_index)
            if best_key is None or key > best_key:
                best_key = key
                best_prediction = prediction
                best_value = value

        detected = best_prediction is not None
        if detected:
            used_prediction_indices.add(best_prediction["prediction_index"])

        matches.append(
            {
                "class_id": class_id,
                "subtest_name": gt_subtest["subtest_name"],
                "ground_truth": gt_subtest["raw"],
                "detected": detected,
                "match_metric": metric_name,
                "match_value": best_value if detected else 0.0,
                "iou_value": (
                    compute_iou(
                        (best_prediction["start_ns"], best_prediction["end_ns"]),
                        (gt_subtest["start_ns"], gt_subtest["end_ns"]),
                    )
                    if detected
                    else None
                ),
                "threshold": threshold,
                "matched_prediction": best_prediction["raw"] if detected else None,
            }
        )

    detected_subtests = _build_detected_subtests(matches)
    return {
        "matches": matches,
        "detected_subtests": detected_subtests,
    }


def split_ground_truth_into_batteries(
    ground_truth_subtests: Iterable[Dict],
) -> List[List[Dict]]:
    """
    Split one day's annotations into continuous target blocks.

    Rows are first sorted by start timestamp. A new block starts whenever the next
    subtest is no longer continuous with the previous one, i.e.:
        current.start_ns != previous.end_ns
    """
    ground_truth = _normalize_ground_truth(ground_truth_subtests)
    if not ground_truth:
        return []

    batteries = []
    current_battery = [ground_truth[0]]

    for subtest in ground_truth[1:]:
        previous = current_battery[-1]
        if subtest["start_ns"] == previous["end_ns"]:
            current_battery.append(subtest)
            continue

        batteries.append([item["raw"] for item in current_battery])
        current_battery = [subtest]

    batteries.append([item["raw"] for item in current_battery])
    return batteries


def _assign_predictions_to_batteries(
    predicted_events: Iterable[Dict],
    batteries: List[List[Dict]],
) -> List[List[Dict]]:
    """Assign each prediction to the continuous GT block with the largest overlap."""
    normalized_predictions = _normalize_predictions(predicted_events)
    assigned_predictions = [[] for _ in batteries]

    battery_windows = []
    for battery in batteries:
        starts = [item["start_ns"] for item in battery]
        ends = [item["end_ns"] for item in battery]
        battery_windows.append((min(starts), max(ends)))

    for prediction in normalized_predictions:
        pred_interval = (prediction["start_ns"], prediction["end_ns"])
        best_battery_index = None
        best_overlap = 0

        for battery_index, window in enumerate(battery_windows):
            overlap = _intersection_length(pred_interval, window)
            if overlap > best_overlap:
                best_overlap = overlap
                best_battery_index = battery_index

        if best_battery_index is not None and best_overlap > 0:
            assigned_predictions[best_battery_index].append(prediction["raw"])

    return assigned_predictions


def _evaluate_target_block(
    predicted_events: Iterable[Dict],
    ground_truth_block: List[Dict],
    block_index: int,
) -> Dict:
    """Evaluate one continuous target block."""
    predicted_events = list(predicted_events)
    if not ground_truth_block:
        overlap_any_fields = _compute_overlap_any_fields(
            predicted_events=predicted_events,
            ground_truth_block=[],
        )
        return {
            "battery_index": block_index,
            "window_start_ns": None,
            "window_end_ns": None,
            "num_ground_truth_subtests": 0,
            "target_status": "empty_block",
            "is_complete_battery_target": False,
            "ground_truth_subtests": [],
            "assigned_predictions": predicted_events,
            "matches": [],
            "detected_subtests": {name: False for name in SUBTEST_NAMES},
            **overlap_any_fields,
            "performed_event_detected_count": 0,
            "performed_event_target_count": 0,
            "performed_event_score": 0.0,
            "continuous_success": False,
            "battery_detected_count": None,
            "battery_target_count": None,
            "battery_score": None,
            "protocol_threshold_success": None,
            "protocol_complete_success": None,
            "moderate_success": False,
            "strict_success": None,
            "structured_success": None,
        }

    match_results = match_subtests(
        predicted_events=predicted_events,
        ground_truth_subtests=ground_truth_block,
    )
    matches = match_results["matches"]
    detected_subtests = match_results["detected_subtests"]
    subtest_iou_stats = _extract_subtest_iou_stats(matches)
    overlap_any_fields = _compute_overlap_any_fields(
        predicted_events=predicted_events,
        ground_truth_block=ground_truth_block,
    )
    performed_event_detected_count = sum(bool(item["detected"]) for item in matches)
    performed_event_target_count = len(matches)
    has_assigned_predictions = len(predicted_events) > 0
    any_hit_success = performed_event_detected_count > 0
    performed_event_score = (
        performed_event_detected_count / performed_event_target_count
        if performed_event_target_count > 0
        else 0.0
    )
    continuous_success = _meets_success_ratio(
        performed_event_detected_count,
        performed_event_target_count,
    )

    normalized_ground_truth_block = _normalize_ground_truth(ground_truth_block)
    starts = [item["start_ns"] for item in normalized_ground_truth_block]
    ends = [item["end_ns"] for item in normalized_ground_truth_block]
    target_status, is_complete_battery_target = _classify_target_block(
        normalized_ground_truth_block
    )

    result = {
        "battery_index": block_index,
        "window_start_ns": min(starts),
        "window_end_ns": max(ends),
        "num_ground_truth_subtests": performed_event_target_count,
        "target_status": target_status,
        "is_complete_battery_target": is_complete_battery_target,
        "ground_truth_subtests": ground_truth_block,
        "assigned_predictions": predicted_events,
        "matches": matches,
        "detected_subtests": detected_subtests,
        "subtest_iou_stats": subtest_iou_stats,
        **overlap_any_fields,
        "has_assigned_predictions": has_assigned_predictions,
        "any_hit_success": any_hit_success,
        "performed_event_detected_count": performed_event_detected_count,
        "performed_event_target_count": performed_event_target_count,
        "performed_event_score": performed_event_score,
        "continuous_success": continuous_success,
        "battery_detected_count": None,
        "battery_target_count": None,
        "battery_score": None,
        "protocol_threshold_success": None,
        "protocol_complete_success": None,
        # moderate_success now means success on any temporally continuous block
        # with at least a 2/3 detection ratio.
        "moderate_success": continuous_success,
        "strict_success": None,
        "structured_success": None,
    }

    if is_complete_battery_target:
        result.update(_build_battery_summary(detected_subtests))

    return result


def summarize_battery_results(battery_results: Iterable[Dict]) -> Dict:
    """Aggregate scores across all continuous target blocks in one split."""
    battery_results = list(battery_results)
    num_blocks = len(battery_results)
    complete_blocks = [item for item in battery_results if item["is_complete_battery_target"]]
    noncompliant_blocks = [item for item in battery_results if not item["is_complete_battery_target"]]

    performed_detected_count = sum(item["performed_event_detected_count"] for item in battery_results)
    performed_target_count = sum(item["performed_event_target_count"] for item in battery_results)
    battery_detected_count = sum(item["battery_detected_count"] or 0 for item in complete_blocks)
    battery_target_count = sum(item["battery_target_count"] or 0 for item in complete_blocks)
    continuous_count = sum(bool(item["continuous_success"]) for item in battery_results)
    protocol_threshold_count = sum(
        bool(item["protocol_threshold_success"]) for item in complete_blocks
    )
    protocol_complete_count = sum(
        bool(item["protocol_complete_success"]) for item in complete_blocks
    )

    session_has_target = performed_target_count > 0
    session_has_predictions = any(bool(item.get("has_assigned_predictions")) for item in battery_results)
    session_any_hit_success = any(bool(item.get("any_hit_success")) for item in battery_results)
    session_any_hit_tp = int(session_has_target and session_any_hit_success)
    session_any_hit_fp = int(session_has_predictions and not session_any_hit_success)
    session_any_hit_fn = int(session_has_target and not session_any_hit_success)
    session_any_hit_prf = _compute_precision_recall_f1(
        session_any_hit_tp,
        session_any_hit_fp,
        session_any_hit_fn,
    )
    session_overlap_any_success = any(
        bool(_read_overlap_any_fields(item)["overlap_any_success"])
        for item in battery_results
    )
    session_overlap_any_has_predictions = any(
        bool(_read_overlap_any_fields(item)["overlap_any_has_assigned_predictions"])
        for item in battery_results
    )
    session_overlap_any_tp = int(session_has_target and session_overlap_any_success)
    session_overlap_any_fp = int(
        session_overlap_any_has_predictions and not session_overlap_any_success
    )
    session_overlap_any_fn = int(session_has_target and not session_overlap_any_success)
    session_overlap_any_prf = _compute_precision_recall_f1(
        session_overlap_any_tp,
        session_overlap_any_fp,
        session_overlap_any_fn,
    )

    battery_any_hit_tp = sum(
        int(bool(item.get("any_hit_success")))
        for item in complete_blocks
    )
    battery_any_hit_fp = sum(
        int(bool(item.get("has_assigned_predictions")) and not bool(item.get("any_hit_success")))
        for item in complete_blocks
    )
    battery_any_hit_fn = sum(
        int(not bool(item.get("any_hit_success")))
        for item in complete_blocks
    )
    battery_any_hit_prf = _compute_precision_recall_f1(
        battery_any_hit_tp,
        battery_any_hit_fp,
        battery_any_hit_fn,
    )
    complete_overlap_any_fields = [
        _read_overlap_any_fields(item)
        for item in complete_blocks
    ]
    battery_overlap_any_tp = sum(
        int(bool(item["overlap_any_success"]))
        for item in complete_overlap_any_fields
    )
    battery_overlap_any_fp = sum(
        int(
            bool(item["overlap_any_has_assigned_predictions"])
            and not bool(item["overlap_any_success"])
        )
        for item in complete_overlap_any_fields
    )
    battery_overlap_any_fn = sum(
        int(not bool(item["overlap_any_success"]))
        for item in complete_overlap_any_fields
    )
    battery_overlap_any_prf = _compute_precision_recall_f1(
        battery_overlap_any_tp,
        battery_overlap_any_fp,
        battery_overlap_any_fn,
    )

    subtest_iou_values = {name: [] for name in SUBTEST_NAMES}
    for battery_result in battery_results:
        for match in battery_result.get("matches", []):
            iou_value = _read_match_iou_value(match)
            if not match.get("detected") or iou_value is None:
                continue
            subtest_iou_values[match["subtest_name"]].append(float(iou_value))
    overall_subtest_iou_values = [
        value
        for values in subtest_iou_values.values()
        for value in values
    ]
    subtest_iou_stats = {
        "overall": _compute_mean_sd(overall_subtest_iou_values),
        "per_subtest": {
            name: _compute_mean_sd(values)
            for name, values in subtest_iou_values.items()
        },
    }

    return {
        "num_blocks": num_blocks,
        "num_complete_battery_targets": len(complete_blocks),
        "num_noncompliant_blocks": len(noncompliant_blocks),
        "target_status_counts": {
            "complete_battery": len(complete_blocks),
            "partial_battery": sum(item["target_status"] == "partial_battery" for item in battery_results),
            "repeated_subtests": sum(item["target_status"] == "repeated_subtests" for item in battery_results),
        },
        "performed_event_detected_count": performed_detected_count,
        "performed_event_target_count": performed_target_count,
        "performed_event_score": (
            performed_detected_count / performed_target_count
            if performed_target_count > 0
            else 0.0
        ),
        "battery_detected_count": battery_detected_count,
        "battery_target_count": battery_target_count,
        "battery_score": (
            battery_detected_count / battery_target_count
            if battery_target_count > 0
            else None
        ),
        "continuous_success_count": continuous_count,
        "continuous_success_rate": (
            continuous_count / num_blocks
            if num_blocks > 0
            else None
        ),
        "protocol_threshold_success_count": protocol_threshold_count,
        "protocol_threshold_success_rate": (
            protocol_threshold_count / len(complete_blocks)
            if complete_blocks
            else None
        ),
        "protocol_complete_success_count": protocol_complete_count,
        "protocol_complete_success_rate": (
            protocol_complete_count / len(complete_blocks)
            if complete_blocks
            else None
        ),
        # Backward-compatible aliases.
        "moderate_success_count": continuous_count,
        "moderate_success_rate": (
            continuous_count / num_blocks
            if num_blocks > 0
            else None
        ),
        "strict_success_count": protocol_complete_count,
        "strict_success_rate": (
            protocol_complete_count / len(complete_blocks)
            if complete_blocks
            else None
        ),
        "structured_success_count": protocol_threshold_count,
        "structured_success_rate": (
            protocol_threshold_count / len(complete_blocks)
            if complete_blocks
            else None
        ),
        "session_any_hit_success": session_any_hit_success,
        "session_any_hit_tp": session_any_hit_tp,
        "session_any_hit_fp": session_any_hit_fp,
        "session_any_hit_fn": session_any_hit_fn,
        "session_any_hit_precision": session_any_hit_prf["precision"],
        "session_any_hit_recall": session_any_hit_prf["recall"],
        "session_any_hit_f1": session_any_hit_prf["f1"],
        "session_overlap_any_success": session_overlap_any_success,
        "session_overlap_any_tp": session_overlap_any_tp,
        "session_overlap_any_fp": session_overlap_any_fp,
        "session_overlap_any_fn": session_overlap_any_fn,
        "session_overlap_any_precision": session_overlap_any_prf["precision"],
        "session_overlap_any_recall": session_overlap_any_prf["recall"],
        "session_overlap_any_f1": session_overlap_any_prf["f1"],
        "battery_any_hit_tp": battery_any_hit_tp,
        "battery_any_hit_fp": battery_any_hit_fp,
        "battery_any_hit_fn": battery_any_hit_fn,
        "battery_any_hit_precision": battery_any_hit_prf["precision"],
        "battery_any_hit_recall": battery_any_hit_prf["recall"],
        "battery_any_hit_f1": battery_any_hit_prf["f1"],
        "battery_overlap_any_tp": battery_overlap_any_tp,
        "battery_overlap_any_fp": battery_overlap_any_fp,
        "battery_overlap_any_fn": battery_overlap_any_fn,
        "battery_overlap_any_precision": battery_overlap_any_prf["precision"],
        "battery_overlap_any_recall": battery_overlap_any_prf["recall"],
        "battery_overlap_any_f1": battery_overlap_any_prf["f1"],
        "subtest_iou_stats": subtest_iou_stats,
    }


def evaluate_battery_collection(
    predicted_events: Iterable[Dict],
    ground_truth_subtests: Iterable[Dict],
) -> Dict:
    """
    Evaluate all continuous target blocks found in one session/day.

    Each block always contributes to performed_event_score.
    Only complete standard batteries contribute to battery_score.
    """
    batteries = split_ground_truth_into_batteries(ground_truth_subtests)
    assigned_predictions = _assign_predictions_to_batteries(predicted_events, batteries)

    battery_results = []
    for battery_index, battery_ground_truth in enumerate(batteries):
        battery_results.append(
            _evaluate_target_block(
                predicted_events=assigned_predictions[battery_index],
                ground_truth_block=battery_ground_truth,
                block_index=battery_index,
            )
        )

    return {
        "batteries": battery_results,
        "summary": summarize_battery_results(battery_results),
    }


def aggregate_battery_collections(session_results: Dict[str, Dict]) -> Dict:
    """Aggregate battery outcomes across a split of multiple sessions."""
    all_batteries = []
    session_any_hit_tp = 0
    session_any_hit_fp = 0
    session_any_hit_fn = 0
    session_overlap_any_tp = 0
    session_overlap_any_fp = 0
    session_overlap_any_fn = 0
    for session_result in session_results.values():
        session_batteries = session_result.get("batteries", [])
        all_batteries.extend(session_batteries)
        session_summary = session_result.get("summary", {})
        if "session_overlap_any_tp" not in session_summary:
            session_summary = dict(session_summary)
            session_summary.update(summarize_battery_results(session_batteries))
        session_any_hit_tp += int(session_summary.get("session_any_hit_tp", 0) or 0)
        session_any_hit_fp += int(session_summary.get("session_any_hit_fp", 0) or 0)
        session_any_hit_fn += int(session_summary.get("session_any_hit_fn", 0) or 0)
        session_overlap_any_tp += int(session_summary.get("session_overlap_any_tp", 0) or 0)
        session_overlap_any_fp += int(session_summary.get("session_overlap_any_fp", 0) or 0)
        session_overlap_any_fn += int(session_summary.get("session_overlap_any_fn", 0) or 0)

    split_summary = summarize_battery_results(all_batteries)
    session_any_hit_prf = _compute_precision_recall_f1(
        session_any_hit_tp,
        session_any_hit_fp,
        session_any_hit_fn,
    )
    session_overlap_any_prf = _compute_precision_recall_f1(
        session_overlap_any_tp,
        session_overlap_any_fp,
        session_overlap_any_fn,
    )
    split_summary["session_any_hit_success"] = None
    split_summary["session_any_hit_tp"] = session_any_hit_tp
    split_summary["session_any_hit_fp"] = session_any_hit_fp
    split_summary["session_any_hit_fn"] = session_any_hit_fn
    split_summary["session_any_hit_precision"] = session_any_hit_prf["precision"]
    split_summary["session_any_hit_recall"] = session_any_hit_prf["recall"]
    split_summary["session_any_hit_f1"] = session_any_hit_prf["f1"]
    split_summary["session_overlap_any_success"] = None
    split_summary["session_overlap_any_tp"] = session_overlap_any_tp
    split_summary["session_overlap_any_fp"] = session_overlap_any_fp
    split_summary["session_overlap_any_fn"] = session_overlap_any_fn
    split_summary["session_overlap_any_precision"] = session_overlap_any_prf["precision"]
    split_summary["session_overlap_any_recall"] = session_overlap_any_prf["recall"]
    split_summary["session_overlap_any_f1"] = session_overlap_any_prf["f1"]
    split_summary["num_sessions"] = len(session_results)
    split_summary["sessions_with_blocks"] = sum(
        1 for session_result in session_results.values() if session_result.get("batteries")
    )
    split_summary["sessions_with_complete_battery_targets"] = sum(
        1
        for session_result in session_results.values()
        if session_result.get("summary", {}).get("num_complete_battery_targets", 0) > 0
    )
    return split_summary


def evaluate_battery(
    predicted_events: Iterable[Dict],
    ground_truth_subtests: Iterable[Dict],
) -> Dict:
    """Evaluate one target block directly without day-level splitting."""
    return _evaluate_target_block(
        predicted_events=_as_records(predicted_events),
        ground_truth_block=[item["raw"] for item in _normalize_ground_truth(ground_truth_subtests)],
        block_index=0,
    )
