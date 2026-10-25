#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gait Detection Post-Processing
------------------------------
Locked final post-processing stack used by the final thesis pipeline:
1. Hamming-weighted overlap-add reconstruction of a continuous probability trace.
2. Temporal smoothing of reconstructed probabilities.
3. Competition-aware score-first decoding with per-class hysteresis thresholds.
4. Gap filling and minimum-duration pruning.
5. Final isolated-segment filtering (`island300`, adopted on 2026-10-08).

The legacy experimental branches (legacy/tuned hysteresis, Viterbi, HSMM,
NMS, and gaitmap-based trimming) have been removed from the runtime path.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FS = 102.4

DECODE_MODE = "competition_aware"
PRIORITY = [4, 3, 1, 2, 0]
SMOOTH_S = 1.5
SWITCH_MARGIN = 0.08
TIE_MARGIN = 0.02
SWITCH_PERSIST_S = 1.0
CLASS_ON_THRESHOLDS = {1: 0.40, 2: 0.40, 3: 0.40, 4: 0.35}
CLASS_OFF_THRESHOLDS = {1: 0.28, 2: 0.28, 3: 0.28, 4: 0.25}
MIN_DUR_S = {1: 15.0, 2: 12.0, 3: 20.0, 4: 70.0}
PRE_MERGE_S = {1: 5.0, 2: 5.0, 3: 5.0, 4: 15.0}
DEFAULT_ISLAND_NEIGHBORHOOD_SEC = 300.0
DEFAULT_ISLAND_MIN_NEIGHBORS = 1
DEFAULT_ISLAND_TARGET_CLASSES = [1, 2, 3, 4]

DEFAULT_ISLAND_FILTER_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "neighborhood_sec": DEFAULT_ISLAND_NEIGHBORHOOD_SEC,
    "min_neighbor_count": DEFAULT_ISLAND_MIN_NEIGHBORS,
    "target_classes": DEFAULT_ISLAND_TARGET_CLASSES,
    "require_different_class_neighbor": False,
}

DEFAULT_POSTPROCESSING_CONFIG: Dict[str, Any] = {
    "priority": PRIORITY,
    "class_on_thresholds": CLASS_ON_THRESHOLDS,
    "class_off_thresholds": CLASS_OFF_THRESHOLDS,
    "min_duration_sec": MIN_DUR_S,
    "gap_fill_sec": PRE_MERGE_S,
    "smooth_probabilities": True,
    "smooth_sec": SMOOTH_S,
    "switch_margin": SWITCH_MARGIN,
    "tie_margin": TIE_MARGIN,
    "switch_persist_sec": SWITCH_PERSIST_S,
    "island_filter": DEFAULT_ISLAND_FILTER_CONFIG,
}


def _merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge final-chain config while preserving per-class mappings."""
    merged = dict(base)
    if not override:
        return merged

    for key, value in override.items():
        if key in {"min_duration_sec", "gap_fill_sec", "class_on_thresholds", "class_off_thresholds", "island_filter"} and isinstance(value, dict):
            current = dict(merged.get(key, {}))
            if key == "island_filter":
                current.update(value)
            else:
                current.update({int(k): float(v) for k, v in value.items()})
            merged[key] = current
        else:
            merged[key] = value
    return merged


def resolve_postprocessing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a concrete config for the locked final post-processing chain."""
    resolved = _merge_config(DEFAULT_POSTPROCESSING_CONFIG, config)

    decoder_mode = str(resolved.get("decoder_mode", DECODE_MODE))
    if decoder_mode != DECODE_MODE:
        raise ValueError("Only the locked final competition-aware decoder is supported.")

    resolved["priority"] = [int(class_id) for class_id in resolved.get("priority", PRIORITY)]
    resolved["min_duration_sec"] = {int(k): float(v) for k, v in resolved.get("min_duration_sec", MIN_DUR_S).items()}
    resolved["gap_fill_sec"] = {int(k): float(v) for k, v in resolved.get("gap_fill_sec", PRE_MERGE_S).items()}
    resolved["class_on_thresholds"] = {int(k): float(v) for k, v in resolved.get("class_on_thresholds", CLASS_ON_THRESHOLDS).items()}
    resolved["class_off_thresholds"] = {int(k): float(v) for k, v in resolved.get("class_off_thresholds", CLASS_OFF_THRESHOLDS).items()}

    smoothing_cfg = config.get("smoothing") if isinstance(config, dict) else None
    if isinstance(smoothing_cfg, dict):
        if "enabled" in smoothing_cfg:
            resolved["smooth_probabilities"] = bool(smoothing_cfg["enabled"])
        if "window_sec" in smoothing_cfg:
            resolved["smooth_sec"] = float(smoothing_cfg["window_sec"])

    island_cfg = _merge_config(DEFAULT_ISLAND_FILTER_CONFIG, resolved.get("island_filter"))
    island_cfg["enabled"] = bool(island_cfg.get("enabled", False))
    island_cfg["neighborhood_sec"] = max(0.0, float(island_cfg.get("neighborhood_sec", DEFAULT_ISLAND_NEIGHBORHOOD_SEC)))
    island_cfg["min_neighbor_count"] = max(0, int(island_cfg.get("min_neighbor_count", DEFAULT_ISLAND_MIN_NEIGHBORS)))
    island_cfg["target_classes"] = sorted({int(class_id) for class_id in island_cfg.get("target_classes", DEFAULT_ISLAND_TARGET_CLASSES)})
    island_cfg["require_different_class_neighbor"] = bool(island_cfg.get("require_different_class_neighbor", False))

    resolved["decoder_mode"] = DECODE_MODE
    resolved["decoder_type"] = DECODE_MODE
    resolved["smooth_probabilities"] = bool(resolved.get("smooth_probabilities", True))
    resolved["smooth_sec"] = float(resolved.get("smooth_sec", SMOOTH_S))
    resolved["switch_margin"] = float(resolved.get("switch_margin", SWITCH_MARGIN))
    resolved["tie_margin"] = float(resolved.get("tie_margin", TIE_MARGIN))
    resolved["switch_persist_sec"] = float(resolved.get("switch_persist_sec", SWITCH_PERSIST_S))
    resolved["island_filter"] = island_cfg
    return resolved


def interval_overlap_ns(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_gap_ns(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0


def merge_intervals_ns(intervals: List[Tuple[int, int]], max_gap_ns: int) -> List[Tuple[int, int]]:
    """Merge adjacent absolute-timestamp intervals. Retained for boundary analysis tools."""
    if not intervals:
        return []

    ordered = sorted((int(start_ns), int(end_ns)) for start_ns, end_ns in intervals if end_ns > start_ns)
    if not ordered:
        return []

    merged: List[List[int]] = [[ordered[0][0], ordered[0][1]]]
    for start_ns, end_ns in ordered[1:]:
        last = merged[-1]
        if start_ns - last[1] <= max_gap_ns:
            last[1] = max(last[1], end_ns)
        else:
            merged.append([start_ns, end_ns])

    return [(int(start_ns), int(end_ns)) for start_ns, end_ns in merged]


def choose_reference_support_block(
    event_interval: Tuple[int, int],
    support_blocks: List[Tuple[int, int]],
) -> Tuple[Optional[Tuple[int, int]], int, Optional[int]]:
    """Choose the best support block by overlap first, then minimal gap."""
    if not support_blocks:
        return None, 0, None

    best_block: Optional[Tuple[int, int]] = None
    best_overlap = -1
    best_gap: Optional[int] = None

    for block in support_blocks:
        overlap = interval_overlap_ns(event_interval, block)
        if overlap > best_overlap:
            best_block = block
            best_overlap = overlap
            best_gap = interval_gap_ns(event_interval, block)
        elif overlap == best_overlap and overlap == 0:
            gap = interval_gap_ns(event_interval, block)
            if best_gap is None or gap < best_gap:
                best_block = block
                best_gap = gap

    return best_block, max(0, best_overlap), best_gap


def seconds_to_samples(seconds: float) -> int:
    return max(1, int(round(seconds * FS)))


def make_hamming_window(length: int) -> np.ndarray:
    if length <= 0:
        return np.ones(0, dtype=np.float32)
    if length == 1:
        return np.ones(1, dtype=np.float32)
    weights = np.hamming(length).astype(np.float32)
    peak = float(weights.max())
    return weights / (peak if peak > 0 else 1.0)


def smooth_probabilities(P: np.ndarray, window_sec: float) -> np.ndarray:
    """Centered moving average followed by row-wise renormalization."""
    if P.size == 0:
        return P.astype(np.float32, copy=True)

    kernel_len = min(seconds_to_samples(window_sec), P.shape[0])
    if kernel_len <= 1:
        return P.astype(np.float32, copy=True)

    kernel = np.ones(kernel_len, dtype=np.float32) / float(kernel_len)
    out = np.empty_like(P, dtype=np.float32)
    for class_id in range(P.shape[1]):
        out[:, class_id] = np.convolve(P[:, class_id], kernel, mode="same")

    row_sums = out.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 1e-8
    out[valid] = out[valid] / row_sums[valid]
    out[~valid] = 0.0
    out[~valid, 0] = 1.0
    return out


def compute_margin_traces(P: np.ndarray) -> np.ndarray:
    if P.size == 0:
        return np.zeros_like(P, dtype=np.float32)

    margins = np.empty_like(P, dtype=np.float32)
    for class_id in range(P.shape[1]):
        mask = np.ones(P.shape[1], dtype=bool)
        mask[class_id] = False
        other_max = P[:, mask].max(axis=1) if mask.any() else 0.0
        margins[:, class_id] = P[:, class_id] - other_max
    return margins


def _threshold_for_class(cfg: Dict[str, Any], class_id: int, key: str, fallback: Dict[int, float]) -> float:
    mapping = cfg.get(key, {})
    if class_id in mapping:
        return float(mapping[class_id])
    return float(fallback[class_id])


def _best_score_first_candidate(
    probs_t: np.ndarray,
    cfg: Dict[str, Any],
    exclude: Optional[List[int]] = None,
) -> int:
    candidates = []
    priority_rank = {class_id: idx for idx, class_id in enumerate(cfg.get("priority", PRIORITY))}
    tie_margin = float(cfg.get("tie_margin", TIE_MARGIN))
    excluded = set(exclude or [])

    for class_id in range(1, len(probs_t)):
        if class_id in excluded:
            continue
        on_threshold = _threshold_for_class(cfg, class_id, "class_on_thresholds", CLASS_ON_THRESHOLDS)
        if probs_t[class_id] >= on_threshold:
            candidates.append((float(probs_t[class_id]), class_id))

    if not candidates:
        return 0

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_prob, best_class = candidates[0]
    if len(candidates) == 1:
        return int(best_class)

    second_prob = candidates[1][0]
    if (best_prob - second_prob) > tie_margin:
        return int(best_class)

    tie_classes = [class_id for prob, class_id in candidates if (best_prob - prob) <= tie_margin]
    tie_classes.sort(key=lambda class_id: priority_rank.get(class_id, 999))
    return int(tie_classes[0])


def competition_aware_decode(
    P: np.ndarray,
    cfg: Dict[str, Any],
    return_debug: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Final score-first decoder with persistent activation/switch/end decisions."""
    L, _ = P.shape
    y = np.zeros(L, dtype=np.int64)
    active = 0
    switch_margin = float(cfg.get("switch_margin", SWITCH_MARGIN))
    persist_frames = seconds_to_samples(float(cfg.get("switch_persist_sec", SWITCH_PERSIST_S)))

    pending_kind: Optional[str] = None
    pending_class = 0
    pending_start = -1
    pending_count = 0
    state_trace: List[Dict[str, Any]] = []

    def reset_pending() -> None:
        nonlocal pending_kind, pending_class, pending_start, pending_count
        pending_kind = None
        pending_class = 0
        pending_start = -1
        pending_count = 0

    def advance_pending(kind: str, class_id: int, frame_idx: int) -> bool:
        nonlocal pending_kind, pending_class, pending_start, pending_count
        if pending_kind == kind and pending_class == class_id:
            pending_count += 1
        else:
            pending_kind = kind
            pending_class = class_id
            pending_start = frame_idx
            pending_count = 1
        return pending_count >= persist_frames

    for t in range(L):
        probs_t = P[t]
        best_candidate = _best_score_first_candidate(probs_t, cfg)
        state_event: Dict[str, Any] = {
            "frame_idx": int(t),
            "active_before": int(active),
            "best_candidate": int(best_candidate),
            "pending_kind_before": pending_kind,
            "pending_class_before": int(pending_class),
            "pending_count_before": int(pending_count),
        }

        if active == 0:
            if best_candidate == 0:
                reset_pending()
                y[t] = 0
                state_event["action"] = "stay_background"
            else:
                confirmed = advance_pending("activate", best_candidate, t)
                y[t] = 0
                if confirmed:
                    y[pending_start:t + 1] = best_candidate
                    active = best_candidate
                    state_event["action"] = "activate_candidate"
                    state_event["activation_start"] = int(pending_start)
                    reset_pending()
                else:
                    state_event["action"] = "wait_activate"

            state_event["active_after"] = int(active)
            state_trace.append(state_event)
            continue

        y[t] = active
        active_score = float(probs_t[active])
        active_off = _threshold_for_class(cfg, active, "class_off_thresholds", CLASS_OFF_THRESHOLDS)
        challenger = _best_score_first_candidate(probs_t, cfg, exclude=[active])

        if challenger != 0:
            challenger_score = float(probs_t[challenger])
            state_event["challenger"] = int(challenger)
            state_event["challenger_score"] = challenger_score
        else:
            challenger_score = 0.0
            state_event["challenger"] = 0
            state_event["challenger_score"] = challenger_score

        if challenger != 0 and challenger_score >= (active_score + switch_margin):
            confirmed = advance_pending("switch", challenger, t)
            if confirmed:
                y[pending_start:t + 1] = challenger
                active = challenger
                state_event["action"] = "switch_to_challenger"
                state_event["switch_start"] = int(pending_start)
                reset_pending()
            else:
                state_event["action"] = "wait_switch"
        elif active_score < active_off:
            confirmed = advance_pending("end", 0, t)
            if confirmed:
                y[pending_start:t + 1] = 0
                active = 0
                state_event["action"] = "end_active"
                state_event["end_start"] = int(pending_start)
                reset_pending()
            else:
                state_event["action"] = "wait_end"
        else:
            reset_pending()
            state_event["action"] = "keep_active"

        state_event["active_after"] = int(active)
        state_trace.append(state_event)

    debug = {"state_trace": state_trace, "persist_frames": int(persist_frames)} if return_debug else None
    return y, debug


def fill_gaps(y: np.ndarray, gap_cfg: Dict[int, float]) -> np.ndarray:
    out = y.copy()
    L = len(out)
    i = 0
    while i < L:
        if out[i] == 0:
            i += 1
            continue

        class_id = int(out[i])
        j = i + 1
        while j < L and out[j] == class_id:
            j += 1

        k = j
        while k < L and out[k] == 0:
            k += 1

        if k < L and out[k] == class_id:
            gap_len = k - j
            max_gap = seconds_to_samples(float(gap_cfg.get(class_id, 0.0)))
            if 0 < gap_len <= max_gap:
                out[j:k] = class_id
        i = j

    return out


def remove_short_segments(y: np.ndarray, min_dur_cfg: Dict[int, float]) -> np.ndarray:
    out = y.copy()
    L = len(out)
    i = 0
    while i < L:
        class_id = int(out[i])
        j = i + 1
        while j < L and out[j] == class_id:
            j += 1

        if class_id != 0:
            min_len = seconds_to_samples(float(min_dur_cfg.get(class_id, 0.0)))
            if (j - i) < min_len:
                out[i:j] = 0
        i = j

    return out


def labels_to_segments(labels: np.ndarray, score_source: np.ndarray) -> List[Tuple[int, int, int, float]]:
    segments: List[Tuple[int, int, int, float]] = []
    i = 0
    while i < len(labels):
        class_id = int(labels[i])
        j = i + 1
        while j < len(labels) and labels[j] == class_id:
            j += 1
        if class_id != 0:
            segments.append((class_id, i, j, float(score_source[i:j, class_id].mean())))
        i = j
    return segments


def decode_pipeline(
    P_seq: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
) -> Tuple[List[Tuple[int, int, int, float]], Optional[Dict[str, Any]]]:
    cfg = resolve_postprocessing_config(config)
    if cfg.get("smooth_probabilities", False):
        score_source = smooth_probabilities(P_seq, float(cfg.get("smooth_sec", SMOOTH_S)))
    else:
        score_source = P_seq.astype(np.float32, copy=True)

    y_raw, decoder_debug = competition_aware_decode(score_source, cfg, return_debug=return_debug)
    y_clean = fill_gaps(y_raw, cfg["gap_fill_sec"])
    y_clean = remove_short_segments(y_clean, cfg["min_duration_sec"])
    y_clean = fill_gaps(y_clean, cfg["gap_fill_sec"])
    segments_before_cleanup = labels_to_segments(y_raw, score_source)
    segments = labels_to_segments(y_clean, score_source)

    if not return_debug:
        return segments, None

    def summarize_segment_list(label_segments: List[Tuple[int, int, int, float]]) -> List[Dict[str, Any]]:
        return [
            {
                "class_id": int(class_id),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "duration_sec": float((end_idx - start_idx) / FS),
                "mean_score": float(mean_score),
            }
            for class_id, start_idx, end_idx, mean_score in label_segments
        ]

    return segments, {
        "decoder_mode": DECODE_MODE,
        "score_source": score_source,
        "margin_traces": compute_margin_traces(score_source),
        "decoded_labels_pre_cleanup": y_raw,
        "decoded_labels_post_cleanup": y_clean,
        "segments_before_cleanup": summarize_segment_list(segments_before_cleanup),
        "segments_after_cleanup": summarize_segment_list(segments),
        "decoder_debug": decoder_debug,
    }


def reconstruct_continuous_sequence(
    window_probs: np.ndarray,
    window_start_times: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    window_confidences: Optional[np.ndarray] = None,
    return_debug: bool = False,
) -> Tuple[np.ndarray, int, Optional[Dict[str, Any]]]:
    """Reconstruct a continuous probability sequence via Hamming-weighted overlap-add."""
    N, Ts, C = window_probs.shape
    if N == 0:
        empty = np.zeros((0, C), dtype=np.float32)
        if return_debug:
            return empty, 0, {
                "weight_sum": np.zeros(0, dtype=np.float32),
                "support_counts": np.zeros(0, dtype=np.int32),
                "window_weights": np.zeros(0, dtype=np.float32),
                "sorted_window_start_times": np.zeros(0, dtype=np.int64),
            }
        return empty, 0, None

    order = np.argsort(window_start_times)
    window_probs = window_probs[order]
    window_start_times = window_start_times[order]
    min_ts = int(window_start_times[0])
    max_ts = int(window_start_times[-1])
    total_samples = int(round(((max_ts - min_ts) / 1e9) * FS)) + Ts

    accum = np.zeros((total_samples, C), dtype=np.float32)
    weight_sum = np.zeros((total_samples,), dtype=np.float32)
    support_counts = np.zeros((total_samples,), dtype=np.int32)
    window_weights = make_hamming_window(Ts)

    for window_index in range(N):
        start_idx = int(round(((int(window_start_times[window_index]) - min_ts) / 1e9) * FS))
        end_idx = start_idx + Ts
        if start_idx < 0 or end_idx > total_samples:
            continue
        accum[start_idx:end_idx] += window_probs[window_index] * window_weights[:, None]
        weight_sum[start_idx:end_idx] += window_weights
        support_counts[start_idx:end_idx] += 1

    probs_full = np.zeros_like(accum, dtype=np.float32)
    nonzero = weight_sum > 1e-8
    probs_full[nonzero] = accum[nonzero] / weight_sum[nonzero, None]
    probs_full[~nonzero, 0] = 1.0

    if not return_debug:
        return probs_full, min_ts, None

    return probs_full, min_ts, {
        "weight_sum": weight_sum,
        "support_counts": support_counts,
        "window_weights": window_weights,
        "sorted_window_start_times": window_start_times,
    }


def _segment_gap_ns(left_segment: Tuple[int, int, int, float], right_segment: Tuple[int, int, int, float]) -> int:
    _, left_start_idx, left_end_idx, _ = left_segment
    _, right_start_idx, right_end_idx, _ = right_segment
    left_start_ns = int(round((left_start_idx / FS) * 1e9))
    left_end_ns = int(round((left_end_idx / FS) * 1e9))
    right_start_ns = int(round((right_start_idx / FS) * 1e9))
    right_end_ns = int(round((right_end_idx / FS) * 1e9))

    if left_end_ns < right_start_ns:
        return right_start_ns - left_end_ns
    if right_end_ns < left_start_ns:
        return left_start_ns - right_end_ns
    return 0


def filter_isolated_segments(
    segments: List[Tuple[int, int, int, float]],
    config: Dict[str, Any],
) -> Tuple[List[Tuple[int, int, int, float]], Dict[str, Any]]:
    """Remove isolated predicted segments with too few neighboring test segments."""
    island_cfg = config.get("island_filter", {})
    if not island_cfg.get("enabled", False) or not segments:
        return list(segments), {
            "enabled": bool(island_cfg.get("enabled", False)),
            "kept_indices": list(range(len(segments))),
            "removed_indices": [],
            "neighbor_counts": [0 for _ in segments],
        }

    neighborhood_ns = int(round(float(island_cfg.get("neighborhood_sec", DEFAULT_ISLAND_NEIGHBORHOOD_SEC)) * 1e9))
    min_neighbor_count = max(0, int(island_cfg.get("min_neighbor_count", DEFAULT_ISLAND_MIN_NEIGHBORS)))
    target_classes = {int(class_id) for class_id in island_cfg.get("target_classes", DEFAULT_ISLAND_TARGET_CLASSES)}
    require_different_class = bool(island_cfg.get("require_different_class_neighbor", False))

    kept_segments: List[Tuple[int, int, int, float]] = []
    kept_indices: List[int] = []
    removed_indices: List[int] = []
    neighbor_counts: List[int] = [0 for _ in segments]

    for idx, segment in enumerate(segments):
        class_id = int(segment[0])
        if class_id not in target_classes:
            kept_segments.append(segment)
            kept_indices.append(idx)
            continue

        neighbor_count = 0
        for other_idx, other_segment in enumerate(segments):
            if other_idx == idx:
                continue
            if require_different_class and int(other_segment[0]) == class_id:
                continue
            if _segment_gap_ns(segment, other_segment) <= neighborhood_ns:
                neighbor_count += 1

        neighbor_counts[idx] = neighbor_count
        if neighbor_count >= min_neighbor_count:
            kept_segments.append(segment)
            kept_indices.append(idx)
        else:
            removed_indices.append(idx)

    return kept_segments, {
        "enabled": True,
        "neighborhood_sec": float(island_cfg.get("neighborhood_sec", DEFAULT_ISLAND_NEIGHBORHOOD_SEC)),
        "min_neighbor_count": min_neighbor_count,
        "target_classes": sorted(target_classes),
        "require_different_class_neighbor": require_different_class,
        "kept_indices": kept_indices,
        "removed_indices": removed_indices,
        "neighbor_counts": neighbor_counts,
    }


def _segment_summary(
    class_id: int,
    start_idx: int,
    end_idx: int,
    min_ts_ns: int,
    score_source: np.ndarray,
    margin_traces: np.ndarray,
) -> Dict[str, Any]:
    start_time_ns = min_ts_ns + int(round((start_idx / FS) * 1e9))
    end_time_ns = min_ts_ns + int(round((end_idx / FS) * 1e9))
    class_scores = score_source[start_idx:end_idx, class_id]
    class_margins = margin_traces[start_idx:end_idx, class_id]
    return {
        "class_id": int(class_id),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "duration_sec": float((end_idx - start_idx) / FS),
        "start_time_ns": int(start_time_ns),
        "end_time_ns": int(end_time_ns),
        "mean_score": float(class_scores.mean()) if class_scores.size else 0.0,
        "max_score": float(class_scores.max()) if class_scores.size else 0.0,
        "mean_margin": float(class_margins.mean()) if class_margins.size else 0.0,
    }


def run_postprocessing(
    window_probs: np.ndarray,
    window_start_times: np.ndarray,
    window_confidences: Optional[np.ndarray] = None,
    postprocessing_config: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
) -> List[Dict[str, Any]]:
    """Locked final post-processing entry point."""
    cfg = resolve_postprocessing_config(postprocessing_config)
    probs_full, min_ts_ns, reconstruction_debug = reconstruct_continuous_sequence(
        window_probs=window_probs,
        window_start_times=window_start_times,
        config=cfg,
        window_confidences=window_confidences,
        return_debug=return_debug,
    )

    segments, decode_debug = decode_pipeline(probs_full, config=cfg, return_debug=return_debug)
    pre_island_segments = list(segments)

    score_source = decode_debug["score_source"] if decode_debug else probs_full
    margin_traces = decode_debug["margin_traces"] if decode_debug else compute_margin_traces(score_source)
    segments, island_debug = filter_isolated_segments(segments, cfg)

    events = []
    segment_summaries = []
    for class_id, start_idx, end_idx, mean_score in segments:
        ts_start = min_ts_ns + int(round((start_idx / FS) * 1e9))
        ts_end = min_ts_ns + int(round((end_idx / FS) * 1e9))
        events.append(
            {
                "class_id": int(class_id),
                "timestamp_start_ns": int(ts_start),
                "timestamp_end_ns": int(ts_end),
                "score": float(mean_score),
            }
        )
        segment_summaries.append(
            _segment_summary(
                class_id=class_id,
                start_idx=start_idx,
                end_idx=end_idx,
                min_ts_ns=min_ts_ns,
                score_source=score_source,
                margin_traces=margin_traces,
            )
        )

    removed_segment_summaries = [
        _segment_summary(
            class_id=pre_island_segments[idx][0],
            start_idx=pre_island_segments[idx][1],
            end_idx=pre_island_segments[idx][2],
            min_ts_ns=min_ts_ns,
            score_source=score_source,
            margin_traces=margin_traces,
        )
        for idx in island_debug.get("removed_indices", [])
    ]

    if not return_debug:
        return events

    return events, {
        "config": cfg,
        "reconstructed_probabilities": probs_full,
        "reconstruction_debug": reconstruction_debug,
        "margin_traces": margin_traces,
        "decoded_labels_pre_cleanup": decode_debug["decoded_labels_pre_cleanup"] if decode_debug else np.zeros(0, dtype=np.int64),
        "decoded_labels_post_cleanup": decode_debug["decoded_labels_post_cleanup"] if decode_debug else np.zeros(0, dtype=np.int64),
        "segments_before_cleanup": decode_debug["segments_before_cleanup"] if decode_debug else [],
        "segments_after_cleanup": decode_debug["segments_after_cleanup"] if decode_debug else [],
        "decoder_debug": decode_debug["decoder_debug"] if decode_debug else None,
        "island_filter_debug": island_debug,
        "segments_before_island_filter": [
            _segment_summary(
                class_id=class_id,
                start_idx=start_idx,
                end_idx=end_idx,
                min_ts_ns=min_ts_ns,
                score_source=score_source,
                margin_traces=margin_traces,
            )
            for class_id, start_idx, end_idx, _ in pre_island_segments
        ],
        "segments_removed_by_island_filter": removed_segment_summaries,
        "final_segments": segment_summaries,
        "events": events,
    }
