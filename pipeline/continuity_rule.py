#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, Tuple

import numpy as np


DEFAULT_CONTINUITY_RULE_CONFIG: Dict[str, Any] = {
    "mode": "legacy_any_positive",
    "gaps_duration_ns": 30_000_000_000,
}

LEGACY_CONTINUITY_RULE_CONFIG: Dict[str, Any] = {
    "mode": "legacy_any_positive",
}


def _group_ranges(group_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return contiguous start and end indices for each continuity group."""
    if group_ids.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    starts = np.concatenate(([0], np.flatnonzero(np.diff(group_ids)) + 1))
    ends = np.concatenate((starts[1:], [len(group_ids)]))
    return starts.astype(np.int64), ends.astype(np.int64)


def apply_continuity_rule(
    timestamps: np.ndarray,
    mask: np.ndarray,
    gaps_duration_ns: int = 30_000_000_000,
    scores: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
) -> np.ndarray:
    """
    Apply the original hard continuity rule.

    A continuity group is defined by timestamp gaps larger than `gaps_duration_ns`.
    If any window inside the group is positive, the entire group is forwarded.

    Notes:
    - `scores` is accepted only for backward compatibility and is ignored.
    - `config` is accepted only for backward compatibility. Only
      `config["gaps_duration_ns"]` is honored if present.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)

    if timestamps.ndim != 1 or mask.ndim != 1:
        raise ValueError("timestamps and mask must be 1D arrays.")
    if len(timestamps) != len(mask):
        raise ValueError("timestamps and mask must have the same length.")

    resolved_gap_ns = int(gaps_duration_ns)
    if config and "gaps_duration_ns" in config:
        resolved_gap_ns = int(config["gaps_duration_ns"])

    if len(timestamps) == 0:
        empty = np.zeros(0, dtype=bool)
        if return_debug:
            return empty, {
                "config": {"mode": "legacy_any_positive", "gaps_duration_ns": resolved_gap_ns},
                "group_ids": [],
                "group_summaries": [],
            }
        return empty

    time_diffs = np.diff(timestamps)
    is_new_group = time_diffs > resolved_gap_ns
    group_ids = np.concatenate(([0], np.cumsum(is_new_group))).astype(np.int64)
    starts, ends = _group_ranges(group_ids)

    forwarded = np.zeros(len(mask), dtype=bool)
    group_summaries = []

    for group_index, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        group_mask = mask[start_idx:end_idx]
        should_forward = bool(group_mask.any())
        if should_forward:
            forwarded[start_idx:end_idx] = True

        group_summaries.append(
            {
                "group_index": int(group_index),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_time_ns": int(timestamps[start_idx]),
                "end_time_ns": int(timestamps[end_idx - 1]),
                "num_windows": int(end_idx - start_idx),
                "num_positive_windows": int(group_mask.sum()),
                "forwarded": bool(should_forward),
                "mode": "legacy_any_positive",
            }
        )

    if return_debug:
        return forwarded, {
            "config": {"mode": "legacy_any_positive", "gaps_duration_ns": resolved_gap_ns},
            "group_ids": group_ids.tolist(),
            "group_summaries": group_summaries,
        }

    return forwarded
