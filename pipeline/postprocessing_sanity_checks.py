#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight sanity checks for the locked final post-processing pipeline.

This is not a full unit-test suite. It is meant to catch obvious regressions in:
- overlap-add reconstruction
- hard continuity grouping
- competition-aware decoding
- gap filling and minimum-duration pruning
- final isolated-segment filtering
"""

import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.continuity_rule import apply_continuity_rule
from pipeline.postprocessing import (
    decode_pipeline,
    fill_gaps,
    reconstruct_continuous_sequence,
    remove_short_segments,
    run_postprocessing,
)


def check_overlap_add_reconstruction() -> None:
    window = np.zeros((16, 5), dtype=np.float32)
    window[:, 2] = 0.75
    window[:, 0] = 0.25
    window_probs = np.stack([window, window], axis=0)
    window_start_times = np.array([0, int(round((8 / 102.4) * 1e9))], dtype=np.int64)

    probs_full, _, debug = reconstruct_continuous_sequence(
        window_probs=window_probs,
        window_start_times=window_start_times,
        return_debug=True,
    )

    supported = debug["support_counts"] > 0
    assert supported.any(), "Reconstruction produced no supported samples."
    assert np.allclose(probs_full[supported, 2], 0.75, atol=1e-5), "Overlap-add changed constant class probability."
    assert np.allclose(probs_full[supported, 0], 0.25, atol=1e-5), "Overlap-add changed constant background probability."


def check_continuity_rule_hard_forwarding() -> None:
    timestamps = np.array([0, 30_000_000_000, 60_000_000_000, 200_000_000_000, 230_000_000_000], dtype=np.int64)
    mask = np.array([0, 1, 0, 0, 0], dtype=bool)

    forwarded = apply_continuity_rule(timestamps, mask)
    assert forwarded[:3].all(), "Positive first group should be fully forwarded."
    assert not forwarded[3:].any(), "All-negative second group should remain filtered out."


def check_competition_aware_activation_persistence() -> None:
    probs = np.zeros((10, 5), dtype=np.float32)
    probs[:, 0] = 0.90
    probs[2:6, 2] = 0.80
    probs[2:6, 0] = 0.20

    _, debug = decode_pipeline(
        probs,
        config={
            "smooth_probabilities": False,
            "switch_persist_sec": 0.03,
        },
        return_debug=True,
    )
    labels = debug["decoded_labels_pre_cleanup"]
    assert np.all(labels[:2] == 0), "Decoder should stay in background before evidence appears."
    assert np.all(labels[2:6] == 2), "Sustained evidence should activate the candidate and recover the full run."
    assert np.all(labels[6:] == 0), "Segment should end after evidence disappears."


def check_competition_aware_switch_persistence() -> None:
    probs = np.zeros((12, 5), dtype=np.float32)
    probs[:, 0] = 0.10
    probs[:, 1] = 0.78
    probs[5:7, 2] = 0.90
    probs[5:7, 1] = 0.65

    _, debug = decode_pipeline(
        probs,
        config={
            "smooth_probabilities": False,
            "switch_margin": 0.08,
            "switch_persist_sec": 0.04,
        },
        return_debug=True,
    )
    labels = debug["decoded_labels_pre_cleanup"]
    assert np.all(labels == 1), "Short challenger bursts should not cause a switch before persistence is met."


def check_gap_fill_and_min_duration() -> None:
    labels = np.array([1, 1, 0, 0, 1, 1, 2, 2, 0, 2], dtype=np.int64)
    filled = fill_gaps(labels, {1: 0.03, 2: 0.02})
    assert np.all(filled[:6] == 1), "Gap filling failed to reconnect same-class segments."

    pruned = remove_short_segments(np.array([0, 3, 3, 0, 2, 2, 2, 0], dtype=np.int64), {3: 0.05, 2: 0.01})
    assert np.all(pruned[1:3] == 0), "Short segment was not removed."
    assert np.all(pruned[4:7] == 2), "Long enough segment should have been kept."


def check_island_filter_removes_isolated_segment() -> None:
    window = np.zeros((24, 5), dtype=np.float32)
    window[:, 0] = 0.10
    window[4:18, 2] = 0.90

    events, debug = run_postprocessing(
        window_probs=np.stack([window, window, window], axis=0),
        window_start_times=np.array([0, int(round(3.0 * 1e9)), int(round(20.0 * 1e9))], dtype=np.int64),
        postprocessing_config={
            "smooth_probabilities": False,
            "switch_persist_sec": 0.03,
            "min_duration_sec": {2: 0.05},
            "gap_fill_sec": {2: 0.02},
            "island_filter": {
                "enabled": True,
                "neighborhood_sec": 5.0,
                "min_neighbor_count": 1,
            },
        },
        return_debug=True,
    )
    assert len(events) == 2, "Island filtering should preserve the close pair and drop the isolated event."
    island_debug = debug["island_filter_debug"]
    assert island_debug["removed_indices"] == [2], "Only the isolated segment should be removed."
    assert len(debug["segments_removed_by_island_filter"]) == 1, "Removed island segment should be exposed in debug output."


def check_event_schema() -> None:
    window = np.zeros((24, 5), dtype=np.float32)
    window[:, 0] = 0.10
    window[4:18, 4] = 0.90
    events, debug = run_postprocessing(
        window_probs=np.stack([window], axis=0),
        window_start_times=np.array([0], dtype=np.int64),
        postprocessing_config={
            "smooth_probabilities": False,
            "switch_persist_sec": 0.03,
            "min_duration_sec": {4: 0.05},
            "gap_fill_sec": {4: 0.02},
            "island_filter": {"enabled": False},
        },
        return_debug=True,
    )
    assert isinstance(events, list), "run_postprocessing should return a list of events."
    assert events, "Toy example should produce at least one event."
    assert set(events[0].keys()) == {"class_id", "timestamp_start_ns", "timestamp_end_ns", "score"}, "Event schema changed."
    assert "segments_before_cleanup" in debug, "Debug output should expose pre-cleanup segments."


def run_all_checks() -> None:
    check_overlap_add_reconstruction()
    check_continuity_rule_hard_forwarding()
    check_competition_aware_activation_persistence()
    check_competition_aware_switch_persistence()
    check_gap_fill_and_min_duration()
    check_island_filter_removes_isolated_segment()
    check_event_schema()
    print("All locked-final post-processing sanity checks passed.")


if __name__ == "__main__":
    run_all_checks()
