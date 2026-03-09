#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def apply_continuity_rule(
    timestamps: np.ndarray, 
    mask: np.ndarray, 
    gaps_duration_ns: int = 30_000_000_000
) -> np.ndarray:
    """
    Applies a continuity rule: if any detection exists within a continuous time segment 
    (separated by gaps > gaps_duration_ns), the entire segment is marked as detected.

    Args:
        timestamps: 1D array of timestamps (ns).
        mask: 1D binary array (0/1).
        gaps_duration_ns: Threshold for time gaps to separate groups.

    Returns:
        Processed binary mask.
    """
    # 1. Identify groups based on time gaps
    time_diffs = np.diff(timestamps)
    is_new_group = time_diffs > gaps_duration_ns
    group_ids = np.concatenate(([0], np.cumsum(is_new_group)))

    # 2. Propagate detection within groups
    # Group by ID -> Take Max (Any True) -> Broadcast back to original shape
    df = pd.DataFrame({'group_id': group_ids, 'signal': mask})
    df['continuity_rule'] = df.groupby('group_id')['signal'].transform('max')

    return df['continuity_rule'].to_numpy(dtype=bool)