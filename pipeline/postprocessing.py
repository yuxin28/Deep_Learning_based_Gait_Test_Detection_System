#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gait Detection Post-Processing
------------------------------
Converts window-level probabilities into discrete gait events.
Steps:
1. Reconstruct continuous signal (Overlap-Add).
2. Hysteresis Decoding (Double Thresholding).
3. Continuity Rules (Gap Filling).
4. Outlier Removal (Minimum Duration).
5. Non-Maximum Suppression (NMS).
"""

import numpy as np
from typing import List, Dict, Tuple, Any

# ==========================================
# 1. Configuration (Optimized Parameters)
# ==========================================

FS = 102.4  # Sampling frequency (Hz)

# Hysteresis Thresholds
HYS_ON = 0.50
HYS_OFF = 0.38

# Class Priority: 4 (2min) > 3 (slow) > 1 (preferred) > 2 (fast) > 0 (background)
PRIORITY = [4, 3, 1, 2, 0]

# Minimum Duration (seconds)
MIN_DUR_S = {1: 15.0, 2: 12.0, 3: 20.0, 4: 70.0}

# Pre-merge Gap (seconds)
PRE_MERGE_S = {1: 5.0, 2: 5.0, 3: 5.0, 4: 15.0}

# NMS IoU Threshold
NMS_IOU = 0.6

# ==========================================
# 2. Utility Functions
# ==========================================

def segment_iou(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Calculates IoU between two 1D segments (start, end)."""
    s = max(a[1], b[1])
    e = min(a[2], b[2])
    inter = max(0, e - s)
    if inter <= 0: return 0.0
    u = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return 0.0 if u <= 0 else inter / u

def make_hamming_window(L: int) -> np.ndarray:
    """Generates a normalized Hamming window."""
    if L <= 0: return np.ones(0, dtype=np.float32)
    w = np.hamming(L).astype(np.float32)
    m = float(w.max())
    return w / (m if m > 0 else 1.0)

# ==========================================
# 3. Core Decoding Logic
# ==========================================

def hysteresis_decode(P: np.ndarray) -> np.ndarray:
    """Decodes probabilities to class labels using hysteresis thresholds."""
    L, C = P.shape
    y = np.zeros(L, dtype=np.int64)
    active = 0
    prio_rank = {c: i for i, c in enumerate(PRIORITY)}
    
    for t in range(L):
        # Identify candidates above ON threshold
        cands = []
        for c in range(1, C):
            if P[t, c] >= HYS_ON:
                cands.append((prio_rank.get(c, 999), -P[t, c], c))
        
        if active == 0:
            if cands:
                cands.sort()
                active = cands[0][2]
        else:
            # Check if active class drops below OFF threshold
            if P[t, active] < HYS_OFF:
                if cands:
                    cands.sort()
                    active = cands[0][2]
                else:
                    active = 0
        y[t] = active
    return y

def fill_gaps(y: np.ndarray, gap_cfg: Dict[int, float]) -> np.ndarray:
    """Merges small gaps between segments of the same class."""
    if not gap_cfg: return y
    out = y.copy()
    L = len(y)
    i = 0
    while i < L:
        if out[i] == 0:
            i += 1
            continue
        c = out[i]
        j = i + 1
        while j < L and out[j] == c: j += 1
        
        k = j
        while k < L and out[k] == 0: k += 1
        
        if k < L and out[k] == c:
            gap_len = k - j
            max_gap = int(round(gap_cfg.get(c, 0) * FS))
            if 0 < gap_len <= max_gap:
                out[j:k] = c
        i = j
    return out

def remove_short_segments(y: np.ndarray, min_dur_cfg: Dict[int, float]) -> np.ndarray:
    """Removes segments shorter than the minimum duration."""
    if not min_dur_cfg: return y
    out = y.copy()
    L = len(y)
    i = 0
    while i < L:
        c = out[i]
        j = i + 1
        while j < L and out[j] == c: j += 1
        
        if c != 0:
            if (j - i) < int(round(min_dur_cfg.get(c, 0) * FS)):
                out[i:j] = 0
        i = j
    return out

def nms_1d(segs: List[Tuple[int, int, int, float]]) -> List[Tuple[int, int, int, float]]:
    """Performs 1D Non-Maximum Suppression."""
    out = []
    by_class = {}
    for item in segs:
        by_class.setdefault(item[0], []).append(item)
    
    for c, lst in by_class.items():
        lst.sort(key=lambda x: x[3], reverse=True)
        while lst:
            curr = lst.pop(0)
            out.append(curr)
            lst = [x for x in lst if segment_iou((c, curr[1], curr[2]), (c, x[1], x[2])) < NMS_IOU]
    return out

def decode_pipeline(P_seq: np.ndarray) -> List[Tuple[int, int, int, float]]:
    """Full decoding pipeline: Hysteresis -> Fill Gaps -> Filter Short -> List."""
    y = hysteresis_decode(P_seq)
    y = fill_gaps(y, PRE_MERGE_S)
    y = remove_short_segments(y, MIN_DUR_S)
    
    segs = []
    L = len(y)
    i = 0
    while i < L:
        c = y[i]
        j = i + 1
        while j < L and y[j] == c: j += 1
        
        if c != 0:
            segs.append((c, i, j, float(P_seq[i:j, c].mean())))
        i = j
    return segs

# ==========================================
# 4. Signal Reconstruction
# ==========================================

def reconstruct_continuous_sequence(
    window_probs: np.ndarray,
    window_start_times: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Reconstructs continuous probabilities using Overlap-Add."""
    N, Ts, C = window_probs.shape
    if N == 0: return np.zeros((0, C), dtype=np.float32), 0

    min_ts = int(window_start_times[0])
    max_ts = int(window_start_times[-1])
    total_samples = int(round(((max_ts - min_ts) / 1e9) * FS)) + Ts
    
    accum = np.zeros((total_samples, C), dtype=np.float32)
    wsum = np.zeros((total_samples,), dtype=np.float32)
    w = make_hamming_window(Ts)

    for i in range(N):
        t0 = window_start_times[i]
        start_idx = int(round(((t0 - min_ts) / 1e9) * FS))
        end_idx = start_idx + Ts
        
        if start_idx < 0 or end_idx > total_samples: continue
        
        valid_len = end_idx - start_idx
        accum[start_idx:end_idx] += (window_probs[i, :valid_len] * w[:valid_len, None])
        wsum[start_idx:end_idx] += w[:valid_len]
        
    probs_full = np.zeros_like(accum)
    nonzero = wsum > 1e-8
    probs_full[nonzero] = accum[nonzero] / wsum[nonzero, None]
    probs_full[~nonzero, 0] = 1.0 # Default to background

    return probs_full, min_ts

# ==========================================
# 5. Main Entry Point
# ==========================================

def run_postprocessing(
    window_probs: np.ndarray,
    window_start_times: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Main Interface: Reconstructs signal, decodes events, and returns absolute timestamps.
    """
    probs_full, min_ts_ns = reconstruct_continuous_sequence(window_probs, window_start_times)
    segs = decode_pipeline(probs_full)
    segs = nms_1d(segs)
    
    all_events = []
    for c, s, e, score in segs:
        ts_start = min_ts_ns + int(round((s / FS) * 1e9))
        ts_end   = min_ts_ns + int(round((e / FS) * 1e9))
        
        all_events.append({
            "class_id": int(c),
            "timestamp_start_ns": ts_start,
            "timestamp_end_ns": ts_end,
            "score": float(score)
        })
            
    return all_events