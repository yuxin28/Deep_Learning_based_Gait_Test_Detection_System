#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting Utility for Gait Detection
-----------------------------------
Generates visualization for detected gait events:
1. Overview plot (Full signal with highlighted events).
2. Detailed subplots (Zoomed-in view for each event).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Any

# Constants for visualization
CLASS_COLORS = {
    1: 'tab:red', 
    2: 'tab:green', 
    3: 'tab:blue', 
    4: 'tab:orange', 
    0: 'black'
}

CLASS_NAMES = { 
    1: '2x10MWT Preferred', 
    2: '2x10MWT Fast',
    3: '2x10MWT Slow', 
    4: '2MWT'
}

def _to_berlin_time(nanoseconds_index: pd.Index) -> pd.DatetimeIndex:
    """Converts nanosecond index to Berlin datetime (localized)."""
    return pd.to_datetime(nanoseconds_index, unit='ns') \
             .tz_localize('UTC') \
             .tz_convert('Europe/Berlin')

def _plot_overview(signal: pd.DataFrame, all_events: List[Dict], axis_name: str) -> None:
    """Generates the full signal overview with color-coded event overlays."""
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    
    full_segment = signal[axis_name]
    if full_segment.empty:
        ax.text(0.5, 0.5, "Signal Empty", ha='center')
        return

    x_full = _to_berlin_time(full_segment.index)
    
    # Plot background signal
    ax.plot(x_full, full_segment.values, color='lightgray', linewidth=1, label='Background', zorder=1)
    
    # Overlay events
    legend_added = set()
    for event in all_events:
        ts_start, ts_end = event["timestamp_start_ns"], event["timestamp_end_ns"]
        cls_id = int(event['class_id'])
        
        event_seg = signal.loc[ts_start:ts_end, axis_name]
        if event_seg.empty:
            continue
            
        x_event = _to_berlin_time(event_seg.index)
        color = CLASS_COLORS.get(cls_id, 'black')
        label = CLASS_NAMES.get(cls_id, f'Class {cls_id}') if cls_id not in legend_added else None
        
        if label:
            legend_added.add(cls_id)
            
        ax.plot(x_event, event_seg.values, color=color, linewidth=2, label=label, zorder=2)
            
    # Formatting
    ax.set_title(f"Full Signal Overview: {axis_name}")
    ax.set_ylabel(axis_name)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    date_fmt = mdates.DateFormatter('%H:%M:%S', tz=x_full.tz)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(loc='upper right')

def plot_detected_events(
    signal: pd.DataFrame, 
    all_events: List[Dict[str, Any]], 
    axis_name: str = 'right_sensor_gyr_y'
) -> None:
    """
    Main plotting function.
    
    Args:
        signal: DataFrame containing the full sensor data.
        all_events: List of detected event dictionaries.
        axis_name: Column name of the sensor axis to plot.
    """
    if not all_events:
        print("No events detected to plot.")
        return
    if axis_name not in signal.columns:
        print(f"Error: Column '{axis_name}' not found in signal.")
        return

    # 1. Overview Plot
    _plot_overview(signal, all_events, axis_name)

    # 2. Detailed Subplots
    num_events = len(all_events)
    fig, axes = plt.subplots(
        nrows=num_events, 
        ncols=1, 
        figsize=(12, 3 * num_events), 
        constrained_layout=True 
    )
    
    if num_events == 1:
        axes = [axes]

    for i, event in enumerate(all_events):
        ts_start, ts_end = event["timestamp_start_ns"], event["timestamp_end_ns"]
        cls_id = int(event['class_id'])
        cls_name = CLASS_NAMES.get(cls_id, 'Unknown')
        
        segment = signal.loc[ts_start:ts_end, axis_name]
        ax = axes[i]
        
        if not segment.empty:
            x_vals = _to_berlin_time(segment.index)
            color = CLASS_COLORS.get(cls_id, 'black')
            duration = (ts_end - ts_start) / 1e9
            
            ax.plot(x_vals, segment.values, color=color)
            
            start_str = x_vals[0].strftime('%H:%M:%S')
            end_str = x_vals[-1].strftime('%H:%M:%S')
            
            ax.set_title(f"Event {i+1}: {cls_name} | {start_str}-{end_str} | {duration:.2f}s", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')
            ax.set_title(f"Event {i+1}: {cls_name} (No Data)")

        ax.set_xticks([]) # Hide x-ticks for cleaner subplots

    fig.supylabel(axis_name, fontsize=12)
    plt.show()