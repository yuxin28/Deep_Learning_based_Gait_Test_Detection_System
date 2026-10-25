#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gait Detection Analysis System - Offline Tkinter GUI
Runs completely offline, no internet connection required.

- Runs the locked final pipeline (TCN -> U-Net+BiGRU -> competition-aware
  decoding -> island300) through `detect_gait_test()`.
- The worker thread never touches Tk widgets; it reports through a queue that
  the main thread polls.
- The recording is read once and shared by the pipeline and the plots.
- Events are grouped into batteries with the 600 s gap rule used in the
  independent battery-detection evaluation.
- All times are shown in Europe/Berlin, matching `plot_detected_events`.
"""

import csv
import json
import os
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ==========================================
# Path Configuration
# ==========================================
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# ==========================================
# Import Pipeline Modules
# ==========================================
PIPELINE_AVAILABLE = False
PIPELINE_IMPORT_ERROR = ""
try:
    import numpy as np
    import pandas as pd
    from gait_detection import detect_gait_test
    from plot_detected_events import plot_detected_events

    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_IMPORT_ERROR = str(e)
    print(f"Warning: Could not import pipeline modules: {e}")


# ==========================================
# Constants
# ==========================================
FINAL_STAGE1 = "tcn"
FINAL_STAGE2 = "unet_bigru"
FINAL_CONFIG_TEXT = "TCN -> U-Net+BiGRU -> competition-aware decoding -> island300"
TIMEZONE = "Europe/Berlin"
BATTERY_GAP_SEC = 600.0  # same rule as end2end_metrics/battery_detection_10min.py

CLASS_NAMES = {
    1: "2×10MWT Preferred",
    2: "2×10MWT Fast",
    3: "2×10MWT Slow",
    4: "2MWT",
}
CLASS_SHORT = {1: "Pref", 2: "Fast", 3: "Slow", 4: "2MWT"}

AXIS_OPTIONS = [
    ("Right gyro Y (recommended)", "right_sensor_gyr_y"),
    ("Right gyro X", "right_sensor_gyr_x"),
    ("Right gyro Z", "right_sensor_gyr_z"),
    ("Right accel X", "right_sensor_acc_x"),
    ("Right accel Y", "right_sensor_acc_y"),
    ("Right accel Z", "right_sensor_acc_z"),
    ("Left gyro Y", "left_sensor_gyr_y"),
    ("Left gyro X", "left_sensor_gyr_x"),
    ("Left gyro Z", "left_sensor_gyr_z"),
    ("Left accel X", "left_sensor_acc_x"),
    ("Left accel Y", "left_sensor_acc_y"),
    ("Left accel Z", "left_sensor_acc_z"),
]

# Progress (percent, label) reported when each pipeline stage starts.
STAGE_PROGRESS = {
    "loading": (5, "Loading recording..."),
    "preprocessing": (15, "Preprocessing: Ullrich gait filter and 90 s windows..."),
    "stage1": (35, "Stage 1: TCN candidate detection..."),
    "continuity": (50, "Applying continuity rule..."),
    "stage2": (60, "Stage 2: U-Net+BiGRU segmentation..."),
    "postprocessing": (90, "Decoding events..."),
}

STATUS_MESSAGES = {
    "ok": "Analysis completed.",
    "missing_signal_path": "The selected file could not be found.",
    "failed_to_read_signal": "The HDF5 signal could not be read.",
    "no_valid_windows": "No valid candidate windows were generated during preprocessing.",
    "no_stage1_positive_windows": "Stage 1 did not forward any candidate gait windows.",
    "no_windows_after_continuity": "No windows remained after the continuity rule.",
    "no_events_after_postprocessing": "The pipeline finished but produced no gait-test events.",
}


class AnalysisCancelled(Exception):
    """Raised from the progress callback to stop the pipeline between stages."""


# ==========================================
# Helpers
# ==========================================
def format_time(ns, with_date=False):
    ts = pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(TIMEZONE)
    return ts.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def group_into_batteries(events, gap_sec=BATTERY_GAP_SEC):
    """Group chronologically sorted events; a gap above `gap_sec` starts a new battery."""
    gap_ns = int(gap_sec * 1e9)
    ordered = sorted(events, key=lambda e: (e["timestamp_start_ns"], e["timestamp_end_ns"]))
    groups = []
    for event in ordered:
        if not groups or event["timestamp_start_ns"] - groups[-1][-1]["timestamp_end_ns"] > gap_ns:
            groups.append([event])
        else:
            groups[-1].append(event)

    batteries = []
    for members in groups:
        class_ids = [int(e["class_id"]) for e in members]
        start_ns = min(e["timestamp_start_ns"] for e in members)
        end_ns = max(e["timestamp_end_ns"] for e in members)
        batteries.append({
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
            "duration_sec": (end_ns - start_ns) / 1e9,
            "class_ids": class_ids,
            "complete": Counter(class_ids) == Counter({1: 1, 2: 1, 3: 1, 4: 1}),
        })
    return batteries


def to_jsonable(value):
    """json.dump fallback for numpy scalars and arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# ==========================================
# GUI
# ==========================================
class GaitDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gait Detection Analysis System")
        self._fit_window()
        self._setup_style()

        self.file_path = None
        self.signal = None
        self.results = None
        self.batteries = []
        self.worker = None
        self.cancel_event = threading.Event()
        self.messages = queue.Queue()
        self.sort_state = {}

        self.axis_var = tk.StringVar(value=AXIS_OPTIONS[0][0])
        self.advanced_var = tk.BooleanVar(value=False)
        self.stage1_model_var = tk.StringVar(value=FINAL_STAGE1)
        self.stage1_use_author_var = tk.BooleanVar(value=True)
        self.stage2_model_var = tk.StringVar(value=FINAL_STAGE2)
        self.stage2_use_author_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Select an HDF5 recording to start.")
        self.step_var = tk.StringVar(value="")
        self.progress_var = tk.IntVar(value=0)

        for var in (self.stage1_model_var, self.stage1_use_author_var,
                    self.stage2_model_var, self.stage2_use_author_var):
            var.trace_add("write", lambda *_: self._update_config_note())

        self._build()
        self._set_mode("no_file")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not PIPELINE_AVAILABLE:
            self.status_var.set("Pipeline modules could not be imported.")
            messagebox.showerror(
                "Pipeline unavailable",
                f"Could not import the pipeline modules:\n{PIPELINE_IMPORT_ERROR}\n\n"
                "Install the dependencies with: pip install -r requirements.txt",
            )
            self.select_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Window and style
    # ------------------------------------------------------------------
    def _fit_window(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        # Tk on Linux does not follow desktop scaling; scale relative to a 1920 px wide screen.
        self.ui_scale = max(1.0, min(screen_w / 1920, screen_h / 1080))
        self.root.tk.call("tk", "scaling", 96 / 72 * self.ui_scale)
        width = min(int(1400 * self.ui_scale), int(screen_w * 0.9))
        height = min(int(900 * self.ui_scale), int(screen_h * 0.85))
        self.root.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{(screen_h - height) // 3}")
        self.root.minsize(min(int(980 * self.ui_scale), screen_w), min(int(620 * self.ui_scale), screen_h))

    def px(self, value):
        return int(value * self.ui_scale)

    def _setup_style(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=11)
        tkfont.nametofont("TkTextFont").configure(size=11)
        family = base.actual("family")
        self.font_title = tkfont.Font(family=family, size=17, weight="bold")
        self.font_section = tkfont.Font(family=family, size=12, weight="bold")
        self.font_small = tkfont.Font(family=family, size=10)
        self.font_mono = tkfont.Font(family="Courier", size=10)

        style.configure("Title.TLabel", font=self.font_title)
        style.configure("Muted.TLabel", foreground="#5b6470", font=self.font_small)
        style.configure("Warn.TLabel", foreground="#9a5b00", font=self.font_small)
        style.configure("Section.TLabelframe.Label", font=self.font_section)
        style.configure("Primary.TButton", font=self.font_section, padding=(10, 8))
        style.configure("TButton", padding=(8, 5))
        style.configure("Treeview", rowheight=int(base.metrics("linespace") * 1.6))
        style.configure("Treeview.Heading", font=self.font_section)
        style.configure("TNotebook.Tab", padding=(12, 5))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="Gait Detection Analysis System", style="Title.TLabel").pack(anchor="w")
        subtitle = ttk.Label(
            header,
            text=f"Offline gait-test detection · final configuration: {FINAL_CONFIG_TEXT}",
            style="Muted.TLabel",
        )
        subtitle.pack(anchor="w", fill=tk.X)
        subtitle.bind("<Configure>", lambda e: subtitle.config(wraplength=max(e.width - 4, 100)))

        left = ttk.Frame(outer, width=self.px(330))
        left.grid(row=1, column=0, sticky="nsw", padx=(0, 12))
        self._build_controls(left)

        right = ttk.Frame(outer)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_results(right)

        status_bar = ttk.Frame(outer)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=tk.LEFT)

    def _build_controls(self, parent):
        # Recording
        file_box = ttk.LabelFrame(parent, text=" Recording ", style="Section.TLabelframe", padding=10)
        file_box.pack(fill=tk.X, pady=(0, 10))
        self.select_btn = ttk.Button(file_box, text="Select HDF5 file...", command=self.select_file)
        self.select_btn.pack(fill=tk.X)
        self.file_label = ttk.Label(file_box, text="No file selected", style="Muted.TLabel", wraplength=self.px(290))
        self.file_label.pack(anchor="w", pady=(8, 0))

        # Plot channel
        axis_box = ttk.LabelFrame(parent, text=" Plot channel ", style="Section.TLabelframe", padding=10)
        axis_box.pack(fill=tk.X, pady=(0, 10))
        self.axis_combo = ttk.Combobox(
            axis_box, textvariable=self.axis_var, values=[label for label, _ in AXIS_OPTIONS], state="readonly"
        )
        self.axis_combo.pack(fill=tk.X)
        ttk.Label(axis_box, text="Used only for plots; detection uses all 12 channels.",
                  style="Muted.TLabel", wraplength=self.px(290)).pack(anchor="w", pady=(6, 0))

        # Models
        model_box = ttk.LabelFrame(parent, text=" Models ", style="Section.TLabelframe", padding=10)
        model_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(model_box, text="Final thesis models with bundled weights.", wraplength=self.px(290)).pack(anchor="w")
        self.advanced_chk = ttk.Checkbutton(
            model_box, text="Show advanced options", variable=self.advanced_var, command=self._toggle_advanced
        )
        self.advanced_chk.pack(anchor="w", pady=(6, 0))

        self.advanced_frame = ttk.Frame(model_box)
        self.advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(self.advanced_frame, text="Stage 1").grid(row=0, column=0, sticky="w", pady=(8, 2))
        self.stage1_combo = ttk.Combobox(self.advanced_frame, textvariable=self.stage1_model_var,
                                         values=["tcn", "tcn_bilstm"], state="readonly", width=14)
        self.stage1_combo.grid(row=0, column=1, sticky="ew", pady=(8, 2))
        self.s1_chk = ttk.Checkbutton(self.advanced_frame, text="Bundled weights",
                                      variable=self.stage1_use_author_var)
        self.s1_chk.grid(row=1, column=1, sticky="w")
        ttk.Label(self.advanced_frame, text="Stage 2").grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.stage2_combo = ttk.Combobox(self.advanced_frame, textvariable=self.stage2_model_var,
                                         values=["unet_bigru", "unet_att_gru"], state="readonly", width=14)
        self.stage2_combo.grid(row=2, column=1, sticky="ew", pady=(8, 2))
        self.s2_chk = ttk.Checkbutton(self.advanced_frame, text="Bundled weights",
                                      variable=self.stage2_use_author_var)
        self.s2_chk.grid(row=3, column=1, sticky="w")
        self.config_note = ttk.Label(self.advanced_frame, text="", style="Warn.TLabel", wraplength=self.px(290))
        self.config_note.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Actions
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(0, 10))
        actions.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(actions, text="Start analysis", style="Primary.TButton",
                                    command=self.start_analysis)
        self.start_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self.cancel_analysis)
        self.cancel_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 3))
        self.reset_btn = ttk.Button(actions, text="Reset", command=self.reset)
        self.reset_btn.grid(row=1, column=1, sticky="ew", pady=(6, 0), padx=(3, 0))

        # Progress
        progress_box = ttk.LabelFrame(parent, text=" Progress ", style="Section.TLabelframe", padding=10)
        progress_box.pack(fill=tk.X)
        ttk.Progressbar(progress_box, variable=self.progress_var, maximum=100).pack(fill=tk.X)
        ttk.Label(progress_box, textvariable=self.step_var, style="Muted.TLabel",
                  wraplength=self.px(290)).pack(anchor="w", pady=(6, 0))

    def _build_results(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Overview
        overview = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(overview, text="Overview")
        overview.columnconfigure(1, weight=1)
        self.overview_rows = {}
        for i, key in enumerate(["File", "Recording", "Status", "Models", "Events", "Batteries"]):
            ttk.Label(overview, text=key, style="Muted.TLabel").grid(row=i, column=0, sticky="nw", padx=(0, 16), pady=3)
            value = ttk.Label(overview, text="-", wraplength=self.px(700), justify="left")
            value.grid(row=i, column=1, sticky="nw", pady=3)
            self.overview_rows[key] = value

        ttk.Separator(overview).grid(row=6, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(overview, text="Events per class", font=self.font_section).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.class_rows = {}
        for i, cid in enumerate(CLASS_NAMES):
            ttk.Label(overview, text=CLASS_NAMES[cid]).grid(row=8 + i, column=0, sticky="w", pady=2)
            value = ttk.Label(overview, text="-")
            value.grid(row=8 + i, column=1, sticky="w", pady=2)
            self.class_rows[cid] = value

        # Events
        events_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(events_tab, text="Events")
        self.events_tree = self._make_tree(events_tab, [
            ("idx", "#", 50, "e"),
            ("class", "Test", 170, "w"),
            ("start", "Start", 100, "center"),
            ("end", "End", 100, "center"),
            ("duration", "Duration (s)", 110, "e"),
            ("score", "Confidence", 110, "e"),
            ("battery", "Battery", 80, "e"),
        ])

        # Batteries
        battery_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(battery_tab, text="Batteries")
        ttk.Label(
            battery_tab,
            text=f"Events closer than {BATTERY_GAP_SEC:.0f} s form one battery. "
                 "Complete = each of the four tests exactly once (order not checked).",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        self.battery_tree = self._make_tree(battery_tab, [
            ("idx", "#", 50, "e"),
            ("start", "Start", 100, "center"),
            ("end", "End", 100, "center"),
            ("duration", "Duration (s)", 110, "e"),
            ("count", "Tests", 70, "e"),
            ("sequence", "Sequence", 260, "w"),
            ("complete", "Complete", 90, "center"),
        ])
        self.battery_tree.tag_configure("incomplete", foreground="#9a5b00")

        # Raw JSON
        raw_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(raw_tab, text="Raw JSON")
        self.raw_text = scrolledtext.ScrolledText(raw_tab, wrap=tk.NONE, font=self.font_mono, state=tk.DISABLED)
        self.raw_text.pack(fill=tk.BOTH, expand=True)

        # Output actions
        out = ttk.Frame(parent)
        out.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.plot_btn = ttk.Button(out, text="Plot signal with events", command=self.visualize_results)
        self.plot_btn.pack(side=tk.LEFT)
        self.export_json_btn = ttk.Button(out, text="Export JSON...", command=self.export_json)
        self.export_json_btn.pack(side=tk.RIGHT)
        self.export_csv_btn = ttk.Button(out, text="Export CSV...", command=self.export_csv)
        self.export_csv_btn.pack(side=tk.RIGHT, padx=(0, 6))

    def _make_tree(self, parent, columns):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", selectmode="browse")
        for key, title, width, anchor in columns:
            tree.heading(key, text=title, command=lambda k=key, t=tree: self._sort_tree(t, k))
            tree.column(key, width=self.px(width), anchor=anchor, stretch=(anchor == "w"))
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        return tree

    def _sort_tree(self, tree, column):
        reverse = self.sort_state.get((str(tree), column), False)
        rows = [(tree.set(item, column), item) for item in tree.get_children("")]

        def key(value):
            try:
                return (0, float(value.rstrip("%")))
            except ValueError:
                return (1, value)

        rows.sort(key=lambda row: key(row[0]), reverse=reverse)
        for index, (_, item) in enumerate(rows):
            tree.move(item, "", index)
        self.sort_state[(str(tree), column)] = not reverse

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _set_mode(self, mode):
        """mode: no_file | ready | running | done"""
        running = mode == "running"
        has_results = self.results is not None and not running
        has_events = has_results and bool(self.results["events"])

        def state(enabled):
            return tk.NORMAL if enabled else tk.DISABLED

        self.select_btn.config(state=state(not running and PIPELINE_AVAILABLE))
        self.start_btn.config(state=state(mode in ("ready", "done") and PIPELINE_AVAILABLE))
        self.cancel_btn.config(state=state(running))
        self.reset_btn.config(state=state(not running))
        self.plot_btn.config(state=state(has_events and self.signal is not None))
        self.export_json_btn.config(state=state(has_results))
        self.export_csv_btn.config(state=state(has_events))
        combo_state = tk.DISABLED if running else "readonly"
        for widget in (self.axis_combo, self.stage1_combo, self.stage2_combo):
            widget.config(state=combo_state)
        for widget in (self.s1_chk, self.s2_chk, self.advanced_chk):
            widget.config(state=state(not running))

    def _toggle_advanced(self):
        if self.advanced_var.get():
            self.advanced_frame.pack(fill=tk.X)
        else:
            self.advanced_frame.pack_forget()
            self.stage1_model_var.set(FINAL_STAGE1)
            self.stage2_model_var.set(FINAL_STAGE2)
            self.stage1_use_author_var.set(True)
            self.stage2_use_author_var.set(True)
        self._update_config_note()

    def _is_final_config(self):
        return (self.stage1_model_var.get() == FINAL_STAGE1 and self.stage2_model_var.get() == FINAL_STAGE2
                and self.stage1_use_author_var.get() and self.stage2_use_author_var.get())

    def _update_config_note(self):
        if self._is_final_config():
            self.config_note.config(text="")
        else:
            self.config_note.config(text="Not the final thesis configuration: results will differ from the reported ones.")

    def _axis_column(self):
        return dict(AXIS_OPTIONS).get(self.axis_var.get(), AXIS_OPTIONS[0][1])

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def select_file(self):
        path = filedialog.askopenfilename(
            title="Select HDF5 recording",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return
        self.file_path = path
        self.signal = None
        self.results = None
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self.file_label.config(text=f"{os.path.basename(path)}\n{size_mb:.1f} MB")
        self._clear_results()
        self.progress_var.set(0)
        self.step_var.set("")
        self.status_var.set("Ready. Click 'Start analysis'.")
        self._set_mode("ready")

    def start_analysis(self):
        if not self.file_path or (self.worker and self.worker.is_alive()):
            return
        config = {
            "file_path": self.file_path,
            "axis": self._axis_column(),
            "s1_backbone": self.stage1_model_var.get(),
            "s1_use_author": self.stage1_use_author_var.get(),
            "s2_model": self.stage2_model_var.get(),
            "s2_use_author": self.stage2_use_author_var.get(),
        }
        self.results = None
        self.signal = None
        self._clear_results()
        self.cancel_event.clear()
        self.progress_var.set(0)
        self.status_var.set("Analysing...")
        self._set_mode("running")
        self.worker = threading.Thread(target=self._run_pipeline, args=(config,), daemon=True)
        self.worker.start()
        self.root.after(100, self._poll_messages)

    def cancel_analysis(self):
        self.cancel_event.set()
        self.cancel_btn.config(state=tk.DISABLED)
        self.step_var.set("Cancelling after the current step...")

    def _run_pipeline(self, config):
        """Worker thread: never touches Tk; reports through self.messages."""

        def report(stage):
            if self.cancel_event.is_set():
                raise AnalysisCancelled()
            self.messages.put(("progress",) + STAGE_PROGRESS[stage])

        try:
            report("loading")
            signal = pd.read_hdf(config["file_path"])
            output = detect_gait_test(
                signal_path=config["file_path"],
                signal=signal,
                axis_name=config["axis"],
                stage1_backbone=config["s1_backbone"],
                stage2_model=config["s2_model"],
                plot_signal=False,
                use_author_weights_stage1=config["s1_use_author"],
                use_author_weights_stage2=config["s2_use_author"],
                return_debug=True,
                progress_callback=report,
            )
            if self.cancel_event.is_set():
                raise AnalysisCancelled()
            if output is None:
                raise RuntimeError("The pipeline returned no output.")
            self.messages.put(("done", output, signal, config))
        except AnalysisCancelled:
            self.messages.put(("cancelled",))
        except Exception as exc:  # reported to the user in the main thread
            self.messages.put(("error", f"{type(exc).__name__}: {exc}"))

    def _poll_messages(self):
        finished = False
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "progress":
                self.progress_var.set(message[1])
                self.step_var.set(message[2])
            elif kind == "done":
                self._on_done(*message[1:])
                finished = True
            elif kind == "cancelled":
                self.progress_var.set(0)
                self.step_var.set("Analysis cancelled.")
                self.status_var.set("Analysis cancelled.")
                self._set_mode("ready")
                finished = True
            elif kind == "error":
                self.progress_var.set(0)
                self.step_var.set(message[1])
                self.status_var.set("Analysis failed.")
                self._set_mode("ready")
                messagebox.showerror("Analysis failed", message[1])
                finished = True
        if not finished:
            self.root.after(100, self._poll_messages)

    def _on_done(self, output, signal, config):
        status = output.get("debug", {}).get("status", "unknown")
        events = sorted(output.get("events", []), key=lambda e: e["timestamp_start_ns"])
        self.signal = signal
        self.batteries = group_into_batteries(events)
        self.results = {
            "file": os.path.basename(config["file_path"]),
            "configuration": FINAL_CONFIG_TEXT if self._config_is_final(config) else "custom (not the final thesis configuration)",
            "stage1": f"{config['s1_backbone']} ({'bundled' if config['s1_use_author'] else 'own'} weights)",
            "stage2": f"{config['s2_model']} ({'bundled' if config['s2_use_author'] else 'own'} weights)",
            "timezone": TIMEZONE,
            "analysed_at": datetime.now().isoformat(timespec="seconds"),
            "pipeline_status": status,
            "status_message": STATUS_MESSAGES.get(status, f"Pipeline finished with status: {status}"),
            "events": events,
            "batteries": self.batteries,
        }
        self.progress_var.set(100)
        self.step_var.set(self.results["status_message"])
        self.status_var.set(self.results["status_message"])
        self._show_results()
        self._set_mode("done")

    @staticmethod
    def _config_is_final(config):
        return (config["s1_backbone"] == FINAL_STAGE1 and config["s2_model"] == FINAL_STAGE2
                and config["s1_use_author"] and config["s2_use_author"])

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------
    def _battery_index_by_event(self):
        index = {}
        for b_idx, battery in enumerate(self.batteries, 1):
            for event in self.results["events"]:
                if battery["start_ns"] <= event["timestamp_start_ns"] and event["timestamp_end_ns"] <= battery["end_ns"]:
                    index.setdefault(id(event), b_idx)
        return index

    def _show_results(self):
        events = self.results["events"]
        rows = self.overview_rows
        rows["File"].config(text=self.results["file"])
        if self.signal is not None and len(self.signal.index):
            rows["Recording"].config(text=f"{format_time(self.signal.index[0], True)} -> "
                                          f"{format_time(self.signal.index[-1], True)} ({TIMEZONE})")
        rows["Status"].config(text=self.results["status_message"])
        rows["Models"].config(text=f"{self.results['stage1']} · {self.results['stage2']}\n{self.results['configuration']}")
        total_sec = sum((e["timestamp_end_ns"] - e["timestamp_start_ns"]) / 1e9 for e in events)
        rows["Events"].config(text=f"{len(events)} events, {total_sec / 60:.1f} min in total" if events else "No events detected")
        complete = sum(b["complete"] for b in self.batteries)
        rows["Batteries"].config(text=f"{len(self.batteries)} batteries, {complete} complete" if self.batteries else "-")
        counts = Counter(int(e["class_id"]) for e in events)
        for cid, label in self.class_rows.items():
            label.config(text=str(counts.get(cid, 0)))

        battery_of = self._battery_index_by_event()
        for i, event in enumerate(events, 1):
            duration = (event["timestamp_end_ns"] - event["timestamp_start_ns"]) / 1e9
            self.events_tree.insert("", tk.END, values=(
                i,
                CLASS_NAMES.get(int(event["class_id"]), f"Class {event['class_id']}"),
                format_time(event["timestamp_start_ns"]),
                format_time(event["timestamp_end_ns"]),
                f"{duration:.1f}",
                f"{float(event.get('score', 0.0)) * 100:.1f}%",
                battery_of.get(id(event), ""),
            ))

        for i, battery in enumerate(self.batteries, 1):
            self.battery_tree.insert("", tk.END, values=(
                i,
                format_time(battery["start_ns"]),
                format_time(battery["end_ns"]),
                f"{battery['duration_sec']:.0f}",
                len(battery["class_ids"]),
                " -> ".join(CLASS_SHORT.get(c, str(c)) for c in battery["class_ids"]),
                "yes" if battery["complete"] else "no",
            ), tags=() if battery["complete"] else ("incomplete",))

        self.raw_text.config(state=tk.NORMAL)
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert("1.0", json.dumps(self.results, indent=2, ensure_ascii=False, default=to_jsonable))
        self.raw_text.config(state=tk.DISABLED)

    def _clear_results(self):
        self.batteries = []
        for label in self.overview_rows.values():
            label.config(text="-")
        for label in self.class_rows.values():
            label.config(text="-")
        for tree in (self.events_tree, self.battery_tree):
            tree.delete(*tree.get_children(""))
        self.raw_text.config(state=tk.NORMAL)
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def visualize_results(self):
        if self.signal is None or not self.results or not self.results["events"]:
            return
        try:
            self.status_var.set("Plotting... close the plot windows to continue.")
            self.root.update_idletasks()
            plot_detected_events(self.signal, self.results["events"], axis_name=self._axis_column())
            self.status_var.set(self.results["status_message"])
        except Exception as exc:
            messagebox.showerror("Plot failed", str(exc))

    def _default_name(self, extension):
        stem = Path(self.results["file"]).stem if self.results else "gait_analysis"
        return f"{stem}_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

    def export_json(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(title="Export results as JSON", defaultextension=".json",
                                            filetypes=[("JSON files", "*.json")],
                                            initialfile=self._default_name("json"))
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=to_jsonable)
            self.status_var.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_csv(self):
        if not self.results or not self.results["events"]:
            return
        path = filedialog.asksaveasfilename(title="Export events as CSV", defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")],
                                            initialfile=self._default_name("csv"))
        if not path:
            return
        battery_of = self._battery_index_by_event()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["event", "class_id", "test", "start_ns", "end_ns", f"start_{TIMEZONE}",
                                 f"end_{TIMEZONE}", "duration_sec", "score", "battery"])
                for i, e in enumerate(self.results["events"], 1):
                    writer.writerow([
                        i, int(e["class_id"]), CLASS_NAMES.get(int(e["class_id"]), ""),
                        int(e["timestamp_start_ns"]), int(e["timestamp_end_ns"]),
                        format_time(e["timestamp_start_ns"], True), format_time(e["timestamp_end_ns"], True),
                        f"{(e['timestamp_end_ns'] - e['timestamp_start_ns']) / 1e9:.2f}",
                        f"{float(e.get('score', 0.0)):.4f}", battery_of.get(id(e), ""),
                    ])
            self.status_var.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def reset(self):
        if self.worker and self.worker.is_alive():
            return
        self.file_path = None
        self.signal = None
        self.results = None
        self.file_label.config(text="No file selected")
        self.progress_var.set(0)
        self.step_var.set("")
        self.status_var.set("Select an HDF5 recording to start.")
        self._clear_results()
        self._set_mode("no_file")

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Analysis running", "An analysis is still running. Quit anyway?"):
                return
            self.cancel_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    GaitDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
