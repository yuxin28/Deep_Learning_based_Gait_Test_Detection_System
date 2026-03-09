#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gait Detection Analysis System - Offline Tkinter GUI
Runs completely offline, no internet connection required.
Updated: 
1. Layout adjusted: 'Author Weights' checkbox moved to new line (vertical layout).
2. Full 12-Channel Visualization Support.
3. Large Fonts for HiDPI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# ==========================================
# Path Configuration
# ==========================================
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
if str(current_dir / 'pipeline') not in sys.path:
    sys.path.append(str(current_dir / 'pipeline'))

# ==========================================
# Import Pipeline Modules
# ==========================================
PIPELINE_AVAILABLE = False
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    from preprocessing import process_single_file
    from stage1 import detect_gait_sequences
    from continuity_rule import apply_continuity_rule
    from stage2 import segment_gait_test
    from postprocessing import run_postprocessing
    from plot_detected_events import plot_detected_events
    
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pipeline modules: {e}")
    print("Ensure preprocessing.py, stage1.py, stage2.py, etc., are in the python path.")


class GaitDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gait Detection Analysis System v2.5")
        self.root.geometry("1600x1100") 
        
        # Setup styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors/fonts (Large Fonts)
        self.style.configure('Title.TLabel', font=('Arial', 30, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 16))
        self.style.configure('Header.TLabel', font=('Arial', 18, 'bold'))
        self.style.configure('TNotebook.Tab', font=('Arial', 14, 'bold'), padding=[15, 8])
        
        # Variables
        self.file_path = None
        self.axis_var = tk.StringVar(value='right_sensor_gyr_y')
        
        # --- Model Configuration Variables ---
        self.stage1_model_var = tk.StringVar(value='tcn')
        self.stage1_use_author_var = tk.BooleanVar(value=True)
        
        self.stage2_model_var = tk.StringVar(value='unet_bigru')
        self.stage2_use_author_var = tk.BooleanVar(value=True)
        # -------------------------------------
        
        self.status_var = tk.StringVar(value='Ready')
        self.progress_var = tk.IntVar(value=0)
        self.current_step_var = tk.StringVar(value='')
        
        # Data storage
        self.results = None
        self.current_signal = None 
        self.is_processing = False
        
        # Class Name Mapping
        self.class_names = {
            1: '2x10MWT Preferred Walk',
            2: '2x10MWT Fast Walk',
            3: '2x10MWT Slow Walk',
            4: '2-Minute Walk Test'
        }
        
        # Create Interface
        self.create_widgets()
        
        if not PIPELINE_AVAILABLE:
            self.show_warning("Pipeline modules unavailable. Check console for ImportErrors.")
            self.status_var.set("Error: Pipeline Missing")
            self.start_btn.config(state=tk.DISABLED)
    
    def create_widgets(self):
        """Create GUI widgets"""
        
        # ====================
        # Top Header
        # ====================
        header_frame = tk.Frame(self.root, bg='#667eea', height=160) 
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="Gait Detection Analysis System",
            font=('Arial', 32, 'bold'), 
            bg='#667eea',
            fg='white'
        )
        title_label.pack(pady=(20, 10))
        
        subtitle_label = tk.Label(
            header_frame,
            text="Deep Learning-based Automated Gait Test Detection and Segmentation",
            font=('Arial', 16), 
            bg='#667eea',
            fg='white'
        )
        subtitle_label.pack()
        
        status_label = tk.Label(
            header_frame,
            textvariable=self.status_var,
            font=('Arial', 14, 'italic'), 
            bg='#667eea',
            fg='white'
        )
        status_label.pack(pady=10)
        
        # ====================
        # Main Container
        # ====================
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel (Widened)
        left_frame = tk.Frame(main_container, width=500) 
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_frame.pack_propagate(False)
        
        # Right Panel
        right_frame = tk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # ====================
        # Left: File Upload Area
        # ====================
        file_frame = tk.LabelFrame(
            left_frame,
            text=" Data File",
            font=('Arial', 18, 'bold'), 
            padx=20,
            pady=20
        )
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.select_file_btn = tk.Button(
            file_frame,
            text="Select HDF5 File",
            command=self.select_file,
            font=('Arial', 20), 
            bg='#667eea',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor='hand2'
        )
        self.select_file_btn.pack(fill=tk.X)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=('Arial', 13), 
            fg='#666',
            wraplength=450
        )
        self.file_label.pack(pady=(15, 0))
        
        # ====================
        # Left: Sensor Axis Selection (UPDATED: ALL 12 CHANNELS)
        # ====================
        axis_frame = tk.LabelFrame(
            left_frame,
            text=" Visualization Axis",
            font=('Arial', 18, 'bold'),
            padx=20,
            pady=20
        )
        axis_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            axis_frame,
            text="Select Axis for Plotting:",
            font=('Arial', 18)
        ).pack(anchor=tk.W)
        
        # --- Updated List: All 12 Channels ---
        axis_options = [
            # Right Sensor
            ('Right Gyro Y (Recommended)', 'right_sensor_gyr_y'),
            ('Right Gyro X', 'right_sensor_gyr_x'),
            ('Right Gyro Z', 'right_sensor_gyr_z'),
            ('Right Accel X', 'right_sensor_acc_x'),
            ('Right Accel Y', 'right_sensor_acc_y'),
            ('Right Accel Z', 'right_sensor_acc_z'),
            
            # Left Sensor
            ('Left Gyro Y', 'left_sensor_gyr_y'),
            ('Left Gyro X', 'left_sensor_gyr_x'),
            ('Left Gyro Z', 'left_sensor_gyr_z'),
            ('Left Accel X', 'left_sensor_acc_x'),
            ('Left Accel Y', 'left_sensor_acc_y'),
            ('Left Accel Z', 'left_sensor_acc_z'),
        ]
        
        self.axis_combo = ttk.Combobox(
            axis_frame,
            textvariable=self.axis_var,
            values=[opt[1] for opt in axis_options],
            state='readonly',
            font=('Arial', 20)
        )
        self.axis_combo.pack(fill=tk.X, pady=(10, 0))
        self.axis_combo.current(0) # Default to Right Gyro Y

        # ====================
        # Left: Model Configuration (Layout Updated: Vertical)
        # ====================
        model_frame = tk.LabelFrame(
            left_frame,
            text=" Model Settings",
            font=('Arial', 18, 'bold'),
            padx=20,
            pady=20
        )
        model_frame.pack(fill=tk.X, pady=(0, 20))

        # --- Stage 1 Group ---
        s1_frame = tk.Frame(model_frame)
        s1_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Row 1: Label + Dropdown
        s1_row1 = tk.Frame(s1_frame)
        s1_row1.pack(fill=tk.X, anchor='w')
        
        tk.Label(s1_row1, text="Stage 1:", font=('Arial', 14, 'bold'), width=8, anchor='w').pack(side=tk.LEFT)
        self.stage1_combo = ttk.Combobox(
            s1_row1,
            textvariable=self.stage1_model_var,
            values=['tcn', 'tcn_bilstm'], 
            state='readonly',
            font=('Arial', 13),
            width=14
        )
        self.stage1_combo.pack(side=tk.LEFT, padx=10)
        
        # Row 2: Checkbox (Next Line)
        self.s1_chk = tk.Checkbutton(
            s1_frame, 
            text="Use Author Weights", 
            variable=self.stage1_use_author_var,
            font=('Arial', 12)
        )
        self.s1_chk.pack(anchor='w', pady=(5, 0)) # Aligned Left, below row 1

        # --- Stage 2 Group ---
        s2_frame = tk.Frame(model_frame)
        s2_frame.pack(fill=tk.X)
        
        # Row 1: Label + Dropdown
        s2_row1 = tk.Frame(s2_frame)
        s2_row1.pack(fill=tk.X, anchor='w')
        
        tk.Label(s2_row1, text="Stage 2:", font=('Arial', 14, 'bold'), width=8, anchor='w').pack(side=tk.LEFT)
        self.stage2_combo = ttk.Combobox(
            s2_row1,
            textvariable=self.stage2_model_var,
            values=['unet_bigru', 'unet_att_gru'], 
            state='readonly',
            font=('Arial', 13),
            width=14
        )
        self.stage2_combo.pack(side=tk.LEFT, padx=10)
        
        # Row 2: Checkbox (Next Line)
        self.s2_chk = tk.Checkbutton(
            s2_frame, 
            text="Use Author Weights", 
            variable=self.stage2_use_author_var,
            font=('Arial', 12)
        )
        self.s2_chk.pack(anchor='w', pady=(5, 0)) # Aligned Left, below row 1
        
        # ====================
        # Left: Action Buttons
        # ====================
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_btn = tk.Button(
            button_frame,
            text="Start Analysis",
            command=self.start_analysis,
            font=('Arial', 18, 'bold'), 
            bg='#10b981',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=15,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 15))

        self.plot_btn = tk.Button(
            button_frame,
            text="Visualize Signals",
            command=self.visualize_results,
            font=('Arial', 15, 'bold'), 
            bg='#3b82f6',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=12,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.plot_btn.pack(fill=tk.X, pady=(0, 15))
        
        reset_btn = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            font=('Arial', 20),
            bg='#e5e7eb',
            fg='#374151',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        reset_btn.pack(fill=tk.X)
        
        # ====================
        # Left: Progress Bar
        # ====================
        progress_frame = tk.LabelFrame(
            left_frame,
            text=" Processing Progress",
            font=('Arial', 18, 'bold'),
            padx=20,
            pady=20
        )
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))
        
        self.progress_label = tk.Label(
            progress_frame,
            textvariable=self.current_step_var,
            font=('Arial', 13),
            fg='#666'
        )
        self.progress_label.pack(pady=(5, 0))
        
        # ====================
        # Right: Results Display Area
        # ====================
        results_frame = tk.LabelFrame(
            right_frame,
            text=" Analysis Results",
            font=('Arial', 20, 'bold'), 
            padx=20,
            pady=20
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Notebook (Tabs)
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Overview
        overview_frame = tk.Frame(self.notebook)
        self.notebook.add(overview_frame, text=' Overview ')
        
        self.overview_text = scrolledtext.ScrolledText(
            overview_frame, wrap=tk.WORD, font=('Arial', 20), state=tk.DISABLED
        )
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 2: Detailed Events
        events_frame = tk.Frame(self.notebook)
        self.notebook.add(events_frame, text=' Detailed Events ')
        
        self.events_text = scrolledtext.ScrolledText(
            events_frame, wrap=tk.WORD, font=('Courier', 14), state=tk.DISABLED
        )
        self.events_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Raw Data
        raw_frame = tk.Frame(self.notebook)
        self.notebook.add(raw_frame, text=' Raw Data ')
        
        self.raw_text = scrolledtext.ScrolledText(
            raw_frame, wrap=tk.WORD, font=('Courier', 14), state=tk.DISABLED
        )
        self.raw_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export Button
        export_btn = tk.Button(
            results_frame,
            text="Export Results (JSON)",
            command=self.export_results,
            font=('Arial', 20),
            bg='#667eea',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor='hand2'
        )
        export_btn.pack(pady=(20, 0))
        
        self.show_empty_state()
    
    def select_file(self):
        """Select file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select HDF5 File",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
        )
        
        if file_path:
            self.file_path = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            
            self.file_label.config(
                text=f"[Selected] {file_name}\nSize: {file_size:.2f} MB",
                fg='#10b981'
            )
            
            if PIPELINE_AVAILABLE:
                self.start_btn.config(state=tk.NORMAL)
            self.status_var.set("Ready")
            self.results = None
            self.plot_btn.config(state=tk.DISABLED)
    
    def start_analysis(self):
        """Start the analysis process"""
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        
        if self.is_processing:
            return
        
        # --- Capture UI values HERE ---
        config = {
            's1_backbone': self.stage1_model_var.get(),
            's1_use_author': self.stage1_use_author_var.get(),
            's2_model': self.stage2_model_var.get(),
            's2_use_author': self.stage2_use_author_var.get(),
            'axis': self.axis_var.get(),
            'file_path': self.file_path
        }

        # UI updates: Disable controls
        self.start_btn.config(state=tk.DISABLED)
        self.select_file_btn.config(state=tk.DISABLED)
        self.plot_btn.config(state=tk.DISABLED)
        self.axis_combo.config(state=tk.DISABLED)
        
        self.stage1_combo.config(state=tk.DISABLED)
        self.s1_chk.config(state=tk.DISABLED)
        
        self.stage2_combo.config(state=tk.DISABLED)
        self.s2_chk.config(state=tk.DISABLED)
        
        self.is_processing = True
        self.status_var.set("Processing...")
        
        # Run in thread
        thread = threading.Thread(target=self.process_file, args=(config,))
        thread.daemon = True
        thread.start()
    
    def process_file(self, config):
        """Executes the gait detection pipeline."""
        try:
            s1_backbone = config['s1_backbone']
            s1_use_author = config['s1_use_author']
            s2_model = config['s2_model']
            s2_use_author = config['s2_use_author']
            fpath = config['file_path']

            self.update_progress(5, "Loading HDF5 Signal...")
            signal = pd.read_hdf(fpath)
            self.current_signal = signal 
            
            self.update_progress(20, "Preprocessing...")
            wins, ts = process_single_file(signal)
            
            if wins is None:
                raise Exception("No gait segments detected (Preprocessing).")
            
            w_src1 = "Author" if s1_use_author else "User"
            self.update_progress(40, f"Stage 1: Detection ({s1_backbone}, {w_src1})...")
            
            mask = detect_gait_sequences(
                wins, 
                backbone_type=s1_backbone, 
                use_author_weights=s1_use_author
            )
            
            if len(mask) == 0 or not np.any(mask):
                raise Exception("No gait tests detected in Stage 1.")
            
            self.update_progress(60, "Continuity Rules...")
            new_mask = apply_continuity_rule(ts, mask)
            
            stage1_result = wins[new_mask]
            new_ts = ts[new_mask]
            
            if len(stage1_result) == 0:
                raise Exception("No valid segments after continuity check.")
            
            w_src2 = "Author" if s2_use_author else "User"
            self.update_progress(75, f"Stage 2: Segmentation ({s2_model}, {w_src2})...")
            
            raw_result = segment_gait_test(
                stage1_result, 
                model_type=s2_model,
                use_author_weights=s2_use_author
            )
            
            self.update_progress(90, "Post-processing...")
            all_events = run_postprocessing(
                window_probs=raw_result,
                window_start_times=new_ts
            )
            
            if not all_events:
                raise Exception("No events found after post-processing.")
            
            self.update_progress(100, "Analysis Completed.")
            
            self.results = {
                'events': all_events,
                'file': os.path.basename(fpath),
                'axis': config['axis'],
                'stage1_info': f"{s1_backbone} ({w_src1})",
                'stage2_info': f"{s2_model} ({w_src2})",
                'timestamp': datetime.now().isoformat()
            }
            
            self.root.after(100, self.display_results)
            self.root.after(100, lambda: self.plot_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            self.root.after(0, lambda: self.show_error(error_msg))
            self.current_signal = None
        finally:
            self.is_processing = False
            self.root.after(0, self.enable_buttons)

    def visualize_results(self):
        if self.current_signal is None or self.results is None:
            messagebox.showwarning("Data Missing", "No analysis data available.")
            return
            
        axis = self.axis_var.get()
        events = self.results['events']
        
        try:
            self.status_var.set("Generating Plots...")
            plot_detected_events(self.current_signal, events, axis_name=axis)
            self.status_var.set("Analysis Completed")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot: {e}")

    def update_progress(self, value, step):
        self.progress_var.set(value)
        self.current_step_var.set(step)
        self.root.update_idletasks()
    
    def display_results(self):
        if not self.results:
            return
        
        events = self.results['events']
        total_events = len(events)
        total_duration = sum((e['timestamp_end_ns'] - e['timestamp_start_ns']) / 1e9 for e in events)
        avg_score = sum(e['score'] for e in events) / total_events if total_events > 0 else 0
        
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        
        overview = f"""
============================================================
                  Analysis Results Overview                
============================================================

File Name:     {self.results['file']}
Stage 1:       {self.results['stage1_info']}
Stage 2:       {self.results['stage2_info']}
Time:          {self.results['timestamp']}

------------------------------------------------------------
[Statistical Summary]
  - Detected Events:  {total_events}
  - Total Duration:   {total_duration:.1f} seconds
  - Avg Confidence:   {avg_score * 100:.1f}%

------------------------------------------------------------
[Event Type Distribution]
"""
        class_counts = {}
        for event in events:
            cid = event['class_id']
            class_counts[cid] = class_counts.get(cid, 0) + 1
        
        for cid in sorted(class_counts.keys()):
            overview += f"\n  {self.class_names.get(cid, f'Unknown Class {cid}')}: {class_counts[cid]}"
        
        self.overview_text.insert(1.0, overview)
        self.overview_text.config(state=tk.DISABLED)
        
        # --- Detailed Events ---
        self.events_text.config(state=tk.NORMAL)
        self.events_text.delete(1.0, tk.END)
        
        for i, event in enumerate(events, 1):
            dur = (event['timestamp_end_ns'] - event['timestamp_start_ns']) / 1e9
            start = self.format_timestamp(event['timestamp_start_ns'])
            end = self.format_timestamp(event['timestamp_end_ns'])
            cls_name = self.class_names.get(event['class_id'], 'Unknown')
            
            info = f"""
Event {i}: {cls_name}
------------------------------------------------------------
  Duration:   {dur:.1f} s
  Confidence: {event['score'] * 100:.1f}%
  Time Range: {start} -> {end}

"""
            self.events_text.insert(tk.END, info)
        self.events_text.config(state=tk.DISABLED)
        
        # --- Raw JSON ---
        self.raw_text.config(state=tk.NORMAL)
        self.raw_text.delete(1.0, tk.END)
        self.raw_text.insert(1.0, json.dumps(self.results, indent=2, ensure_ascii=False))
        self.raw_text.config(state=tk.DISABLED)
        
        self.status_var.set("Analysis Completed")
    
    def format_timestamp(self, ns):
        dt = datetime.fromtimestamp(ns / 1e9)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    
    def show_empty_state(self):
        empty_text = "\n\n            [ Results will appear here ]\n\n            Select a file and click 'Start Analysis'"
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, empty_text)
        self.overview_text.config(state=tk.DISABLED)
    
    def show_error(self, message):
        messagebox.showerror("Error", message)
        self.status_var.set("Error Occurred")
        self.current_step_var.set(message)
    
    def show_warning(self, message):
        messagebox.showwarning("Warning", message)
    
    def enable_buttons(self):
        """Re-enable all controls after processing"""
        if self.file_path and PIPELINE_AVAILABLE:
            self.start_btn.config(state=tk.NORMAL)
        self.select_file_btn.config(state=tk.NORMAL)
        
        self.axis_combo.config(state='readonly')
        self.stage1_combo.config(state='readonly')
        self.stage2_combo.config(state='readonly')
        
        self.s1_chk.config(state=tk.NORMAL)
        self.s2_chk.config(state=tk.NORMAL)
    
    def export_results(self):
        if not self.results:
            return
        
        fpath = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"gait_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if fpath:
            try:
                with open(fpath, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2)
                messagebox.showinfo("Success", f"Saved to {fpath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def reset(self):
        if self.is_processing:
            return
        
        self.file_path = None
        self.results = None
        self.current_signal = None
        self.file_label.config(text="No file selected", fg='#666')
        self.start_btn.config(state=tk.DISABLED)
        self.plot_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.current_step_var.set('')
        self.status_var.set("Ready")
        self.show_empty_state()
        
        for widget in [self.events_text, self.raw_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)
            widget.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = GaitDetectionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()