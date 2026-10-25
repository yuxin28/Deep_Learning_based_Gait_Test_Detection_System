# Deep Learning-based Gait Test Detection System

An end-to-end deep learning framework for the automated detection and fine-grained segmentation of clinical gait tests from raw foot-worn IMU data. A two-stage architecture first separates gait activity from background movement and then labels each time step with the specific test protocol (e.g., 2-Minute Walk Test, 10-Meter Walk Tests).

## 📋 Table of Contents

* [System Overview](#system-overview)
* [Data Availability](#data-availability)
* [Data Specifications](#data-specifications)
  * [System Input (Raw Sensor Data)](#1-system-input-raw-sensor-data)
  * [Deep Learning Model I/O](#2-deep-learning-model-io)
  * [System Output (Event List)](#3-system-output-event-list)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Quick Start: Inference](#quick-start-inference)
  * [GUI Application](#gui-application)
  * [Programmatic API](#programmatic-api)
* [Pipeline Architecture](#pipeline-architecture)
  * [Preprocessing](#preprocessing)
  * [Stage 1: Binary Detection](#stage-1-binary-detection)
  * [Stage 2: Multi-Class Segmentation](#stage-2-multi-class-segmentation)
  * [Continuity Rule](#continuity-rule)
  * [Post-Processing](#post-processing)
* [Final Configuration](#final-configuration)
* [Training & Model Management](#training--model-management)
* [Evaluation](#evaluation)
  * [Event and Battery Metrics](#event-and-battery-metrics-metrics_pipelinepy)
  * [Independent Battery Detection](#independent-battery-detection-battery_detection_10minpy)
  * [External 4×10MWT Dataset](#external-dataset-with-410mwt-batteries-evaluate_4x10mwt_datasetpy)
* [References](#references)
* [Contact](#contact)

---

## System Overview

The pipeline processes **12-channel IMU data** (accelerometer and gyroscope from the left and right foot) and automates the annotation of unsupervised, home-based gait tests:

1. **Preprocessing:** `gaitmap`'s `UllrichGaitSequenceDetection` runs once on the whole recording (both feet, merged). The recording is cut into 90 s windows with a 30 s stride, and only windows that overlap a detected gait sequence are kept as candidates.
2. **Stage 1 (Detection):** A **TCN** / **TCN-BiLSTM** classifier decides for each candidate window whether it contains a gait test.
3. **Continuity Rule:** Candidate windows are grouped into temporally continuous blocks; if any window of a block is positive, the whole block is forwarded to Stage 2.
4. **Stage 2 (Segmentation):** A **DS-UNet** (BiGRU bottleneck, optional attention gates) classifies every time step into a gait test type.
5. **Post-Processing:** Hamming-weighted overlap-add, temporal smoothing, competition-aware decoding with per-class hysteresis thresholds, gap filling, minimum-duration pruning and an isolated-segment filter (`island300`) turn frame-level probabilities into a list of events.

The default arguments of all modules reproduce the final configuration (TCN → U-Net+BiGRU → competition-aware decoding → island300), also listed in [`pipeline/final_config.json`](pipeline/final_config.json).

### Supported Classes

* **1:** Preferred Walk (2×10MWT)
* **2:** Fast Walk (2×10MWT)
* **3:** Slow Walk (2×10MWT)
* **4:** 2-Minute Walk Test (2MWT)
* **0:** Background/Null

---

## Data Availability

The models were developed on clinical patient recordings that **cannot be made public**. This repository therefore contains **no sensor recordings, annotations, sample files or dataset split lists**. Pre-trained weights and the normalization statistics are included so that the pipeline can be run on your own data in the format described below.

---

## Data Specifications

### 1. System Input (Raw Sensor Data)

Raw IMU data is loaded from an HDF5 file (via `pandas.read_hdf`).

* **Format:** Pandas DataFrame indexed by timestamp (ns)
* **Sampling Rate:** 102.4 Hz
* **Required Columns (12 Channels):**

| **Sensor Location** | **Modality**  | **Axis** | **Column Names**                                                 |
| ------------------- | ------------- | -------- | ---------------------------------------------------------------- |
| **Right Foot**      | Accelerometer | X, Y, Z  | `right_sensor_acc_x`, `right_sensor_acc_y`, `right_sensor_acc_z` |
| **Right Foot**      | Gyroscope     | X, Y, Z  | `right_sensor_gyr_x`, `right_sensor_gyr_y`, `right_sensor_gyr_z` |
| **Left Foot**       | Accelerometer | X, Y, Z  | `left_sensor_acc_x`, `left_sensor_acc_y`, `left_sensor_acc_z`    |
| **Left Foot**       | Gyroscope     | X, Y, Z  | `left_sensor_gyr_x`, `left_sensor_gyr_y`, `left_sensor_gyr_z`    |

> **Note:** Data is Z-score normalized internally with the statistics in `models_training/*/config/norm_params.json`.

### 2. Deep Learning Model I/O

#### Stage 1: Binary Detection Model

* **Input:** `(B, 12, 9216)` *(Batch, Channels, Time Steps)*
  * **9216:** 90 s window × 102.4 Hz
* **Output:** `(B, 1)`
  * Probability *p* ∈ [0, 1] that the window contains a gait test

#### Stage 2: Segmentation Model

* **Input:** `(B, 12, 9216)`
* **Output:** `(B, 5, 9216)`
  * **5:** Class probabilities
  * **9216:** Dense frame-level predictions

### 3. System Output (Event List)

```python
[
  {
    "class_id": 4,                          # Predicted class (e.g., 2MWT)
    "timestamp_start_ns": 1635412800000..., # Absolute start time (ns)
    "timestamp_end_ns": 1635412870000...,   # Absolute end time (ns)
    "score": 0.95                           # Confidence score
  }
]
```

---

## Project Structure

```text
Project_Root/
├── models_training/
│   ├── stage1/                          # Stage 1: Binary Gait Detection
│   │   ├── best_model/                  # Pre-trained weights (tcn, tcn_bilstm)
│   │   ├── config/
│   │   │   ├── config.json              # Training configuration
│   │   │   ├── train_files.json         # Training file list
│   │   │   ├── val_files.json           # Validation file list
│   │   │   └── norm_params.json         # Normalization parameters
│   │   ├── backbone_model.py            # TCN & BiLSTM architectures
│   │   ├── dataset.py                   # Dataset loader
│   │   ├── focal_loss.py                # Focal Loss
│   │   ├── model.py                     # Classification head
│   │   ├── train.py                     # Training script
│   │   ├── test.py                      # Window-level evaluation script
│   │   └── utils.py                     # Metrics utilities
│   │
│   └── stage2/                          # Stage 2: Frame-level Segmentation
│       ├── best_model/                  # Pre-trained weights (unet_bigru, unet_att_gru)
│       ├── config/
│       │   ├── config.json
│       │   ├── train_files.json
│       │   ├── val_files.json
│       │   └── norm_params.json
│       ├── combined_loss.py             # CE + Dice loss
│       ├── dataset.py                   # UNet dataset & augmentation
│       ├── model.py                     # GaitSegUNet architecture
│       ├── train.py                     # Training script
│       ├── test.py                      # Frame-level evaluation script
│       └── utils.py                     # Metrics utilities
│
├── pipeline/                            # End-to-end inference system
│   ├── end2end_metrics/
│   │   ├── metrics_pipeline.py          # End-to-end evaluation (event + battery metrics)
│   │   ├── battery_metrics.py           # Battery-level metrics
│   │   ├── battery_detection_10min.py   # Independent battery detection (600 s grouping)
│   │   ├── evaluate_4x10mwt_dataset.py  # Evaluation on an external 4×10MWT dataset
│   │   └── utils.py                     # IoU matching & aggregation
│   ├── final_config.json                # Final pipeline configuration
│   ├── preprocessing.py                 # Ullrich gait detection & sliding windows
│   ├── stage1.py                        # Stage 1 inference wrapper
│   ├── continuity_rule.py               # Continuity rule
│   ├── stage2.py                        # Stage 2 inference wrapper
│   ├── postprocessing.py                # OLA, smoothing, competition-aware decoding, island filter
│   ├── postprocessing_sanity_checks.py  # Synthetic sanity checks for post-processing
│   ├── gait_detection.py                # Programmatic API entry point
│   ├── gui_app.py                       # Desktop GUI
│   └── plot_detected_events.py          # Visualization utilities
│
├── requirements.txt
└── README.md
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yuxin28/Deep_Learning_based_Gait_Test_Detection_System.git
   cd Deep_Learning_based_Gait_Test_Detection_System
   ```

2. **Install dependencies** (Python 3.10+, virtual environment recommended):

   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start: Inference

### GUI Application

1. **Run the application:**
   ```bash
   python pipeline/gui_app.py
   ```

2. **Configuration:**
   * **Data File:** Select an `.h5` file in the input format above.
   * **Visualization Axis:** Choose a sensor channel (e.g., `right_sensor_gyr_y`).
   * **Model Settings:** Select architectures and toggle "Author Weights". The defaults (`tcn`, `unet_bigru`, author weights) are the final configuration; the GUI marks any other combination as custom.

3. **Analyze:** Click "Start Analysis".

4. **Visualize:** Click "Visualize Signals" to open interactive plots of the raw signal and the detected events.

### Programmatic API

```python
from pipeline.gait_detection import detect_gait_test

events = detect_gait_test(
    signal_path="path/to/your_recording.h5",
    axis_name="right_sensor_gyr_y",

    # Model selection
    stage1_backbone="tcn",          # 'tcn' | 'tcn_bilstm'
    stage2_model="unet_bigru",      # 'unet_bigru' | 'unet_att_gru'

    # Weight source (True = author's pre-trained model, False = your own training run)
    use_author_weights_stage1=True,
    use_author_weights_stage2=True,

    plot_signal=True,

    # Optional overrides (defaults = final configuration)
    preprocessing_config=None,      # e.g. {"stride_s": 30, "ullrich_config": {"peak_prominence": 12.0}}
    continuity_config=None,         # e.g. {"gaps_duration_ns": 30_000_000_000}
    postprocessing_config=None,     # e.g. {"island_filter": {"neighborhood_sec": 300.0}}
    return_debug=False,             # True -> {"events": ..., "debug": ...}
)

print(f"Detected {len(events) if events else 0} events.")
```

An already loaded DataFrame can be passed with `signal=...` instead of reading `signal_path`, and `progress_callback` is called with the name of each stage before it starts.

#### `use_author_weights`

| **Value**   | **Behavior**                          | **Weights Path**                                |
| ----------- | ------------------------------------- | ----------------------------------------------- |
| **`True`**  | Loads the provided pre-trained weights | `models_training/stageX/best_model/[model]/`    |
| **`False`** | Loads weights from your own training   | `models_training/stageX/stageX_output/[model]/` |

> **Note:** With `False`, the weights of your most recent training run (as defined in `config.json`) are used.

---

## Pipeline Architecture

### Preprocessing

Implemented in `pipeline/preprocessing.py` (`process_single_file`):

1. `UllrichGaitSequenceDetection` (gaitmap) is run once on the full recording.
2. The recording is cut into 90 s windows (stride 30 s).
3. Windows that overlap at least one detected gait sequence (and contain no missing values) are kept as candidates.

The Ullrich parameters were selected on the validation split and are the defaults (`DEFAULT_ULLRICH_CONFIG`):

| **Parameter**                      | **Final value** |
| ---------------------------------- | --------------- |
| `sensor_scope`                     | `"both"`        |
| `merge_gait_sequences_from_sensors`| `True`          |
| `sensor_channel_config`            | `"gyr_ml"`      |
| `peak_prominence`                  | `12.0`          |
| `active_signal_threshold`          | `None`          |
| `additional_margin_s`              | `0.0`           |

Override them with `preprocessing_config={"ullrich_config": {...}}` in `detect_gait_test`, or `ullrich_config=...` in `process_single_file`.

### Stage 1: Binary Detection

Distinguishes windows containing a gait test (class 1) from other activity (class 0).

* **Models:** `TCN` or `TCN-BiLSTM`
* **Dataset:** `GaitSensorDataset` (Z-score normalized)
* **Loss:** Focal Loss (α=0.75, γ=2.0) for class imbalance

### Stage 2: Multi-Class Segmentation

Frame-level classification into the gait test types.

* **Model:** `GaitSegUNet` (Depthwise Separable UNet)
  * **Encoder:** Depthwise separable convolutions
  * **Bottleneck:** BiGRU for long-range temporal context
  * **Decoder:** Optional attention gates
* **Loss:** Cross-Entropy + Dice

### Continuity Rule

Implemented in `pipeline/continuity_rule.py`. Consecutive candidate windows whose start times are at most 30 s apart form one block. If Stage 1 marks any window of a block as positive, all windows of that block are forwarded to Stage 2.

### Post-Processing

Implemented in `pipeline/postprocessing.py` (`run_postprocessing`):

1. **Overlap-Add:** Hamming-weighted reconstruction of one continuous probability trace from overlapping windows.
2. **Smoothing:** Centered 1.5 s moving average, renormalized per frame.
3. **Competition-Aware Decoding:** Score-first decoder with per-class hysteresis thresholds. A class switch requires a margin of 0.08 that persists for 1 s; ties within 0.02 are resolved by class priority (2MWT > slow > preferred > fast).

   | **Class**                 | **ON** | **OFF** | **Gap fill** | **Min. duration** |
   | ------------------------- | ------ | ------- | ------------ | ----------------- |
   | 1 Preferred               | 0.40   | 0.28    | 5 s          | 15 s              |
   | 2 Fast                    | 0.40   | 0.28    | 5 s          | 12 s              |
   | 3 Slow                    | 0.40   | 0.28    | 5 s          | 20 s              |
   | 4 2MWT                    | 0.35   | 0.25    | 15 s         | 70 s              |

4. **Gap Filling & Minimum-Duration Pruning:** Same-class segments separated by short gaps are merged, then segments shorter than the class minimum are removed.
5. **Island Filter (`island300`):** A detected test segment is removed if no other test segment lies within 300 s of it, since standardized gait tests are performed as a battery.

Check the post-processing logic on synthetic inputs with:

```bash
python pipeline/postprocessing_sanity_checks.py
```

---

## Final Configuration

[`pipeline/final_config.json`](pipeline/final_config.json) lists the complete final configuration (models, preprocessing, continuity rule and post-processing). These values are already the code defaults; the file documents them and can be used as a starting point for overrides:

```python
import json
from pipeline.gait_detection import detect_gait_test

cfg = json.load(open("pipeline/final_config.json"))
events = detect_gait_test(
    "path/to/your_recording.h5",
    stage1_backbone=cfg["models"]["stage1_backbone"],
    stage2_model=cfg["models"]["stage2_model"],
    preprocessing_config=cfg["preprocessing_config"],
    continuity_config=cfg["continuity_config"],
    postprocessing_config=cfg["postprocessing_config"],
    plot_signal=False,
)
```

---

## Training & Model Management

The inference pipeline loads hyperparameters and weights produced by training.

### Dataset File Lists

`train_files.json` is a list of `.npy` segment files:

```json
[
  "/path/to/data/subject_01_session_01.npy",
  "/path/to/data/subject_01_session_02.npy"
]
```

`val_files.json` is a dictionary with `between_subject` and `within_subject` keys:

```json
{
  "between_subject": ["/path/to/data/subject_03_session_01.npy"],
  "within_subject":  ["/path/to/data/subject_01_session_03.npy"]
}
```

> The provided file lists contain placeholder paths that only document the expected format. Replace them with lists of your own files.

### Training Stage 1

1. Configure `models_training/stage1/config/config.json` (set `enable: true` for the models to train).
2. Run:
   ```bash
   python models_training/stage1/train.py
   ```
3. Weights are saved to `models_training/stage1/stage1_output/`.
4. Use them in inference with `use_author_weights_stage1=False`.

### Training Stage 2

1. Configure `models_training/stage2/config/config.json`.
2. Run:
   ```bash
   python models_training/stage2/train.py
   ```
3. Weights and `train_params.json` are saved to `models_training/stage2/stage2_output/`.
4. Use them in inference with `use_author_weights_stage2=False`.

---

## Evaluation

No evaluation data is shipped with this repository. All scripts below expect your own annotated recordings. Annotation CSVs need the columns `type` (`preferred_walk`, `fast_walk`, `slow_walk`, `2min_walk`), `time_stamp_start` and `time_stamp_end` (ns).

### Dataset Info Files

`metrics_pipeline.py` and `battery_detection_10min.py` read `pipeline/end2end_metrics/<split>_dataset_info/<split>_between_paths.json` and `<split>_within_paths.json`, where `<split>` is `--dataset-source` (`test` or `val`). These directories are git-ignored. Each file maps a recording ID (`<subjectID>_<session>`) to its annotation CSV and sensor H5 file:

```json
{
  "subject001_session01": {
    "csv_files": ["/path/to/subject001/session01/annotations.csv"],
    "h5_files":  ["/path/to/subject001/session01/sensor_data.h5"]
  }
}
```

### Event and Battery Metrics (`metrics_pipeline.py`)

```bash
python pipeline/end2end_metrics/metrics_pipeline.py --dataset-source val --num-workers 4 \
    --postprocessing-cache-dir path/to/cache
```

Reported separately for the between-subject and within-subject splits:

* **Event level:** per-class Precision, Recall and F1 with greedy one-to-one matching (same class, IoU > 0.5). Predicted classes without annotations count as false positives.
* **Battery level** (`battery_metrics.py`): one day of annotations is split into continuous blocks (each subtest starts where the previous one ends), and every prediction is assigned to the block it overlaps most. A subtest counts as detected if a same-class prediction reaches IoU ≥ 0.3 (preferred, fast, slow) or covers ≥ 50% of it (2MWT).

  | **Metric**                        | **Definition**                                              | **Blocks**         |
  | --------------------------------- | ----------------------------------------------------------- | ------------------ |
  | `performed_event_score`           | detected subtests / annotated subtests                      | all                |
  | `continuous_success_rate`         | share of blocks with ≥ 2/3 of subtests detected             | all                |
  | `battery_score`                   | detected subtests / 4                                       | complete batteries |
  | `protocol_threshold_success_rate` | share with ≥ 3 of 4 subtests detected                       | complete batteries |
  | `protocol_complete_success_rate`  | share with 4 of 4 subtests detected                         | complete batteries |
  | subtest IoU                       | mean / SD IoU of detected subtests, overall and per class   | all                |
  | any-hit P/R/F1                    | ≥ 1 subtest detected, per block and per day                 | all                |
  | overlap-any P/R/F1                | any test prediction overlaps an annotation, class ignored   | all                |

  A complete battery has exactly one preferred, fast, slow and 2MWT subtest.

`--postprocessing-cache-dir` stores the Stage 1/2 outputs so that later runs only repeat the post-processing. Results are written to `pipeline/end2end_metrics/results/` (or `--results-dir-name`).

### Independent Battery Detection (`battery_detection_10min.py`)

Evaluates whether whole test batteries are found, independent of subtest classes:

1. Predicted events (all classes) and annotated subtests are grouped **separately** into batteries: an event joins the previous battery if it starts ≤ 600 s after the previous event ends.
2. Predicted and annotated batteries are matched one-to-one (greedy, largest IoU first). A pair counts as a match on any positive temporal overlap.
3. Reported for all annotated batteries and for complete batteries only: TP/FP/FN, Precision, Recall, F1, battery IoU and start/end errors.

```bash
python pipeline/end2end_metrics/battery_detection_10min.py --dataset-source test \
    --results-dir path/to/battery_detection_results --num-workers 4
```

If your path files were written for another location, `--dataset-root` remaps them: paths are taken relative to `CONFIGURED_DATASET_ROOT` (set at the top of the script) and joined with the new root.

### External Dataset with 4×10MWT Batteries (`evaluate_4x10mwt_dataset.py`)

Runs the unchanged final pipeline on a dataset with a different protocol (three batteries per day, each a 4×10MWT at preferred, fast and slow speed, no 2MWT). Predicted 2MWT events count as false positives and are also reported separately. Expected layout (one session per date folder):

```text
<dataset-root>/.../YYYY-MM-DD/<recording>.h5
<dataset-root>/.../YYYY-MM-DD/<annotations>.csv
```

Main metrics:

* Event level (same class, IoU > 0.5) per class and macro over preferred, fast and slow, plus 2MWT false alarms
* Localization of matched events (IoU, start/end error) and a confusion table per annotated test
* Battery recovery on complete batteries (same class, IoU ≥ 0.3; ≥ 2/3 and 3/3 success)
* Independent battery detection (600 s grouping, as above)

Diagnostics:

* Relaxed matching (IoU ≥ 0.3, ≥ 0.1, any overlap) and a control with predicted 2MWT events removed
* Per-day stage counts (candidate windows, Stage 1 positives, forwarded windows, events) and the stage at which each annotated test was lost
* Stage 1 window-level P/R/F1 before and after the continuity rule (a 90 s window is positive when its central 30 s overlaps an annotated test, as in training), and the size of the continuity groups
* Stage 2 probabilities inside each annotated test
* Post-processing variants recomputed from cached Stage 2 probabilities (island filter off, lower thresholds). These are diagnostics only, not alternative final configurations.
* Class-agnostic detection at event level (any overlap) and time level (overlap of the annotated and predicted time spans)
* Whole battery as the target, for every post-processing variant: 600 s grouping, matching at any overlap, IoU ≥ 0.3 and IoU ≥ 0.5 (all and complete batteries), battery IoU, start/end errors, time level on battery spans, and the predicted batteries without an annotated match (candidates for missing annotations)

```bash
# Final configuration; saves predictions and the Stage 2 probability cache
python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root path/to/dataset \
    --output-dir output/external_4x10mwt

# Recompute metrics and variants from saved predictions (no model run)
python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root path/to/dataset \
    --output-dir output/external_4x10mwt --reuse-predictions

# Ablation: continuity rule off (only Stage 1 positive windows forwarded)
python pipeline/end2end_metrics/evaluate_4x10mwt_dataset.py --dataset-root path/to/dataset \
    --no-continuity --output-dir output/external_4x10mwt_nocont
```

Main outputs: `METRICS_SUMMARY.md`, `metrics.json`, `sessions.csv`, `stage_diagnostics_per_day.csv`, `class_agnostic_per_day.csv`, `gt_stage2_probabilities.csv`, `matched_event_pairs.csv`, `predicted_2mwt_false_alarms.csv`. The `output/` directory is git-ignored.

---

## References

### Gaitmap (Preprocessing)

```bibtex
@ARTICLE{10411039,
  author={Küderle, Arne and Ullrich, Martin and Roth, Nils and et al.},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  title={Gaitmap—An Open Ecosystem for IMU-Based Human Gait Analysis and Algorithm Benchmarking},
  year={2024},
  volume={5},
  pages={163-172},
  doi={10.1109/OJEMB.2024.3356791}
}
```

---

## Contact

For questions or support regarding this Gait Test Detection System, please contact the maintainers.
