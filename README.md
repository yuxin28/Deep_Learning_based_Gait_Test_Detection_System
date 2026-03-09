# Deep Learning-based Gait Test Detection System

A robust, end-to-end deep learning framework designed for the automated detection and fine-grained segmentation of clinical gait tests from raw IMU sensor data. This system utilizes a two-stage architecture to distinguish gait activity from background noise and classify specific test protocols (e.g., 2-Minute Walk Test, 10-Meter Walk Tests) with frame-level precision.

## 📋 Table of Contents

* [System Overview](#system-overview)
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
  * [Stage 1: Binary Detection](#stage-1-binary-detection)
  * [Stage 2: Multi-Class Segmentation](#stage-2-multi-class-segmentation)
  * [Post-Processing](#post-processing)
* [Training & Model Management](#training--model-management)
  * [Dataset JSON Structure](#dataset-json-structure)
  * [Training Stage 1](#training-stage-1)
  * [Training Stage 2](#training-stage-2)
* [Evaluation Framework](#evaluation-framework)
  * [Evaluation Data Structure](#evaluation-data-structure)
  * [Running Evaluation](#running-evaluation)
* [References](#references)
* [Contact](#contact)

---

## System Overview

This repository implements a pipeline for processing **12-channel IMU data** (Accelerometer/Gyroscope from Left & Right feet). It automates the annotation of unsupervised home-based gait tests through the following steps:

1. **Preprocessing:** Candidate extraction using `gaitmap.gait_detection.UllrichGaitSequenceDetection` and sliding window generation.
2. **Stage 1 (Detection):** A **TCN** / **TCN-BiLSTM** model filters false positives, distinguishing actual gait from random movement.
3. **Continuity Rule:** Grouping logic to bridge small gaps in binary detections.
4. **Stage 2 (Segmentation):** A **DS-UNet** (with BiGRU Bottleneck) classifies every time step into specific gait test types.
5. **Post-Processing:** Robust reconstruction using Overlap-Add (OLA), Hysteresis Thresholding, and Non-Maximum Suppression (NMS).

### Supported Classes

The system segments data into the following classes:

* **1:** Preferred Walk (2×10MWT)
* **2:** Fast Walk (2×10MWT)
* **3:** Slow Walk (2×10MWT)
* **4:** 2-Minute Walk Test (2MWT)
* **0:** Background/Null

---

## Data Specifications

### 1. System Input (Raw Sensor Data)

The system expects raw IMU data loaded from an HDF5 file (via `pandas`).

* **Format:** Pandas DataFrame
* **Sampling Rate:** 102.4 Hz
* **Required Columns (12 Channels):**

| **Sensor Location** | **Modality**  | **Axis** | **Column Names**                                               |
| ------------------- | ------------- | -------- | -------------------------------------------------------------- |
| **Right Foot**      | Accelerometer | X, Y, Z  | `right_sensor_acc_x`, `right_sensor_acc_y`, `right_sensor_acc_z` |
| **Right Foot**      | Gyroscope     | X, Y, Z  | `right_sensor_gyr_x`, `right_sensor_gyr_y`, `right_sensor_gyr_z` |
| **Left Foot**       | Accelerometer | X, Y, Z  | `left_sensor_acc_x`, `left_sensor_acc_y`, `left_sensor_acc_z`    |
| **Left Foot**       | Gyroscope     | X, Y, Z  | `left_sensor_gyr_x`, `left_sensor_gyr_y`, `left_sensor_gyr_z`    |

> **Note:** Data is internally normalized (Z-score) using pre-calculated mean/std statistics found in `models_training/*/config/norm_params.json`.

### 2. Deep Learning Model I/O

The pipeline processes data in fixed-length sliding windows.

#### Stage 1: Binary Detection Model

* **Input:** `(B, 12, 9216)` *(Batch, Channels, Time Steps)*
  * **9216:** Fixed window of 90 seconds × 102.4 Hz
* **Output:** `(B, 1)`
  * Scalar probability *p* ∈ [0, 1] indicating if the entire window contains a valid gait sequence

#### Stage 2: Segmentation Model

* **Input:** `(B, 12, 9216)`
* **Output:** `(B, 5, 9216)`
  * **5:** Probability distribution over the 5 classes
  * **9216:** Dense frame-level predictions

### 3. System Output (Event List)

A list of dictionaries containing the detected events:

```python
[
  {
    "class_id": 4,                          # Predicted Class (e.g., 2MWT)
    "timestamp_start_ns": 1635412800000..., # Absolute Start Time (ns)
    "timestamp_end_ns": 1635412870000...,   # Absolute End Time (ns)
    "score": 0.95                           # Confidence Score
  }
]
```

---

## Project Structure

```text
Project_Root/
├── models_training/                 # Model Training & Evaluation
│   ├── stage1/                      # Stage 1: Binary Gait Detection
│   │   ├── best_model/              # Pre-trained weights (tcn, tcn_bilstm)
│   │   ├── config/
│   │   │   ├── config.json          # Training Configuration
│   │   │   ├── train_files.json    # Training dataset list
│   │   │   ├── val_files.json      # Validation dataset list
│   │   │   ├── test_files.json     # Testing dataset list
│   │   │   └── norm_params.json    # Normalization parameters
│   │   ├── backbone_model.py        # TCN & BiLSTM architectures
│   │   ├── dataset.py               # Dataset loader
│   │   ├── focal_loss.py            # Custom Focal Loss
│   │   ├── model.py                 # Classification Head
│   │   ├── train.py                 # Training script
│   │   ├── test.py                  # Evaluation script
│   │   └── utils.py                 # Metrics utilities
│   │
│   └── stage2/                      # Stage 2: Frame-level Segmentation
│       ├── best_model/              # Pre-trained weights
│       ├── config/
│       │   ├── config.json          # Training Configuration
│       │   ├── train_files.json
│       │   ├── val_files.json
│       │   ├── test_files.json
│       │   └── norm_params.json
│       ├── combined_loss.py         # Hybrid Loss (CE + Dice)
│       ├── dataset.py               # UNet Dataset & Augmentation
│       ├── model.py                 # GaitSegUNet Architecture
│       ├── train.py                 # Training script
│       ├── test.py                  # Evaluation script
│       └── utils.py                 # Metrics utilities
│
├── pipeline/                        # End-to-end Inference System
│   ├── end2end_metrics/             # Event-level evaluation
│   │   ├── results/                 # Saved metrics (JSON)
│   │   ├── utils.py                 # IoU calculation & aggregation
│   │   └── metrics_pipeline.py     # Main evaluation script
│   ├── samples/                     # Example HDF5 data
│   ├── continuity_rule.py           # Temporal continuity logic
│   ├── gui_app.py                   # Desktop GUI Application
│   ├── gait_detection.py            # Programmatic API entry point
│   ├── plot_detected_events.py      # Visualization utilities
│   ├── postprocessing.py            # OLA, Hysteresis, NMS fusion
│   ├── preprocessing.py             # Sliding window generation
│   ├── stage1.py                    # Stage 1 inference wrapper
│   └── stage2.py                    # Stage 2 inference wrapper
│
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://mad-srv.informatik.uni-erlangen.de/ox31ykoc/yuxin_gait_detection
   cd Yuxin_Gait_detection
   ```

2. **Install dependencies:**

   It is recommended to use a virtual environment (Python 3.10+).

   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start: Inference

### GUI Application

For easy visualization and testing without writing code, use the offline Tkinter GUI.

1. **Run the application:**
   ```bash
   python pipeline/gui_app.py
   ```

2. **Configuration:**
   * **Data File:** Select an `.h5` file.
   * **Visualization Axis:** Choose a sensor channel (e.g., `right_sensor_gyr_y`).
   * **Model Settings:** Select architectures (e.g., `tcn`, `unet_bigru`) and toggle "Author Weights".

3. **Analyze:** Click "Start Analysis".

4. **Visualization:** Once analysis is complete, click "Visualize Signals" to open interactive plots showing the raw signal and detected gait events.

### Programmatic API

To integrate the pipeline into your own scripts, use `detect_gait_test`.

```python
from pipeline.gait_detection import detect_gait_test

# Run the pipeline
events = detect_gait_test(
    signal_path="pipeline/samples/patient_data.h5",
    axis_name="right_sensor_gyr_y",
  
    # Model Selection
    stage1_backbone="tcn",          # Options: 'tcn', 'tcn_bilstm'
    stage2_model="unet_bigru",      # Options: 'unet_bigru', 'unet_att_gru'
  
    # Weight Source (True = Author's Best Model, False = User's Trained Model)
    use_author_weights_stage1=True, 
    use_author_weights_stage2=True,
  
    plot_signal=True                # Auto-plot results
)

print(f"Detected {len(events) if events else 0} events.")
```

#### ⚠️ Parameter Explanation: `use_author_weights`

This boolean flag determines the source of the model weights (`best_model.pth`):

| **Value**   | **Behavior**                                       | **Weights Path**                                |
| ----------- | -------------------------------------------------- | ----------------------------------------------- |
| **`True`**  | Loads **Author Provided Weights** (Pre-trained)   | `models_training/stageX/best_model/[model]/`    |
| **`False`** | Loads **User Trained Weights** (From local training) | `models_training/stageX/stageX_output/[model]/` |

> **Note:** If set to `False`, the system will use the weights from your most recent training run as defined in `config.json`.

---

## Pipeline Architecture

### Stage 1: Binary Detection

**Objective:** Filter out noise from continuous sensor streams by distinguishing valid gait sequences (Class 1) from non-gait background activities (Class 0).

* **Models:** `TCN` (Temporal Convolutional Network) or `TCN-BiLSTM`.
* **Dataset:** `GaitSensorDataset` (Z-score normalized).
* **Loss Function:** **Focal Loss** (α=0.75, γ=2.0) to handle class imbalance.

### Stage 2: Multi-Class Segmentation

**Objective:** Perform precise, frame-level classification of gait activities into specific test types.

* **Model:** `GaitSegUNet` (Depthwise Separable UNet).
* **Features:**
  * **Encoder:** Depthwise Separable Convolutions.
  * **Bottleneck:** BiGRU for global temporal dependencies.
  * **Decoder:** Optional Attention Gates.
* **Loss Function:** Combined Loss (Cross-Entropy + Dice Loss).

### Post-Processing

Implemented in `pipeline/postprocessing.py`:

1. **Overlap-Add (OLA):** Reconstructs continuous probability signals.
2. **Hysteresis Decoding:** Uses dual thresholds (ON=0.50, OFF=0.38) to prevent flickering.
3. **Gap Filling:** Merges segments of the same class separated by small gaps (<5-15s).
4. **Non-Maximum Suppression (NMS):** Resolves overlapping events based on confidence scores.

---

## Training & Model Management

The inference pipeline dynamically loads hyperparameters and weights generated during training.

### Dataset JSON Structure

To ensure correct loading during training and testing, the JSON configuration files must follow these structures:

* **Training Data (train_files.json):**
  
  Must be a List of file path strings.

  ```json
  [
    "/path/to/data/subject_01_session_01.npy",
    "/path/to/data/subject_01_session_02.npy"
  ]
  ```

* **Validation / Testing Data (val_files.json, test_files.json):**
  
  Must be a Dictionary containing `between_subject` and `within_subject` keys.

  ```json
  {
    "between_subject": [
      "/path/to/data/subject_03_session_01.npy"
    ],
    "within_subject": [
      "/path/to/data/subject_01_session_03.npy"
    ]
  }
  ```

### Training Stage 1

1. **Configure** `models_training/stage1/config/config.json`:
   * Set `enable: true` for models you want to train.

2. **Run Training:**
   ```bash
   python models_training/stage1/train.py
   ```

3. **Result:** Weights are saved to `models_training/stage1/stage1_output/`.

4. **Inference:** Set `use_author_weights_stage1=False` to use this model.

### Training Stage 2

1. **Configure** `models_training/stage2/config/config.json`.

2. **Run Training:**
   ```bash
   python models_training/stage2/train.py
   ```

3. **Result:** Weights and `train_params.json` are saved to `models_training/stage2/stage2_output/`.

4. **Inference:** Set `use_author_weights_stage2=False` to use this model.

---

## Evaluation Framework

The `end2end_metrics` module provides comprehensive event-level evaluation against ground truth annotations.

### Evaluation Data Structure

The `metrics_pipeline.py` script requires a JSON file mapping patient sessions to their Ground Truth CSV and raw H5 files.

**Structure:** A **Dictionary** where keys are unique session identifiers.

```json
{
    "subject_001_session_01": {
        "csv_files": [
            "/dataset/anonymous/subject_001/session_01/annotations.csv"
        ],
        "h5_files": [
            "/dataset/anonymous/subject_001/session_01/sensor_data.h5"
        ]
    },
    "subject_001_session_02": {
        "csv_files": [
            "/dataset/anonymous/subject_001/session_02/annotations.csv"
        ],
        "h5_files": [
            "/dataset/anonymous/subject_001/session_02/sensor_data.h5"
        ]
    }
}
```

### Running Evaluation

1. Ensure your `test_dataset_info/*.json` files match the structure above.

2. Run the evaluation pipeline:
   ```bash
   python pipeline/end2end_metrics/metrics_pipeline.py
   ```

3. **Metrics:** Precision, Recall, and F1-Score (IoU Threshold > 0.5) are calculated for "Within-Subject" and "Between-Subject" splits.

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