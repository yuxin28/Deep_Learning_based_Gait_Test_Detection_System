#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 - Gait Detection Inference
----------------------------------
Runs binary classification on sliding windows to identify gait sequences.
Features:
- Robust parameter mapping (syncs with test.py).
- Supports both TCN and TCN-BiLSTM backbones.
- Handles User vs Author weight loading logic.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# 1. Path Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # pipeline/stage1.py -> Project_Root 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models_training.stage1.backbone_model import TCNBackbone, TCN_BiLSTM_Backbone
from models_training.stage1.model import GaitBinaryDetector

# -------------------------------------------------------------------------
# Dataset Wrapper
# -------------------------------------------------------------------------
class Stage1Dataset(Dataset):
    """
    Simple Dataset wrapper for inference.
    Normalizes input windows using provided mean/std.
    """
    def __init__(self, data: np.ndarray, norm_params: Dict[str, List[float]]):
        self.data = torch.tensor(data, dtype=torch.float32)
        # Convert mean/std to tensors for efficient broadcasting
        self.mean = torch.tensor(norm_params["mean"], dtype=torch.float32)
        self.std = torch.tensor(norm_params["std"], dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Input shape: (window_size, num_channels)
        # Transpose to (num_channels, window_size) for TCN input format
        sample = self.data[idx].T  
        
        # Normalize: (C, T) - (C, 1) / (C, 1)
        x = (sample - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return x

# -------------------------------------------------------------------------
# Main Detection Function
# -------------------------------------------------------------------------
def detect_gait_sequences(
    signal_windows: np.ndarray,
    backbone_type: str = "tcn",
    use_author_weights: bool = True
) -> np.ndarray:
    """
    Runs binary classification on sliding windows to identify gait sequences.

    Args:
        signal_windows (np.ndarray): Shape (N, window_size, channels).
        backbone_type (str): "tcn" or "tcn_bilstm".
        use_author_weights (bool): If True, loads the best pre-trained weights.

    Returns:
        np.ndarray: A binary array (N,) where 1 indicates Gait, 0 indicates Non-Gait.
    """
    
    # 1. Load Normalization Parameters
    norm_path = ROOT / "models_training" / "stage1" / "config" / "norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization params not found at {norm_path}")

    with open(norm_path, 'r') as f: 
        norm_data = json.load(f)
        # Handle case where json is flat or nested
        norm_params = norm_data.get("normalization_params", norm_data)

    # 2. Determine Config & Weights Path (Synced with test.py logic)
    if use_author_weights:
        base_dir = ROOT / "models_training" / "stage1" / "best_model"
        config_path = base_dir / "config.json"
        
        if not config_path.exists():
             raise FileNotFoundError(f"Author config not found at {config_path}")

        with open(config_path, 'r') as f: 
            full_config = json.load(f)
            
        if backbone_type == "tcn":
            output_dir = base_dir / "tcn"
            model_config = full_config["models"]["A"]["model_params"]
        elif backbone_type == "tcn_bilstm":
            output_dir = base_dir / "tcn_bilstm"
            model_config = full_config["models"]["B"]["model_params"]
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    else:
        # Load from user training output directory
        user_config_path = ROOT / "models_training" / "stage1" / "config" / "config.json"
        with open(user_config_path, 'r') as f:
            full_config = json.load(f)
            
        if backbone_type == "tcn":
            rel_path = full_config["models"]["A"]["save_dir"]
        elif backbone_type == "tcn_bilstm":
            rel_path = full_config["models"]["B"]["save_dir"]
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
        output_dir = ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            
        # Load the specific training params
        train_params_path = output_dir / "train_params.json"
        if not train_params_path.exists():
             raise FileNotFoundError(f"Training params not found at {train_params_path}")
             
        with open(train_params_path, 'r') as f:
            model_config = json.load(f)["model_params"]

    # 3. Setup Data Loader
    dataset = Stage1Dataset(signal_windows, norm_params)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Instantiate Model (With Robust Mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if backbone_type == "tcn":
        backbone = TCNBackbone(
            in_channels = model_config.get("in_channels", 12),
            channels = model_config.get("channels", [64, 64, 128, 128]),
            kernel_size = model_config.get("kernel_size", 15),
            dilations = model_config.get("dilations", [8, 16, 32, 64, 128]),
            num_stacks = model_config.get("num_stacks", 1),
            dropout = model_config.get("dropout", 0.1),
            norm = model_config.get("norm", "BN"),
            causal = model_config.get("causal", False),
            use_skip = model_config.get("use_skip", True),
            proj_dim = model_config.get("proj_dim", 128)
        )
    else:  # "tcn_bilstm"
        backbone = TCN_BiLSTM_Backbone(
            in_channels=model_config.get("in_channels", 12),
            tcn_channels=model_config.get("tcn_channels", [64, 64, 128]),
            tcn_kernel_size=model_config.get("tcn_kernel_size", 15),
            tcn_dilations=model_config.get("tcn_dilations", [8, 16, 32]),
            # Robust: Check "tcn_causal" first, then "causal", default False
            tcn_causal=model_config.get("tcn_causal", model_config.get("causal", False)),
            lstm_hidden=model_config.get("lstm_hidden", 128),
            lstm_bidirectional=model_config.get("lstm_bidirectional", True),
            # Robust: Check "lstm_proj_out" first, then "proj_dim", default 128
            lstm_proj_out=model_config.get("lstm_proj_out", model_config.get("proj_dim", 128))
        )

    model = GaitBinaryDetector(backbone=backbone).to(device)

    # 5. Load Weights
    ckpt_path = output_dir / 'best_model.pth'
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}")
        return np.zeros(len(signal_windows), dtype=int) 
    
    state = torch.load(ckpt_path, map_location=device)
    
    # Robust loading: Handle case where state is dict or direct state_dict
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    model.eval()

    # 6. Inference Loop
    preds_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)         # (B, C, T)
            logits = model(batch)            # (B, 1)
            probs = torch.sigmoid(logits).squeeze(-1) # (B,)

            # Threshold at 0.5 for binary classification
            preds = (probs >= 0.5).cpu().numpy().astype(int) 
            preds_list.extend(preds.tolist())
            
    return np.array(preds_list)