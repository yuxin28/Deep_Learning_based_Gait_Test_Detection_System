#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 - Gait Segmentation Inference
-------------------------------------
Segments gait cycles from sliding windows using UNet-based models.
Features:
- Robust parameter mapping (syncs with test.py).
- Supports both Standard (BiGRU) and Attention-based UNet backbones.
- Handles User vs Author weight loading logic.
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# 1. Path Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # pipeline/stage2.py -> Project_Root 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models_training.stage2.model import GaitSegUNet

# -------------------------------------------------------------------------
# Dataset Wrapper
# -------------------------------------------------------------------------
class Stage2Dataset(Dataset):
    """
    Dataset wrapper for gait segmentation inference.
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
        # Transpose to (num_channels, window_size) for UNet (C, T)
        sample = self.data[idx].T  
        
        # Normalize: (C, T) - (C, 1) / (C, 1)
        x = (sample - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return x

# -------------------------------------------------------------------------
# Main Segmentation Function
# -------------------------------------------------------------------------
def segment_gait_test(
    signal_windows: np.ndarray,
    model_type: str = "unet_bigru",
    use_author_weights: bool = True
) -> np.ndarray:
    """
    Perform gait segmentation on test data using a pretrained UNet model.

    Args:
        signal_windows (np.ndarray): Preprocessed signal windows of shape (N, T, C).
        model_type (str): "unet_bigru" (Standard) or "unet_att_gru" (Attention).
        use_author_weights (bool): If True, loads the best pre-trained weights.

    Returns:
        np.ndarray: Probability of each gait test for each time step (N, T, n_classes).
    """
    
    # 1. Load Normalization Parameters
    norm_path = ROOT / "models_training" / "stage2" / "config" / "norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization params not found at {norm_path}")

    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
        norm_params = norm_data.get("normalization_params", norm_data)

    # 2. Determine Config & Weights Path
    if use_author_weights:
        base_dir = ROOT / "models_training" / "stage2" / "best_model"
        config_path = base_dir / "config.json"
        
        if not config_path.exists():
             raise FileNotFoundError(f"Author config not found at {config_path}")

        with open(config_path, 'r') as f:
            author_config = json.load(f)

        if "unet_bigru" in model_type:
            output_dir = base_dir / "unet_bigru"
            # Attempt to grab params from config if they exist, else rely on defaults in class
            model_config = author_config.get("models", {}).get("unet_bigru", {}).get("model_params", {})
        elif "unet_att_gru" in model_type:
            output_dir = base_dir / "unet_att_gru"
            model_config = author_config.get("models", {}).get("unet_att_gru", {}).get("model_params", {})
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    else:
        # Load from user training output directory
        user_config_path = ROOT / "models_training" / "stage2" / "config" / "config.json"
        with open(user_config_path, 'r') as f:
            full_config = json.load(f)
            
        if "unet_bigru" in model_type:
            model_key = "unet_bigru"
            # Fallback to key 'A' if named that way in user config, logic depends on config structure
            if model_key not in full_config["models"]: model_key = "A"
        elif "unet_att_gru" in model_type:
            model_key = "unet_att_gru"
            if model_key not in full_config["models"]: model_key = "B"
        else:
             raise ValueError(f"Unsupported model type: {model_type}")

        save_dir_rel = full_config["models"][model_key].get("save_dir", f"stage2_output/{model_key}")
        output_dir = ROOT / save_dir_rel if not Path(save_dir_rel).is_absolute() else Path(save_dir_rel)

        # Load train_params.json which contains the exact hyperparameters used
        train_params_path = output_dir / "train_params.json"
        if not train_params_path.exists():
            raise FileNotFoundError(f"Training params not found at {train_params_path}")
            
        with open(train_params_path, 'r') as f:
            loaded_json = json.load(f)
            # train_params.json usually dumps the whole config dict or just model_params
            model_config = loaded_json.get("model_params", loaded_json)

    # 3. Setup Data Loader
    dataset = Stage2Dataset(signal_windows, norm_params)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Instantiate Model (Robust Mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Defaults provided here serve as fallbacks if config is missing keys
    model = GaitSegUNet(
        n_channels=model_config.get("n_channels", model_config.get("in_channels", 12)),
        n_classes=model_config.get("n_classes", model_config.get("num_classes", 5)),
        base_filter=model_config.get("base_filter", 64),
        dropout_rate=model_config.get("dropout_rate", 0.1),
        use_attention=model_config.get("use_attention", False),
        use_recurrent_bottleneck=model_config.get("use_recurrent_bottleneck", True),
        use_fusion=model_config.get("use_fusion", False),
    ).to(device)

    # 5. Load Weights
    ckpt_path = output_dir / 'best_model.pth'

    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}")
        return np.zeros((signal_windows.shape[0], signal_windows.shape[1], model_config.get("num_classes", 5)))
    
    state = torch.load(ckpt_path, map_location=device)
    
    # Robust loading: Handle case where state is dict or direct state_dict
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
        
    model.eval()

    # 6. Inference Loop
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)            # (B, C, T)
            logits = model(batch)               # (B, n_classes, T)
            outputs = F.softmax(logits, dim=1)  # (B, n_classes, T)
            
            # Transpose back to (B, T, n_classes) for easy processing
            all_predictions.append(outputs.cpu().numpy().transpose(0, 2, 1))

    return np.concatenate(all_predictions, axis=0)  # (N, T, n_classes)

if __name__ == "__main__":
    # Simple test case (Requires proper data setup)
    dummy_windows = np.random.rand(10, 256, 12)  # 10 windows, 256 time steps, 12 channels
    results = segment_gait_test(
        signal_windows=dummy_windows,
        model_type="unet_bigru",
        use_author_weights=False
    )
    print("Segmentation Results Shape:", results.shape)