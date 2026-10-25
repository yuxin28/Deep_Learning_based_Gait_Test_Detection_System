#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 - Frame-level UNet Testing
----------------------------------
Evaluates trained UNet models for gait segmentation.
Handles consistent path resolution, weight loading logic (User vs Author),
and metric extraction.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------------------------------------------------
# 1. Project Root Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Adjust based on file location

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models_training.stage2.dataset import UNetDataset
from models_training.stage2.model import GaitSegUNet
from models_training.stage2.utils import update_confmat, metrics_from_cm

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def get_scalar(val):
    """Safely extracts a scalar from a tensor or numpy array."""
    if isinstance(val, (np.ndarray, list, torch.Tensor)):
        val = np.array(val)
        return val.item() if val.size == 1 else val
    return val

def resolve_path(p: str) -> Path:
    """Resolves a path string to a Path object relative to ROOT if not absolute."""
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj

def eval_split(model, loader, device, num_classes):
    """Runs inference on a dataloader and calculates the Confusion Matrix."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=100, desc="Eval", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            
            # Handle potential size mismatch due to pooling/upsampling (off by 1 pixel)
            if logits.shape[-1] != y.shape[-1]:
                logits = F.interpolate(logits, size=y.shape[-1], mode="nearest")

            probs = F.softmax(logits, dim=1)
            y_pred = probs.argmax(dim=1)

            update_confmat(cm, y, y_pred, num_classes)
    return cm

# -------------------------------------------------------------------------
# Main Testing Logic
# -------------------------------------------------------------------------

def test_one_model(model_key: str, model_cfg: Dict[str, Any], global_cfg: Dict[str, Any], use_author_weights: bool):
    print("\n" + "=" * 70)
    print(f"Testing Stage2 model {model_key}: {model_cfg.get('model_type', 'Unknown')}")
    print("=" * 70)

    device = torch.device(global_cfg["device"])
    test_cfg = global_cfg["training"]
    paths_cfg = global_cfg["paths"]

    # --- 1. Load Data Configuration ---
    norm_json_path = resolve_path(paths_cfg["norm_json"])
    test_dict_path = resolve_path(paths_cfg["test_dict"])

    with open(norm_json_path, "r") as f:
        norm_params = json.load(f)
    if "normalization_params" in norm_params:
        norm_params = norm_params["normalization_params"]

    with open(test_dict_path, "r") as f:
        test_dict = json.load(f)

    # --- 2. Initialize Datasets & Loaders ---
    common_args = {
        "norm_params": norm_params,
        "frequency": test_cfg.get("frequency", 100),
        "use_augmentation": False
    }
    loader_args = {
        "batch_size": model_cfg["batch_size"],
        "shuffle": False,
        "num_workers": 4
    }

    ds_within = UNetDataset(npy_paths=test_dict.get("within_subject", []), **common_args)
    ds_between = UNetDataset(npy_paths=test_dict.get("between_subject", []), **common_args)

    loader_within = DataLoader(ds_within, **loader_args)
    loader_between = DataLoader(ds_between, **loader_args)

    # --- 3. Determine Parameters & Paths ---
    model_type = model_cfg.get("model_type", "unet_bigru") 

    if use_author_weights:
        print(">> Using Author's Best Weights...")
        best_model_root = ROOT / "models_training" / "stage2" / "best_model"
        
        # Load unified config if available, otherwise assume direct mapping
        config_path = best_model_root / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                author_config = json.load(f)
            # Use author_config parameters if they exist, otherwise placeholders
            m_params = author_config.get("models", {}).get(model_key, {}).get("model_params", {})
        else:
             m_params = {} # Fallback

        if "unet_att_gru" in model_type or model_key == "unet_att_gru":
            output_dir = best_model_root / "unet_att_gru"
        else:
            output_dir = best_model_root / "unet_bigru"
    else:
        print(">> Using User Trained Weights...")
        save_dir = model_cfg.get("save_dir", f"models_training/stage2/stage2_output/{model_key}")
        output_dir = resolve_path(save_dir)
        
        with open(output_dir / "train_params.json", 'r') as f:
            m_params = json.load(f)

    # --- 4. Build Model (Robust Mapping) ---
    num_classes = m_params.get("n_classes", m_params.get("num_classes", 5))
    
    model = GaitSegUNet(
        n_channels=m_params.get("n_channels", m_params.get("in_channels", 12)),
        n_classes=num_classes,
        base_filter=m_params.get("base_filter", 64),
        dropout_rate=m_params.get("dropout_rate", 0.1),
        use_attention=m_params.get("use_attention", False),
        use_recurrent_bottleneck=m_params.get("use_recurrent_bottleneck", True),
        use_fusion=m_params.get("use_fusion", False),
    ).to(device)

    # --- 5. Load Checkpoint ---
    ckpt_path = output_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        return

    print(f"Loading checkpoint -> {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle state dict wrapped in 'model_state' or standalone
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # --- 6. Run Evaluation ---
    eval_tasks = [("Within-Subject", loader_within), ("Between-Subject", loader_between)]
    save_out = resolve_path(model_cfg.get("save_dir", f"./stage2_output/{model_key}"))
    save_out.mkdir(parents=True, exist_ok=True)

    for name, loader in eval_tasks:
        print(f"\n[{name} Evaluation]")
        cm = eval_split(model, loader, device, num_classes)
        acc, macro, wf1 = metrics_from_cm(cm)
        
        print(f"Acc={get_scalar(acc):.4f} MacroF1={get_scalar(macro):.4f} WeightedF1={get_scalar(wf1):.4f}")
        
        suffix = "within" if "Within" in name else "between"
        np.save(save_out / f"cm_{suffix}.npy", cm)

    print(f"\nConfusion matrices saved to {save_out}")

def main():
    config_path = ROOT / "models_training" / "stage2" / "config" / "config.json"
    if not config_path.exists():
        print(f"[Error] Config not found: {config_path}")
        return

    print(f"Reading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    for model_key, model_cfg in cfg["models"].items():
        if model_cfg.get("enable", False):
            use_weights = model_cfg.get("use_author_weights", True)
            test_one_model(model_key, model_cfg, cfg, use_author_weights=use_weights)

if __name__ == "__main__":
    main()