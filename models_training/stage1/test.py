#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 - Binary Gait Detection Testing
---------------------------------------
Evaluates trained Binary Models (TCN or TCN-BiLSTM).
Handles path resolution, safe parameter loading, and metric extraction.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any

# -------------------------------------------------------------------------
# 1. Project Root Setup
# -------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Adjust based on file location

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models_training.stage1.dataset import GaitSensorDataset
from models_training.stage1.model import GaitBinaryDetector
from models_training.stage1.backbone_model import TCN_BiLSTM_Backbone, TCNBackbone
from models_training.stage1.utils import update_confmat, metrics_from_cm

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def get_scalar(val):
    """Safely extracts a scalar float from a tensor or numpy array."""
    if isinstance(val, (np.ndarray, list, torch.Tensor)):
        val = np.array(val)
        if val.size > 1:
            return val[-1]  # Return the positive class (Gait)
        return val.item()
    return val

def resolve_path(p: str) -> Path:
    """Resolves a path string to a Path object relative to ROOT if not absolute."""
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj

def eval_split(model, loader, device, obj_thresh=0.5):
    """Runs inference on a dataloader and calculates the Confusion Matrix."""
    model.eval()
    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=100, desc="Eval", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long().view(-1)

            logits = model(x)
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > obj_thresh).long()

            update_confmat(cm, y, preds, num_classes=2)
    return cm

# -------------------------------------------------------------------------
# Main Testing Logic
# -------------------------------------------------------------------------

def test_one_model(model_key: str, model_cfg: Dict[str, Any], global_cfg: Dict[str, Any], use_author_weights: bool):
    print("\n" + "=" * 70)
    print(f"Testing Stage1 Model {model_key}: {model_cfg.get('name', 'Unknown')}")
    print("=" * 70)

    device = torch.device(global_cfg["device"])
    paths_cfg = global_cfg["paths"]
    test_cfg = global_cfg["training"]

    # --- 1. Load Data Configuration ---
    norm_path = resolve_path(paths_cfg["norm_json"])
    test_dict_path = resolve_path(paths_cfg["test_dict"])

    print(f"Loading normalization params from: {norm_path}")
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    norm_params = norm_data.get("normalization_params", norm_data)

    print(f"Loading test dataset list from: {test_dict_path}")
    with open(test_dict_path, "r") as f:
        test_npy = json.load(f)

    # --- 2. Initialize Datasets & Loaders ---
    common_ds_args = {
        "norm_params": norm_params,
        "segment_duration": test_cfg.get("segment_duration", 90),
        "decision_duration": test_cfg.get("decision_duration", 5)
    }
    loader_args = {
        "batch_size": model_cfg["batch_size"],
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": True
    }

    ds_within = GaitSensorDataset(npy_paths=test_npy["within_subject"], **common_ds_args)
    ds_between = GaitSensorDataset(npy_paths=test_npy["between_subject"], **common_ds_args)

    loader_within = DataLoader(ds_within, **loader_args)
    loader_between = DataLoader(ds_between, **loader_args)

    # --- 3. Determine Model Paths & Parameters ---
    backbone_type = model_cfg.get("backbone_type", "tcn")
    
    if use_author_weights:
        print(">> Using Author's Best Weights...")
        best_model_root = ROOT / "models_training" / "stage1" / "best_model"
        
        # Load author's config to get model params
        with open(best_model_root / "config.json", 'r') as f: 
            author_config = json.load(f)
            
        if backbone_type == "tcn":
            output_dir = best_model_root / "tcn"
            m_params = author_config["models"]["A"]["model_params"] 
        else:
            output_dir = best_model_root / "tcn_bilstm"
            m_params = author_config["models"]["B"]["model_params"]
    else:
        print(">> Using User Trained Weights...")
        default_save = f"models_training/stage1/stage1_output/model_{model_key}_{backbone_type}"
        output_dir = resolve_path(model_cfg.get("save_dir", default_save))
        
        with open(output_dir / "train_params.json", 'r') as f:
            m_params = json.load(f)["model_params"]

    # --- 4. Instantiate Model (Safe Parameter Mapping) ---
    if backbone_type == "tcn":
        backbone = TCNBackbone(**m_params) 
    elif backbone_type == "tcn_bilstm":
        backbone = TCN_BiLSTM_Backbone(
            in_channels=m_params.get("in_channels", 12),
            tcn_channels=m_params.get("tcn_channels", [64, 64, 128]),
            tcn_kernel_size=m_params.get("tcn_kernel_size", 15),
            tcn_dilations=m_params.get("tcn_dilations", [8, 16, 32]),
            tcn_causal=m_params.get("tcn_causal", m_params.get("causal", False)),
            lstm_hidden=m_params.get("lstm_hidden", 128),
            lstm_bidirectional=m_params.get("lstm_bidirectional", True),
            lstm_proj_out=m_params.get("lstm_proj_out", m_params.get("proj_dim", 128))
        )
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    model = GaitBinaryDetector(backbone).to(device)

    # --- 5. Load Checkpoint ---
    ckpt_path = output_dir / 'best_model.pth'
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint -> {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state", state))

    # --- 6. Run Evaluation ---
    obj_thresh = model_cfg.get("obj_thresh", 0.5)
    eval_tasks = [("Within-Subject", loader_within), ("Between-Subject", loader_between)]

    # Determine save directory
    save_dir = resolve_path(model_cfg.get("save_dir", f"./stage1_output/{model_key}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, loader in eval_tasks:
        print(f"\n[{name} Evaluation]")
        cm = eval_split(model, loader, device, obj_thresh)
        prec, rec, f1 = metrics_from_cm(cm)
        
        print(f"Precision (Gait)={get_scalar(prec):.4f} "
              f"Recall (Gait)={get_scalar(rec):.4f} "
              f"F1 (Gait)={get_scalar(f1):.4f}")
        
        suffix = "within" if "Within" in name else "between"
        np.save(save_dir / f"cm_{suffix}.npy", cm)

    print(f"\nResults saved to: {save_dir}")

def main():
    config_path = ROOT / "models_training" / "stage1" / "config" / "config.json"
    if not config_path.exists():
        print(f"[Error] Config file not found: {config_path}")
        return

    print(f"Reading configuration from: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    for key, model_cfg in cfg["models"].items():
        if model_cfg.get("enable", False):
            use_weights = model_cfg.get("use_author_weights", True)
            test_one_model(key, model_cfg, cfg, use_author_weights=use_weights)

if __name__ == "__main__":
    main()