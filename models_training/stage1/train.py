#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 - Binary Gait Detection Training
----------------------------------------
Trains TCN or TCN-BiLSTM models for binary classification (Gait vs Non-Gait).
Features real-time loss display and Positive Class (Gait) metric tracking.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

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
from models_training.stage1.focal_loss import FocalLoss
from models_training.stage1.utils import update_confmat, metrics_from_cm

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def resolve_path(p: str) -> Path:
    """Resolves a path string to a Path object relative to ROOT if not absolute."""
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj

def get_pos_scalar(arr):
    """
    Extracts the metric for the Positive Class (Class 1 / Gait).
    Handles scalar conversion from tensor/numpy array.
    """
    if isinstance(arr, (np.ndarray, list, torch.Tensor)):
        arr = np.array(arr)
        if arr.size > 1:
            return arr[1].item() # Return Class 1
        return arr.item()
    return arr

# -------------------------------------------------------------------------
# Training / Evaluation Loop
# -------------------------------------------------------------------------

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    obj_thresh: float = 0.5,
    scaler: GradScaler = None,
    amp_enabled: bool = True,
    is_train: bool = True,
) -> Dict[str, Any]:
    
    if is_train:
        model.train()
        if not hasattr(model, "optimizer"):
            raise RuntimeError("Model has no optimizer attached.")
    else:
        model.eval()

    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, ncols=100, desc="Train" if is_train else "Val", leave=False)
    
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        if is_train:
            model.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()

        current_loss = loss.item()
        running_loss += current_loss
        n_batches += 1
        
        # Real-time loss display in progress bar
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > obj_thresh).long()
            update_confmat(cm, y.long(), preds, num_classes=2)

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "cm": cm}

# -------------------------------------------------------------------------
# Main Training Logic
# -------------------------------------------------------------------------

def train_one_model(model_key: str, model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"Training Stage1 Model {model_key}: {model_cfg.get('name', '')}")
    print("=" * 70)

    device = torch.device(global_cfg["device"])
    train_cfg = global_cfg["training"]
    paths_cfg = global_cfg["paths"]

    # --- 1. Load Data Configuration ---
    norm_path = resolve_path(paths_cfg["norm_json"])
    train_list_path = resolve_path(paths_cfg["train_list"])
    val_dict_path = resolve_path(paths_cfg["val_dict"])

    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    norm_params = norm_data.get("normalization_params", norm_data)

    with open(train_list_path, "r") as f:
        train_npy = json.load(f)
    with open(val_dict_path, "r") as f:
        val_npy = json.load(f)

    # --- 2. Dataset & Loaders ---
    common_args = {
        "norm_params": norm_params,
        "segment_duration": train_cfg.get("segment_duration", 90),
        "decision_duration": train_cfg.get("decision_duration", 5)
    }
    loader_args = {
        "batch_size": model_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True
    }

    train_ds = GaitSensorDataset(train_npy, **common_args)
    val_within_ds = GaitSensorDataset(val_npy["within_subject"], **common_args)
    val_between_ds = GaitSensorDataset(val_npy["between_subject"], **common_args)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_within_loader = DataLoader(val_within_ds, shuffle=False, **loader_args)
    val_between_loader = DataLoader(val_between_ds, shuffle=False, **loader_args)

    # --- 3. Initialize Model ---
    backbone_type = model_cfg.get("backbone_type", "tcn_bilstm")
    m_params = model_cfg.get("model_params", {})

    if backbone_type == "tcn":
        backbone = TCNBackbone(
            in_channels = m_params.get("in_channels", 12),
            channels = m_params.get("channels", [64, 64, 128, 128]),
            kernel_size = m_params.get("kernel_size", 15),
            dilations = m_params.get("dilations", [8, 16, 32, 64, 128]),
            num_stacks = m_params.get("num_stacks", 1),
            dropout = m_params.get("dropout", 0.1),
            norm = m_params.get("norm", "BN"),
            causal = m_params.get("causal", False),
            use_skip = m_params.get("use_skip", True),
            proj_dim = m_params.get("proj_dim", 128)
        )
    elif backbone_type == "tcn_bilstm":
        backbone = TCN_BiLSTM_Backbone(
            in_channels=m_params.get("in_channels", 12),
            tcn_channels=m_params.get("tcn_channels", (64, 64, 128)),
            tcn_kernel_size=m_params.get("tcn_kernel_size", 15),
            tcn_dilations=m_params.get("tcn_dilations", (8, 16, 32)),
            tcn_causal=m_params.get("tcn_causal", m_params.get("causal", False)), 
            lstm_hidden=m_params.get("lstm_hidden", 128),
            lstm_bidirectional=m_params.get("lstm_bidirectional", True),
            lstm_proj_out=m_params.get("lstm_proj_out", m_params.get("proj_dim", 128))
        )
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    backbone = backbone.to(device)
    model = GaitBinaryDetector(backbone=backbone).to(device)
    
    # --- 4. Optimizer & Loss ---
    model.optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=model_cfg.get("lr", 1e-4), 
        weight_decay=model_cfg.get("weight_decay", 0.0)
    )

    focal_cfg = model_cfg.get("focal_loss", {"alpha": 0.75, "gamma": 2.0})
    criterion = FocalLoss(alpha=focal_cfg["alpha"], gamma=focal_cfg["gamma"])
    scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", True))

    # --- 5. Training Loop ---
    epochs = model_cfg.get("epochs", 50)
    save_every = train_cfg.get("save_every", 5)
    patience = train_cfg.get("early_stopping_patience", 10)
    obj_thresh = model_cfg.get("obj_thresh", 0.5)

    save_dir = resolve_path(model_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "train_params.json", "w") as f:
        json.dump(model_cfg, f, indent=4)

    best_val_pos_f1 = -1.0 
    epochs_since_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch [{epoch}/{epochs}] -----")

        # Train
        train_stats = run_one_epoch(model, train_loader, device, criterion, obj_thresh, scaler, train_cfg.get("amp", True), is_train=True)
        _, _, f1_tr_arr = metrics_from_cm(train_stats["cm"])
        
        # Val Within
        val_w_stats = run_one_epoch(model, val_within_loader, device, criterion, obj_thresh, scaler, train_cfg.get("amp", True), is_train=False)
        _, _, f1_w_arr = metrics_from_cm(val_w_stats["cm"])

        # Val Between
        val_b_stats = run_one_epoch(model, val_between_loader, device, criterion, obj_thresh, scaler, train_cfg.get("amp", True), is_train=False)
        _, _, f1_b_arr = metrics_from_cm(val_b_stats["cm"])

        # Extract Positive Class F1
        f1_tr = get_pos_scalar(f1_tr_arr)
        f1_w = get_pos_scalar(f1_w_arr)
        f1_b = get_pos_scalar(f1_b_arr)

        current_val_score = (f1_w + f1_b) / 2.0
        
        print(f"Train      | Loss={train_stats['loss']:.4f} F1(Gait)={f1_tr:.4f}")
        print(f"Val Within | Loss={val_w_stats['loss']:.4f} F1(Gait)={f1_w:.4f}")
        print(f"Val Between| Loss={val_b_stats['loss']:.4f} F1(Gait)={f1_b:.4f}")
        print(f"==> Avg Val F1: {current_val_score:.4f} (Best: {best_val_pos_f1:.4f})")

        # Checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), save_dir / f"ckpt_epoch_{epoch:03d}.pth")

        # Early Stopping
        if current_val_score > best_val_pos_f1:
            best_val_pos_f1 = current_val_score
            best_epoch = epoch
            epochs_since_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_positive_f1': best_val_pos_f1
            }, save_dir / "best_model.pth")
            print(f"★ [Best Model] Saved.")
        else:
            epochs_since_improve += 1
            
        if epochs_since_improve >= patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}. Best F1: {best_val_pos_f1:.4f}")
            break

def main():
    config_path = ROOT / "models_training" / "stage1" / "config" / "config.json"
    if not config_path.exists():
        print(f"[Error] Config not found at {config_path}")
        return

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    for key, model_cfg in cfg["models"].items():
        if model_cfg.get("enable", False):
            train_one_model(key, model_cfg, cfg)

if __name__ == "__main__":
    main()