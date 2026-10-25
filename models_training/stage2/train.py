#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 - Frame-level UNet Training
-----------------------------------
Trains UNet models for multi-class gait segmentation.
Features:
- Robust parameter loading for different UNet variants.
- Real-time loss display.
- Consistent hyperparameter saving for test script compatibility.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from models_training.stage2.dataset import UNetDataset
from models_training.stage2.model import GaitSegUNet
from models_training.stage2.utils import update_confmat, metrics_from_cm 

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def get_scalar(val):
    """Safely extracts a scalar float from a tensor or numpy array."""
    if isinstance(val, (np.ndarray, list, torch.Tensor)):
        val = np.array(val)
        return val.item() if val.size == 1 else val
    return val

def resolve_path(p: str) -> Path:
    """Resolves a path string to a Path object relative to ROOT if not absolute."""
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj

# -------------------------------------------------------------------------
# Training / Evaluation Loop
# -------------------------------------------------------------------------

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
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

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, ncols=100, desc="Train" if is_train else "Val", leave=False)
    
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            model.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                
                # Handle potential size mismatch (e.g. odd sequence lengths)
                if logits.shape[-1] != y.shape[-1]:
                    logits = F.interpolate(logits, size=y.shape[-1], mode="nearest")
                
                loss = criterion(logits, y)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()

        # Update Metrics
        current_loss = loss.item()
        running_loss += current_loss
        n_batches += 1
        
        # Real-time loss display
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            y_pred = probs.argmax(dim=1)
            update_confmat(cm, y, y_pred, num_classes)

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "cm": cm}


# -------------------------------------------------------------------------
# Main Training Logic
# -------------------------------------------------------------------------

def train_one_model(model_key: str, model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"Training Stage2 model {model_key}: {model_cfg.get('name', '')}")
    print("=" * 70)

    device = torch.device(global_cfg["device"])
    train_cfg = global_cfg["training"]
    paths_cfg = global_cfg["paths"]

    # --- 1. Load Data Configuration ---
    norm_path = resolve_path(paths_cfg["norm_json"])
    train_list_path = resolve_path(paths_cfg["train_list"])
    val_dict_path = resolve_path(paths_cfg["val_dict"])

    with open(norm_path, "r") as f:
        norm_params = json.load(f)
    if "normalization_params" in norm_params:
        norm_params = norm_params["normalization_params"]

    with open(train_list_path, "r") as f:
        train_list = json.load(f)
    with open(val_dict_path, "r") as f:
        val_dict = json.load(f)

    # --- 2. Dataset & Loaders ---
    common_args = {
        "norm_params": norm_params,
        "frequency": train_cfg.get("frequency", 100)
    }
    
    # Enable augmentation only for training
    train_ds = UNetDataset(train_list, use_augmentation=True, **common_args)
    val_within = UNetDataset(val_dict.get("within_subject", []), use_augmentation=False, **common_args)
    val_between = UNetDataset(val_dict.get("between_subject", []), use_augmentation=False, **common_args)

    loader_args = {
        "batch_size": model_cfg["batch_size"],
        "num_workers": train_cfg.get("num_workers", 4),
        "pin_memory": True
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader_within = DataLoader(val_within, shuffle=False, **loader_args)
    val_loader_between = DataLoader(val_between, shuffle=False, **loader_args)

    # --- 3. Initialize Model ---
    # Handle potentially nested or flat param structures
    m_params = model_cfg.get("model_params", model_cfg) 
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

    # --- 4. Optimizer & Loss ---
    model.optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=model_cfg.get("lr", 1e-3), 
        weight_decay=model_cfg.get("weight_decay", 0.0)
    )

    class_weights = model_cfg.get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=model_cfg.get("ignore_index", -100))
    scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", True))

    # --- 5. Save Configuration ---
    save_dir = resolve_path(model_cfg.get("save_dir", f"./stage2_output/{model_key}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the entire config dict so test.py can find 'model_params' easily
    print(f"[Info] Saving train_params.json to {save_dir}")
    with open(save_dir / "train_params.json", "w") as f:
        json.dump(model_cfg, f, indent=4)

    # --- 6. Training Loop ---
    best_val_macro = -1.0 
    epochs_no_improve = 0
    epochs = model_cfg.get("epochs", 50)
    patience = train_cfg.get("early_stopping_patience", 10)
    save_every = train_cfg.get("save_every", 5)

    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch [{epoch}/{epochs}] -----")
        
        # Train
        train_stats = run_one_epoch(model, train_loader, device, num_classes, criterion, scaler, train_cfg.get("amp", True), is_train=True)
        acc_tr, macro_tr, _ = metrics_from_cm(train_stats["cm"])

        # Val
        val_stats_within = run_one_epoch(model, val_loader_within, device, num_classes, criterion, scaler, train_cfg.get("amp", True), is_train=False)
        val_stats_between = run_one_epoch(model, val_loader_between, device, num_classes, criterion, scaler, train_cfg.get("amp", True), is_train=False)

        _, macro_in, _ = metrics_from_cm(val_stats_within["cm"])
        _, macro_out, _ = metrics_from_cm(val_stats_between["cm"])

        # Average Macro F1 for Model Selection
        current_val_macro = (macro_in + macro_out) / 2.0

        print(f"Train      | Loss={train_stats['loss']:.4f} Acc={get_scalar(acc_tr):.4f} MacroF1={get_scalar(macro_tr):.4f}")
        print(f"Val Within | Loss={val_stats_within['loss']:.4f} MacroF1={get_scalar(macro_in):.4f}")
        print(f"Val Between| Loss={val_stats_between['loss']:.4f} MacroF1={get_scalar(macro_out):.4f}")
        print(f"==> Overall Val Macro F1: {current_val_macro:.4f} (Best: {best_val_macro:.4f})")

        # Save Interval Checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), save_dir / f"ckpt_epoch_{epoch:03d}.pth")

        # Early Stopping & Best Model
        if current_val_macro > best_val_macro:
            best_val_macro = current_val_macro
            epochs_no_improve = 0
            
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_macro,
            }, save_dir / "best_model.pth")
            print(f"★ [Best Model] Updated! Macro F1: {best_val_macro:.4f}")
        else:
            epochs_no_improve += 1
            print(f"[Info] No improvement for {epochs_no_improve}/{patience} epochs.")

        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
            break

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
            train_one_model(model_key, model_cfg, cfg)

if __name__ == "__main__":
    main()