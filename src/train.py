"""
train.py

Production-quality CO₂ factor forecasting with Deep Learning.
Predicts next 168 hours (7 days, 1h resolution).

Architecture:
  - PRIMARY:  Temporal Fusion Transformer (TFT) — state-of-the-art for multivariate
              time series with known future covariates (weather forecast)
  - BASELINE: NHiTS — for comparison and report

Framework: PyTorch Lightning + PyTorch Forecasting

Usage:
    python -m src.train --config config.yaml --model tft
    python -m src.train --config config.yaml --model nhits
"""
import argparse
import logging
import os
import warnings

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from src.data import (
    add_time_features,
    build_datasets,
    build_datasets_nhits,
    fill_covariate_nans,
    load_and_prepare,
    make_splits,
)
from src.models import build_nhits, build_tft

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ══════════════════════════════════════════════════════════════════
# PART 3: TRAINING
# ══════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, cfg, model_name="tft"):
    """Train with EarlyStopping, LR scheduling, and checkpointing."""
    save_dir = cfg["save_dir"]
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor   = "val_loss",
            patience  = cfg["early_stopping_patience"],
            mode      = "min",
            verbose   = True,
        ),
        ModelCheckpoint(
            dirpath   = f"{save_dir}/checkpoints",
            filename  = f"{model_name}_best",
            monitor   = "val_loss",
            mode      = "min",
            save_top_k = 1,
            verbose   = True,
        ),
        ModelCheckpoint(
            dirpath    = f"{save_dir}/checkpoints",
            filename   = f"{model_name}_last",
            save_last  = True,
            save_top_k = 0,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs          = cfg["max_epochs"],
        accelerator         = cfg["accelerator"],
        devices             = 1,
        gradient_clip_val   = cfg["gradient_clip_val"],
        callbacks           = callbacks,
        logger              = CSVLogger(f"{save_dir}/logs", name=model_name),
        enable_progress_bar = True,
        log_every_n_steps   = 50,
    )

    best_ckpt   = f"{save_dir}/checkpoints/{model_name}_best.ckpt"
    resume_ckpt = f"{save_dir}/checkpoints/{model_name}_last.ckpt"

    if os.path.exists(resume_ckpt):
        print(f"Resuming {model_name} from last epoch...")
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    else:
        print(f"Starting {model_name} fresh...")
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

    print(f"\nBest checkpoint: {callbacks[1].best_model_path}")
    if callbacks[1].best_model_score is not None:
        print(f"Best val_loss:   {callbacks[1].best_model_score:.4f}")
    return trainer, best_ckpt


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["tft", "nhits"], required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg["seed"], workers=True)

    # ── 1. Load data ──
    df = load_and_prepare(cfg["master_dataset"], cfg["target"])

    # Fill covariate NaNs after loading (was in original load_and_prepare,
    # extracted so the function stays focused on loading + indexing)
    df = fill_covariate_nans(df, cfg["past_covariates"] + cfg["future_covariates"])

    df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx = make_splits(df, cfg)

    print("\nBuilding TimeSeriesDataSets...")

    if args.model == "tft":
        training_ds, train_loader, val_loader, _ = build_datasets(
            df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg)
        model = build_tft(training_ds, cfg)
    else:
        training_ds, train_loader, val_loader, _ = build_datasets_nhits(
            df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg)
        model = build_nhits(training_ds, cfg)
 
    print("\n" + "="*50)
    print(f"TRAINING: {args.model.upper()}")
    print("="*50)
    train_model(model, train_loader, val_loader, cfg, model_name=args.model)


if __name__ == "__main__":
    main()
