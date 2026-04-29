"""
evaluate.py

PART 4: EVALUATION

Computes MAE, RMSE, MAPE per forecast horizon, seasonal naive baseline,
worst forecast windows, and saves plots.

Usage:
    python -m src.evaluate --config config.yaml
    python -m src.evaluate --config config.yaml --models tft
"""
import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models import NHiTS

from src.data import (
    add_time_features,
    build_datasets,
    build_datasets_nhits,
    fill_covariate_nans,
    load_and_prepare,
    make_splits,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def evaluate(model, test_loader, df_test, cfg, model_name="tft"):
    """
    Evaluate on test set. Computes MAE, RMSE, MAPE per forecast horizon.
    Saves plots: predictions vs actuals + error by horizon step.
    """
    HORIZON = cfg["horizon"]
    save_dir = cfg["save_dir"]
    os.makedirs(f"{save_dir}/predictions", exist_ok=True)

    # Get raw predictions (median = q50)
    predictions = model.predict(
        test_loader,
        return_y=True,
        return_x=True,
        trainer_kwargs={"accelerator": "auto"},
    )
    print("x keys:", list(predictions.x.keys()))
    y_true = predictions.y[0].cpu().numpy()
    if predictions.output.dim() == 3:
        y_pred = predictions.output[:, :, 1].cpu().numpy()   # TFT — [q10, q50, q90]
    else:
        y_pred = predictions.output.cpu().numpy()             # NHiTS

    abs_err = np.abs(y_pred - y_true)

    # =========================
    # Seasonal naive baseline (t-24) — padding-safe
    # =========================
    x = predictions.x
    encoder_target = x["encoder_target"].detach().cpu().numpy()      # [N, max_enc_len]
    encoder_lengths = x["encoder_lengths"].detach().cpu().numpy()     # [N]
    y_true = predictions.y[0].detach().cpu().numpy()                  # [N, H]

    y_naive = np.zeros_like(y_true)

    for h_idx in range(HORIZON):
        h = h_idx + 1
        if h <= 24:
            # t-(24-h+1) inside the encoder, indexed by encoder_length
            offset = (24 - h + 1)                       # 24..1
            idx = encoder_lengths - offset              # [N] indexes in encoder_target
            idx = np.clip(idx, 0, encoder_target.shape[1] - 1)
            y_naive[:, h_idx] = encoder_target[np.arange(len(idx)), idx]
        else:
            y_naive[:, h_idx] = y_true[:, h_idx - 24]

    mae_naive  = np.nanmean(np.abs(y_naive - y_true))
    rmse_naive = np.sqrt(np.nanmean((y_naive - y_true) ** 2))

    print(f"  Seasonal Naive (t-24) MAE:  {mae_naive:.4f} kg CO₂/kWh")
    print(f"  Seasonal Naive (t-24) RMSE: {rmse_naive:.4f} kg CO₂/kWh")

    # Overall metrics
    mae  = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    mape = (np.abs((y_pred - y_true) / (y_true + 1e-8))).mean() * 100

    print(f"\n{'='*50}")
    print(f"TEST SET METRICS — {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  MAE:  {mae:.4f} kg CO₂/kWh")
    print(f"  RMSE: {rmse:.4f} kg CO₂/kWh")
    print(f"  MAPE: {mape:.2f}%")

    # Per-horizon metrics (MAE at each step h=1..168)
    mae_per_h  = abs_err.mean(axis=0)
    rmse_per_h = np.sqrt((abs_err ** 2).mean(axis=0))

    print(f"  MAE 1–24h:    {mae_per_h[:24].mean():.4f}")
    print(f"  MAE 25–72h:   {mae_per_h[24:72].mean():.4f}")
    print(f"  MAE 73–168h:  {mae_per_h[72:].mean():.4f}")

    true_level = y_true.reshape(-1)
    err_level = abs_err.reshape(-1)

    bins = pd.qcut(true_level, q=10, duplicates="drop")
    df_level = pd.DataFrame({"bin": bins.astype(str), "abs_err": err_level})
    level_mae = df_level.groupby("bin")["abs_err"].mean().reset_index()
    level_mae.to_csv(f"{save_dir}/predictions/{model_name}_mae_by_true_decile.csv", index=False)

    # ==========================================
    # Worst forecast windows (top-10 by MAE)
    # ==========================================

    window_mae = abs_err.mean(axis=1)  # MAE per 168h window

    worst_idx = np.argsort(-window_mae)[:10]
    worst_table = pd.DataFrame({
        "window_rank": np.arange(1, 11),
        "window_id": worst_idx,
        "window_mae": window_mae[worst_idx],
    })

    worst_table.to_csv(f"{save_dir}/predictions/{model_name}_worst_windows.csv", index=False)
    print(f"  Worst windows saved to {save_dir}/predictions/{model_name}_worst_windows.csv")

    # Save 3 worst windows as figures
    # FIX: original code had savefig OUTSIDE the loop, so only the last figure
    # was saved (and overwritten each iteration). Now each window gets its own file.
    for j, wi in enumerate(worst_idx[:3], start=1):
        plt.figure(figsize=(14, 5))
        plt.plot(y_true[wi], label="Actual", linewidth=1.5)
        plt.plot(y_pred[wi], label="Predicted", linestyle="--", linewidth=1.5)

        # Q10/Q90 for TFT
        if predictions.output.dim() == 3:
            q10 = predictions.output[wi, :, 0].detach().cpu().numpy()
            q90 = predictions.output[wi, :, 2].detach().cpu().numpy()
            plt.fill_between(np.arange(HORIZON), q10, q90, alpha=0.2, label="80% interval")

        plt.title(f"{model_name.upper()} — Worst window #{j} (MAE={window_mae[wi]:.4f})")
        plt.xlabel("Hour ahead")
        plt.ylabel("CO₂ factor")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/predictions/{model_name}_worst_window_{j}.png", dpi=150)
        plt.close()

    # ── Plot 1: Predictions vs Actuals (first 2 weeks of test) ──
    # FIX: original code had the entire plotting block (including savefig)
    # nested inside `if predictions.output.dim() == 3`, so for NHiTS the
    # plot was never saved. Now savefig runs unconditionally; only the
    # quantile band is conditional.
    fig, ax = plt.subplots(figsize=(14, 5))
    sample = 0   # representative window
    ax.plot(y_true[sample], label="Actual CO₂ factor", color="steelblue", linewidth=1.5)
    ax.plot(y_pred[sample], label="Predicted (median)", color="tomato",   linewidth=1.5, linestyle="--")
    if predictions.output.dim() == 3:
        q10 = predictions.output[sample, :, 0].cpu().numpy()
        q90 = predictions.output[sample, :, 2].cpu().numpy()
        ax.fill_between(range(HORIZON), q10, q90, alpha=0.2, color="tomato", label="80% interval")
    ax.set_title(f"{model_name.upper()} — Predictions vs Actuals (168h window)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("CO₂ factor [kg/kWh]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/predictions/{model_name}_pred_vs_actual.png", dpi=150)
    plt.close()

    # ── Plot 2: MAE by forecast horizon step ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(1, HORIZON + 1), mae_per_h, color="steelblue", linewidth=1.5)
    ax.axvline(24,  color="gray", linestyle="--", alpha=0.5, label="24h")
    ax.axvline(72,  color="gray", linestyle=":",  alpha=0.5, label="72h")
    ax.axvline(168, color="gray", linestyle="-.", alpha=0.5, label="168h")
    ax.set_title(f"{model_name.upper()} — MAE by Forecast Horizon")
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE [kg CO₂/kWh]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/predictions/{model_name}_mae_by_horizon.png", dpi=150)
    plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        "horizon_h": range(1, HORIZON + 1),
        "mae":  mae_per_h,
        "rmse": rmse_per_h,
    })
    metrics_df.to_csv(f"{save_dir}/predictions/{model_name}_metrics_by_horizon.csv", index=False)

    print(f"  Saved plots to {save_dir}/predictions/{model_name}_*.png")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def compare_models(metrics_tft, metrics_nhits):
    """Print side-by-side comparison table."""
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Metric':<10} {'TFT':>12} {'NHiTS':>12}")
    print("-" * 36)
    for k in ["mae", "rmse", "mape"]:
        unit = "%" if k == "mape" else "kg/kWh"
        print(f"{k.upper():<10} {metrics_tft[k]:>12.4f} {metrics_nhits[k]:>12.4f}  {unit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", choices=["tft", "nhits"], default=["tft", "nhits"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["save_dir"]

    df = load_and_prepare(cfg["master_dataset"], cfg["target"])
    df = fill_covariate_nans(df, cfg["past_covariates"] + cfg["future_covariates"])
    df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx = make_splits(df, cfg)

    metrics_tft = None
    metrics_nhits = None

    if "tft" in args.models:
        ckpt_tft = f"{save_dir}/checkpoints/tft_best.ckpt"
        if not os.path.exists(ckpt_tft):
            print(f"Skipping TFT: {ckpt_tft} not found")
        else:
            _, _, _, test_loader = build_datasets(
                df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg)
            best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_tft)
            print("\nEvaluating TFT on test set...")
            metrics_tft = evaluate(best_tft, test_loader, df, cfg, model_name="tft")

    if "nhits" in args.models:
        ckpt_nhits = f"{save_dir}/checkpoints/nhits_best.ckpt"
        if not os.path.exists(ckpt_nhits):
            print(f"Skipping NHiTS: {ckpt_nhits} not found")
        else:
            _, _, _, test_loader_nhits = build_datasets_nhits(
                df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg)
            best_nhits = NHiTS.load_from_checkpoint(ckpt_nhits)
            print("\nEvaluating NHiTS on test set...")
            metrics_nhits = evaluate(best_nhits, test_loader_nhits, df, cfg, model_name="nhits")

    if metrics_tft and metrics_nhits:
        compare_models(metrics_tft, metrics_nhits)


if __name__ == "__main__":
    main()
