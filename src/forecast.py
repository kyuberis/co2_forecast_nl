"""
forecast.py

PART 5: INFERENCE (actual 7-day forecast)

Run a 168h forecast using the trained TFT and an Open-Meteo weather forecast.

Usage:
    python -m src.forecast --config config.yaml
"""
import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.data import (
    add_time_features,
    build_datasets,
    fill_covariate_nans,
    get_needed_cols,
    load_and_prepare,
    make_splits,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def run_inference_with_weather_forecast(
    best_ckpt_path,
    training_dataset,
    df_full,
    weather_forecast_csv,
    cfg,
    country_col="country",
    country_value="NL",
    time_col="timestamp",
):
    """
    168h real inference WITH weather forecast:
      - encoder: last `lookback` hours from df_full
      - decoder: next `horizon` hours with Open-Meteo forecast weather_* + deterministic time features
    """
    target_col = cfg["target"]
    lookback   = cfg["lookback"]
    horizon    = cfg["horizon"]
    save_dir   = cfg["save_dir"]

    # ---- 1) Load model
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt_path)
    model.eval()

    # ---- 2) History (context)
    df_hist = df_full.sort_values(time_col).copy()
    df_context = df_hist.tail(lookback).copy()
    last_ts = pd.to_datetime(df_hist[time_col].iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")

    # ---- 3) Future timestamps
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
        tz="UTC",
    )
    df_future = pd.DataFrame({time_col: future_ts})
    df_future[country_col] = country_value
    df_future[target_col] = np.nan  # unknown future target

    # ---- 4) Load Open-Meteo 7-day forecast and join
    df_wfc = pd.read_csv(weather_forecast_csv)
    if "validfrom" in df_wfc.columns:
        df_wfc["validfrom"] = pd.to_datetime(df_wfc["validfrom"], utc=True)
        df_wfc = df_wfc.rename(columns={"validfrom": time_col})
    elif "time" in df_wfc.columns:
        df_wfc["time"] = pd.to_datetime(df_wfc["time"], utc=True)
        df_wfc = df_wfc.rename(columns={"time": time_col})
    elif time_col in df_wfc.columns:
        df_wfc[time_col] = pd.to_datetime(df_wfc[time_col], utc=True)
    else:
        # common case: first unnamed column is datetime index
        first = df_wfc.columns[0]
        df_wfc[first] = pd.to_datetime(df_wfc[first], utc=True)
        df_wfc = df_wfc.rename(columns={first: time_col})

    # keep only weather_*
    weather_cols = [c for c in df_wfc.columns if c.startswith("weather_")]
    df_wfc = df_wfc[[time_col] + weather_cols].drop_duplicates(subset=[time_col])

    # align to exactly the horizon timestamps
    df_future = df_future.merge(df_wfc, on=time_col, how="left")

    # If forecast does not fully cover horizon (rare), forward-fill then 0-fill
    for c in weather_cols:
        df_future[c] = df_future[c].ffill().fillna(0)

    # ---- 5) Add deterministic time features for future rows (must match training)
    df_future = add_time_features(df_future.set_index(time_col), timezone=cfg["timezone"]).reset_index()

    # ---- 6) Create consistent time_idx (same reference as training)
    min_ts = pd.to_datetime(df_hist[time_col].min())
    if min_ts.tzinfo is None:
        min_ts = min_ts.tz_localize("UTC")
    else:
        min_ts = min_ts.tz_convert("UTC")

    def _make_time_idx(s):
        s = pd.to_datetime(s)
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return (
            (s - min_ts)
            .dt.total_seconds()
            .div(3600)
            .round()
            .astype(int)
        )

    if "time_idx" not in df_context.columns:
        df_context["time_idx"] = _make_time_idx(df_context[time_col])

    df_future["time_idx"] = _make_time_idx(df_future[time_col])

    # ---- 7) Combine context + future
    df_combined = pd.concat([df_context, df_future], ignore_index=True)
    last_y = float(df_context[target_col].iloc[-1])
    df_combined[target_col] = df_combined[target_col].fillna(last_y)

    # ---- 8) Ensure all columns required by training_dataset exist
    needed_cols = get_needed_cols(training_dataset)
    for col in needed_cols:
        if col not in df_combined.columns:
            df_combined[col] = 0

    # Fill NaNs in non-target numeric columns
    for col in df_combined.columns:
        if col == target_col:
            continue
        if df_combined[col].dtype.kind in "biufc":
            df_combined[col] = df_combined[col].fillna(0)

    # ---- 9) Build inference dataset/loader
    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_combined,
        predict=True,
        stop_randomization=True,
    )
    inference_loader = inference_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    # ---- 10) Predict
    preds = model.predict(inference_loader, mode="quantiles")
    if hasattr(preds, "dim") and preds.dim() == 3 and preds.shape[-1] >= 3:
        y_q10 = preds[0, :, 0].detach().cpu().numpy()
        y_q50 = preds[0, :, 1].detach().cpu().numpy()
        y_q90 = preds[0, :, 2].detach().cpu().numpy()
    else:
        y_q50 = preds[0].detach().cpu().numpy() if hasattr(preds, "detach") else np.array(preds).squeeze()
        y_q10 = y_q50
        y_q90 = y_q50

    result = pd.DataFrame({
        "timestamp": future_ts,
        "co2_forecast_q10": y_q10[:horizon],
        "co2_forecast_q50": y_q50[:horizon],
        "co2_forecast_q90": y_q90[:horizon],
    })

    os.makedirs(f"{save_dir}/predictions", exist_ok=True)
    result.to_csv(f"{save_dir}/predictions/co2_forecast_168h_with_weather.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result["timestamp"], result["co2_forecast_q50"], label="Forecast (median)", linewidth=2)
    ax.fill_between(
        result["timestamp"],
        result["co2_forecast_q10"],
        result["co2_forecast_q90"],
        alpha=0.2,
        label="80% interval"
    )
    ax.set_title("CO₂ Factor — 7-Day Forecast (With Open-Meteo Weather Forecast)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("CO₂ factor [kg/kWh]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/predictions/co2_forecast_7days_with_weather.png", dpi=150)
    plt.close()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["save_dir"]
    ckpt_tft = f"{save_dir}/checkpoints/tft_best.ckpt"
    if not os.path.exists(ckpt_tft):
        raise FileNotFoundError(f"TFT checkpoint not found at {ckpt_tft}. Train it first.")

    df = load_and_prepare(cfg["master_dataset"], cfg["target"])
    df = fill_covariate_nans(df, cfg["past_covariates"] + cfg["future_covariates"])
    df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx = make_splits(df, cfg)

    training_ds, _, _, _ = build_datasets(
        df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg)

    forecast_df = run_inference_with_weather_forecast(
        best_ckpt_path=ckpt_tft,
        training_dataset=training_ds,
        df_full=df,
        weather_forecast_csv=cfg["weather_forecast"],
        cfg=cfg,
    )
    print(forecast_df.head())


if __name__ == "__main__":
    main()
