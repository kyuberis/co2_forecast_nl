"""
data.py

Data loading, time-feature engineering, chronological splitting,
and TimeSeriesDataSet construction for TFT and NHiTS.
"""

import logging

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════════
# PART 1: DATA PREPARATION
# ══════════════════════════════════════════════════════════════════


def load_and_prepare(path, target):
    """
    Load master dataset, add required columns for TimeSeriesDataSet.
    """
    print(f"Loading {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Drop rows where target is still NaN after fill (long gaps)
    before = len(df)
    df = df.dropna(subset=[target])
    print(f"  Dropped {before - len(df)} rows with NaN target. Remaining: {len(df)}")

    df = df.reset_index()
    if "validfrom" in df.columns:
        pass
    elif "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "timestamp"})
    else:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df = df.rename(columns={"validfrom": "timestamp"})

    df["time_idx"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().div(3600).round().astype(int)
    )
    df["country"] = "NL"  # single group — required by TimeSeriesDataSet

    # Ensure correct dtypes
    df["country"] = df["country"].astype(str)
    df["hour"] = df["hour"].astype(int)
    df["dow"] = df["dow"].astype(int)
    df["month"] = df["month"].astype(int)

    print(f"  Final shape: {df.shape}")
    print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def fill_covariate_nans(df, covariate_cols):
    """Fill any remaining NaNs in covariates"""
    for c in covariate_cols:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    return df


def make_splits(df, cfg):
    """Chronological train/val/test split — NO shuffling.
    Return the indices so that the window's val/test starts strictly after the boundaries.
    """
    train_end = pd.Timestamp(cfg["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["val_end"], tz="UTC")

    train_df = df[df["timestamp"] < train_end].copy()
    val_df = df[df["timestamp"] < val_end].copy()

    max_train_idx = int(train_df["time_idx"].max())
    max_val_idx = int(val_df["time_idx"].max())

    min_val_pred_idx = int(df.loc[df["timestamp"] >= train_end, "time_idx"].min())
    min_test_pred_idx = int(df.loc[df["timestamp"] >= val_end, "time_idx"].min())

    print(f"  Train: {len(train_df)} rows (up to {cfg['train_end']})")
    print(
        f"  Val:   {len(val_df) - len(train_df)} new rows ({cfg['train_end']} → {cfg['val_end']})"
    )
    print(f"  Test:  {len(df) - len(val_df)} new rows ({cfg['val_end']} → end)")
    print(f"  min_val_pred_idx:  {min_val_pred_idx}  (val windows start at TRAIN_END)")
    print(f"  min_test_pred_idx: {min_test_pred_idx} (test windows start at VAL_END)")

    return df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx


def build_datasets(df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg):
    """
    Create PyTorch Forecasting TimeSeriesDataSet objects for TFT.

    IMPORTANT:
      - validation/test -> predict=False (so y_true stays correct)
      - min_prediction_idx -> windows start strictly after boundaries
    """
    target = cfg["target"]
    past_cols = [c for c in cfg["past_covariates"] if c in df.columns]
    future_cols = [c for c in cfg["future_covariates"] if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= max_train_idx],
        time_idx="time_idx",
        target=target,
        group_ids=["country"],
        min_encoder_length=cfg["lookback"] // 2,
        max_encoder_length=cfg["lookback"],
        min_prediction_length=cfg["horizon"],
        max_prediction_length=cfg["horizon"],
        static_categoricals=["country"],
        time_varying_known_reals=future_cols,
        time_varying_unknown_reals=[target] + past_cols,
        target_normalizer=TorchNormalizer(method="robust", center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= max_val_idx],
        predict=False,
        stop_randomization=True,
        min_prediction_idx=min_val_pred_idx,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=min_test_pred_idx,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=cfg["batch_size"], num_workers=2, pin_memory=True
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=cfg["batch_size"] * 2, num_workers=2
    )
    test_loader = test.to_dataloader(train=False, batch_size=cfg["batch_size"] * 2, num_workers=2)

    print(
        f"  Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )
    return training, train_loader, val_loader, test_loader


def build_datasets_nhits(df, max_train_idx, max_val_idx, min_val_pred_idx, min_test_pred_idx, cfg):
    """Separate dataset for NHiTS with fixed encoder length."""
    target = cfg["target"]
    past_cols = [c for c in cfg["past_covariates"] if c in df.columns]
    future_cols = [c for c in cfg["future_covariates"] if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= max_train_idx],
        time_idx="time_idx",
        target=target,
        group_ids=["country"],
        min_encoder_length=cfg["lookback"],
        max_encoder_length=cfg["lookback"],
        min_prediction_length=cfg["horizon"],
        max_prediction_length=cfg["horizon"],
        static_categoricals=["country"],
        time_varying_known_reals=future_cols,
        time_varying_unknown_reals=[target] + past_cols,
        target_normalizer=TorchNormalizer(method="robust", center=True),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= max_val_idx],
        predict=False,
        stop_randomization=True,
        min_prediction_idx=min_val_pred_idx,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=min_test_pred_idx,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=cfg["batch_size"], num_workers=2, pin_memory=True
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=cfg["batch_size"] * 2, num_workers=2
    )
    test_loader = test.to_dataloader(train=False, batch_size=cfg["batch_size"] * 2, num_workers=2)

    return training, train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# TIME FEATURES
# ══════════════════════════════════════════════════════════════════


def add_time_features(df, timezone="Europe/Amsterdam"):
    """
    Add cyclical time features and solar angle proxies.
    These are always known for any future timestamp — safe for forecasting.
    Expects df indexed by a tz-aware DatetimeIndex (UTC or any tz).
    """
    df = df.copy()

    if df.index.tz is None:
        raise ValueError(
            "add_time_features expects a timezone-aware DatetimeIndex. "
            "Localize first, e.g. df.index = df.index.tz_localize('UTC')"
        )

    local_idx = df.index.tz_convert(timezone)

    hour = local_idx.hour
    dow = local_idx.dayofweek
    doy = local_idx.dayofyear
    month = local_idx.month

    # Hour of day (cyclical)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclical)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    # Day of year (cyclical)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Month (cyclical)
    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)

    # Raw
    df["hour"] = hour
    df["dow"] = dow
    df["month"] = month

    # Daylight binary flag (NL proxy)
    summer_offset = np.cos(2 * np.pi * (doy - 172) / 365.25)
    sunrise = 4.5 + 2.0 * (1 - summer_offset) / 2
    sunset = 21.5 - 2.0 * (1 - summer_offset) / 2
    df["is_daylight"] = ((hour >= sunrise) & (hour <= sunset)).astype("int8")

    return df


def get_needed_cols(ds):
    """Return all columns that a TimeSeriesDataSet requires."""
    cols = []
    cols += list(getattr(ds, "group_ids", []) or [])
    cols += [getattr(ds, "time_idx", "time_idx")]

    target = getattr(ds, "target", None)
    if isinstance(target, str):
        cols += [target]
    elif target is not None:
        cols += list(target)

    cols += list(getattr(ds, "static_categoricals", []) or [])
    cols += list(getattr(ds, "static_reals", []) or [])

    cols += list(getattr(ds, "time_varying_known_categoricals", []) or [])
    cols += list(getattr(ds, "time_varying_known_reals", []) or [])

    cols += list(getattr(ds, "time_varying_unknown_categoricals", []) or [])
    cols += list(getattr(ds, "time_varying_unknown_reals", []) or [])

    return set([c for c in cols if c is not None])
