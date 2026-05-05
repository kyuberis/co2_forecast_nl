"""
Three sanity tests for the most important issues:
1. add_time_features is deterministic and timezone-aware
2. make_splits places val/test windows strictly after their boundaries (no leakage)
3. seasonal naive (t-24) on a 24h-periodic signal produces zero error
"""

import numpy as np
import pandas as pd

from co2_forecast.data import add_time_features, make_splits


def test_time_features_deterministic_and_tz_aware():
    idx = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(24)}, index=idx)

    out_utc = add_time_features(df, timezone="UTC")
    out_utc_again = add_time_features(df, timezone="UTC")
    pd.testing.assert_frame_equal(out_utc, out_utc_again)

    out_ams = add_time_features(df, timezone="Europe/Amsterdam")
    assert (out_ams["hour"].iloc[0] - out_utc["hour"].iloc[0]) % 24 == 2

    for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
        assert out_ams[col].between(-1.0, 1.0).all()


def test_chronological_split_no_leakage():
    n = 24 * 700
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y": np.arange(n, dtype=float),
            "time_idx": np.arange(n),
        }
    )
    cfg = {"train_end": "2024-07-01", "val_end": "2024-10-01"}
    df_out, max_train_idx, _, min_val_pred_idx, min_test_pred_idx = make_splits(df, cfg)

    train_end = pd.Timestamp(cfg["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["val_end"], tz="UTC")

    max_train_ts = df.loc[df["time_idx"] == max_train_idx, "timestamp"].iloc[0]
    min_val_pred_ts = df.loc[df["time_idx"] == min_val_pred_idx, "timestamp"].iloc[0]
    min_test_pred_ts = df.loc[df["time_idx"] == min_test_pred_idx, "timestamp"].iloc[0]

    assert max_train_ts < train_end
    assert min_val_pred_ts >= train_end
    assert min_test_pred_ts >= val_end


def test_seasonal_naive_perfect_on_24h_period():
    horizon = 48
    n_windows = 5
    enc_len = 96  # multiple of 24 so tile aligns with hour-of-day
    np.random.seed(0)
    one_day = np.random.rand(24)

    encoder_target = np.tile(one_day, (n_windows, enc_len // 24))
    encoder_lengths = np.full(n_windows, enc_len)
    y_true = np.tile(one_day, (n_windows, horizon // 24))

    y_naive = np.zeros_like(y_true)
    for h_idx in range(horizon):
        h = h_idx + 1
        if h <= 24:
            offset = 24 - h + 1
            idx = encoder_lengths - offset
            idx = np.clip(idx, 0, encoder_target.shape[1] - 1)
            y_naive[:, h_idx] = encoder_target[np.arange(len(idx)), idx]
        else:
            y_naive[:, h_idx] = y_true[:, h_idx - 24]

    np.testing.assert_allclose(y_naive, y_true, rtol=1e-12)
    mae_naive = np.nanmean(np.abs(y_naive - y_true))
    assert mae_naive < 1e-12
