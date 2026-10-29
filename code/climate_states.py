"""Lagged daily CCPU with training-window quartiles frozen for each test window."""
from __future__ import annotations
import numpy as np
import pandas as pd


def market_calendar(features):
    return pd.DatetimeIndex(sorted(pd.to_datetime(features.datetime).unique()))


def window_boundaries(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics = metrics[(metrics["Track"] == "main") & (metrics["Model"] == "XGBoost") & (metrics["Features"] == "tech")]
    columns = ["Window", "TrainStart", "TrainEnd", "TestStart", "TestEnd"]
    windows = metrics[columns].drop_duplicates().sort_values("Window")
    for column in columns[1:]:
        windows[column] = pd.to_datetime(windows[column]).dt.normalize()
    return windows


def add_ccpu_states(
    panel: pd.DataFrame,
    inputs: dict,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    lag_days: int,
    low_quantile: float,
    high_quantile: float,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccpu = inputs["ccpu"].set_index("date")["CCPU"]
    calendar = market_calendar(features)
    calendar_frame = pd.DataFrame({"datetime": calendar})
    calendar_frame[f"{prefix}_date"] = calendar_frame["datetime"] - pd.to_timedelta(lag_days, unit="D")
    calendar_frame[f"{prefix}_value"] = calendar_frame[f"{prefix}_date"].map(ccpu)
    if calendar_frame[f"{prefix}_value"].isna().any():
        raise AssertionError(f"Missing aligned CCPU values for {prefix}")

    windows = window_boundaries(metrics)
    thresholds: list[dict] = []
    for row in windows.itertuples(index=False):
        train = calendar_frame[calendar_frame["datetime"].between(row.TrainStart, row.TrainEnd)]
        values = train[f"{prefix}_value"]
        thresholds.append(
            {
                "window_id": int(row.Window),
                "lag_days": int(lag_days),
                "low_quantile": float(low_quantile),
                "high_quantile": float(high_quantile),
                "low_threshold": float(values.quantile(low_quantile)),
                "high_threshold": float(values.quantile(high_quantile)),
                "train_start": row.TrainStart,
                "train_end": row.TrainEnd,
                "training_dates": len(values),
            }
        )
    threshold_frame = pd.DataFrame(thresholds)
    output = panel.merge(
        calendar_frame[["datetime", f"{prefix}_date", f"{prefix}_value"]],
        on="datetime",
        how="left",
        validate="many_to_one",
    ).merge(threshold_frame, on="window_id", how="left", validate="many_to_one")
    state = np.full(len(output), "middle", dtype=object)
    state[output[f"{prefix}_value"].to_numpy() <= output["low_threshold"].to_numpy()] = "low"
    state[output[f"{prefix}_value"].to_numpy() >= output["high_threshold"].to_numpy()] = "high"
    output[f"{prefix}_state"] = state
    if output[f"{prefix}_value"].isna().any():
        raise AssertionError(f"Prediction panel has missing CCPU values for {prefix}")
    return output, threshold_frame
