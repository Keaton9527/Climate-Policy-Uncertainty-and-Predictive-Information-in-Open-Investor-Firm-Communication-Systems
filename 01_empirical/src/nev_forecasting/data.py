"""Read the processed firm-day panel and construct next-observation targets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .common import resolve

@dataclass(frozen=True)
class Window:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def prepare_panel(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = resolve(config["paths"]["baseline"])
    raw = pd.read_parquet(path)
    raw["stock_code"] = raw["stock_code"].astype(str).str.zfill(6)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values(["stock_code", "datetime"]).reset_index(drop=True)

    expected_stocks = int(config["sample"]["expected_stocks"])
    if raw["stock_code"].nunique() != expected_stocks:
        raise AssertionError(
            f"Expected {expected_stocks} stocks, found {raw['stock_code'].nunique()}"
        )
    if raw.duplicated(["stock_code", "datetime"]).any():
        raise AssertionError("Duplicate stock-date keys detected in input baseline")
    if not (raw.loc[raw["ques_counts"] == 0, "ques_sentiment_score"] == 0).all():
        raise AssertionError("Question sentiment is nonzero where ques_counts is zero")
    if not (raw.loc[raw["ans_counts"] == 0, "ans_sentiment_score"] == 0).all():
        raise AssertionError("Answer sentiment is nonzero where ans_counts is zero")

    raw["target_close"] = raw.groupby("stock_code", sort=False)["close"].shift(-1)
    raw["target_date"] = raw.groupby("stock_code", sort=False)["datetime"].shift(-1)
    raw["target"] = (raw["target_close"] > raw["close"]).astype("Int8")

    start = pd.Timestamp(config["sample"]["start"])
    end = pd.Timestamp(config["sample"]["end"])
    sample = raw[
        raw["datetime"].between(start, end)
        & raw["target_date"].notna()
        & (raw["target_date"] <= end)
    ].copy()
    sample["target"] = sample["target"].astype(int)
    sample = sample.sort_values(["datetime", "stock_code"]).reset_index(drop=True)
    if sample["stock_code"].nunique() != expected_stocks:
        raise AssertionError("The requested sample does not retain the expected stocks")
    return raw, sample


def generate_windows(config: dict) -> list[Window]:
    sample_cfg = config["sample"]
    val_start = pd.Timestamp(sample_cfg["first_validation_start"])
    sample_end = pd.Timestamp(sample_cfg["end"])
    windows: list[Window] = []
    window_id = 1
    while True:
        train_start = val_start - pd.DateOffset(months=int(sample_cfg["train_months"]))
        train_end = val_start - pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=int(sample_cfg["validation_months"])) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=int(sample_cfg["test_months"])) - pd.Timedelta(days=1)
        if test_end > sample_end:
            break
        windows.append(
            Window(window_id, train_start, train_end, val_start, val_end, test_start, test_end)
        )
        val_start += pd.DateOffset(months=int(sample_cfg["step_months"]))
        window_id += 1
    expected = int(sample_cfg["expected_windows"])
    if len(windows) != expected:
        raise AssertionError(f"Expected {expected} windows, generated {len(windows)}")
    for window in windows:
        if not (window.train_end < window.val_start <= window.val_end < window.test_start):
            raise AssertionError(f"Overlapping periods in window {window.window_id}")
    return windows


def window_frame(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return panel[panel["datetime"].between(start, end)].copy()
