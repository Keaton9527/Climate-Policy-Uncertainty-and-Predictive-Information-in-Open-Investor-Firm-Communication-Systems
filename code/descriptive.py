"""Ex-post market regimes and interaction descriptives (Tables 1, 2 and A3)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from panel import Window

def market_regimes(index_df: pd.DataFrame, windows: list[Window]) -> pd.DataFrame:
    index_df = index_df.rename(columns={"Date": "datetime"})
    index_df["datetime"] = pd.to_datetime(index_df["datetime"])
    rows = []
    for window in windows:
        segment = index_df[index_df["datetime"].between(window.test_start, window.test_end)].sort_values("datetime")
        if len(segment) < 2:
            raise AssertionError(f"Missing NEV index data for window {window.window_id}")
        # The paper defines tau in elapsed calendar days, not trading-day row numbers.
        elapsed_days = (segment["datetime"] - segment["datetime"].min()).dt.days.to_numpy()
        slope = float(np.polyfit(elapsed_days, segment["close"].astype(float), 1)[0])
        regime = "Uptrend" if slope > 2 else "Downtrend" if slope < -2 else "Sideways"
        rows.append(
            {
                "Window": window.window_id,
                "TrainingPeriod": f"{window.train_start.date()} to {window.train_end.date()}",
                "ValidationPeriod": f"{window.val_start.date()} to {window.val_end.date()}",
                "TestPeriod": f"{window.test_start.date()} to {window.test_end.date()}",
                "Slope": slope,
                "Trend": regime,
            }
        )
    return pd.DataFrame(rows)


def assign_regime(dates: pd.Series, regimes: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="object")
    normalized = pd.to_datetime(dates)
    for _, row in regimes.iterrows():
        start_text, end_text = row["TestPeriod"].split(" to ")
        mask = normalized.between(pd.Timestamp(start_text), pd.Timestamp(end_text))
        result.loc[mask] = row["Trend"]
    return result


def build_interaction_table(
    panel: pd.DataFrame, interactions: pd.DataFrame, config: dict, regimes: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    interactions = interactions.copy()
    interactions["stock_code"] = interactions["stock_code"].astype(str).str.zfill(6)
    interactions["datetime"] = pd.to_datetime(interactions["datetime"], format="mixed", errors="coerce")
    interactions["reply_time"] = pd.to_datetime(interactions["reply_time"], format="mixed", errors="coerce")
    stocks = set(panel["stock_code"].unique())
    start = pd.Timestamp(config["sample"]["start"])
    end = pd.Timestamp(config["sample"]["end"])
    interactions = interactions[
        interactions["stock_code"].isin(stocks) & interactions["datetime"].between(start, end)
    ].copy()
    interaction_count = len(interactions)
    interactions["question_regime"] = assign_regime(interactions["datetime"], regimes)
    interactions["reply_regime"] = assign_regime(interactions["reply_time"], regimes)
    interactions["answered"] = (interactions["whether_answer"].eq("已回复") | pd.to_numeric(interactions["whether_answer"], errors="coerce").fillna(0).gt(0))
    interactions["observed_lag"] = ((interactions["reply_time"] - interactions["datetime"]).dt.total_seconds() / 86400).round(2)

    order = ["Downtrend", "Sideways", "Uptrend"]
    records: list[dict[str, object]] = []
    for regime in order:
        q = interactions[interactions["question_regime"] == regime]
        regime_windows = regimes[regimes["Trend"] == regime]
        calendar_days = sum(
            (pd.Timestamp(row["TestPeriod"].split(" to ")[1]) - pd.Timestamp(row["TestPeriod"].split(" to ")[0])).days + 1
            for _, row in regime_windows.iterrows()
        )
        records.append(
            {
                "Trend": regime,
                "HalfYearWindows": len(regime_windows),
                "QuestionsPerCalendarDay": len(q) / calendar_days if calendar_days else np.nan,
                "EventualAnswerRatePct": 100 * q["answered"].mean() if len(q) else np.nan,
                "MedianResponseLagDays": q.loc[q["answered"], "observed_lag"].median(),
            }
        )
    return pd.DataFrame(records), interaction_count


def run_descriptive(raw, panel, windows, inputs, config):
    regimes = market_regimes(inputs['nev_index'], windows)
    interactions, _ = build_interaction_table(panel, inputs['qa'], config, regimes)
    features = config['features']['question'] + config['features']['answer']
    stats = panel[features].describe(percentiles=[.25, .5, .75]).T.reset_index()
    stats = stats.rename(columns={'index':'Variable','25%':'P25','50%':'Median','75%':'P75'})
    return regimes, {'table_01_market_regimes':regimes, 'table_02_interactions':interactions, 'table_A3_daily_features':stats}
