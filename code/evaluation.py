"""Add daily IC/RankIC to the in-memory rolling-window classification results."""
import numpy as np
import pandas as pd
from metrics import cross_sectional_ic


def evaluate_ranking(features, metrics, forecasts, config):
    baseline = features[["stock_code", "datetime", "close"]].copy()
    baseline["stock_code"] = baseline.stock_code.astype(str).str.zfill(6)
    baseline["datetime"] = pd.to_datetime(baseline.datetime)
    if baseline.duplicated(["stock_code", "datetime"]).any():
        raise ValueError("Duplicate baseline keys")
    baseline = baseline.sort_values(["stock_code", "datetime"])
    baseline["forward_return"] = baseline.groupby("stock_code").close.shift(-1) / baseline.close - 1
    baseline["target_date"] = baseline.groupby("stock_code").datetime.shift(-1)
    returns = baseline[["stock_code", "datetime", "forward_return", "target_date"]]
    if len(metrics) != config["sample"]["expected_windows"] * 12 or metrics.duplicated(["Window", "Model", "Features"]).any():
        raise ValueError("Unexpected main-track metric inventory")
    rows, daily_rows, all_dates = [], [], set()
    for record in metrics.to_dict("records"):
        window, model, spec = record["Window"], record["Model"], record["Features"]
        pred = forecasts[(window, model, spec)].copy()
        pred["stock_code"] = pred.stock_code.astype(str).str.zfill(6)
        pred["datetime"] = pd.to_datetime(pred.datetime)
        if pred.empty or pred.duplicated(["stock_code", "datetime"]).any():
            raise ValueError(f"Empty or duplicate forecasts: {window}/{model}/{spec}")
        if not (pred.window_id.eq(window).all() and pred.model.eq(model).all() and pred.feature_set.eq(spec).all()):
            raise ValueError(f"Forecast metadata mismatch: {window}/{model}/{spec}")
        if not pred.y_prob.between(0, 1).all() or not pred.y_pred.isin([0, 1]).all():
            raise ValueError(f"Invalid forecasts: {window}/{model}/{spec}")
        data = pred.merge(returns, on=["stock_code", "datetime"], how="left", validate="one_to_one")
        if data.forward_return.isna().any() or not data.target_date.gt(data.datetime).all():
            raise ValueError(f"Missing or misaligned forward returns: {window}/{model}/{spec}")
        if not data.forward_return.gt(0).astype(int).eq(data.y_true).all():
            raise ValueError(f"Direction labels disagree with returns: {window}/{model}/{spec}")
        values = []
        for date, group in data.groupby("datetime", sort=True):
            value = cross_sectional_ic(group.y_prob.reset_index(drop=True), group.forward_return.reset_index(drop=True))
            values.append(value)
            daily_rows.append(dict(Window=window, Model=model, Features=spec, date=date, **value))
            all_dates.add(date)
        daily = pd.DataFrame(values)
        rows.append({**record, "IC": daily.ic.mean(), "RankIC": daily.rank_ic.mean(),
                     "TestDates": len(daily), "ValidICDates": int(daily.ic_valid.sum()),
                     "ValidRankICDates": int(daily.rank_ic_valid.sum())})
    return pd.DataFrame(rows)
