"""Add daily IC/RankIC to the saved rolling-window classification results."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import resolve, write_json
from .metrics import cross_sectional_ic


def evaluate_ranking(config, dirs):
    baseline = pd.read_parquet(resolve(config["paths"]["baseline"]), columns=["stock_code", "datetime", "close"])
    baseline["stock_code"] = baseline.stock_code.astype(str).str.zfill(6)
    baseline["datetime"] = pd.to_datetime(baseline.datetime)
    if baseline.duplicated(["stock_code", "datetime"]).any():
        raise ValueError("Duplicate baseline keys")
    baseline = baseline.sort_values(["stock_code", "datetime"])
    baseline["forward_return"] = baseline.groupby("stock_code").close.shift(-1) / baseline.close - 1
    baseline["target_date"] = baseline.groupby("stock_code").datetime.shift(-1)
    returns = baseline[["stock_code", "datetime", "forward_return", "target_date"]]
    metrics = pd.read_csv(dirs["main_interim"] / "window_metrics.csv")
    if len(metrics) != config["sample"]["expected_windows"] * 12 or metrics.duplicated(["Window", "Model", "Features"]).any():
        raise ValueError("Unexpected main-track metric inventory")
    rows, daily_rows, all_dates = [], [], set()
    for record in metrics.to_dict("records"):
        window, model, spec = record["Window"], record["Model"], record["Features"]
        path = dirs["main_interim"] / "predictions" / f"window_{window:03d}_{model}_{spec}.parquet"
        pred = pd.read_parquet(path)
        pred["stock_code"] = pred.stock_code.astype(str).str.zfill(6)
        pred["datetime"] = pd.to_datetime(pred.datetime)
        if pred.empty or pred.duplicated(["stock_code", "datetime"]).any():
            raise ValueError(f"Empty or duplicate forecasts: {path.name}")
        if not (pred.window_id.eq(window).all() and pred.model.eq(model).all() and pred.feature_set.eq(spec).all()):
            raise ValueError(f"Forecast metadata mismatch: {path.name}")
        if not pred.y_prob.between(0, 1).all() or not pred.y_pred.isin([0, 1]).all():
            raise ValueError(f"Invalid forecasts: {path.name}")
        data = pred.merge(returns, on=["stock_code", "datetime"], how="left", validate="one_to_one")
        if data.forward_return.isna().any() or not data.target_date.gt(data.datetime).all():
            raise ValueError(f"Missing or misaligned forward returns: {path.name}")
        if not data.forward_return.gt(0).astype(int).eq(data.y_true).all():
            raise ValueError(f"Direction labels disagree with returns: {path.name}")
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
    if len(all_dates) != config["sample"]["expected_prediction_dates"]:
        raise ValueError("Unexpected test date coverage")
    result = pd.DataFrame(rows)
    result.to_csv(dirs["main_interim"] / "window_metrics_with_ranking.csv", index=False)
    pd.DataFrame(daily_rows).to_parquet(dirs["main_interim"] / "daily_ranking.parquet", index=False)
    write_json(dirs["main_interim"] / "ranking_coverage.json", {
        "specification_windows": len(result), "test_dates": len(all_dates),
        "min_valid_ic_dates": int(result.ValidICDates.min()),
        "target_signs_verified": True})
    return result
