"""Daily classification metrics, validation thresholds and cross-sectional ranking."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

def mean_daily_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    dates: Iterable,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    )
    values: list[dict[str, float]] = []
    for _, group in frame.groupby("date", sort=True):
        truth = group["y_true"].to_numpy()
        pred = group["y_pred"].to_numpy()
        prob = group["y_prob"].to_numpy()
        auc = roc_auc_score(truth, prob) if len(np.unique(truth)) == 2 else 0.5
        values.append(
            {
                "MCC": matthews_corrcoef(truth, pred) if len(np.unique(pred)) > 1 else 0.0,
                "DA": accuracy_score(truth, pred),
                "Precision": precision_score(truth, pred, zero_division=0),
                "Recall": recall_score(truth, pred, zero_division=0),
                "F1": f1_score(truth, pred, zero_division=0),
                "AUC": auc,
            }
        )
    return pd.DataFrame(values).mean().to_dict()


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, dates: Iterable, config: dict) -> float:
    cfg = config["threshold"]
    thresholds = np.arange(float(cfg["minimum"]), float(cfg["maximum"]) + 1e-9, float(cfg["step"]))
    truth = np.asarray(y_true, dtype=np.int8)
    date_codes, unique_dates = pd.factorize(pd.to_datetime(dates), sort=True)
    group_count = len(unique_dates)
    scored: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(np.int8)
        tp = np.bincount(date_codes, weights=((truth == 1) & (pred == 1)), minlength=group_count)
        tn = np.bincount(date_codes, weights=((truth == 0) & (pred == 0)), minlength=group_count)
        fp = np.bincount(date_codes, weights=((truth == 0) & (pred == 1)), minlength=group_count)
        fn = np.bincount(date_codes, weights=((truth == 1) & (pred == 0)), minlength=group_count)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        daily_mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator), where=denominator > 0)
        mcc = float(daily_mcc.mean())
        scored.append((float(mcc), -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[2]

def cross_sectional_ic(probabilities: Iterable[float], returns: Iterable[float]) -> dict:
    frame = pd.DataFrame(
        {
            "probability": pd.to_numeric(pd.Series(probabilities), errors="coerce"),
            "return": pd.to_numeric(pd.Series(returns), errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    n_stocks = int(len(frame))
    result = {
        "n_stocks": n_stocks,
        "ic": np.nan,
        "rank_ic": np.nan,
        "ic_valid": False,
        "rank_ic_valid": False,
        "invalid_reason": "",
    }
    if n_stocks < 3:
        result["invalid_reason"] = "fewer_than_3_stocks"
        return result
    if frame["probability"].nunique(dropna=True) < 2:
        result["invalid_reason"] = "constant_probability"
        return result
    if frame["return"].nunique(dropna=True) < 2:
        result["invalid_reason"] = "constant_return"
        return result
    ic = float(pearsonr(frame["probability"], frame["return"]).statistic)
    rank_ic = float(spearmanr(frame["probability"], frame["return"]).statistic)
    if not (np.isfinite(ic) and np.isfinite(rank_ic)):
        result["invalid_reason"] = "nonfinite_correlation"
        return result
    result.update(
        {
            "ic": ic,
            "rank_ic": rank_ic,
            "ic_valid": True,
            "rank_ic_valid": True,
            "invalid_reason": "valid",
        }
    )
    return result
