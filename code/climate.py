"""Paired forecasts and bootstrap primitives shared by Section 4.4 analyses."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from climate_states import add_ccpu_states

SPECS = ["tech", "tech_answer", "tech_question", "tech_all"]

def block_counts(n, reps=5000, block=20, seed=42):
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(reps, int(np.ceil(n/block))))
    idx = ((starts[:,:,None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    counts = np.empty((reps+1, n), dtype=np.int16)
    counts[0] = 1
    for k in range(reps):
        counts[k+1] = np.bincount(idx[k], minlength=n)
    return counts


def holm(values):
    p = np.asarray(values, float)
    out = np.full(len(p), np.nan)
    ix = np.where(np.isfinite(p))[0]
    order = ix[np.argsort(p[ix])]
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return out


def uncertainty(x):
    estimate, boot = x[0], x[1:]
    valid = boot[np.isfinite(boot)]
    if not len(valid):
        return dict(estimate=estimate, ci_low=np.nan, ci_high=np.nan, p_value=np.nan, valid_repetitions=0)
    # Centered bootstrap test of H0: contrast=0; two-sided, finite-simulation correction.
    p = (1 + np.count_nonzero(np.abs(valid-estimate) >= abs(estimate))) / (len(valid)+1)
    return dict(estimate=estimate, ci_low=np.quantile(valid,.025), ci_high=np.quantile(valid,.975), p_value=p, valid_repetitions=len(valid))


def load_forecasts(forecasts, expected_windows):
    keys = ["window_id", "stock_code", "datetime"]
    panel = None
    for spec in SPECS:
        frames = [v for (w, model, name), v in sorted(forecasts.items()) if model == 'XGBoost' and name == spec]
        if len(frames) != expected_windows:
            raise ValueError(f'{spec}: missing rolling forecasts')
        frame = pd.concat(frames, ignore_index=True)
        frame["stock_code"] = frame.stock_code.astype(str).str.zfill(6)
        frame["datetime"] = pd.to_datetime(frame.datetime)
        if frame.duplicated(keys).any() or frame.duplicated(["stock_code", "datetime"]).any():
            raise ValueError("Forecast stock/date keys are not unique")
        frame = frame[keys + ["y_true", "y_prob", "y_pred"]]
        frame = frame.rename(columns={"y_prob": f"y_prob_{spec}", "y_pred": f"y_pred_{spec}"})
        if panel is None:
            panel = frame
        else:
            merged = panel.merge(frame, on=keys, how="outer", validate="one_to_one", suffixes=("", "_check"), indicator=True)
            if not merged._merge.eq("both").all() or not merged.y_true.eq(merged.y_true_check).all():
                raise ValueError("Paired specifications have different keys or labels")
            panel = merged.drop(columns=["y_true_check", "_merge"])
    return panel.sort_values(["datetime", "stock_code"]).reset_index(drop=True)
