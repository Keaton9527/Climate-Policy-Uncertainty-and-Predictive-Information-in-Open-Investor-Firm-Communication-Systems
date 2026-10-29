"""Five-method technical feature selection within each outer training window."""
from __future__ import annotations
import math
import logging
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from .data import Window, window_frame
from .common import write_json

def tier_scores(scores: dict[str, float]) -> dict[str, str]:
    values = sorted((float(value) for value in scores.values()), reverse=True)
    strong_cut = values[max(0, math.ceil(len(values) / 3) - 1)]
    moderate_cut = values[max(0, math.ceil(2 * len(values) / 3) - 1)]
    levels: dict[str, str] = {}
    for feature, value in scores.items():
        if value >= strong_cut:
            levels[feature] = "S"
        elif value >= moderate_cut:
            levels[feature] = "M"
        else:
            levels[feature] = "W"
    return levels


def select_features(train: pd.DataFrame, config: dict, window_id: int) -> tuple[pd.DataFrame, list[str]]:
    features = list(config["features"]["technical_candidates"])
    data = train.dropna(subset=features + ["target"]).sort_values(["datetime", "stock_code"])
    X = data[features].astype(float)
    y = data["target"].astype(int)
    if y.nunique() != 2:
        raise AssertionError(f"Training target has one class in window {window_id}")

    pearson_scores: dict[str, float] = {}
    for feature in features:
        if X[feature].nunique() <= 1:
            pearson_scores[feature] = 0.0
        else:
            score, _ = pearsonr(X[feature], y)
            pearson_scores[feature] = abs(float(score)) if np.isfinite(score) else 0.0

    mi_values = mutual_info_classif(X, y, random_state=int(config["seed"]))
    mi_scores = dict(zip(features, map(float, mi_values)))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    l1 = LogisticRegressionCV(
        Cs=10,
        cv=TimeSeriesSplit(n_splits=int(config["selection"]["l1_cv_splits"])),
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        max_iter=1000,
        random_state=int(config["seed"]),
        n_jobs=int(config["jobs"]),
    )
    l1.fit(X_scaled, y)
    l1_scores = dict(zip(features, np.abs(l1.coef_[0]).astype(float)))

    rf_params = {
        "n_estimators": int(config["selection"]["random_forest_trees"]),
        "max_depth": int(config["selection"]["random_forest_max_depth"]),
        "random_state": int(config["seed"]),
        "n_jobs": int(config["jobs"]),
        "class_weight": "balanced_subsample",
    }
    rfe = RFE(RandomForestClassifier(**rf_params), n_features_to_select=1, step=1)
    rfe.fit(X, y)
    max_rank = int(max(rfe.ranking_))
    rfe_scores = {
        feature: float((max_rank - rank + 1) / max_rank)
        for feature, rank in zip(features, rfe.ranking_)
    }

    real_rf = RandomForestClassifier(**rf_params)
    real_rf.fit(X, y)
    real_importance = real_rf.feature_importances_.astype(float)
    rng = np.random.default_rng(int(config["seed"]) + window_id)
    null_values: list[np.ndarray] = []
    for iteration in range(int(config["selection"]["null_iterations"])):
        null_rf = RandomForestClassifier(**{**rf_params, "random_state": int(config["seed"]) + iteration + 1})
        null_rf.fit(X, rng.permutation(y.to_numpy()))
        null_values.append(null_rf.feature_importances_.astype(float))
    null_mean = np.mean(np.vstack(null_values), axis=0)
    null_ratio = np.divide(real_importance, null_mean, out=np.zeros_like(real_importance), where=null_mean > 0)
    null_scores = dict(zip(features, null_ratio.astype(float)))

    method_scores = {
        "Pearson": pearson_scores,
        "MI": mi_scores,
        "L1": l1_scores,
        "RFE": rfe_scores,
        "Null": null_scores,
    }
    method_levels = {name: tier_scores(scores) for name, scores in method_scores.items()}
    rows = []
    required = int(config["selection"]["strong_methods_required"])
    for feature in features:
        row: dict[str, object] = {"Window": window_id, "Feature": feature}
        strong_count = 0
        for method in method_scores:
            row[f"{method}_Score"] = method_scores[method][feature]
            row[f"{method}_Level"] = method_levels[method][feature]
            strong_count += int(method_levels[method][feature] == "S")
        row["S_sum"] = strong_count
        row["Selected"] = strong_count >= required
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["S_sum", "Feature"], ascending=[False, True])
    selected = result.loc[result["Selected"], "Feature"].tolist()
    if not selected:
        selected = result.head(4)["Feature"].tolist()
        result.loc[result["Feature"].isin(selected), "Selected"] = True
    return result.reset_index(drop=True), selected


def run_selection(panel: pd.DataFrame, windows: list[Window], config: dict, dirs: dict, logger: logging.Logger) -> dict[int, list[str]]:
    selection_dir = dirs["main_interim"] / "feature_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selected_map: dict[int, list[str]] = {}
    for window in windows:
        logger.info("Feature selection: window %02d", window.window_id)
        train = window_frame(panel, window.train_start, window.train_end)
        result, selected = select_features(train, config, window.window_id)
        result.to_csv(selection_dir / f"window_{window.window_id:03d}_feature_selection.csv", index=False)
        selected_map[window.window_id] = selected
        logger.info("Window %02d selected: %s", window.window_id, ", ".join(selected))
    write_json(selection_dir / "selected_features.json", {str(k): v for k, v in selected_map.items()})
    return selected_map


def load_selected_map(dirs: dict) -> dict[int, list[str]]:
    path = dirs["main_interim"] / "feature_selection/selected_features.json"
    if not path.exists():
        raise FileNotFoundError("Run --phase selection before --phase models")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(key): list(value) for key, value in payload.items()}
