"""Five classifiers with the manuscript hyperparameters and validation-only thresholds."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .metrics import choose_threshold, mean_daily_metrics

def make_model(name: str, config: dict):
    seed = int(config["seed"])
    jobs = int(config["jobs"])
    cfg = config["models"]
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=int(cfg["logistic_regression"]["max_iter"]), random_state=seed)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(cfg["mlp"]["hidden_layer_sizes"]),
            max_iter=int(cfg["mlp"]["max_iter"]),
            early_stopping=bool(cfg["mlp"]["early_stopping"]),
            n_iter_no_change=10,
            random_state=seed,
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=int(cfg["random_forest"]["n_estimators"]),
            random_state=seed,
            n_jobs=jobs,
        )
    if name == "SVC":
        return SVC(
            kernel=str(cfg["svc"]["kernel"]),
            C=float(cfg["svc"]["c"]),
            probability=True,
            max_iter=int(cfg["svc"]["max_iter"]),
            random_state=seed,
        )
    if name == "XGBoost":
        xcfg = cfg["xgboost"]
        return XGBClassifier(
            n_estimators=int(xcfg["n_estimators"]),
            learning_rate=float(xcfg["learning_rate"]),
            max_depth=int(xcfg["max_depth"]),
            min_child_weight=float(xcfg["min_child_weight"]),
            subsample=float(xcfg["subsample"]),
            colsample_bytree=float(xcfg["colsample_bytree"]),
            gamma=float(xcfg["gamma"]),
            early_stopping_rounds=int(xcfg["early_stopping_rounds"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=jobs,
            tree_method="hist",
        )
    raise KeyError(name)


def fit_one(
    model_name: str,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
) -> tuple[dict[str, float], pd.DataFrame, float, int | None]:
    required = feature_cols + ["target", "datetime", "stock_code"]
    train = train.dropna(subset=required).sort_values(["datetime", "stock_code"])
    val = val.dropna(subset=required).sort_values(["datetime", "stock_code"])
    test = test.dropna(subset=required).sort_values(["datetime", "stock_code"])
    if min(len(train), len(val), len(test)) == 0:
        raise AssertionError(f"Empty split for {model_name} and {feature_cols}")
    if not train["datetime"].max() < val["datetime"].min() or not val["datetime"].max() < test["datetime"].min():
        raise AssertionError("Train/validation/test leakage detected")

    X_train = train[feature_cols].to_numpy(dtype=float)
    X_val = val[feature_cols].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    y_val = val["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)

    if model_name in {"LogisticRegression", "MLP", "SVC"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model = make_model(model_name, config)
    if model_name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iteration = int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None
    else:
        model.fit(X_train, y_train)
        if model_name == "MLP":
            iteration_value = getattr(model, "n_iter_", 0)
            best_iteration = int(iteration_value[0] if np.ndim(iteration_value) else iteration_value)
        else:
            best_iteration = None

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob, val["datetime"], config)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = mean_daily_metrics(y_test, test_pred, test_prob, test["datetime"])
    predictions = test[["stock_code", "datetime"]].copy()
    predictions["y_true"] = y_test
    predictions["y_prob"] = test_prob
    predictions["y_pred"] = test_pred
    return metrics, predictions, threshold, best_iteration
