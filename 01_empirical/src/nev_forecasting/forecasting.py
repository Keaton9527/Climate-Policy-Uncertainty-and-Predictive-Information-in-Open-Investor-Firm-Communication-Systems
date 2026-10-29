"""Rolling training and saved test forecasts for dynamic and fixed feature tracks."""
from __future__ import annotations
import logging
import time
import pandas as pd
from .common import MODEL_NAMES
from .data import Window, window_frame
from .models import fit_one

def run_model_track(
    track: str,
    panel: pd.DataFrame,
    windows: list[Window],
    selected_map: dict[int, list[str]],
    config: dict,
    dirs: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    is_main = track == "main"
    track_dir = dirs["main_interim"] if is_main else dirs["fixed_interim"]
    prediction_dir = track_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    question = list(config["features"]["question"])
    answer = list(config["features"]["answer"])
    fixed = list(config["features"]["fixed_technical"])
    rows: list[dict[str, object]] = []

    for window in windows:
        tech = selected_map[window.window_id] if is_main else fixed
        feature_sets = {"tech": tech, "tech_all": tech + question + answer}
        if is_main:
            feature_sets.update({"tech_question": tech + question, "tech_answer": tech + answer})
        train = window_frame(panel, window.train_start, window.train_end)
        val = window_frame(panel, window.val_start, window.val_end)
        test = window_frame(panel, window.test_start, window.test_end)
        for model_name in MODEL_NAMES:
            names = ["tech", "tech_all"]
            if is_main and model_name == "XGBoost":
                names += ["tech_question", "tech_answer"]
            for feature_name in names:
                logger.info(
                    "%s window %02d %s %s", track, window.window_id, model_name, feature_name
                )
                started = time.time()
                metrics, predictions, threshold, best_iteration = fit_one(
                    model_name, feature_sets[feature_name], train, val, test, config
                )
                predictions.insert(0, "window_id", window.window_id)
                predictions.insert(1, "track", track)
                predictions.insert(2, "model", model_name)
                predictions.insert(3, "feature_set", feature_name)
                pred_path = prediction_dir / f"window_{window.window_id:03d}_{model_name}_{feature_name}.parquet"
                predictions.to_parquet(pred_path, index=False)
                row: dict[str, object] = {
                    "Window": window.window_id,
                    "Track": track,
                    "Model": model_name,
                    "Features": feature_name,
                    "FeatureColumns": "|".join(feature_sets[feature_name]),
                    "TrainStart": window.train_start.date(),
                    "TrainEnd": window.train_end.date(),
                    "ValidationStart": window.val_start.date(),
                    "ValidationEnd": window.val_end.date(),
                    "TestStart": window.test_start.date(),
                    "TestEnd": window.test_end.date(),
                    "TrainSamples": len(train.dropna(subset=feature_sets[feature_name] + ["target"])),
                    "ValidationSamples": len(val.dropna(subset=feature_sets[feature_name] + ["target"])),
                    "TestSamples": len(predictions),
                    "Threshold": threshold,
                    "BestIteration": best_iteration,
                    "DurationSeconds": time.time() - started,
                    **metrics,
                }
                rows.append(row)
                logger.info("Completed MCC=%.4f DA=%.4f threshold=%.2f", metrics["MCC"], metrics["DA"], threshold)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(track_dir / "window_metrics.csv", index=False)
    expected = len(windows) * (12 if is_main else 10)
    if len(metrics_df) != expected:
        raise AssertionError(f"{track} expected {expected} metric rows, found {len(metrics_df)}")
    return metrics_df
