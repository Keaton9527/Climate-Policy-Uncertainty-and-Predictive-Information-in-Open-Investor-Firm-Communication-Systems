"""Portable paths, configuration and output locations."""
from pathlib import Path
import json
import logging
import sys
import yaml

METRICS = ["MCC", "DA", "Precision", "Recall", "F1", "AUC"]
MODEL_NAMES = ["LogisticRegression", "MLP", "RandomForest", "SVC", "XGBoost"]

def resolve(value):
    return Path(value).expanduser().resolve()

def load_config(path):
    path = resolve(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in config["paths"].items():
        p = Path(value).expanduser()
        config["paths"][key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return config

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def output_dirs(config):
    root = resolve(config["paths"]["output"])
    dirs = {"main_interim": root / "forecasting/dynamic",
            "fixed_interim": root / "forecasting/fixed",
            "desc_interim": root / "descriptive",
            "main_results": root / "tables",
            "fixed_results": root / "tables",
            "climate": root / "climate",
            "source": root / "source_comparisons",
            "residual": root / "residual_ccpu",
            "exposure": root / "exposure"}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def setup_logging(config):
    root = resolve(config["paths"]["output"])
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nev_forecasting")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(root / "run.log", encoding="utf-8")]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
