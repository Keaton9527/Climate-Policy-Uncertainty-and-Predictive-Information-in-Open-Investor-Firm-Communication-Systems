#!/usr/bin/env python3
"""Run the core manuscript pipeline; see README.md for inputs and stages."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import platform
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nev_forecasting.common import load_config, output_dirs, setup_logging, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/paper.yaml")
    parser.add_argument("--phase", choices=["all", "descriptive", "selection", "models", "ranking", "tables", "climate", "source", "residual", "exposure"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    dirs = output_dirs(config)
    logger = setup_logging(config)
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "PyYAML", "pyarrow", "openpyxl"]
    audit = {"phase": args.phase, "python": platform.python_version(),
             "packages": {name: importlib.metadata.version(name) for name in packages},
             "config": config, "input_sha256": {}}
    for name, value in config["paths"].items():
        path = Path(value)
        if name != "output" and path.is_file():
            with path.open("rb") as handle:
                audit["input_sha256"][name] = hashlib.file_digest(handle, "sha256").hexdigest()
    prediction_root = dirs["main_interim"] / "predictions"
    audit["prediction_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(prediction_root.glob("*.parquet"))}
    write_json(Path(config["paths"]["output"]) / f"run_{args.phase}.json", audit)
    if args.phase in {"all", "descriptive", "selection", "models"}:
        from nev_forecasting.data import prepare_panel, generate_windows
        raw, panel = prepare_panel(config)
        windows = generate_windows(config)
    if args.phase in {"all", "descriptive"}:
        from nev_forecasting.descriptive import run_descriptive
        run_descriptive(raw, panel, windows, config, dirs)
        logger.info("Descriptive tables completed")
    if args.phase in {"all", "selection"}:
        from nev_forecasting.selection import run_selection
        run_selection(panel, windows, config, dirs, logger)
    if args.phase in {"all", "models"}:
        from nev_forecasting.selection import load_selected_map
        from nev_forecasting.forecasting import run_model_track
        selected = load_selected_map(dirs)
        run_model_track("main", panel, windows, selected, config, dirs, logger)
        run_model_track("fixed", panel, windows, selected, config, dirs, logger)
    if args.phase in {"all", "ranking"}:
        from nev_forecasting.evaluation import evaluate_ranking
        evaluate_ranking(config, dirs)
        logger.info("IC and RankIC completed")
    if args.phase in {"all", "tables"}:
        from nev_forecasting.tables import export_tables
        export_tables(config, dirs)
        logger.info("Forecast tables completed")
    if args.phase in {"all", "climate"}:
        from nev_forecasting.supplementary import run_joint
        run_joint(config, dirs)
        logger.info("CCPU contrasts completed")
    if args.phase in {"all", "source"}:
        from nev_forecasting.source_comparisons import run_source_comparisons
        run_source_comparisons(config, dirs)
    if args.phase in {"all", "residual"}:
        from nev_forecasting.supplementary import run_residual
        run_residual(config, dirs)
    if args.phase in {"all", "exposure"}:
        from nev_forecasting.supplementary import run_exposure
        run_exposure(config, dirs)
    logger.info("Phase %s completed", args.phase)

if __name__ == "__main__":
    main()
