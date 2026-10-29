#!/usr/bin/env python3
"""Run the paper workflow from source data to final experiment tables."""
import argparse
from pathlib import Path
from common import load_config, progress_logger
from inputs import read_inputs
from text_features import run_text
from market import calculate_technical_indicators
from daily import build_daily
from panel import prepare_panel, generate_windows
from descriptive import run_descriptive
from selection import run_selection
from forecasting import run_model_track
from evaluation import evaluate_ranking
from tables import export_tables
from source_comparisons import run_source_comparisons
from supplementary import run_supplementary


def run_pipeline(inputs, config):
    logger = progress_logger()
    logger.info('Text models and record features')
    records, text_table = run_text(inputs['qa'], inputs['annotations'], config)
    market = calculate_technical_indicators(inputs['market'])
    features = build_daily(records, market)
    raw, panel = prepare_panel(features, config)
    windows = generate_windows(config)
    regimes, results = run_descriptive(raw, panel, windows, inputs, config)
    results['table_A2_language_models'] = text_table
    selected, selection = run_selection(panel, windows, config, logger)
    main, forecasts = run_model_track('main', panel, windows, selected, config, logger)
    fixed, _ = run_model_track('fixed', panel, windows, selected, config, logger)
    ranked = evaluate_ranking(raw, main, forecasts, config)
    results.update(export_tables(ranked, fixed, regimes, selection, config))
    logger.info('Source comparisons and policy uncertainty analyses')
    results.update(run_source_comparisons(forecasts, regimes, config))
    results.update(run_supplementary(raw, forecasts, main, regimes, inputs, config))
    return results


def save_results(results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(directory / f'result_{name}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = read_inputs(config)
    results = run_pipeline(inputs, config)
    save_results(results,config['paths']['output'])
    print(f'Completed: {len(results)} result tables in {config["paths"]["output"]}')


if __name__ == '__main__':
    main()
