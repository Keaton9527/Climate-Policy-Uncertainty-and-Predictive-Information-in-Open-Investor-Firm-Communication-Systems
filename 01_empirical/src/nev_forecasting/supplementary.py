"""Configured entry points for the analyses reported in Section 4.4."""
from pathlib import Path
import pandas as pd
import numpy as np
from . import robustness as engine
from .climate import load_forecasts
from .climate_states import add_ccpu_states


def state_panel(config, dirs):
    panel = load_forecasts(dirs, config['sample']['expected_windows'])
    local = {**config, 'paths': {**config['paths'], 'window_metrics': str(dirs['main_interim'] / 'window_metrics.csv')}}
    ccpu = config['ccpu']
    panel, thresholds = add_ccpu_states(panel, local, ccpu['lag_days'], ccpu['low_quantile'], ccpu['high_quantile'], 'ccpu_primary')
    regimes = pd.read_csv(dirs['desc_interim'] / 'market_regimes.csv').rename(columns={'Window': 'window_id', 'Trend': 'regime'})
    panel = panel.merge(regimes[['window_id', 'regime']], on='window_id', how='left', validate='many_to_one')
    baseline = pd.read_parquet(config['paths']['baseline'], columns=['stock_code', 'datetime', 'close'])
    baseline['stock_code'] = baseline.stock_code.astype(str).str.zfill(6)
    baseline['datetime'] = pd.to_datetime(baseline.datetime)
    baseline = baseline.sort_values(['stock_code', 'datetime'])
    if baseline.duplicated(['stock_code', 'datetime']).any():
        raise ValueError('Duplicate baseline firm/date keys')
    baseline['forward_return'] = baseline.groupby('stock_code').close.shift(-1) / baseline.close - 1
    panel = panel.merge(baseline[['stock_code', 'datetime', 'forward_return']], on=['stock_code', 'datetime'], how='left', validate='one_to_one')
    if panel[['forward_return', 'regime']].isna().any().any():
        raise ValueError('Missing returns or market regimes')
    if len(panel) != 73528 or panel.datetime.nunique() != 1212:
        raise ValueError('The paper requires 73,528 paired firm-days over 1,212 dates')
    np.testing.assert_array_equal(panel.forward_return.gt(0).astype(int), panel.y_true)
    panel['raw_state'] = panel.ccpu_primary_state
    panel['ccpu'] = panel.ccpu_primary_value
    for spec in ['tech_answer', 'tech_question', 'tech_all']:
        panel[f'd_{spec}'] = (panel.y_true-panel.y_prob_tech)**2 - (panel.y_true-panel[f'y_prob_{spec}'])**2
    thresholds.to_csv(dirs['main_results'] / 'table_A6_ccpu_thresholds.csv', index=False)
    return panel.sort_values(['datetime', 'stock_code']).reset_index(drop=True), baseline


def run_joint(config, dirs):
    panel, _ = state_panel(config, dirs)
    engine.SPECS = ['tech', 'tech_answer', 'tech_question', 'tech_all']
    engine.METRICS = ['MCC', 'AUC', 'IC', 'RankIC']
    cells, contrasts = engine.grouped_analysis(panel, 'regime', 'raw_state', dirs['climate'], 'joint_four_metrics', config['bootstrap'])
    for frame in [cells, contrasts]:
        frame['holm_p_within_metric'] = np.nan
        for ix in frame.groupby('metric').groups.values():
            frame.loc[ix, 'holm_p_within_metric'] = engine.holm(frame.loc[ix, 'p_value'])
        frame['holm_p_all_metrics'] = engine.holm(frame.p_value)
    engine.save(cells, dirs['climate'] / 'joint_four_metrics_metrics.csv')
    engine.save(contrasts, dirs['climate'] / 'joint_four_metrics_contrasts.csv')
    table = contrasts[contrasts.metric.eq('MCC')].copy()
    for state in ['low', 'high']:
        values = cells[cells.metric.eq('MCC') & cells.state.eq(state)][['group', 'specification', 'estimate']]
        table = table.merge(values.rename(columns={'estimate': 'delta_'+state}), on=['group', 'specification'], validate='one_to_one')
    engine.save(table, dirs['main_results'] / 'table_07_ccpu_contrasts.csv')
    panel.to_parquet(dirs['climate'] / 'paired_state_panel.parquet', index=False)


def run_residual(config, dirs):
    panel, baseline = state_panel(config, dirs)
    engine.SPECS, engine.METRICS = ['tech', 'tech_answer'], ['MCC']
    controls, coefficients, calendar = engine.residualize(panel, baseline, config, 2)
    controls['all_sample'] = 'all'
    controls.to_parquet(dirs['residual'] / 'predictions_controls.parquet', index=False)
    engine.save(coefficients, dirs['residual'] / 'residual_training_coefficients.csv')
    engine.save(calendar, dirs['residual'] / 'daily_controls.csv')
    engine.grouped_analysis(controls, 'all_sample', 'residual_state', dirs['residual'], 'residual', config['bootstrap'])
    engine.regress(controls, 'd_tech_answer', ['ccpu','epu','nev_rv','nev_ret20','trend'], dirs['residual'] / 'brier_regression_raw.csv')


def run_exposure(config, dirs):
    panel, baseline = state_panel(config, dirs)
    engine.SPECS, engine.METRICS = ['tech', 'tech_answer'], ['MCC']
    exposure = engine.audit_exposure(dirs['exposure'], dirs['exposure'], config)
    panel, thresholds = engine.attach_actual_exposure(panel, baseline, exposure)
    engine.save(thresholds, dirs['exposure'] / 'exposure_training_medians.csv')
    coverage = panel.assign(missing=panel.exposure.isna()).groupby('window_id').agg(n_obs=('stock_code','size'), missing=('missing','sum'), n_firms=('stock_code','nunique')).reset_index()
    engine.save(coverage, dirs['exposure'] / 'exposure_merge_coverage.csv')
    panel.to_parquet(dirs['exposure'] / 'predictions_exposure.parquet', index=False)
    engine.grouped_analysis(panel.dropna(subset=['exposure']), 'exposure_group', 'raw_state', dirs['exposure'], 'exposure_raw', config['bootstrap'])
