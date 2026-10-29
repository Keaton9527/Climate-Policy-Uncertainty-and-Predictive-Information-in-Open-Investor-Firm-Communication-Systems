"""Policy uncertainty and exposure analyses using in-memory forecasts."""
import numpy as np
import pandas as pd
import robustness as engine
from climate import load_forecasts
from climate_states import add_ccpu_states


def state_panel(features, forecasts, metrics, regimes, inputs, config):
    panel = load_forecasts(forecasts, config['sample']['expected_windows'])
    ccpu = config['ccpu']
    panel, thresholds = add_ccpu_states(panel, inputs, features, metrics,
        ccpu['lag_days'], ccpu['low_quantile'], ccpu['high_quantile'], 'ccpu_primary')
    regimes = regimes.rename(columns={'Window':'window_id','Trend':'regime'})
    panel = panel.merge(regimes[['window_id','regime']], on='window_id', validate='many_to_one')
    prices = features[['stock_code','datetime','close']].sort_values(['stock_code','datetime']).copy()
    prices['forward_return'] = prices.groupby('stock_code').close.shift(-1) / prices.close - 1
    panel = panel.merge(prices[['stock_code','datetime','forward_return']], on=['stock_code','datetime'], how='left', validate='one_to_one')
    if panel[['forward_return','regime']].isna().any().any():
        raise ValueError('Missing forward returns or market regimes')
    panel['raw_state'] = panel.ccpu_primary_state
    panel['ccpu'] = panel.ccpu_primary_value
    for spec in ['tech_answer','tech_question','tech_all']:
        panel[f'd_{spec}'] = (panel.y_true-panel.y_prob_tech)**2 - (panel.y_true-panel[f'y_prob_{spec}'])**2
    return panel.sort_values(['datetime','stock_code']).reset_index(drop=True), prices, thresholds


def run_supplementary(features, forecasts, metrics, regimes, inputs, config):
    panel, prices, thresholds = state_panel(features, forecasts, metrics, regimes, inputs, config)
    specs = ['tech','tech_answer','tech_question','tech_all']
    cells, contrasts = engine.grouped_analysis(panel, 'regime', 'raw_state', config['bootstrap'], specs, ['MCC','AUC','IC','RankIC'])
    for frame in [cells, contrasts]:
        frame['holm_p_within_metric'] = np.nan
        for ix in frame.groupby('metric').groups.values():
            frame.loc[ix,'holm_p_within_metric'] = engine.holm(frame.loc[ix,'p_value'])
        frame['holm_p_all_metrics'] = engine.holm(frame.p_value)
    table = contrasts[contrasts.metric.eq('MCC')].copy()
    for state in ['low','high']:
        values = cells[cells.metric.eq('MCC') & cells.state.eq(state)][['group','specification','estimate']]
        table = table.merge(values.rename(columns={'estimate':'delta_'+state}), on=['group','specification'], validate='one_to_one')
    results = {'table_07_ccpu_contrasts':table, 'table_A6_ccpu_thresholds':thresholds,
               'joint_metrics':cells, 'joint_contrasts':contrasts}
    controls, _, _ = engine.residualize(panel, prices, inputs, 2)
    controls['all_sample'] = 'all'
    cells, contrasts = engine.grouped_analysis(controls, 'all_sample', 'residual_state', config['bootstrap'], ['tech','tech_answer'], ['MCC'])
    results.update(residual_ccpu_metrics=cells, residual_ccpu_contrasts=contrasts,
        brier_regression=engine.regress(controls, 'd_tech_answer', ['ccpu','epu','nev_rv','nev_ret20','trend']))
    exposure = engine.prepare_exposure(inputs)
    exposed, _ = engine.attach_actual_exposure(panel, prices, exposure)
    cells, contrasts = engine.grouped_analysis(exposed.dropna(subset=['exposure']), 'exposure_group', 'raw_state', config['bootstrap'], ['tech','tech_answer'], ['MCC'])
    results.update(exposure_metrics=cells, exposure_contrasts=contrasts)
    return results
