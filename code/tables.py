"""Create machine-readable tables in manuscript numbering."""
import pandas as pd
from common import METRICS


def summarize(frame, groups, metrics):
    means = frame.groupby(groups, sort=False)[metrics].mean()
    std = frame.groupby(groups, sort=False)[metrics].std(ddof=1).add_suffix("_Std")
    return means.join(std).reset_index()


def export_tables(main, fixed, regimes, selection, config):
    results = {}
    metrics = METRICS + ["IC", "RankIC"]
    regimes = regimes[["Window", "Trend"]]
    main = main.merge(regimes, on="Window", how="left", validate="many_to_one")
    if main.Trend.isna().any():
        raise ValueError("Some forecast windows have no market regime")
    baseline = main[main.Features.isin(["tech", "tech_all"])]
    results["table_04_model_comparison"] = summarize(baseline, ["Model", "Features"], metrics)
    xgb = main[main.Model.eq("XGBoost")]
    results["table_05_xgboost_regimes"] = summarize(xgb[xgb.Features.isin(["tech", "tech_all"])], ["Trend", "Features"], metrics)
    full = summarize(xgb, ["Features"], metrics)
    full.insert(0, "Trend", "Full sample")
    grouped = summarize(xgb, ["Trend", "Features"], metrics)
    results["table_06_source_ablation"] = pd.concat([full, grouped], ignore_index=True)
    # Equal windows within each model/regime, then equal models.
    per_model = baseline.groupby(["Trend", "Features", "Model"], sort=False)[METRICS].mean().reset_index()
    results["table_A4_model_average"] = per_model.groupby(["Trend", "Features"], sort=False)[METRICS].mean().reset_index()
    if len(fixed) != config["sample"]["expected_windows"] * 10:
        raise ValueError("Unexpected fixed-track metric inventory")
    results["table_A5_fixed_features"] = summarize(fixed, ["Model", "Features"], METRICS)
    rows = []
    for feature, group in selection.groupby("Feature", sort=False):
        row = dict(Feature=feature, SelectedWindows=int(group.Selected.astype(bool).sum()), MeanStrongMethodCount=group.S_sum.mean())
        for method in ["Pearson", "MI", "L1", "RFE", "Null"]:
            row[method + "_StrongWindows"] = int(group[method + "_Level"].eq("S").sum())
        rows.append(row)
    results["table_03_feature_selection"] = pd.DataFrame(rows).sort_values(["SelectedWindows", "MeanStrongMethodCount", "Feature"], ascending=[False, False, True])
    return results
