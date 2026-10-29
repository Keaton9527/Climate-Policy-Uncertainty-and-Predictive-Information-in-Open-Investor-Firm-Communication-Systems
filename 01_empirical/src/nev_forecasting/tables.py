"""Create machine-readable tables in manuscript numbering."""
import pandas as pd
from .common import METRICS


def summarize(frame, groups, metrics):
    means = frame.groupby(groups, sort=False)[metrics].mean()
    std = frame.groupby(groups, sort=False)[metrics].std(ddof=1).add_suffix("_Std")
    return means.join(std).reset_index()


def export_tables(config, dirs):
    output = dirs["main_results"]
    metrics = METRICS + ["IC", "RankIC"]
    main = pd.read_csv(dirs["main_interim"] / "window_metrics_with_ranking.csv")
    regimes = pd.read_csv(dirs["desc_interim"] / "market_regimes.csv")[["Window", "Trend"]]
    main = main.merge(regimes, on="Window", how="left", validate="many_to_one")
    if main.Trend.isna().any():
        raise ValueError("Some forecast windows have no market regime")
    baseline = main[main.Features.isin(["tech", "tech_all"])]
    summarize(baseline, ["Model", "Features"], metrics).to_csv(output / "table_04_model_comparison.csv", index=False)
    xgb = main[main.Model.eq("XGBoost")]
    summarize(xgb[xgb.Features.isin(["tech", "tech_all"])], ["Trend", "Features"], metrics).to_csv(
        output / "table_05_xgboost_regimes.csv", index=False)
    full = summarize(xgb, ["Features"], metrics)
    full.insert(0, "Trend", "Full sample")
    grouped = summarize(xgb, ["Trend", "Features"], metrics)
    pd.concat([full, grouped], ignore_index=True).to_csv(output / "table_06_source_ablation.csv", index=False)
    # Equal windows within each model/regime, then equal models.
    per_model = baseline.groupby(["Trend", "Features", "Model"], sort=False)[METRICS].mean().reset_index()
    per_model.groupby(["Trend", "Features"], sort=False)[METRICS].mean().reset_index().to_csv(
        output / "table_A4_model_average.csv", index=False)
    fixed = pd.read_csv(dirs["fixed_interim"] / "window_metrics.csv")
    if len(fixed) != config["sample"]["expected_windows"] * 10:
        raise ValueError("Unexpected fixed-track metric inventory")
    summarize(fixed, ["Model", "Features"], METRICS).to_csv(output / "table_A5_fixed_features.csv", index=False)
    files = sorted((dirs["main_interim"] / "feature_selection").glob("window_*_feature_selection.csv"))
    if len(files) != config["sample"]["expected_windows"]:
        raise ValueError("Missing feature selection tables")
    selection = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    rows = []
    for feature, group in selection.groupby("Feature", sort=False):
        row = dict(Feature=feature, SelectedWindows=int(group.Selected.astype(bool).sum()), MeanStrongMethodCount=group.S_sum.mean())
        for method in ["Pearson", "MI", "L1", "RFE", "Null"]:
            row[method + "_StrongWindows"] = int(group[method + "_Level"].eq("S").sum())
        rows.append(row)
    pd.DataFrame(rows).sort_values(["SelectedWindows", "MeanStrongMethodCount", "Feature"], ascending=[False, False, True]).to_csv(
        output / "table_03_feature_selection.csv", index=False)
