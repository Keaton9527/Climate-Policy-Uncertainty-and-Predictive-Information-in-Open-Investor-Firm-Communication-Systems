"""Optional merge of precomputed market/text features and updated sentiment scores."""
from pathlib import Path
import pandas as pd

def normalize_stock_code(series: pd.Series) -> pd.Series:
    """Extract six-digit stock identifiers, preserving leading zeros."""
    return series.astype(str).str.extract(r"(\d{6})", expand=False)


def build_baseline(old_data_path, sentiment_path, output_path) -> dict[str, int]:
    output_path = Path(output_path)
    old = pd.read_csv(old_data_path, converters={"stock_code": str})
    new = pd.read_parquet(sentiment_path)

    old["stock_code"] = normalize_stock_code(old["stock_code"])
    new["stock_code"] = normalize_stock_code(new["stock_code"])
    old["datetime"] = pd.to_datetime(old["datetime"])
    new["trade_date"] = pd.to_datetime(new["trade_date"])

    common_codes = sorted(set(old["stock_code"].dropna()) & set(new["stock_code"].dropna()))
    old_common = old[old["stock_code"].isin(common_codes)].copy()
    new_common = new[new["stock_code"].isin(common_codes)].copy()

    if old_common.duplicated(["stock_code", "datetime"]).any():
        raise ValueError("Duplicate stock/date keys in the feature panel")
    if new_common.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/date keys in the sentiment panel")

    sentiment = new_common[[
        "stock_code",
        "trade_date",
        "investor_sentiment",
        "company_reply_sentiment",
    ]].rename(columns={
        "investor_sentiment": "new_ques_sentiment_score",
        "company_reply_sentiment": "new_ans_sentiment_score",
    })

    baseline = old_common.merge(
        sentiment,
        left_on=["stock_code", "datetime"],
        right_on=["stock_code", "trade_date"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    baseline["ques_sentiment_score"] = baseline["new_ques_sentiment_score"]
    baseline["ans_sentiment_score"] = baseline["new_ans_sentiment_score"]

    ques_rule_changed = baseline["ques_counts"].eq(0) & baseline["ques_sentiment_score"].ne(0)
    ans_rule_changed = baseline["ans_counts"].eq(0) & baseline["ans_sentiment_score"].ne(0)
    baseline.loc[baseline["ques_counts"].eq(0), "ques_sentiment_score"] = 0.0
    baseline.loc[baseline["ans_counts"].eq(0), "ans_sentiment_score"] = 0.0

    baseline = baseline.drop(columns=[
        "trade_date",
        "new_ques_sentiment_score",
        "new_ans_sentiment_score",
    ])
    baseline = baseline[old.columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(output_path, index=False)

    return {
        "matched_stocks": len(common_codes),
        "output_rows": len(baseline),
        "ques_rule_changed_rows": int(ques_rule_changed.sum()),
        "ans_rule_changed_rows": int(ans_rule_changed.sum()),
    }
