# Input data contracts

Configure paths in `config/paper.local.yaml`. Preserve six-digit stock identifiers and unique firm/date keys. The new code release contains no private text, annotations, feature panels, forecasts, model weights, or external policy/exposure datasets.

## Baseline panel

`baseline.parquet` has one row per stock and trading date, including the original observations preceding the study period.

| Columns | Definition |
| --- | --- |
| `stock_code`, `datetime`, `close` | Six-digit identifier, signal date, closing price |
| `change`, `pct_chg`, `volume`, `amount`, `swing` | Market features |
| `DIFF`, `DEA`, `MACD`, `MA10`, `RSI`, `CCI`, `K`, `D`, `J`, `ROCP` | Technical indicators |
| `ques_counts`, `ques_length`, `ques_sentiment_score` | Daily question activity, length, sentiment |
| `ans_counts`, `ans_length`, `ans_sentiment_score`, `ans_time_diff`, `ans_similarity` | Reply activity, length, sentiment, recoded response interval, semantic similarity |

The historical input contains 193,266 rows for 61 stocks. Target construction and the January 2017–June 2024 period produce 108,244 modeling rows. The target compares the next observed closing price with the current close. Only observations whose target date is inside the study period are retained. Sentiment is zero whenever the corresponding activity is zero. Missing feature rows are excluded separately by model specification.

## NEV industry index

`nev_index.xlsx` has first-sheet columns `Date` and `close`. The full original series is needed for 20-trading-day controls during training, as well as test-window regime slopes. The market files previously uploaded to this repository cover different dates/universes; use the historical study input for exact reproduction.

## Interaction archive for Table 2

`interactions.csv` contains `stock_code`, `datetime`, `reply_time`, and `whether_answer`. Question and reply timestamps are parsed independently. `whether_answer` may be the original string `已回复` or a positive numeric answered flag. Only the 61 baseline stocks and study-period question records are retained. Observed reply lag is computed from the original timestamps in calendar days and rounded to two decimals before the median, matching the retained source table. Eventual answer status describes the archive; it is not used as a question-date forecasting input.

## CCPU

`ccpu.xlsx`, sheet `Daily`, has exactly the two columns `date` and `CCPU`. The historical version contains 9,497 daily observations from 2000-01-01 through 2025-12-31. The loader verifies that version; dates are aligned with a one-calendar-day lag.

## EPU

`epu.xlsx`, sheet `EPU 2000 onwards`, contains `year`, `month`, and `EPU`. The historical file is `China_Mainland_Paper_EPU_20260914.xlsx`. Only the `residual` stage requires it. Monthly EPU is lagged two months.

## Firm transition exposure

The `exposure` phase requires two historical workbooks, supplied by the user.

- `exposure.xlsx` contains `stkcd`, `year`, `转型风险指数`, `转型风险总词频`, and `年报全文总词数`.
- `exposure_words.xlsx` contains `文件名`, `总词数`, and the original consecutive transition-term columns from `节能` through `集约`, inclusive. Preserve the original order and all 34 columns, including repeated dictionary entries. Each filename starts with the six-digit stock code, has the report year at positions 8–11, and ends with a publication date in `YYYY-M-D` format.

The script reconstructs transition word frequency and checks `exposure = 100 × transition words / total words` against the provided index. It then aligns reports from the day after publication. The historical version contains 67,111 firm-year records. Late publication of an older fiscal-year report does not replace a newer available fiscal year.

## Optional preparation inputs

`market.csv` is a Tushare-format daily CSV with `stock_code`, `datetime`, `open`, `high`, `low`, `close`, `change`, `pct_chg`, `volume`, and `amount`. The optional `pre_close` and `stock_name` columns are ignored. The existing raw stock Excel files require source-specific conversion and overlap checks before use; their schema is not silently treated as the Tushare schema.

`record_features.csv` is the historical record-level feature output with `stock_code`, `datetime`, `question`, `question_length`, `q_sentiment_score`, `q_sentiment_label`, `whether_answer`, `reply_time`, `reply`, `reply_length`, `a_sentiment_score`, `a_sentiment_label`, `time_diff`, and `similarity`. Here `whether_answer` is numeric; `time_diff` is elapsed calendar days. `prepare_daily.py` preserves the historical 5% question-activity exclusion and aggregation rules. This input schema differs from the newly uploaded `question_features.csv` and `reply_features.csv`; the historical scores and record features are required for exact panel reconstruction.

`features.csv` is the daily merged output of `prepare_daily.py`. `sentiment.parquet` contains `stock_code`, `trade_date`, `investor_sentiment`, and `company_reply_sentiment`. The baseline builder keeps common firm/date keys and replaces the two continuous sentiment scores. These are the separately precomputed series used in the manuscript.

## Frozen forecast schema

Each forecast file has `window_id`, `track`, `model`, `feature_set`, `stock_code`, `datetime`, `y_true`, `y_prob`, and `y_pred`. Probabilities refer to upward moves; classifications use the validation-selected threshold. XGBoost feature sets are `tech`, `tech_question`, `tech_answer`, and `tech_all`. All four share 73,528 firm-days across 1,212 dates and 10 windows. Files are named `window_NNN_XGBoost_<feature_set>.parquet`.

The dynamic `window_metrics.csv` contains the training/testing boundaries (`TrainStart`, `TrainEnd`, `TestStart`, `TestEnd`), `Window`, `Track`, `Model`, `Features`, thresholds, and daily-average classification metrics. The regime CSV contains `Window`, `Trend`, and `TestPeriod`, with regime values `Downtrend`, `Sideways`, and `Uptrend`.
