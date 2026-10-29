# Systems paper reproduction

Code for **Climate Policy Uncertainty and Predictive Information in Open Investor–Firm Communication Systems**.

All Python files are in `code/`. Source data, downloaded model weights, and final results are kept under `data/`. Features, selected variables, forecasts, and statistical inputs pass directly between functions in memory.

## Run

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/run.py --config config.yaml
```

Set the input paths in `config.yaml` before running. The command downloads the configured text models, fine-tunes MacBERT, evaluates the text classifiers, extracts and aggregates features, runs rolling forecasts and statistical analyses, and saves `data/result_*.csv`. No intermediate data files are required or generated.

To use existing MacBERT weights, set `text.checkpoint` to their directory. Otherwise, leave it `null` to train. Model caches are stored in `data/models/`. Configuration paths are relative to the configuration file.

## Inputs

Place the following source files in `data/` or update their paths in `config.yaml`.

| Input | Required fields |
| --- | --- |
| `qa.csv` | `stock_code`, `datetime`, `question`, `reply_time`, `reply`; optional `whether_answer` (1/0 or 已回复/未回复) |
| `annotations.xlsx` | `text`, `label` (−1, 0, 1); CSV is also supported |
| Market prices | Included `TRD_FwardQuotation*.xlsx`, or a CSV with `stock_code`, `datetime`, `open`, `high`, `low`, `close`, `change`, `pct_chg`, `volume`, `amount` |
| `nev_index.xlsx` | `Date`, `close` |
| `ccpu.xlsx` | Sheet `Daily`: `date`, `CCPU` |
| `epu.xlsx` | Sheet `EPU 2000 onwards`: `year`, `month`, `EPU` |
| `exposure.xlsx` | `stkcd`, `year`, `转型风险指数`, `转型风险总词频`, `年报全文总词数` |
| `exposure_words.xlsx` | `文件名`, `总词数`, and transition-word columns from `节能` through `集约` |

Stock codes use six digits. Market CSV units are percent for `pct_chg`, lots of 100 shares for `volume`, and thousands of yuan for `amount`; the included workbooks are converted automatically. Exposure filenames contain the stock code, fiscal year at characters 8–11, and a trailing publication date (`YYYY-M-D`). Supply the remaining source data according to the paper's Data Availability Statement.

## Results

The pipeline produces Tables 1–7 and A2–A7, plus the residual-CCPU, probability-loss, and exposure analyses. Technical-indicator formulas (Table A1), rolling periods, model settings, and bootstrap parameters are implemented in `code/` and `config.yaml`. Figure 1 uses the rolling periods reported in Table 1. Numerical results can vary with model initialization and computing environment.

## 中文

修改 `config.yaml` 中的数据路径后运行上述命令，即可连续完成模型下载、文本特征提取和实证实验。代码集中在 `code/`，数据集中在 `data/`。中间数据通过变量传递，最终结果保存为 `data/result_*.csv`。
