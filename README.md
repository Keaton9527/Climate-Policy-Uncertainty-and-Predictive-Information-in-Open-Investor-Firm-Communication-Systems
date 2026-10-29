# Climate Policy Uncertainty and Predictive Information in Open Investor–Firm Communication Systems

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

The included QA file is losslessly compressed; the pipeline reads it directly. Stock codes use six digits. Market CSV units are percent for `pct_chg`, lots of 100 shares for `volume`, and thousands of yuan for `amount`; the included workbooks are converted automatically. Exposure filenames contain the stock code, fiscal year at characters 8–11, and a trailing publication date (`YYYY-M-D`). Supply the remaining source data according to the paper's Data Availability Statement.

## Results

The pipeline produces Tables 1–7 and A2–A7, plus the residual-CCPU, probability-loss, and exposure analyses. Technical-indicator formulas (Table A1), rolling periods, model settings, and bootstrap parameters are implemented in `code/` and `config.yaml`. Figure 1 uses the rolling periods reported in Table 1. Numerical results can vary with model initialization and computing environment.
