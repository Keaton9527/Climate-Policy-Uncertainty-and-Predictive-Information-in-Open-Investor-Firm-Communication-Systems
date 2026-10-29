# Systems paper reproduction

Code for **Climate Policy Uncertainty and Predictive Information in Open Investor–Firm Communication Systems**.

This repository combines the existing text-feature workflow with the forecasting and statistical analyses reported in the manuscript. The empirical release is in [`01_empirical/`](01_empirical/README.md), with an explicit map from manuscript tables to executable stages.

## Start here

- **Forecasting and statistical reproduction** — [installation, inputs, commands, and table map](01_empirical/README.md).
- **Input schemas and historical data versions** — [data dictionary](01_empirical/docs/data_dictionary.md).
- **Checks performed for this release** — [validation record](01_empirical/docs/VALIDATION.md).
- **Existing text-feature pipeline** — [original instructions](docs/TEXT_WORKFLOW.md), [`scripts/workflow.py`](scripts/workflow.py), and [`docs/FEATURES.md`](docs/FEATURES.md).

From the repository root, install the empirical requirements in a dedicated Python 3.12 environment, copy `01_empirical/config/paper.yaml` to `paper.local.yaml` in the same directory, set the local input paths, then run:

```sh
python -m pip install -r 01_empirical/requirements.txt
python 01_empirical/run.py --config 01_empirical/config/paper.local.yaml --phase all
```

The empirical package includes technical indicators and daily-panel preparation, five-method feature selection, rolling forecasts with five classifiers, question/reply ablation, market regimes, MCC/AUC/IC/RankIC, the CCPU joint-state comparison, the residual-CCPU and probability-loss checks, the firm-exposure comparison, and the paired source/environment comparisons. Its settings and source lineage are recorded in [`manifest.yaml`](01_empirical/manifest.yaml).

The new empirical package contains only analyses reported in the submitted manuscript and their preparation dependencies. Portfolio backtests, SHAP analyses, withdrawn experiments, the paused annotation study, and unreported sensitivity branches are excluded. The previously uploaded text implementation, historical files, and raw market workbooks are preserved; `legacy/` is historical source material, not an additional empirical reproduction stage.

## Data access

Private interaction text, annotations, precomputed sentiment, forecasting panels, predictions, and the policy/exposure workbooks are supplied locally. See the manuscript's Data Availability Statement for access conditions. The original market workbooks already present under `data/raw/` retain their existing provenance and coverage; no additional dataset is published in this update.

Exact historical reproduction starts from the study panel or its original record-feature and daily-sentiment inputs. Newly extracted text features are useful for rerunning the workflow on new inputs, but do not by themselves recover the separately precomputed historical sentiment series.

## 中文说明

论文实际使用的预测与统计检验代码位于 `01_empirical/`。运行顺序、正文表格对应关系、输入格式和本次核验结果均在该目录的说明中。已有文本特征代码和市场数据保持不变；本次只补充正文涉及的实证分析及必要的数据准备代码，不上传私有问答、标注、模型、预测或中间数据。
