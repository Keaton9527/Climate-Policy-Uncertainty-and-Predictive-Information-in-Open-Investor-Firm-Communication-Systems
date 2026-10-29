# Systems

# 投资者论坛：Hugging Face 模型比较与 MacBERT 文本特征实验

本包包含可衔接运行的实验代码、三份原始股票Excel及一份Wind指数Excel。**按2026-10-29最新要求，不附带QA、人工标注文本、文本样例数据集或任何实验中间数据。** 使用者在本地提供文本与标注后运行。

本机解压目录为 `F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository`。所有可执行入口从配置读取绝对路径，迁移机器时用配置生成器更新，不修改Python源码。

## 1. 文件组织

| 目录 | 内容 |
|---|---|
| scripts/workflow.py | 统一8阶段入口及一键执行 |
| scripts/configure.py | 生成适用于当前电脑的绝对路径配置 |
| configs/workflow.json | 默认实验参数、本机路径示例及固定事件词表 |
| data/raw/stock | 原始股票文件，未合并、未清洗、未改写 |
| data/raw/wind | 原始Wind指数文件，未改写 |
| legacy | 既有模型尝试源码记录；`.py.txt`和清除输出的Notebook供追溯 |
| tests/test_workflow.py | 使用真实小型Transformer的离线全流程测试，数据在临时目录生成 |
| docs | 数据来源与哈希、验证记录、历史选择依据、特征说明 |

原始股票文件保留73家公司的源覆盖范围，日期为2014-08-25至2026-08-20；未在缺少名单时任意删选为61家。三份文件的日期边界存在2018-08-22重叠，后续合并需检查重复键；本包保留原文件。Wind指数有1,772条有效日期记录，覆盖2018-05-18至2025-09-01。QA公司数量由 `expected_companies` 控制，默认 `null`；需要复现61家样本时设为61，并在本地提供相应QA。本包不包含教学半合成数据。

## 2. 安装与配置

推荐Python 3.12。安装命令中替换为自己的Python解释器绝对路径：

```powershell
python -m pip install -r "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\requirements.txt"
```

提供两个本地私有输入：

- 标注Excel/CSV：列为 `text,label`，标签为负面−1、中性0、正面1。标签来源、人工标注质量及使用权限由数据提供者确认。代码不把模型输出当成人工标签。
- QA CSV或只包含QA CSV的目录：至少有 `stock_code,datetime,question,reply_time,reply`。股票代码必须为六位字符串；问题时间有效；有回复时回复时间有效且不早于问题时间。空回复用空字符串；可选 `whether_answer` 与正文不得矛盾。不得把未知回复时间猜成问题时间。

先生成本地配置。以下命令省略 `--labels`/`--qa` 时会使用本目录下 `private/annotations.xlsx` 和 `private/qa.csv` 的绝对路径；也可显式传入已有私有文件的绝对路径：

```powershell
python "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\scripts\configure.py" --root "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository"
```

生成 `configs/workflow.local.json`。`private/`、本地配置和 `work/` 均被Git忽略。设备默认 `auto`，可设为 `cpu` 或 `cuda`。修改实验参数、输入标注或模型后，设置新的 `work_dir`，避免混用旧结果。

本项目已确认的标注源为 `F:\002论文\投资者论坛\dataset\text_data\沪深300股票问答数据-4000条.xlsx`。该文件不随包分发。在原电脑上可用以下命令接入：

```powershell
python "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\scripts\configure.py" --root "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository" --labels "F:\002论文\投资者论坛\dataset\text_data\沪深300股票问答数据-4000条.xlsx" --work-dir "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\local_runs\annotated_4000"
```

原表4,020行，字段为 `text,label`；缺失、冲突及重复处理后保留3,733条，固定划分为训练2,239条、验证747条、测试747条。处理只发生在运行目录，原始标注文件保持不变。使用该配置运行 `benchmark` 即可开展模型比较；`all` 还需提供私有QA路径。

## 3. 一键执行

```powershell
python "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\scripts\workflow.py" all --config "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\configs\workflow.local.json"
```

只做标注集模型比较时，把 `all` 改为 `benchmark`；这会运行前6步，不要求QA文件存在。也可把 `all` 改为下表单个阶段名，按顺序执行。

| 阶段 | 输入 | 输出与后续用途 |
|---|---|---|
| prepare | 私有标注数据 | 先按规范化文本去重、排除冲突标签，再固定分层划分60%训练/20%验证/20%测试；冻结配置和输入哈希 |
| download | Hugging Face模型ID | 下载MacBERT、中文FinBERT、hw2942三种模型；锁定Hub提交版本和文件哈希 |
| train | MacBERT基础模型、训练集 | 固定轮次微调三分类头；统一标签为negative/neutral/positive；保存模型与tokenizer |
| compare | 三种模型、同一个验证集 | 准确率、macro-F1、balanced accuracy、逐类召回率和混淆矩阵 |
| select | 验证比较表、预先设定的选择模式 | 固化所选模型及其哈希，同时保存验证集第一名 |
| evaluate | 固化模型、独立测试集 | 一次性测试指标；重复运行校验后复用，不据测试结果选模型 |
| encode | 固化模型、私有QA | 问题/回复向量；掩码均值池化、L2归一化，空文本为零向量，绑定QA顺序和哈希 |
| features | 同一QA、向量与模型 | 问题特征表、回复特征表和自然日计数，全部保存在本地work目录 |

各步骤的receipt记录输出哈希；缺失前置步骤、配置改变、输入或产物被替换时会明确停止，不静默混用结果。

## 4. “选择MacBERT”的准确含义

默认 `selection_mode="historical_macbert"`：重建用户指定的历史MacBERT路线，同时如实报告当前验证集排名。如果新排名第一并非MacBERT，结果会同时记录两者，不声称MacBERT胜出。

如要开展新的验证集选优实验，在运行前设置 `selection_mode="validation_best"`。按macro-F1选第一名，再完成测试、向量化和特征提取，后续自动使用该模型。

本协议沿用历史比较方式：MacBERT在本地标注训练集上微调，其余两个模型使用已有金融情绪分类头。训练预算不同，不能将结果解释为公平训练预算下的架构优越性。默认不自动微调其他模型，以免悄悄改变历史问题。

FinBERT标签顺序为中性/正面/负面，hw2942为负面/中性/正面。代码读取模型 `id2label` 后重排概率；遇到无语义的 `LABEL_0` 等标签会停止，要求填写经核实的 `label_maps`，不会猜测。

Erlangshen与Qwen的旧尝试代码保存在legacy。它们涉及不同类别体系或生成式/LoRA训练，不纳入统一三分类默认入口；本次未声称已修复并重跑这些历史尝试。可执行的新实验以 `scripts/workflow.py` 为准，legacy不再是第二套默认入口。

## 5. 特征及时间约束

输出包括文本长度、正负情绪概率差、问答数量、回复时滞、问答相关性、模板比例、历史回复新颖度和九类固定关键词事件标记。定义见 `docs/FEATURES.md`。

问题特征在问题时间可用；回复和交互特征在回复时间可用，分表输出。回复新颖度只使用同公司严格早于当前回复日期的历史回复，同日回复不互相进入历史池。数量表是自然日描述统计，预测交易日收益之前仍须按交易所日历和截止时刻映射；不能直接将未来回复回填问题日。

这套实验提供文本特征，未把市场指数的事后状态加入文本编码器，也不包含股价预测模型。当前RDIGF的Fin-Retriever、正式Q12/A19与本历史路线是不同实验。

## 6. 验证与发布

运行离线测试：

```powershell
python "F:\002论文\20260807投资者论坛RDIGF\outputs\github_release_20261029\repository\tests\test_workflow.py"
```

本次已用三个小型真实BERT检查全部8阶段、去重隔离、空回复、同日历史排除、测试结果复用及文件变更拒绝；另外检查真实标注数据的清洗与划分，但没有分发该数据。小型模型验证是工程验证，不是原模型性能复现。完整预训练模型下载及正式训练未执行。

发布包仅含原始市场数据、代码与说明，无文本/标注数据、模型权重、特征面板、运行结果和缓存。股票与Wind数据的原始来源和哈希记录在 `docs/RAW_DATA_MANIFEST.json`；第三方数据和模型仍适用其原有使用与再分发条款，本整理不产生额外授权。GitHub仓库：[Keaton9527/Systems](https://github.com/Keaton9527/Systems)。
