# 历史选择记录

来源：`F:\002论文\投资者论坛\result\20250528更换MacBert模型的微调效果\model_comparison.csv`。

| 模型 | 历史准确率 | 负面召回率 | 中性召回率 | 正面召回率 |
|---|---:|---:|---:|---:|
| macbert-finetuned | 0.773333 | 0.80 | 0.74 | 0.78 |
| finbert | 0.576667 | 0.28 | 0.84 | 0.61 |
| hw2942 | 0.440000 | 0.24 | 0.94 | 0.14 |

原脚本将逐类召回率称作逐类准确率，此处按计算定义命名。这些是已有历史文件中的数值，本次未重跑模型，不将原预测CSV作为原始数据打包。

历史MacBERT代码按每类最多500条取样，再按80%/20%划分，并在训练过程中评价该20%数据。因此该结果不足以证明独立测试集上的泛化优越性。当前复现入口采用不同的60%/20%/20%协议，是修正后的新实验，不能期待逐值复现历史表。

新版流程同时保存验证第一名与实际所选模型。默认historical_macbert重建历史MacBERT路线；validation_best按新验证集macro-F1选优。默认路线不会把新验证结果改写成MacBERT胜出。

模型标签配置核对（2026-10-29）：
- [FinBERT配置](https://huggingface.co/yiyanghkust/finbert-tone-chinese/raw/main/config.json)：0=Neutral、1=Positive、2=Negative。
- [hw2942配置](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2/raw/main/config.json)：0=Negative、1=Neutral、2=Positive。

运行时仍读取下载快照的配置，并锁定实际Hub版本及文件哈希。
