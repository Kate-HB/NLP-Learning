# P1 中文文本分类

目标：训练一个中文文本分类模型，输入一句中文文本，输出类别和置信度。

## 学习目的

- 理解 BERT 如何用于文本分类。
- 掌握 `AutoTokenizer`、`AutoModelForSequenceClassification`、`Trainer`。
- 学会评估指标和错误分析。

## 推荐模型

- `bert-base-chinese`
- `hfl/chinese-roberta-wwm-ext`

## 文件规划

```text
p1-text-classification/
├── README.md
├── prepare_data.py
├── train.py
├── evaluate.py
├── predict.py
└── error_analysis.md
```

## 最低功能

1. 加载中文分类数据。
2. 划分 train/eval/test。
3. tokenizer 编码。
4. Trainer 微调。
5. 输出 accuracy、F1。
6. 单句预测。

## 验收问题

1. 为什么分类任务使用 `[CLS]`。
2. logits 和 label 如何对应。
3. 为什么要有 baseline。
4. 错误样本主要错在哪里。

