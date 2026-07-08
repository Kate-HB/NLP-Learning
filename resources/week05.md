# Week 5：BERT

本周目标：理解 BERT 的 Encoder-only 架构，掌握 Hugging Face `Trainer` 微调流程，完成文本分类或 NER 项目。

---

## 1. 本周学习什么

### 1.1 BERT 原理

- Encoder-only
- Bidirectional context
- `[CLS]`
- `[SEP]`
- Masked Language Modeling
- Next Sentence Prediction
- Fine-tuning

### 1.2 文本分类

- `AutoModelForSequenceClassification`
- label id
- `Trainer`
- `TrainingArguments`
- `compute_metrics`
- Accuracy
- F1

### 1.3 NER 入门

- token classification
- BIO 标注
- label alignment
- entity-level F1

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| BERT Paper | BERT 原论文 | https://arxiv.org/abs/1810.04805 |
| The Illustrated BERT | BERT 图解 | https://jalammar.github.io/illustrated-bert/ |
| Hugging Face Course Chapter 3 | 微调预训练模型 | https://huggingface.co/learn/nlp-course/chapter3/1 |
| Transformers Trainer Docs | Trainer 用法 | https://huggingface.co/docs/transformers/main_classes/trainer |
| NLP-LOVE/ML-NLP | BERT 理论补充 | https://github.com/NLP-LOVE/ML-NLP |

---

## 3. 本周最终产出

```text
05-BERT/
├── bert_notes.md
├── train_classifier.py
├── predict.py
└── evaluate_model.py

09-Projects/
└── sentiment-analysis/
    ├── README.md
    ├── train.py
    └── predict.py

outputs/
└── bert_classification_result.md
```

---

## 4. 每日计划

## Day 21：BERT 原理

### 学习目标

- 理解 BERT 架构
- 理解 MLM 和 NSP

### 任务

1. 阅读 The Illustrated BERT。
2. 阅读 BERT Paper 摘要和模型部分。
3. 整理 `[CLS]`、MLM、NSP。
4. 写 BERT 笔记。

### 输出文件

`05-BERT/bert_notes.md`

### 验收标准

- 能解释 BERT 为什么适合理解任务
- 能解释 MLM

---

## Day 22：准备分类数据

### 学习目标

- 准备 BERT 分类数据
- 使用 tokenizer 编码数据

### 任务

1. 准备小型情感分析数据。
2. 划分训练集和验证集。
3. 使用 tokenizer 编码。
4. 检查 label。

### 代码文件

`05-BERT/prepare_data.py`

### 验收标准

- 能得到可训练 dataset
- 能解释 label 和 logits 的对应关系

---

## Day 23：Trainer 微调

### 学习目标

- 使用 Hugging Face Trainer 训练分类模型

### 任务

1. 加载 `AutoModelForSequenceClassification`。
2. 设置 `TrainingArguments`。
3. 定义 `compute_metrics`。
4. 启动训练。
5. 保存模型。

### 代码文件

`05-BERT/train_classifier.py`

### 验收标准

- 能跑通训练
- 能保存 checkpoint
- 能输出指标

---

## Day 24：预测与评估

### 学习目标

- 加载训练好的模型做预测
- 输出分类结果

### 任务

1. 加载保存模型。
2. 输入单句文本。
3. 输出标签和概率。
4. 评估验证集结果。

### 代码文件

`05-BERT/predict.py`

### 验收标准

- 能预测单条文本
- 能输出分类概率

---

## Day 25：整理项目 README

### 学习目标

- 将 BERT 分类整理成可展示项目

### 任务

1. 写项目背景。
2. 写数据集说明。
3. 写模型方法。
4. 写运行命令。
5. 写结果和错误分析。

### 输出文件

`09-Projects/sentiment-analysis/README.md`

### 验收标准

- README 能让别人复现
- 有指标和错误分析

---

## 5. 本周必须掌握的问题

1. BERT 是 Encoder-only 还是 Decoder-only？
2. `[CLS]` 的作用是什么？
3. MLM 是什么？
4. Fine-tuning 和 Pretraining 有什么区别？
5. `Trainer` 做了什么？
6. `TrainingArguments` 做什么？
7. `compute_metrics` 做什么？
8. logits 怎么变成 label？
9. BERT 为什么不适合直接长文本生成？
10. BERT 分类项目如何复现？

---

## 6. Week 5 完成标准

- 能解释 BERT 基本原理
- 能用 Trainer 微调模型
- 能预测单条文本
- 能写项目 README
- 能完成错误分析
