# Week 8：项目实战

本周目标：把前 7 周学习内容整理成可展示项目，完成代码、README、实验结果、错误分析和保研/面试讲解材料。

---

## 1. 本周学习什么

### 1.1 项目整理

- 项目目录规范
- README 写作
- 运行命令
- 依赖说明
- 数据集说明
- 模型说明
- 结果表格
- 错误分析

### 1.2 项目实战

- 新闻分类
- 情感分析
- 命名实体识别
- RAG 文档问答
- 论文智能分析系统

### 1.3 展示能力

- GitHub 项目展示
- 项目讲解稿
- 面试问答
- 方法对比
- 改进方向

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| Hugging Face Examples | 项目代码参考 | https://github.com/huggingface/transformers/tree/main/examples |
| Hugging Face Course | 项目流程参考 | https://huggingface.co/learn/llm-course/chapter1/1 |
| NLP-LOVE/ML-NLP | 面试知识点 | https://github.com/NLP-LOVE/ML-NLP |
| GitHub Docs | README 写作 | https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes |

---

## 3. 本周最终产出

```text
09-Projects/
├── news-classification/
│   ├── README.md
│   ├── train_tfidf.py
│   └── train_bert.py
├── sentiment-analysis/
│   ├── README.md
│   ├── train.py
│   └── predict.py
├── named-entity-recognition/
│   ├── README.md
│   └── train.py
└── rag-demo/
    ├── README.md
    ├── ingest.py
    ├── retrieve.py
    └── ask.py

Interview/
├── nlp_questions.md
└── project_explanation.md
```

---

## 4. 每日计划

## Day 36：新闻分类项目整理

### 学习目标

- 整理 Week 2 baseline
- 增加 BERT 对比方案

### 任务

1. 创建 `09-Projects/news-classification/`。
2. 整理 TF-IDF baseline。
3. 写数据集说明。
4. 写指标表。
5. 写错误分析。

### 验收标准

- README 能复现项目
- 有 baseline 结果
- 有错误案例

---

## Day 37：情感分析项目整理

### 学习目标

- 整理 BERT 分类项目

### 任务

1. 创建 `09-Projects/sentiment-analysis/`。
2. 整理训练脚本。
3. 整理预测脚本。
4. 写模型说明。
5. 写运行命令。

### 验收标准

- 能单句预测
- README 有完整流程

---

## Day 38：NER 项目整理

### 学习目标

- 整理命名实体识别方案
- 明确 BIO 标注和 token alignment

### 任务

1. 创建 `09-Projects/named-entity-recognition/`。
2. 写 NER 项目 README。
3. 整理 BIO 标注说明。
4. 整理评估指标。
5. 写常见错误。

### 验收标准

- 能解释 BIO
- 能解释 token-label 对齐

---

## Day 39：RAG 文档问答项目

### 学习目标

- 整理语义检索和生成流程
- 形成 RAG 项目 demo

### 任务

1. 创建 `09-Projects/rag-demo/`。
2. 写 `ingest.py` 设计。
3. 写 `retrieve.py` 设计。
4. 写 `ask.py` 设计。
5. README 写 RAG 流程。

### 验收标准

- 能解释 chunk、embedding、retrieve、generate
- 答案要求带引用来源

---

## Day 40：面试讲解与总复盘

### 学习目标

- 把项目讲清楚
- 准备保研/面试问题

### 任务

1. 写 `Interview/project_explanation.md`。
2. 写 `Interview/nlp_questions.md`。
3. 整理总 README。
4. 检查所有项目是否可复现。
5. 写 Week 8 总结。

### 验收标准

- 每个项目能 3 分钟讲清楚
- 每个项目有 3 个改进方向
- 面试题能覆盖 NLP 基础、Transformer、BERT、GPT、RAG

---

## 5. 项目 README 模板

```md
# 项目名称

## 1. 项目背景

## 2. 数据集

## 3. 方法

## 4. 实验设置

## 5. 结果

## 6. 错误分析

## 7. 运行方式

## 8. 后续改进
```

---

## 6. 项目讲解模板

```md
# 项目讲解：项目名称

## 项目解决什么问题

## 为什么选择这个方法

## 数据如何处理

## 模型如何训练

## 指标结果如何

## 错误主要来自哪里

## 后续怎么改进
```

---

## 7. 本周必须掌握的问题

1. 你的项目解决什么问题？
2. 为什么选择这个模型？
3. 数据集怎么处理？
4. baseline 是什么？
5. 指标如何解释？
6. 错误分析发现了什么？
7. 项目有什么不足？
8. 如何进一步改进？
9. RAG 为什么需要检索？
10. 你的项目如何复现？

---

## 8. Week 8 完成标准

- 至少完成 3 个项目 README
- 至少完成 1 个可运行项目
- 每个项目有结果和错误分析
- 完成面试讲解稿
- 完成总复盘

