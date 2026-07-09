# Week 2：NLP 基础

本周目标：掌握传统 NLP 的文本表示方法，完成一个 TF-IDF 文本分类 baseline，为后续 BERT 微调建立对照组。

---

## 1. 本周学习什么

### 1.1 文本处理

- 文本清洗
- 大小写处理
- 标点处理
- 停用词
- 中文分词
- 英文 tokenization
- 训练集、验证集、测试集

### 1.2 传统文本表示

- Bag of Words
- TF-IDF
- N-Gram
- 稀疏向量
- 余弦相似度
- Word2Vec 思想
- GloVe 思想

### 1.3 传统分类流程

- 数据读取
- 标签编码
- 特征提取
- Logistic Regression
- Naive Bayes
- Accuracy
- Precision
- Recall
- F1
- 错误分析

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| Hugging Face LLM Course Chapter 1 | NLP 任务概览 | https://huggingface.co/learn/llm-course/chapter1/1 |
| scikit-learn Text Feature Extraction | TF-IDF、CountVectorizer | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction |
| NLP-LOVE/ML-NLP | NLP 基础、词向量 | https://github.com/NLP-LOVE/ML-NLP |
| Stanford CS224N | Word Vectors | https://web.stanford.edu/class/cs224n/ |

---

## 3. 本周最终产出

```text
03-NLP-Basic/
├── text_preprocess.py
├── tfidf_demo.py
├── tfidf_classifier.py
└── word_vector_notes.md

notes/
├── day06.md
├── day07.md
├── day08.md
├── day09.md
├── day10.md
└── week02_summary.md

outputs/
├── tfidf_result.md
└── tfidf_error_cases.md
```

---

## 4. 每日计划

## Day 6：文本预处理

### 学习目标

- 理解文本为什么要预处理
- 能写基础清洗函数
- 能区分中文和英文处理差异

### 任务

1. 准备 10 条中文文本和 10 条英文文本。
2. 去除多余空格。
3. 处理大小写。
4. 去除或保留标点。
5. 保存清洗前后对比。

### 代码文件

`03-NLP-Basic/text_preprocess.py`

### 验收标准

- 能解释文本清洗的目的
- 代码能输入原文本并输出清洗文本
- 能说明哪些清洗可能损失语义

---

## Day 7：Bag of Words 与 TF-IDF

### 学习目标

- 理解词袋模型
- 理解 TF-IDF
- 能使用 sklearn 提取文本特征

### 任务

1. 使用 `CountVectorizer`。
2. 使用 `TfidfVectorizer`。
3. 查看 vocabulary。
4. 查看向量维度。
5. 比较 BoW 和 TF-IDF 输出。

### 代码文件

`03-NLP-Basic/tfidf_demo.py`

### 验收标准

- 能解释 TF、IDF
- 能解释为什么 TF-IDF 是稀疏向量
- 能输出文本向量维度

---

## Day 8：文本分类 baseline

### 学习目标

- 用 TF-IDF 做文本分类
- 理解 baseline 的价值
- 会计算分类指标

### 任务

1. 准备一个小型分类数据集。
2. 划分训练集和测试集。
3. 使用 TF-IDF 提取特征。
4. 训练 Logistic Regression。
5. 输出 Accuracy、Precision、Recall、F1。

### 代码文件

`03-NLP-Basic/tfidf_classifier.py`

### 验收标准

- 能跑通训练和预测
- 能输出指标
- 能解释 baseline 是后续模型对比对象

---

## Day 9：错误分析

### 学习目标

- 学会看模型错在哪里
- 理解错误分析比单纯看分数更重要

### 任务

1. 找出预测错误样本。
2. 记录真实标签和预测标签。
3. 人工分析错误原因。
4. 总结至少 3 类错误。

### 输出文件

`outputs/tfidf_error_cases.md`

### 验收标准

- 至少记录 10 条错误样本
- 每条错误有原因
- 能提出改进方向

---

## Day 10：Word2Vec 与 GloVe 思想

### 学习目标

- 理解词向量和 TF-IDF 的区别
- 理解 Word2Vec 基本思想
- 理解 GloVe 基本思想

### 任务

1. 阅读 NLP-LOVE/ML-NLP 词向量内容。
2. 整理 CBOW 和 Skip-gram 区别。
3. 整理 TF-IDF 和 Word2Vec 区别。
4. 写 Week 2 总结。

### 输出文件

`03-NLP-Basic/word_vector_notes.md`

### 验收标准

- 能解释词向量为什么能表示语义
- 能解释上下文窗口
- 能说明传统方法的局限

---

## 5. 本周必须掌握的问题

1. 什么是 token？
2. 什么是文本清洗？
3. Bag of Words 的缺点是什么？
4. TF-IDF 中 TF 和 IDF 分别是什么？
5. N-Gram 有什么作用？
6. Word2Vec 和 TF-IDF 有什么区别？
7. 为什么要划分训练集和测试集？
8. Accuracy 和 F1 有什么区别？
9. 为什么需要错误分析？
10. baseline 有什么价值？

---

## 6. Week 2 完成标准

- 能独立完成 TF-IDF 分类
- 能解释传统文本表示方法
- 能输出分类指标
- 能写错误分析
- 能说明为什么后续要学 BERT

