# Week 5：传统 NLP、BERT 与序列标注

目标：理解 baseline 的价值，并扩展到 NER。

## 本周资源

| 资源 | 链接 |
|---|---|
| sklearn Text Feature Extraction | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction |
| TfidfVectorizer | https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html |
| classification_report | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html |
| Token Classification | https://huggingface.co/docs/transformers/tasks/token_classification |
| seqeval | https://huggingface.co/spaces/evaluate-metric/seqeval |

## 每日计划

| Day | 学习内容 | 实验任务 | 笔记重点 |
|---|---|---|---|
| Day 29 | Bag of Words、TF-IDF | 做 TF-IDF 分类 baseline | baseline 为什么重要 |
| Day 30 | baseline 对比 BERT | 比较指标和错例 | 传统方法局限 |
| Day 31 | BIO 标注 | 手写 5 条 BIO 样例 | B/I/O 含义 |
| Day 32 | Token classification | 跑 NER demo | token-label 对齐 |
| Day 33 | NER 评估 | 用 seqeval 看实体级 F1 | token 级 vs 实体级 |
| Day 34 | BERT 总结 | 整理分类、NER、MLM | Encoder-only 能做什么 |
| Day 35 | 复盘 | 补齐 Week 4 项目说明 | 分类和序列标注区别 |

## 本周产出

```text
03-NLP-Basic/
├── tfidf_classifier.py
└── tfidf_error_analysis.md

05-BERT/
├── ner_notes.md
└── token_classification_demo.py
```

## 必须掌握

1. TF-IDF 和 BERT 表示的区别。
2. baseline 的作用。
3. BIO 标注规则。
4. token classification 和 sequence classification 的区别。
5. NER 为什么要做 label alignment。

