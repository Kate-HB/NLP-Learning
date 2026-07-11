# Week 1：Hugging Face 入门与 Tokenizer

目标：理解 `pipeline` 背后的完整流程：文本输入 -> tokenizer -> 张量 -> model -> logits -> 后处理结果。

## 本周资源

| 资源 | 链接 | 用法 |
|---|---|---|
| HF Chapter 1 | https://huggingface.co/learn/llm-course/zh-CN/chapter1/1 | 复盘 NLP、Transformer、pipeline |
| HF Chapter 2.2 | https://huggingface.co/learn/llm-course/zh-CN/chapter2/2 | 理解 pipeline 三步 |
| HF Chapter 2.3 | https://huggingface.co/learn/llm-course/zh-CN/chapter2/3 | 理解模型加载 |
| HF Chapter 2.4 | https://huggingface.co/learn/llm-course/zh-CN/chapter2/4 | 理解 tokenizer |
| HF Chapter 2.5 | https://huggingface.co/learn/llm-course/zh-CN/chapter2/5 | 理解 padding、truncation、batch |
| HF Chapter 2.6 | https://huggingface.co/learn/llm-course/zh-CN/chapter2/6 | 手动推理 |
| Transformers Quicktour | https://huggingface.co/docs/transformers/quicktour | API 查询 |

## 每日计划

| Day | 学习内容 | 代码任务 | 笔记重点 |
|---|---|---|---|
| Day 01 | NLP、LLM、Transformer、pipeline | 重跑 `sentiment-analysis` | pipeline 是什么，不是什么 |
| Day 02 | pipeline 背后三步 | 跑情感分析、生成、零样本分类 | 预处理、模型、后处理 |
| Day 03 | AutoModel、AutoConfig | 打印模型结构和 config | checkpoint、config、权重 |
| Day 04 | Tokenizer 基础 | 中文、英文转 `input_ids` | token、id、vocab |
| Day 05 | padding、truncation、batch | 3 条不同长度文本组成 batch | `attention_mask` 为什么需要 |
| Day 06 | 手动推理 | tokenizer + model + softmax | logits 到 label 的过程 |
| Day 07 | 复盘 | 重跑 Day 02 到 Day 06 | 画出完整推理流程 |

## 本周产出

```text
03-NLP-Basic/
├── automodel_demo.py
├── tokenizer_demo.py
└── manual_inference.py

outputs/
└── tokenizer_batch.md

notes/
├── day01.md
...
└── week01_summary.md
```

## 必须掌握

1. `pipeline` 内部做了哪三步。
2. `input_ids` 是什么。
3. `attention_mask` 是什么。
4. logits 怎么变成预测标签。
5. tokenizer 和 model 为什么必须匹配。

