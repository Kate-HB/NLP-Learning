# Week 4：BERT 微调与中文文本分类项目

目标：完成第一个完整项目 P1：中文文本分类。

## 本周资源

| 资源 | 链接 |
|---|---|
| BERT Paper | https://arxiv.org/abs/1810.04805 |
| The Illustrated BERT | https://jalammar.github.io/illustrated-bert/ |
| HF Chapter 3.1 | https://huggingface.co/learn/llm-course/zh-CN/chapter3/1 |
| HF Chapter 3.2 | https://huggingface.co/learn/llm-course/zh-CN/chapter3/2 |
| HF Chapter 3.3 | https://huggingface.co/learn/llm-course/zh-CN/chapter3/3 |
| Sequence Classification | https://huggingface.co/docs/transformers/tasks/sequence_classification |
| Trainer Docs | https://huggingface.co/docs/transformers/main_classes/trainer |

## 每日计划

| Day | 学习内容 | 项目任务 | 产出 |
|---|---|---|---|
| Day 22 | BERT、MLM、`[CLS]` | 写 BERT 原理笔记 | `05-BERT/bert_notes.md` |
| Day 23 | 数据集与 tokenizer | 准备中文分类数据 | `projects/p1-text-classification/prepare_data.py` |
| Day 24 | Trainer 微调 | 训练 `bert-base-chinese` | `projects/p1-text-classification/train.py` |
| Day 25 | 评估与错误分析 | 输出指标和错例 | `projects/p1-text-classification/evaluate.py` |
| Day 26 | 推理脚本 | 单句中文预测 | `projects/p1-text-classification/predict.py` |
| Day 27 | 项目 README | 写复现命令和结果 | `projects/p1-text-classification/README.md` |
| Day 28 | 复盘 | 讲清项目流程 | `notes/week04_summary.md` |

## P1 最低要求

1. 数据说明。
2. 训练脚本。
3. 评估脚本。
4. 推理脚本。
5. 错误分析。
6. README 能复现。

## 必须掌握

1. BERT 为什么适合文本理解。
2. `[CLS]` 用在哪里。
3. `Trainer` 帮你做了什么。
4. logits 如何对应标签。
5. 微调和预训练的区别。

