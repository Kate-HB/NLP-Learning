# Week 6：GPT、生成与解码策略

目标：理解 GPT 生成流程和常见生成参数，不追求训练 GPT。

## 本周资源

| 资源 | 链接 |
|---|---|
| GPT-2 Paper | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| GPT-2 Docs | https://huggingface.co/docs/transformers/model_doc/gpt2 |
| Generation Docs | https://huggingface.co/docs/transformers/main_classes/text_generation |
| Generation Strategies | https://huggingface.co/docs/transformers/generation_strategies |
| How to generate | https://huggingface.co/blog/how-to-generate |

## 每日计划

| Day | 学习内容 | 实验任务 | 笔记重点 |
|---|---|---|---|
| Day 36 | GPT、Decoder-only | 整理 GPT 和 BERT 区别 | causal mask |
| Day 37 | 自回归生成 | 跑 text-generation pipeline | next token prediction |
| Day 38 | greedy、beam | 固定 prompt 对比输出 | 稳定性和多样性 |
| Day 39 | top-k、top-p | 对比采样结果 | 随机性来源 |
| Day 40 | temperature、repetition penalty | 观察重复和发散 | 参数如何影响文本 |
| Day 41 | 生成错误分析 | 记录跑题、重复、幻觉 | 生成模型局限 |
| Day 42 | 复盘 | 写生成策略对比表 | GPT 适合什么任务 |

## 本周产出

```text
06-GPT/
├── gpt_notes.md
├── generation_pipeline.py
├── decoding_compare.py
└── prompt_experiments.py

outputs/
├── generation_comparison.md
└── generation_error_analysis.md
```

## 必须掌握

1. GPT 为什么是 Decoder-only。
2. 自回归生成是什么。
3. greedy、beam、top-k、top-p 的区别。
4. temperature 控制什么。
5. 为什么会重复和幻觉。

