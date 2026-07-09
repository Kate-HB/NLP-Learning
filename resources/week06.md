# Week 6：GPT

本周目标：理解 GPT 的 Decoder-only 架构和自回归生成流程，掌握常见解码策略，完成文本生成 demo。

---

## 1. 本周学习什么

### 1.1 GPT 原理

- Decoder-only
- Causal Language Modeling
- Causal Mask
- 自回归生成
- next token prediction

### 1.2 解码策略

- greedy search
- beam search
- top-k sampling
- top-p sampling
- temperature
- repetition penalty
- max_new_tokens

### 1.3 生成问题

- 重复生成
- 跑题
- 幻觉
- prompt 敏感
- 长文本退化

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| GPT-2 Paper | GPT 生成模型 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Transformers Generation Docs | 生成参数 | https://huggingface.co/docs/transformers/main_classes/text_generation |
| Hugging Face Course | Causal LM | https://huggingface.co/learn/llm-course/chapter7/6 |
| NLP-LOVE/ML-NLP | 语言模型补充 | https://github.com/NLP-LOVE/ML-NLP |

---

## 3. 本周最终产出

```text
06-GPT/
├── gpt_notes.md
├── generation_pipeline.py
├── decoding_compare.py
└── prompt_experiments.py

outputs/
├── generation_comparison.md
└── generation_error_analysis.md

notes/
└── week06_summary.md
```

---

## 4. 每日计划

## Day 26：GPT 原理

### 学习目标

- 理解 Decoder-only
- 理解自回归生成

### 任务

1. 阅读 GPT-2 Paper 摘要。
2. 整理 GPT 和 BERT 区别。
3. 理解 causal mask。
4. 写 GPT 笔记。

### 输出文件

`06-GPT/gpt_notes.md`

### 验收标准

- 能解释 GPT 为什么适合生成
- 能解释 causal mask

---

## Day 27：文本生成 pipeline

### 学习目标

- 跑通 text-generation pipeline

### 任务

1. 加载 `gpt2` 或中文 GPT-2。
2. 输入 prompt。
3. 生成文本。
4. 修改 `max_new_tokens`。
5. 记录输出。

### 代码文件

`06-GPT/generation_pipeline.py`

### 验收标准

- 能成功生成文本
- 能解释 prompt 的作用

---

## Day 28：解码策略对比

### 学习目标

- 对比 greedy、beam、top-k、top-p

### 任务

1. 固定同一个 prompt。
2. 使用 greedy 生成。
3. 使用 beam search 生成。
4. 使用 top-k 生成。
5. 使用 top-p 生成。
6. 记录差异。

### 代码文件

`06-GPT/decoding_compare.py`

### 验收标准

- 能解释 top-k 和 top-p 区别
- 能说明 beam search 更稳定但可能保守

---

## Day 29：temperature 与重复问题

### 学习目标

- 理解 temperature
- 观察重复生成

### 任务

1. 设置不同 temperature。
2. 设置 repetition penalty。
3. 观察文本质量变化。
4. 记录重复案例。

### 输出文件

`outputs/generation_error_analysis.md`

### 验收标准

- 能解释 temperature 高低影响
- 能分析生成错误

---

## Day 30：生成实验总结

### 学习目标

- 整理生成模型实验
- 总结 GPT 局限

### 任务

1. 汇总不同解码结果。
2. 写输出质量分析。
3. 写 Week 6 总结。
4. 准备进入 RAG 学习。

### 输出文件

`outputs/generation_comparison.md`

### 验收标准

- 有参数对比表
- 有错误分析
- 能解释生成模型幻觉

---

## 5. 本周必须掌握的问题

1. GPT 是什么结构？
2. 自回归生成是什么？
3. causal mask 是什么？
4. greedy search 有什么问题？
5. beam search 有什么特点？
6. top-k 和 top-p 有什么区别？
7. temperature 影响什么？
8. 为什么会重复生成？
9. 为什么会幻觉？
10. GPT 和 BERT 的核心区别是什么？

---

## 6. Week 6 完成标准

- 能跑通文本生成
- 能对比解码策略
- 能解释生成参数
- 能写生成错误分析

