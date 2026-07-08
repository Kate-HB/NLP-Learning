# Week 7：现代 LLM

本周目标：了解现代大模型训练和应用的关键概念，重点掌握 LoRA、PEFT、Instruction Tuning、RLHF 思想，并完成语义检索小实验。

---

## 1. 本周学习什么

### 1.1 参数高效微调

- Fine-tuning
- LoRA
- PEFT
- QLoRA 基础
- adapter 思想
- 冻结主模型
- 训练少量参数

### 1.2 指令微调与对齐

- Instruction Tuning
- SFT
- RLHF
- reward model
- preference data
- alignment

### 1.3 语义向量

- Sentence Embedding
- Bi-Encoder
- Cross-Encoder
- cosine similarity
- top-k retrieval
- embedding model

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| PEFT Docs | LoRA、PEFT | https://huggingface.co/docs/peft/index |
| Sentence Transformers | 语义向量 | https://github.com/UKPLab/sentence-transformers |
| Hugging Face Model Hub | 查找模型 | https://huggingface.co/models |
| NLP-LOVE/ML-NLP | 大模型相关补充 | https://github.com/NLP-LOVE/ML-NLP |

---

## 3. 本周最终产出

```text
07-LLM/
├── lora_notes.md
├── peft_notes.md
├── instruction_tuning_notes.md
└── rlhf_notes.md

08-RAG/
├── embedding_demo.py
└── semantic_search.py

outputs/
└── semantic_search_result.md
```

---

## 4. 每日计划

## Day 31：LoRA 与 PEFT

### 学习目标

- 理解为什么需要参数高效微调
- 理解 LoRA 的基本思想

### 任务

1. 阅读 PEFT Docs 入门。
2. 整理 full fine-tuning 和 LoRA 区别。
3. 理解冻结参数。
4. 写 LoRA 笔记。

### 输出文件

`07-LLM/lora_notes.md`

### 验收标准

- 能解释 LoRA 为什么省显存
- 能说明训练哪些参数

---

## Day 32：Instruction Tuning

### 学习目标

- 理解指令微调
- 理解 SFT 数据格式

### 任务

1. 整理 instruction、input、output。
2. 找 5 条指令数据例子。
3. 说明普通预训练和指令微调区别。

### 输出文件

`07-LLM/instruction_tuning_notes.md`

### 验收标准

- 能解释指令微调的目的
- 能写一条训练样本格式

---

## Day 33：RLHF 思想

### 学习目标

- 理解 RLHF 的大概流程
- 不要求实现

### 任务

1. 整理 SFT、Reward Model、PPO 三阶段。
2. 理解 preference data。
3. 写 RLHF 笔记。

### 输出文件

`07-LLM/rlhf_notes.md`

### 验收标准

- 能说清 RLHF 的目的
- 能说清人工偏好数据的作用

---

## Day 34：Sentence Embedding

### 学习目标

- 使用 sentence-transformers 生成句向量
- 计算相似度

### 任务

1. 准备 10 个句子。
2. 加载 embedding 模型。
3. 生成向量。
4. 计算余弦相似度。
5. 找最相似句子。

### 代码文件

`08-RAG/embedding_demo.py`

### 验收标准

- 能解释 embedding
- 能解释余弦相似度

---

## Day 35：语义搜索

### 学习目标

- 完成 Top-k 语义检索

### 任务

1. 准备 20 条文档片段。
2. 为每条文档生成 embedding。
3. 输入查询。
4. 返回 Top-5。
5. 分析错误结果。

### 代码文件

`08-RAG/semantic_search.py`

### 验收标准

- 能返回 Top-k 文档
- 有相似度分数
- 有错误分析

---

## 5. 本周必须掌握的问题

1. 为什么大模型微调成本高？
2. LoRA 的核心思想是什么？
3. PEFT 是什么？
4. QLoRA 大概解决什么问题？
5. Instruction Tuning 是什么？
6. RLHF 是什么？
7. Sentence Embedding 是什么？
8. Bi-Encoder 和 Cross-Encoder 有什么区别？
9. 余弦相似度是什么？
10. 语义检索和关键词检索有什么区别？

---

## 6. Week 7 完成标准

- 能解释 LoRA
- 能解释 Instruction Tuning
- 能解释 RLHF 思想
- 能生成句向量
- 能完成语义搜索 demo
