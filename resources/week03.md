# Week 3：Hugging Face 与 Tokenizer

本周目标：掌握 Hugging Face 基础用法，理解 tokenizer、model、pipeline 的关系，能手动完成一次模型推理。

---

## 1. 本周学习什么

### 1.1 Hugging Face 基础

- Model Hub
- Dataset Hub
- `pipeline`
- checkpoint
- model revision
- 模型缓存
- 指定模型名称

### 1.2 Tokenizer

- `AutoTokenizer`
- token
- vocabulary
- `input_ids`
- `attention_mask`
- `token_type_ids`
- padding
- truncation
- decode
- batch encoding

### 1.3 Model

- `AutoModel`
- `AutoModelForSequenceClassification`
- logits
- softmax
- label id
- label name

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| Hugging Face LLM Course Chapter 2 | tokenizer、model、pipeline | https://huggingface.co/learn/llm-course/chapter2/1 |
| Transformers Quicktour | 快速使用模型 | https://huggingface.co/docs/transformers/quicktour |
| Tokenizers Docs | tokenizer 细节 | https://huggingface.co/docs/tokenizers/index |
| Transformers GitHub | 生态源码入口 | https://github.com/huggingface/transformers |

---

## 3. 本周最终产出

```text
03-NLP-Basic/
├── tokenizer_demo.py
├── tokenizer_compare.py
└── manual_inference.py

notes/
├── day11.md
├── day12.md
├── day13.md
├── day14.md
├── day15.md
└── week03_summary.md

outputs/
├── tokenizer_outputs.md
└── manual_inference_result.md
```

---

## 4. 每日计划

## Day 11：认识 tokenizer

### 学习目标

- 理解 tokenizer 不是普通分词
- 查看 `input_ids` 和 `attention_mask`

### 任务

1. 加载 `distilbert/distilbert-base-uncased` tokenizer。
2. 输入英文句子。
3. 打印 token。
4. 打印 `input_ids`。
5. 使用 decode 还原。

### 代码文件

`03-NLP-Basic/tokenizer_demo.py`

### 验收标准

- 能解释 `input_ids`
- 能解释 `attention_mask`
- 能使用 `decode`

---

## Day 12：中文 tokenizer 对比

### 学习目标

- 理解中文分词和英文分词差异
- 对比不同模型 tokenizer 输出

### 任务

1. 使用 `bert-base-chinese`。
2. 使用英文 DistilBERT。
3. 输入中文句子和英文句子。
4. 比较 token 输出。
5. 记录差异。

### 代码文件

`03-NLP-Basic/tokenizer_compare.py`

### 验收标准

- 能说明中文模型和英文模型 tokenizer 不同
- 能解释 `[CLS]`、`[SEP]`

---

## Day 13：padding 与 truncation

### 学习目标

- 理解 batch 输入为什么需要 padding
- 理解长文本为什么要 truncation

### 任务

1. 准备长短不同的 3 句话。
2. 使用 padding。
3. 使用 truncation。
4. 设置 `max_length`。
5. 观察 attention mask。

### 输出文件

`outputs/tokenizer_outputs.md`

### 验收标准

- 能解释 padding 位置为什么 mask 为 0
- 能解释 max_length 的作用

---

## Day 14：手动推理

### 学习目标

- 不用 `pipeline` 完成一次分类推理
- 理解 tokenizer + model + softmax 流程

### 任务

1. 加载 tokenizer。
2. 加载分类模型。
3. 编码文本。
4. 前向传播得到 logits。
5. softmax 得到概率。
6. 输出预测标签。

### 代码文件

`03-NLP-Basic/manual_inference.py`

### 验收标准

- 能解释 logits
- 能解释 softmax
- 能说明 pipeline 内部大概流程

---

## Day 15：整理 Hugging Face 入门笔记

### 学习目标

- 整理 Hugging Face 基础概念
- 为 Week 5 微调做准备

### 任务

1. 总结 pipeline。
2. 总结 tokenizer。
3. 总结 AutoModel。
4. 总结 logits。
5. 写 Week 3 总结。

### 输出文件

`notes/week03_summary.md`

### 验收标准

- 能画出推理流程
- 能说明模型输入输出
- 能独立换模型运行

---

## 5. 本周必须掌握的问题

1. Hugging Face checkpoint 是什么？
2. tokenizer 的作用是什么？
3. `input_ids` 是什么？
4. `attention_mask` 是什么？
5. padding 和 truncation 分别解决什么问题？
6. logits 是什么？
7. softmax 做什么？
8. `pipeline` 内部做了哪些步骤？
9. 中文模型和英文模型能不能随便混用？
10. 为什么生产环境要指定模型名？

---

## 6. Week 3 完成标准

- 能使用 tokenizer
- 能解释模型输入
- 能手动完成推理
- 能对比不同 tokenizer
- 能写出 Hugging Face 推理流程

