# 47 三类架构：Encoder-only、Decoder-only、Encoder-Decoder

> 一句话：BERT 是 Encoder-only、GPT 是 Decoder-only、T5 是 Encoder-Decoder；三者的本质差别是"信息可见范围"和"输出方式"，而不是模型名字。


## 前置知识

- 先读 [33-三类架构的数据流](33-三类架构的数据流.md)、[44-Encoder-的完整结构](44-Encoder-的完整结构.md)、[45-Decoder-的完整结构](45-Decoder-的完整结构.md)。
- 先读 [33-三类架构的数据流](33-data-flow-of-three-architectures.md)、[44-Encoder-的完整结构](44-encoder-architecture.md)、[45-Decoder-的完整结构](45-decoder-architecture.md)。

## 一、直观含义

- **Encoder-only（BERT 家族）**："通读全文再答题"的理解者——适合判断、抽取，不适合长篇续写。
- **Decoder-only（GPT 家族）**："只看过去、一路往下写"的写作者——适合生成、对话、续写。
- **Encoder-Decoder（T5、BART）**："先读原文、再照着写"的翻译官——适合翻译、摘要这类"输入输出都是序列但角色不同"的任务。

## 二、具体例子

- 情感分类、NER、抽取式问答 → BERT 类。
- 聊天、写代码、续写 → GPT 类。
- 英译中、长文摘要 → T5 / 翻译模型。

## 三、技术要点（严谨部分）

### 1. Encoder-only

- 只使用 Encoder Block。
- 代表：**BERT**（双向 Encoder 预训练上下文表示）、**RoBERTa**（调整 BERT 预训练策略）、**DistilBERT**（知识蒸馏压缩 BERT——教师模型指导学生模型，用更少参数换更快推理）。
- 特点：通常利用**双向上下文**，擅长理解类任务：文本分类、句子对判断、NER、抽取式问答、Embedding、掩码填充。
- 局限：**不天然适合长篇自回归生成**，因为训练结构不是逐步生成目标序列。

### 2. Decoder-only

- 只使用带**因果掩码**的 Decoder Block。
- 代表：**GPT 系列**和许多现代 LLM。
- **GPT（Generative Pre-trained Transformer）**：用因果语言建模预训练，通过前文预测后续 Token。"Generative" = 生成式，"Pre-trained" = 先大规模预训练。
- 特点：把各种任务统一成"**根据前文继续生成**"——分类、问答、翻译都可以通过提示词转成文本生成问题。
- 擅长：开放式生成、对话、续写、代码生成。
- 代价：每个新 Token 的推理依赖前面已生成内容（串行）。

### 3. Encoder-Decoder

- Encoder 先理解完整输入，Decoder 再条件生成输出。
- 代表：**T5**（把不同 NLP 任务统一表达为文本输入 → 文本输出）、**BART**（双向 Encoder + 自回归 Decoder，通过破坏并重建文本学习 Seq2Seq 表示）、许多翻译模型。
- 适合：输入输出都是序列、但二者作用不同的任务——翻译、摘要、语音识别、条件文本生成。

### 4. 三类架构的本质差异

> 最核心的区别不是模型名字，而是**信息可见范围**和**输出方式**。

| 架构 | 输入可见性 | 输出方式 | 典型优势 |
|---|---|---|---|
| Encoder-only | 通常双向读取完整输入 | 输出表示或任务标签 | 理解与抽取 |
| Decoder-only | 位置 i 读取输入中的位置 ≤ i，并预测位置 i+1 | 自回归生成 | 开放式生成 |
| Encoder-Decoder | Encoder 双向理解，Decoder 因果生成并读取 Encoder | 条件生成 | 输入到输出转换 |

## 四、一句话总结

BERT 读全文答题、GPT 看过去续写、T5 读原文再写译文；选哪种架构，看你要的是"理解"、"生成"还是"条件转换"。
