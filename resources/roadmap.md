# NLP 系统学习路线：56 天

定位：系统理解 NLP，同时每阶段做项目验证。每天按 5 小时设计。

## 每天时间分配

| 模块 | 时间 | 要求 |
|---|---:|---|
| 理解 | 3 小时 | 读主资源，整理概念、输入输出、关键公式 |
| 代码 | 1.5 小时 | 跑通官方示例，再改一个参数或样例 |
| 笔记 | 0.5 小时 | 写 `notes/dayXX.md`，末尾必须有 `## 今日知识总结` |

## 8 周总览

| 周 | 主题 | 理解重点 | 项目/实验 |
|---|---|---|---|
| Week 1 | Hugging Face 入门与 Tokenizer | pipeline、tokenizer、model、logits | 手动推理脚本 |
| Week 2 | PyTorch 最小训练基础 | tensor、autograd、Dataset、训练循环 | 最小分类训练 |
| Week 3 | 深度学习概念与 Transformer | softmax、cross entropy、QKV、mask | Attention 小实验 |
| Week 4 | BERT 微调 | BERT、Trainer、评估指标 | P1 中文文本分类 |
| Week 5 | 传统 NLP、BERT 与序列标注 | TF-IDF、baseline、BIO、NER | 分类 baseline + NER demo |
| Week 6 | GPT 与生成 | Decoder-only、自回归、解码策略 | 生成参数对比 |
| Week 7 | Embedding 与 RAG | embedding、向量检索、chunk、引用 | P2 RAG 文档问答 |
| Week 8 | LLM 应用整合 | 分类 + RAG + UI + README | P3 LLM 应用 Demo |

## 每日路线

| Day | 主题 | 计划 |
|---|---|---|
| 01 | NLP / LLM / Transformer 总览 | [Week 1](weeks/week01.md) |
| 02 | pipeline 背后流程 | [Week 1](weeks/week01.md) |
| 03 | AutoTokenizer | [Week 1](weeks/week01.md) |
| 04 | AutoModel 与 logits | [Week 1](weeks/week01.md) |
| 05 | padding、truncation、batch | [Week 1](weeks/week01.md) |
| 06 | 手动推理 | [Week 1](weeks/week01.md) |
| 07 | Week 1 复盘 | [Week 1](weeks/week01.md) |
| 08 | Tensor、shape、device | [Week 2](weeks/week02.md) |
| 09 | Dataset 和 DataLoader | [Week 2](weeks/week02.md) |
| 10 | nn.Module | [Week 2](weeks/week02.md) |
| 11 | Autograd | [Week 2](weeks/week02.md) |
| 12 | Loss 和 Optimizer | [Week 2](weeks/week02.md) |
| 13 | 训练循环 | [Week 2](weeks/week02.md) |
| 14 | Week 2 复盘 | [Week 2](weeks/week02.md) |
| 15 | NumPy shape、axis、广播 | [Week 3](weeks/week03.md) |
| 16 | softmax 和 cross entropy | [Week 3](weeks/week03.md) |
| 17 | Embedding | [Week 3](weeks/week03.md) |
| 18 | Attention 与 QKV | [Week 3](weeks/week03.md) |
| 19 | Multi-Head Attention | [Week 3](weeks/week03.md) |
| 20 | Transformer Encoder | [Week 3](weeks/week03.md) |
| 21 | Week 3 复盘 | [Week 3](weeks/week03.md) |
| 22 | BERT 原理 | [Week 4](weeks/week04.md) |
| 23 | 数据集与 tokenizer | [Week 4](weeks/week04.md) |
| 24 | Trainer 微调 | [Week 4](weeks/week04.md) |
| 25 | 评估与错误分析 | [Week 4](weeks/week04.md) |
| 26 | P1 推理脚本 | [Week 4](weeks/week04.md) |
| 27 | P1 README | [Week 4](weeks/week04.md) |
| 28 | Week 4 复盘 | [Week 4](weeks/week04.md) |
| 29 | TF-IDF baseline | [Week 5](weeks/week05.md) |
| 30 | baseline 对比 BERT | [Week 5](weeks/week05.md) |
| 31 | BIO 标注 | [Week 5](weeks/week05.md) |
| 32 | Token classification | [Week 5](weeks/week05.md) |
| 33 | NER 错误分析 | [Week 5](weeks/week05.md) |
| 34 | BERT 总结 | [Week 5](weeks/week05.md) |
| 35 | Week 5 复盘 | [Week 5](weeks/week05.md) |
| 36 | GPT 与 Decoder-only | [Week 6](weeks/week06.md) |
| 37 | 自回归生成 | [Week 6](weeks/week06.md) |
| 38 | greedy / beam | [Week 6](weeks/week06.md) |
| 39 | top-k / top-p | [Week 6](weeks/week06.md) |
| 40 | temperature 与 repetition penalty | [Week 6](weeks/week06.md) |
| 41 | 生成错误分析 | [Week 6](weeks/week06.md) |
| 42 | Week 6 复盘 | [Week 6](weeks/week06.md) |
| 43 | Sentence Embedding | [Week 7](weeks/week07.md) |
| 44 | 语义相似度 | [Week 7](weeks/week07.md) |
| 45 | Top-k 检索 | [Week 7](weeks/week07.md) |
| 46 | chunk 与索引 | [Week 7](weeks/week07.md) |
| 47 | RAG 拼接上下文 | [Week 7](weeks/week07.md) |
| 48 | P2 RAG 项目 | [Week 7](weeks/week07.md) |
| 49 | Week 7 复盘 | [Week 7](weeks/week07.md) |
| 50 | P3 应用设计 | [Week 8](weeks/week08.md) |
| 51 | 分类模块接入 | [Week 8](weeks/week08.md) |
| 52 | RAG 模块接入 | [Week 8](weeks/week08.md) |
| 53 | Gradio UI | [Week 8](weeks/week08.md) |
| 54 | 结果展示和错误分析 | [Week 8](weeks/week08.md) |
| 55 | 项目 README 和讲解稿 | [Week 8](weeks/week08.md) |
| 56 | 总复盘 | [Week 8](weeks/week08.md) |

## 最终产出

1. `projects/p1-text-classification`：中文文本分类。
2. `projects/p2-rag-qa`：RAG 文档问答。
3. `projects/p3-llm-app`：整合分类和 RAG 的 LLM Demo。
4. `notes/day01.md` 到 `notes/day56.md`。
5. 每周总结：`notes/week01_summary.md` 到 `notes/week08_summary.md`。

