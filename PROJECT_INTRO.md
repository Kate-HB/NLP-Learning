# NLP 系统学习与项目实战

## 摘要

本项目是一份面向 NLP 初学者的 56 天系统学习计划，覆盖从 Hugging Face 工具链入门到 LLM 应用整合的完整路径。学习路线以 Transformer 为核心，依次深入 Tokenizer、PyTorch 训练、BERT 微调、GPT 生成、Embedding 检索与 RAG 架构，每阶段配合可运行的代码实验和项目产出验证理解。最终交付三个完整项目：中文文本分类、RAG 文档问答和 LLM 应用整合 Demo。

## 完整介绍

### 背景与定位

自然语言处理（NLP）是当前人工智能领域发展最快的方向之一。以 Transformer 为基础架构的大语言模型（LLM）已经深刻改变了文本理解、生成和检索的方式。然而，从零开始学习 NLP 面临两个主要挑战：一是概念体系庞杂，Tokenizer、Attention、预训练、微调等术语常常交织在一起；二是工具链更新迅速，Hugging Face、PyTorch 等生态的 API 变动频繁，新手容易在环境配置阶段就受挫。

本项目正是为解决这些问题而设计的。它不追求百科全书式的理论覆盖，而是以"理解核心概念、跑通关键代码、产出可运行项目"为主线，帮助学习者在 8 周内建立从文本输入到应用交付的完整认知。

### 学习路线与阶段划分

整个学习计划分为 8 周，每周聚焦一个主题，每天约 5 小时（3 小时概念理解、1.5 小时代码实验、0.5 小时笔记整理）。

**Week 1：Hugging Face 入门与 Tokenizer。** 从环境配置和 Pipeline 体验入手，拆解 `pipeline()` 内部的三个步骤（Tokenizer 预处理、模型计算、后处理），理解文本如何被切分为 Token 并转换为模型可处理的数字 ID。重点掌握三种分词粒度（基于单词、基于字符、基于子词）的取舍，以及 Padding、Truncation 和 Attention Mask 在批处理中的作用。通过手动构造 tensor 并观察有无 attention mask 时 logits 的变化，建立对"模型输入格式"的直觉。

**Week 2：PyTorch 最小训练基础。** 不追求系统学完 PyTorch，只聚焦微调必须掌握的概念：Tensor 的 shape/dtype/device、Dataset 与 DataLoader 的关系、`nn.Module` 的 forward 机制、autograd 的梯度计算、以及 `zero_grad → backward → step` 的标准训练循环。用假数据跑通一个最小二分类训练，为后续 BERT 微调打下工程基础。

**Week 3：深度学习概念与 Transformer。** 补足理解 Transformer 所需的数学和结构知识。从 NumPy 矩阵操作开始，逐步学习 softmax 与交叉熵损失、Embedding 查表机制，然后进入 Attention 核心：手算 Q、K、V 的注意力分数，理解 Multi-Head Attention 为何有效，梳理 Transformer Encoder 的完整数据流（MHA → FFN → LayerNorm → Residual）。

**Week 4：BERT 微调与中文文本分类项目（P1）。** 完成第一个完整项目。学习 BERT 的掩码语言模型和 `[CLS]` 表征机制，使用 `bert-base-chinese` 和 Hugging Face Trainer 在中文分类数据集上微调，输出评估指标和错误分析，交付可独立运行的推理脚本。

**Week 5：传统 NLP、BERT 与序列标注。** 引入 TF-IDF 作为 baseline，对比传统方法与 BERT 在文本分类上的表现差异。扩展到序列标注任务，学习 BIO 标注规则，使用 BERT 做命名实体识别（NER），理解 token classification 和 sequence classification 的区别。

**Week 6：GPT 与生成。** 聚焦 Decoder-only 架构和自回归生成。系统对比 greedy、beam search、top-k、top-p 等解码策略，观察 temperature 和 repetition penalty 对生成文本的影响。记录并分析跑题、重复、幻觉等生成模型的典型局限。

**Week 7：Embedding、语义检索与 RAG 项目（P2）。** 学习 Sentence Embedding 和语义相似度计算，使用 FAISS 构建向量索引，实现文档切分、Top-k 检索和上下文拼接。完成 RAG 文档问答项目，回答必须带引用来源，理解检索如何减少模型胡编。

**Week 8：LLM 应用整合项目（P3）。** 将前七周的能力整合为一个 Gradio 应用，包含文本分类和 RAG 问答两个功能模块。交付完整的项目 README，覆盖安装、运行、数据、模型和限制说明。

### 当前进度

截至 Week 1 Day 04，已完成环境搭建（Conda + PyTorch CUDA 12.8 + Transformers 4.57.3）、Pipeline 多任务体验、Transformer 三类架构与注意力机制学习、Tokenizer 三种分词粒度与批处理全流程实验。已产出 4 篇每日笔记（含知识总结）、4 个实验 Notebook、2 篇主题文档和 1 篇周志。

## 涉及技术

### 框架与工具

| 技术 | 用途 |
|---|---|
| **PyTorch**（2.11.0, CUDA 12.8） | 深度学习框架，张量计算、自动微分、模型训练 |
| **Hugging Face Transformers**（4.57.3） | 预训练模型加载、Pipeline 推理、Trainer 微调 |
| **Hugging Face Datasets**（5.0.0） | 数据集加载与预处理 |
| **Hugging Face Hub** | 模型托管、版本管理与分发 |

### 模型架构

| 架构 | 代表模型 | 学习重点 |
|---|---|---|
| **Encoder-only** | BERT、DistilBERT、RoBERTa | 掩码语言模型、双向注意力、文本分类与 NER |
| **Decoder-only** | GPT、GPT-2 | 因果语言模型、自回归生成、解码策略 |
| **Encoder-Decoder** | T5、BART、Whisper | 序列到序列、摘要与翻译、多模态 |

### 核心技术概念

- **Tokenizer**：Word-based / Character-based / Subword-based（BPE、WordPiece、SentencePiece）分词算法，tokenize → convert_tokens_to_ids → decode 原子操作，`[CLS]`/`[SEP]`/`[UNK]`/`[PAD]` 等特殊 Token，padding 策略与 attention mask 机制
- **Attention**：Self-Attention 与 Cross-Attention，因果掩码（Causal Mask），Multi-Head Attention，Q/K/V 投影，KV 缓存优化
- **训练与微调**：自监督预训练（MLM / CLM），迁移学习，Teacher Forcing，Trainer API，学习率调度与评估指标
- **生成策略**：Greedy decoding、Beam search、Top-k / Top-p sampling、Temperature、Repetition penalty
- **Embedding 与检索**：Sentence Embedding、余弦相似度、FAISS 向量索引、Top-k 检索、RAG 架构
- **应用整合**：Gradio 界面、模型推理封装、错误分析与结果展示

### 工程实践

- **环境管理**：Conda 虚拟环境、VS Code + Jupyter Notebook、CUDA 与 GPU 加速
- **版本控制**：Git 分支管理，每日笔记与代码同步提交
- **文档规范**：每日知识总结、每周周志、主题文档分类整理、README 可复现说明
