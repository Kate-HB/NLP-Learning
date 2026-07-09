# NLP 学习路线：56 Days Roadmap

本仓库用于系统学习 NLP。主线参考 Hugging Face，理论补充参考 NLP-LOVE/ML-NLP。

目标：8 周内完成 NLP 基础、Transformer、BERT、GPT、现代 LLM、RAG 和项目实战，最终产出可运行代码、学习笔记、项目 README、实验结果和面试讲解材料。

---

## 1. 我要学习什么

### 1.1 基础能力

- Python 基础
- NumPy 基础
- PyTorch 基础
- Git 与 GitHub
- Markdown 文档写作
- Conda 环境管理
- CUDA 与 GPU 验证

### 1.2 NLP 基础

- 文本预处理
- Tokenization
- Bag of Words
- TF-IDF
- N-Gram
- Word2Vec
- GloVe
- 文本分类
- 序列标注
- 问答任务
- 文本生成
- 语义检索

### 1.3 深度学习 NLP

- Embedding
- RNN 基础了解
- Attention
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder
- Transformer Decoder
- LayerNorm
- Residual Connection

### 1.4 Hugging Face 生态

- `pipeline`
- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `AutoModelForTokenClassification`
- `AutoModelForQuestionAnswering`
- `Trainer`
- `TrainingArguments`
- `datasets`
- `evaluate`
- `tokenizers`
- `sentence-transformers`
- `peft`

### 1.5 经典模型

- BERT
- RoBERTa
- MacBERT
- GPT
- GPT-2
- Sentence-BERT
- BGE Embedding

### 1.6 现代 LLM 入门

- LoRA
- PEFT
- QLoRA 基础概念
- Instruction Tuning
- RLHF 思想
- RAG 基础流程
- 文档切分
- 向量检索
- Prompt 拼接
- 引用来源
- 幻觉分析

---

## 2. 学习资料

### 2.1 主线资料

| 资料 | 地址 | 用途 |
|---|---|---|
| Hugging Face LLM Course | https://huggingface.co/learn/llm-course/chapter1/1 | 主线课程，优先学习 |
| Transformers GitHub | https://github.com/huggingface/transformers | 模型加载、推理、微调 |
| Transformers Docs | https://huggingface.co/docs/transformers/index | API 查询 |
| Datasets GitHub | https://github.com/huggingface/datasets | 数据集加载和处理 |
| Datasets Docs | https://huggingface.co/docs/datasets/index | `datasets` 用法 |
| Tokenizers GitHub | https://github.com/huggingface/tokenizers | 分词器原理和实现 |
| Evaluate Docs | https://huggingface.co/docs/evaluate/index | 指标计算 |
| PEFT Docs | https://huggingface.co/docs/peft/index | LoRA 与参数高效微调 |
| Sentence Transformers | https://github.com/UKPLab/sentence-transformers | 语义向量和检索 |

### 2.2 理论补充资料

| 资料 | 地址 | 用途 |
|---|---|---|
| NLP-LOVE/ML-NLP | https://github.com/NLP-LOVE/ML-NLP | 机器学习、深度学习、NLP 理论和面试复习 |
| PyTorch Tutorials | https://pytorch.org/tutorials/ | PyTorch 基础 |
| Dive into Deep Learning | https://d2l.ai/ | 深度学习基础 |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | Transformer 图解 |
| The Illustrated BERT | https://jalammar.github.io/illustrated-bert/ | BERT 图解 |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | Transformer 原论文 |
| BERT Paper | https://arxiv.org/abs/1810.04805 | BERT 原论文 |
| GPT-2 Paper | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | GPT 生成模型论文 |


### 2.4 每周详细计划

每周计划已拆分到 `resources/` 目录，按顺序学习即可。

| 周次 | 主题 | 文档 | 核心产出 |
|---|---|---|---|
| Week 1 | 基础准备 | [resources/week01.md](resources/week01.md) | 环境记录、PyTorch demo、pipeline 结果 |
| Week 2 | NLP 基础 | [resources/week02.md](resources/week02.md) | TF-IDF baseline、错误分析 |
| Week 3 | Hugging Face 与 Tokenizer | [resources/week03.md](resources/week03.md) | tokenizer demo、手动推理 |
| Week 4 | Transformer | [resources/week04.md](resources/week04.md) | Self-Attention、Mini Transformer |
| Week 5 | BERT | [resources/week05.md](resources/week05.md) | BERT 微调、情感分析项目 |
| Week 6 | GPT | [resources/week06.md](resources/week06.md) | 文本生成 demo、解码策略对比 |
| Week 7 | 现代 LLM | [resources/week07.md](resources/week07.md) | LoRA 笔记、语义搜索 demo |
| Week 8 | 项目实战 | [resources/week08.md](resources/week08.md) | 项目 README、面试讲解稿 |
### 2.3 NLP-LOVE/ML-NLP 使用方式

- Week 1：看机器学习基础、深度学习基础
- Week 2：看 NLP 基础、文本表示、词向量
- Week 3~4：看 Attention、Transformer
- Week 5：看 BERT 相关内容
- Week 6：看 GPT、语言模型相关内容
- Week 7：看大模型、微调、RAG 相关补充
- Week 8：看面试题和项目总结

---

## 3. 最终要产出什么

### 3.1 每日产出

- 1 篇学习笔记
- 1 个可运行代码片段
- 1 次问题记录
- 1 次 Git 提交

### 3.2 每周产出

| 周 | 产出 |
|---|---|
| Week 1 | 环境配置记录、PyTorch 基础代码、Hugging Face pipeline demo |
| Week 2 | TF-IDF 文本分类 baseline、NLP 基础笔记 |
| Week 3 | Tokenizer 实验、Hugging Face 推理脚本 |
| Week 4 | Mini Transformer 关键模块 |
| Week 5 | BERT 文本分类或 NER 微调项目 |
| Week 6 | GPT 文本生成 demo |
| Week 7 | LoRA/PEFT/RAG 基础笔记和小实验 |
| Week 8 | 完整项目 README、结果表、错误分析、面试讲解稿 |

### 3.3 最终项目产出

至少完成 5 个项目：

1. 情感分析
2. 新闻分类
3. 命名实体识别
4. 抽取式问答
5. RAG 文档问答

每个项目必须包含：

- 项目背景
- 数据集说明
- 模型选择
- 训练方法
- 运行命令
- 评估指标
- 实验结果
- 错误分析
- 后续改进

### 3.4 最终仓库产出

- `README.md`：总路线和项目说明
- `notes/`：每日学习笔记
- `notebooks/`：实验 notebook
- `09-Projects/`：项目代码
- `Papers/`：论文笔记
- `Resources/`：学习资料整理
- `Interview/`：面试题和项目讲解稿

---

## 4. 项目结构

```text
NLP-Learning/
├── README.md
├── NLP-HuggingFace-Learning-Plan.md
├── 00-Environment/
├── 01-Python/
├── 02-PyTorch/
├── 03-NLP-Basic/
├── 04-Transformer/
├── 05-BERT/
├── 06-GPT/
├── 07-LLM/
├── 08-RAG/
├── 09-Projects/
├── Papers/
├── Resources/
│   ├── README.md
│   ├── week01.md
│   ├── week02.md
│   ├── week03.md
│   ├── week04.md
│   ├── week05.md
│   ├── week06.md
│   ├── week07.md
│   └── week08.md
├── Interview/
├── notes/
├── notebooks/
├── data/
├── outputs/
└── projects/
```

### 4.1 目录说明

| 目录 | 内容 |
|---|---|
| `00-Environment/` | Conda、CUDA、PyTorch、依赖安装记录 |
| `01-Python/` | Python、NumPy、基础代码 |
| `02-PyTorch/` | Tensor、Autograd、Dataset、DataLoader、训练循环 |
| `03-NLP-Basic/` | 文本预处理、TF-IDF、N-Gram、Word2Vec |
| `04-Transformer/` | Attention、Self-Attention、Mini Transformer |
| `05-BERT/` | BERT 原理、分类、NER、QA 微调 |
| `06-GPT/` | GPT 原理、文本生成、解码策略 |
| `07-LLM/` | LoRA、PEFT、Instruction Tuning、RLHF |
| `08-RAG/` | 文档切分、向量检索、RAG 问答 |
| `09-Projects/` | 最终展示项目 |
| `Papers/` | 论文阅读笔记 |
| `Resources/` | 资料链接和学习路线 |
| `Interview/` | 保研面试题、项目讲解稿 |
| `notes/` | 每日学习笔记 |
| `notebooks/` | notebook 实验 |
| `data/` | 本地数据集，不提交大文件 |
| `outputs/` | 实验结果、图表、模型输出 |

---

## 5. 8 周学习路线

## Week 1：基础准备

### 学习内容

- Python
- NumPy
- PyTorch
- Git
- Markdown
- Conda 环境
- CUDA 验证
- Hugging Face pipeline

### 学习目标

- 能管理 Conda 环境
- 能验证 GPU 是否可用
- 能跑通 PyTorch
- 能跑通 Hugging Face `pipeline`
- 能写第一篇学习笔记

### 学习资料

- PyTorch Tutorials
- Hugging Face LLM Course Chapter 1
- NLP-LOVE/ML-NLP 机器学习基础部分

### 任务

- 安装 PyTorch CUDA 版
- 运行 `torch.cuda.is_available()`
- 运行 Hugging Face 情感分析 pipeline
- 创建项目目录
- 写 `notes/day01.md`

### 产出

- `00-Environment/env.md`
- `02-PyTorch/tensor_demo.py`
- `notes/day01.md`
- pipeline 运行截图或输出记录

---

## Week 2：NLP 基础

### 学习内容

- 文本清洗
- 分词
- Tokenization
- Bag of Words
- TF-IDF
- N-Gram
- Word2Vec
- GloVe
- 余弦相似度

### 学习目标

- 理解传统文本表示方法
- 能用 TF-IDF 做文本分类
- 能解释词向量的作用
- 能完成一个传统 baseline

### 学习资料

- Hugging Face LLM Course Chapter 1
- NLP-LOVE/ML-NLP NLP 基础部分
- scikit-learn 文本特征提取文档

### 任务

- 实现文本清洗函数
- 使用 TF-IDF 表示文本
- 使用 Logistic Regression 做分类
- 统计 Accuracy、Precision、Recall、F1
- 记录 10 条错误样本

### 产出

- `03-NLP-Basic/text_preprocess.py`
- `03-NLP-Basic/tfidf_baseline.py`
- `notes/week02_nlp_basic.md`
- `outputs/tfidf_result.md`

---

## Week 3：Hugging Face 与 Tokenizer

### 学习内容

- `pipeline`
- `AutoTokenizer`
- `AutoModel`
- `input_ids`
- `attention_mask`
- padding
- truncation
- decode
- batch encoding

### 学习目标

- 理解 tokenizer 的作用
- 能查看模型输入
- 能手动完成 tokenizer + model 推理
- 能解释 `input_ids` 和 `attention_mask`

### 学习资料

- Hugging Face LLM Course Chapter 2
- Transformers Docs Quicktour
- Tokenizers GitHub

### 任务

- 对英文和中文文本做 tokenizer
- 查看 token、id、attention mask
- 使用 `decode` 还原文本
- 对比不同 tokenizer 的输出
- 手写一个不用 `pipeline` 的推理流程

### 产出

- `03-NLP-Basic/tokenizer_demo.py`
- `notebooks/week03_tokenizer.ipynb`
- `notes/week03_tokenizer.md`

---

## Week 4：Transformer

### 学习内容

- Attention
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Encoder
- Decoder
- LayerNorm
- Residual Connection

### 学习目标

- 理解 Transformer 的整体结构
- 能解释 Q、K、V
- 能写 Self-Attention
- 能实现 Mini Transformer Encoder

### 学习资料

- Attention Is All You Need
- The Illustrated Transformer
- NLP-LOVE/ML-NLP Transformer 部分

### 任务

- 推导 Attention 计算过程
- 用 PyTorch 写 Self-Attention
- 写 Multi-Head Attention
- 写 Positional Encoding
- 组装 Mini Transformer

### 产出

- `04-Transformer/self_attention.py`
- `04-Transformer/mini_transformer.py`
- `notes/week04_transformer.md`
- Transformer 结构图或文字说明

---

## Week 5：BERT

### 学习内容

- BERT 架构
- MLM
- NSP
- `[CLS]`
- Fine-tuning
- 文本分类
- NER
- QA
- Trainer

### 学习目标

- 理解 BERT 适合做理解类任务的原因
- 能用 Hugging Face 微调 BERT
- 能完成文本分类或 NER 项目
- 能写项目 README

### 学习资料

- BERT Paper
- The Illustrated BERT
- Hugging Face LLM Course Chapter 3
- NLP-LOVE/ML-NLP BERT 部分

### 任务

- 加载 `bert-base-chinese`
- 准备分类数据集
- 使用 `Trainer` 微调
- 计算 Accuracy 和 F1
- 保存模型
- 写预测脚本

### 产出

- `05-BERT/train_classifier.py`
- `05-BERT/predict.py`
- `09-Projects/sentiment-analysis/README.md`
- `outputs/bert_classification_result.md`

---

## Week 6：GPT

### 学习内容

- Decoder-only
- Causal Mask
- 自回归生成
- greedy search
- beam search
- top-k
- top-p
- temperature

### 学习目标

- 理解 GPT 的生成流程
- 能运行文本生成模型
- 能比较不同解码策略
- 能解释生成重复和幻觉问题

### 学习资料

- GPT-2 Paper
- Transformers Generation Docs
- NLP-LOVE/ML-NLP 语言模型部分

### 任务

- 跑通 text-generation pipeline
- 对比 greedy、beam、top-k、top-p
- 调整 temperature
- 记录不同输出
- 分析生成错误

### 产出

- `06-GPT/generation_demo.py`
- `notes/week06_gpt.md`
- `outputs/generation_comparison.md`

---

## Week 7：现代 LLM

### 学习内容

- LoRA
- PEFT
- QLoRA 基础
- Instruction Tuning
- RLHF 思想
- RAG 基础
- Sentence Embedding
- 向量检索

### 学习目标

- 理解大模型微调的基本方向
- 理解参数高效微调
- 理解 RAG 为什么需要检索
- 能做语义搜索 demo

### 学习资料

- PEFT Docs
- Sentence Transformers
- Hugging Face Model Hub
- NLP-LOVE/ML-NLP LLM/RAG 相关内容

### 任务

- 使用 sentence-transformers 生成句向量
- 计算余弦相似度
- 实现 Top-k 检索
- 阅读 LoRA 基本原理
- 写 RAG 流程笔记

### 产出

- `07-LLM/lora_notes.md`
- `08-RAG/semantic_search.py`
- `notes/week07_llm_rag.md`
- `outputs/semantic_search_result.md`

---

## Week 8：项目实战

### 学习内容

- 新闻分类
- 情感分析
- 命名实体识别
- RAG 文档问答
- 论文智能分析系统
- 项目 README
- 错误分析
- 面试讲解

### 学习目标

- 整理至少 3 个完整项目
- 每个项目能复现
- 每个项目有结果表
- 每个项目有错误分析
- 能讲清楚项目方案

### 学习资料

- Hugging Face Examples
- NLP-LOVE/ML-NLP 面试部分
- GitHub 优秀项目 README

### 任务

- 整理项目代码
- 写每个项目 README
- 写运行命令
- 写结果表
- 写错误分析
- 写保研面试讲解稿

### 产出

- `09-Projects/news-classification/`
- `09-Projects/sentiment-analysis/`
- `09-Projects/named-entity-recognition/`
- `09-Projects/rag-demo/`
- `Interview/project_explanation.md`
- `Interview/nlp_questions.md`

---

## 6. 项目清单

| 项目 | 学习重点 | 核心库 | 产出 |
|---|---|---|---|
| 情感分析 | 文本分类、BERT 微调 | transformers、datasets | 分类模型、预测脚本、README |
| 新闻分类 | TF-IDF baseline、BERT 对比 | sklearn、transformers | baseline、模型对比、错误分析 |
| 命名实体识别 | BIO 标注、token classification | transformers、seqeval | NER 模型、实体抽取脚本 |
| 抽取式问答 | start/end logits、answer span | transformers、datasets | QA demo、EM/F1 结果 |
| 语义搜索 | embedding、余弦相似度 | sentence-transformers | Top-k 检索脚本 |
| RAG 文档问答 | 检索增强生成 | sentence-transformers、transformers | 文档问答 demo、引用来源 |

---

## 7. 每个项目 README 模板

```text
# 项目名称

## 1. 项目背景
说明任务是什么，为什么做。

## 2. 数据集
说明来源、字段、样本量、标签分布。

## 3. 方法
说明 baseline、模型、tokenizer、训练方式。

## 4. 实验设置
说明 epoch、batch size、learning rate、max length。

## 5. 结果
给出 Accuracy、Precision、Recall、F1、EM 等指标。

## 6. 错误分析
列出典型错误样本和原因。

## 7. 运行方式
给出安装、训练、预测命令。

## 8. 后续改进
列出可改进点。
```

---

## 8. 学习笔记模板

```text
# Day XX

## 今日目标

## 学习内容

## 关键概念

## 代码实验

## 运行结果

## 遇到的问题

## 解决方法

## 明日任务
```

---

## 9. 实验记录模板

```text
# 实验名称

## 实验目标

## 数据集

## 模型

## 参数

- learning_rate:
- batch_size:
- epoch:
- max_length:

## 指标

| 指标 | 结果 |
|---|---:|
| Accuracy | |
| Precision | |
| Recall | |
| F1 | |

## 错误样本

| 文本 | 真实标签 | 预测标签 | 原因 |
|---|---|---|---|

## 结论
```

---

## 10. 当前进度

- [x] Day 01：配置 D 盘 Conda 环境
- [x] Day 01：安装 PyTorch CUDA 版
- [x] Day 01：修复依赖冲突
- [x] Day 01：跑通 Hugging Face sentiment-analysis pipeline
- [x] Day 01：创建项目目录
- [ ] Day 02：学习 Tokenizer
- [ ] Week 1：完成 PyTorch 基础
- [ ] Week 2：完成 TF-IDF baseline
- [ ] Week 3：完成 tokenizer 实验
- [ ] Week 4：完成 Mini Transformer
- [ ] Week 5：完成 BERT 微调
- [ ] Week 6：完成 GPT 生成 demo
- [ ] Week 7：完成语义搜索 demo
- [ ] Week 8：完成项目整理

---

## 11. Day 01 记录

环境：

```text
Python：D:\Soft\Conda
PyTorch：2.11.0+cu128
CUDA：12.8
Transformers：可用
Datasets：可用
```

验证命令：

```powershell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love NLP'))"
```

输出：

```text
[{'label': 'POSITIVE', 'score': 0.9997692704200745}]
```

学到的概念：

- `pipeline` 是 Hugging Face 的高级封装
- 它会自动完成 tokenizer、模型加载、推理和结果格式化
- 不指定模型时，会使用默认模型

---

## 12. Day 02 任务

目标：学习 tokenizer。

任务：

1. 使用 `AutoTokenizer` 加载 tokenizer。
2. 输入英文句子：`I love NLP`。
3. 输入中文句子：`我正在学习自然语言处理`。
4. 查看 `input_ids`。
5. 查看 `attention_mask`。
6. 使用 `decode` 还原文本。
7. 记录到 `notes/day02.md`。

示例代码：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

text = "I love NLP"
encoded = tokenizer(text)

print(encoded)
print(tokenizer.decode(encoded["input_ids"]))
```

---

## 13. 结课检查清单

- [ ] 能解释 NLP 常见任务
- [ ] 能解释 Tokenization
- [ ] 能解释 TF-IDF
- [ ] 能解释 Word2Vec
- [ ] 能解释 Attention
- [ ] 能解释 Transformer
- [ ] 能解释 BERT
- [ ] 能解释 GPT
- [ ] 能使用 Hugging Face pipeline
- [ ] 能使用 AutoTokenizer
- [ ] 能使用 AutoModel
- [ ] 能使用 datasets
- [ ] 能使用 Trainer 微调
- [ ] 能完成文本分类项目
- [ ] 能完成 NER 项目
- [ ] 能完成 RAG 项目
- [ ] 能写项目 README
- [ ] 能做错误分析
- [ ] 能讲清楚项目


