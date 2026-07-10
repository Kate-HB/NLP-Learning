# Day 01：环境配置与 Hugging Face 入门

## 今日进度

- 已学习：[Hugging Face LLM Course Chapter 1/4](https://huggingface.co/learn/llm-course/zh-CN/chapter1/4)
- 已完成环境配置、Pipeline 实验和 Transformer 架构入门

## 完成事项

- 配置 D 盘 Conda `base` 环境
- 配置 VS Code Python 解释器
- 安装并验证 PyTorch CUDA 版
- 修复 Python 依赖冲突
- 跑通 Hugging Face `sentiment-analysis` Pipeline
- 学会在 VS Code 中使用 Jupyter Notebook
- 登录 Hugging Face Hub
- 了解模型下载、缓存与模型格式

## 环境信息

```text
Python: D:\Soft\Conda\python.exe
PyTorch: 2.11.0+cu128
CUDA available: True
```

检查当前 Notebook 环境：

```python
import sys
import torch
import transformers

print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available())
print(transformers.__version__)
```

## NLP 基础

自然语言处理（NLP）研究如何让计算机理解、分析和生成人类语言。模型不仅需要识别单词，还需要根据上下文理解含义。

常见 NLP 任务：

| 任务 | 作用 | Pipeline |
|---|---|---|
| 文本分类 | 情感分析、垃圾邮件识别 | `text-classification` |
| 零样本分类 | 根据候选标签直接分类 | `zero-shot-classification` |
| Token 分类 | 命名实体识别、词性标注 | `token-classification` |
| 文本生成 | 根据提示续写文本 | `text-generation` |
| 掩码填充 | 预测句子中缺失的词 | `fill-mask` |
| 问答 | 从上下文中提取答案 | `question-answering` |
| 摘要 | 压缩长文本 | `summarization` |
| 翻译 | 将文本转换为另一种语言 | `translation` |

## Pipeline

`pipeline()` 是 Transformers 的高级推理接口，自动连接预处理、模型推理和后处理：

```text
原始文本
→ Tokenizer 将文本转换为模型输入
→ 模型计算预测结果
→ 后处理将结果转换为可读格式
```

情感分析示例：

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

classifier("I love NLP")
```

运行结果：

```text
[{'label': 'POSITIVE', 'score': 0.9997692704200745}]
```

### Pipeline 的主要参数

- `task`：任务类型
- `model`：模型仓库名称
- 输入：待处理文本
- 输出：标签、分数、答案或生成文本

### 模型下载与缓存

首次加载模型时，Transformers 会从 Hugging Face Hub 下载模型。之后会读取本地缓存，不再重复下载。

Windows 默认缓存位置通常为：

```text
C:\Users\27729\.cache\huggingface\hub
```

建议明确指定模型，避免默认模型发生变化：

```python
generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)
```

## 掩码填充

`fill-mask` 输入必须包含模型对应的掩码符号：

```python
from transformers import AutoTokenizer, pipeline

model_id = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(tokenizer.mask_token)
print(tokenizer.mask_token_id)

unmasker = pipeline("fill-mask", model=model_id)
unmasker(f"Paris is the {tokenizer.mask_token} of France.")
```

若 `tokenizer.mask_token` 为 `None`，该模型不能直接用于 `fill-mask`。

## 预训练与微调

### 预训练

模型从随机权重开始，在大规模原始文本上通过自监督学习掌握通用语言规律。

自监督表示训练目标可以由原始文本自动构造，不需要人工逐条标注。

### 微调

在预训练模型基础上，使用较小的人工标注数据集继续训练，使模型适应分类、问答、NER 等具体任务。

微调的优势：

- 所需数据更少
- 训练时间更短
- 计算成本更低
- 能利用预训练获得的通用语言知识

## 模型格式与框架

Hugging Face Hub 中并非所有模型都属于 Transformers。

| 模型格式 | 常用运行方式 |
|---|---|
| Transformers | `pipeline()`、`AutoModel` |
| Flair | `SequenceTagger.load()` |
| GGUF | llama.cpp、Ollama、`llama-cpp-python` |

Transformers 模型通常包含带 `model_type` 的 `config.json`。若出现以下错误：

```text
Should have a model_type key in its config.json
```

通常说明模型格式与 Transformers 不兼容。

## Jupyter 使用

启动浏览器版 JupyterLab：

```powershell
conda activate base
jupyter lab
```

VS Code 中可直接打开 `.ipynb`。右上角内核应选择：

```text
D:\Soft\Conda\python.exe
```

常用操作：

- `Shift + Enter`：运行当前单元格
- Restart Kernel：重启内核
- Clear Outputs：清除旧输出

安装或切换 Python 包版本后必须重启内核，否则内存中可能仍保留旧版本。

## Hugging Face 登录

在 VS Code 终端执行：

```powershell
hf auth login
hf auth whoami
```

未登录仍可访问公开模型，但请求速率更低。Token 不能写入代码或提交到 Git。

## 常见问题记录

### `No mask_token ([MASK]) found`

原因：`fill-mask` 输入没有包含对应 Tokenizer 的掩码符号。

解决：读取 `tokenizer.mask_token`，并将其放入输入文本。

### `Unknown task question-answering`

原因：当前 Transformers 新版本未注册该 Pipeline，而课程部分代码仍基于 4.x。

若要直接运行课程示例，可固定：

```powershell
pip install "transformers==4.57.3"
```

安装后需要重启 Notebook 内核。

### Windows 符号链接警告

这是 Hugging Face 缓存优化警告，不影响模型运行。开启 Windows 开发者模式后可减少缓存中的重复文件。

## Transformer 学习内容

Transformer 架构、Encoder、Decoder、注意力机制、Cross-Attention、因果掩码、Teacher Forcing 和并行训练已单独整理：

[Transformer 基础](../04-Transformer/transformer-basics.md)

## 今日产出

- `notebooks/day01.ipynb`
- `notes/day01.md`
- `04-Transformer/transformer-basics.md`
- 可运行的情感分析 Pipeline

## 明日任务

学习 Tokenizer：

- 文本如何切分为 Token
- Token 如何转换为 ID
- 特殊 Token
- Padding 与 Truncation
- Attention Mask

---

## 今日知识总结

### NLP 与任务类型

自然语言处理（NLP）的目标是让计算机能够处理、理解和生成人类语言。理解语言不能只分析孤立的单词，还需要结合单词在句子中的位置和上下文关系。NLP 任务可以按输出形式分为几类：文本分类对整段文本给出标签，例如情感分析；Token 分类对句子中的每个词进行判断，例如词性标注和命名实体识别；文本生成根据已有提示预测后续内容；掩码填充根据左右上下文恢复缺失词；抽取式问答从给定上下文中定位答案；翻译和摘要则根据一段输入生成新的目标序列。

### Hugging Face Pipeline

Hugging Face 的 `pipeline()` 是封装模型推理流程的高级接口。输入文本进入 Pipeline 后，Tokenizer 先把文本切分成 Token 并转换为数字 ID，模型再根据这些数字进行计算，最后由后处理步骤把模型输出转换为标签、置信度、答案或生成文本。Pipeline 适合快速体验模型和验证任务，但它隐藏了 Tokenizer、模型加载和后处理细节。后续深入学习时，需要分别使用 `AutoTokenizer` 和 `AutoModel`，观察文本如何变成模型输入，以及模型原始输出如何被解释。

Pipeline 的任务类型和模型能力必须匹配。例如，`fill-mask` 需要带有掩码语言建模能力的模型，输入中还必须包含该 Tokenizer 定义的 `mask_token`。GPT 类模型通常没有掩码符号，因此不能直接用于 `fill-mask`。Hugging Face Hub 也不只存放 Transformers 模型，Flair 模型需要 Flair 框架，GGUF 模型通常需要 llama.cpp 或 Ollama。模型位于 Hub 上，并不代表它一定能传给 Transformers 的 `pipeline()`。

### 模型下载与缓存

第一次加载指定模型时，Transformers 会从 Hugging Face Hub 下载配置、Tokenizer 和模型权重，并保存到本地缓存。以后再次使用相同模型时，程序会优先读取缓存，而不是重复下载。模型仓库可能包含几百 MB 到几十 GB 的文件，因此使用前需要检查模型格式、参数规模和权重文件大小。登录 Hugging Face 后可以获得更高的请求限制，但登录只影响 Hub 访问和下载，不会提高模型在本机上的推理速度。

### 预训练、自监督学习与微调

Transformer 语言模型通常先进行预训练。预训练从随机参数开始，使用大规模文本学习语言中的统计规律。训练目标可以直接从原始文本自动构造，例如遮住一个词让模型恢复它，或者让模型根据前文预测下一个词，因此这种方法称为自监督学习。自监督学习不需要人工为每条文本添加任务标签，但训练规模大、时间长、计算成本高。

预训练模型获得的是通用语言知识，还不能直接在所有具体任务上达到最佳效果。微调是在预训练参数基础上，使用规模较小的任务数据继续训练，使模型适应情感分析、问答或命名实体识别等任务。因为模型已经掌握了大量语言规律，微调通常比从零训练需要更少的数据、更短的时间和更低的计算成本。这种把已有知识迁移到新任务的过程属于迁移学习。

### Transformer 的三类架构

Transformer 可以按 Encoder 和 Decoder 的使用方式分为三类。Encoder-only 模型以 BERT 为代表，它读取完整输入，并为每个 Token 构造结合上下文的表示，适合文本分类、命名实体识别和掩码填充等理解任务。Decoder-only 模型以 GPT 为代表，它根据已有前文逐 Token 预测后续内容，适合文本生成和对话。Encoder-Decoder 模型以 T5、BART 为代表，Encoder 负责理解完整源序列，Decoder 根据 Encoder 输出生成目标序列，适合翻译、摘要等输入和输出都为序列的任务。

### 注意力机制

注意力机制用于判断当前 Token 应该关注序列中的哪些 Token，以及各自应占多大权重。它使每个 Token 的表示能够融合其他位置的信息，从而理解指代、语义依赖和上下文关系。Encoder 中的自注意力通常可以读取完整输入序列，因此处理某个词时既能参考前文，也能参考后文。

Encoder-Decoder 模型的 Decoder 中存在两种注意力。Masked Self-Attention 处理目标序列，只允许当前位置使用已经出现的目标前文；Cross-Attention 则让 Decoder 读取 Encoder 对完整源句生成的表示。以翻译为例，预测一个目标词时，Decoder 一方面参考已经生成的目标词，另一方面可以查看完整源句，因此能够处理两种语言语序不同，或需要依赖源句后半部分信息的情况。

### 目标右移与因果掩码

训练 Decoder 时，目标序列需要右移一位。假设正确目标是“我 喜欢 NLP”，解码器输入为“`<BOS>` 我 喜欢”，监督目标仍为“我 喜欢 NLP”。这样，第一个位置根据起始符预测“我”，第二个位置根据“`<BOS>` 我”预测“喜欢”，第三个位置根据“`<BOS>` 我 喜欢”预测“NLP”。右移的作用是防止模型在预测当前位置时直接看到当前位置的正确答案，并建立“使用前文预测下一个词”的输入输出对齐关系。

因果掩码负责遮住当前位置之后的未来词。仍以预测“喜欢”为例，“喜欢”是当前正确答案，“NLP”是未来词，“我”是过去词。目标右移防止模型直接读取当前答案“喜欢”，因果掩码防止模型读取未来词“NLP”。标准因果掩码通常允许一个输入位置关注自身，因此只有因果掩码而不右移目标序列，当前答案仍可能泄漏。目标右移和因果掩码解决的是相关但不同的问题，两者需要配合。

### Teacher Forcing 与并行训练

训练时，数据集已经提供完整的正确目标句，因此每个预测位置都可以使用真实目标前文，而不必使用模型在前一个位置的预测结果。这种方法称为 Teacher Forcing。即使第一个位置预测错误，第二个位置仍然使用真实的“我”去预测“喜欢”；即使第二个位置预测错误，第三个位置仍然使用真实的“我 喜欢”去预测“NLP”。各位置的预测分别与对应正确答案比较，得到损失后共同更新模型参数。

完整目标句在训练开始前已经已知，所以所有位置需要的真实前文都能提前构造。Transformer 可以把这些位置组成矩阵，一次交给 GPU 并行计算。因果掩码只控制每个位置能够读取的信息范围，不要求计算必须按词依次执行。推理时则没有标准答案，模型必须先生成第一个词，再把实际生成结果作为下一步输入，因此自回归推理通常只能逐 Token 进行，无法像训练一样在序列位置上完全并行。
