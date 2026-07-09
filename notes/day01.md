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
