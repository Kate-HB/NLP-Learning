# Day 03：拆解 pipeline、模型与 Tokenizer

## 今日目标

- 理解 `pipeline()` 内部的预处理、模型计算和后处理。
- 理解 `checkpoint`、`AutoTokenizer`、`AutoModel`、`AutoModelForSequenceClassification` 的区别。
- 理解 hidden states、logits、softmax、label 的关系。
- 理解 tokenizer 输出中的 `input_ids`、`token_type_ids`、`attention_mask`。
- 理解 padding、truncation、`return_tensors="pt"`。

## 学习资源

- Hugging Face LLM Course Chapter 2.2：https://huggingface.co/learn/llm-course/zh-CN/chapter2/2
- Hugging Face LLM Course Chapter 2.3：https://huggingface.co/learn/llm-course/zh-CN/chapter2/3
- Hugging Face LLM Course Chapter 2.4：https://huggingface.co/learn/llm-course/zh-CN/chapter2/4
- Hugging Face LLM Course Chapter 2.5：https://huggingface.co/learn/llm-course/zh-CN/chapter2/5
- 代码：[day03-pipeline.ipynb](../notebooks/day03-pipeline.ipynb)
- 代码：[day03-createtransformer.ipynb](../notebooks/day03-createtransformer.ipynb)

## 学习内容

### Transformers 库

Transformers 库的目标是提供统一 API，让用户可以加载、训练、保存和使用不同 Transformer 模型。

它的核心特点：

- 易用：几行代码即可下载模型并推理。
- 灵活：模型本质上仍是 PyTorch `nn.Module` 或 TensorFlow/Keras 模型。
- 简单：模型结构尽量集中在对应模型文件里，便于阅读和修改。

今天重点不是训练模型，而是拆开 `pipeline()`，理解一次 NLP 推理从文本到结果的完整流程。

## pipeline 内部流程

图示：[full_nlp_pipeline.svg](../03-NLP-Basic/pictures/full_nlp_pipeline.svg)

`pipeline()` 主要包含三步：

```text
原始文本
-> tokenizer 预处理
-> model 计算
-> 后处理
-> 人能看懂的结果
```

以情感分析为例：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

这段代码内部大致等于：

```text
文本
-> input_ids / attention_mask
-> 模型输出 logits
-> softmax 转概率
-> 取最大概率类别
-> POSITIVE / NEGATIVE
```

## checkpoint

`checkpoint` 可以理解为“模型存档地址”。

例如：

```python
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
```

这个 checkpoint 通常包含：

```text
config.json          模型结构配置
model.safetensors    模型权重
tokenizer.json       tokenizer 配置
vocab.txt            词表
```

`checkpoint` 不是模型对象本身，它只是告诉 Transformers 去哪里加载模型文件。

## tokenizer

Transformer 模型不能直接处理原始文本。Tokenizer 负责把文本转换成模型能处理的数字。

Tokenizer 主要做三件事：

1. 把文本切成 token。
2. 把 token 映射成 token id。
3. 添加模型需要的特殊输入，例如 `[CLS]`、`[SEP]`、`attention_mask`。

示例：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

输出包含：

```text
input_ids：token 的数字 ID
token_type_ids：区分句子 A 和句子 B
attention_mask：标记哪些 token 需要被模型关注
```

## model

`AutoModel.from_pretrained(checkpoint)` 会加载模型主体。

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
```

它输出的是 hidden states，不是具体任务结果。

hidden states 的常见形状：

```text
(batch_size, sequence_length, hidden_size)
```

例如：

```text
torch.Size([2, 16, 768])
```

含义：

```text
2：一次输入 2 个句子
16：每个句子被 padding 到 16 个 token
768：每个 token 的上下文化向量维度
```

## 模型头 Head

图示：[transformer_and_head.svg](../04-Transformer/pictures/transformer_and_head.svg)

Transformer 主体先输出 hidden states，然后不同任务会接不同的 head。

流程：

```text
input_ids
-> Embedding 层
-> Transformer layers
-> hidden states
-> Head
-> logits
```

常见 head：

- 分类 head：文本分类。
- MLM head：预测 `[MASK]`。
- QA head：预测答案开始和结束位置。
- LM head：预测下一个 token。

所以：

```python
AutoModel
```

输出 hidden states。

```python
AutoModelForSequenceClassification
```

输出分类 logits。

## logits 与 softmax

模型输出的 logits 是原始分数，不是概率。

示例：

```text
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]])
```

每一行对应一个输入句子，每一列对应一个类别。

softmax 会把 logits 转成概率：

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

结果类似：

```text
第一句：NEGATIVE 0.0402，POSITIVE 0.9598
第二句：NEGATIVE 0.9995，POSITIVE 0.0005
```

最后再用：

```python
model.config.id2label
```

把类别 ID 映射成人能读懂的标签。

## return_tensors="pt"

不加 `return_tensors="pt"` 时，tokenizer 返回 Python list。

```python
tokenizer("How are you?")
```

加上后，返回 PyTorch tensor。

```python
tokenizer("How are you?", return_tensors="pt")
```

区别：

```text
list：方便查看
tensor：可以直接传给 PyTorch 模型
```

注意：加 `return_tensors="pt"` 后会多一层 batch 维度。

```text
不加：长度 6
加：shape = (1, 6)
```

## 句子对与 batch

这句代码是“句子对输入”，不是两个独立样本：

```python
tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
```

它会拼成：

```text
[CLS] How are you ? [SEP] I'm fine , thank you ! [SEP]
```

并用 `token_type_ids` 区分：

```text
0：第一句
1：第二句
```

如果要传多个独立句子，应该传 list：

```python
tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    return_tensors="pt",
)
```

这才是 batch。

## padding 与 attention_mask

PyTorch tensor 必须是矩形。多个句子长度不同，不能直接组成一个规则 tensor。

例如：

```text
第 1 句：6 个 token
第 2 句：10 个 token
```

使用：

```python
tokenizer(texts, padding=True, return_tensors="pt")
```

会把短句补齐到同样长度。

补齐部分的 `input_ids` 通常是 `0`，对应的 `attention_mask` 是 `0`。

```text
attention_mask = 1：真实 token
attention_mask = 0：padding token
```

所以 `attention_mask` 的作用是防止模型把 padding 当成真实文本。

## truncation

BERT 通常最多处理 512 个 token。输入太长时，需要截断。

```python
encoded_input = tokenizer(
    text,
    truncation=True,
    max_length=10,
)
```

如果文本超过 `max_length`，tokenizer 会保留前面的有效 token，并保留特殊 token，例如 `[CLS]` 和 `[SEP]`。

判断是否截断：

```python
len(encoded_input["input_ids"]) == max_length
```

## 今天解决的问题

- 明确了 `checkpoint` 和 `model` 的区别：checkpoint 是文件地址，model 是加载后的模型对象。
- 明确了 `AutoModel` 和 `AutoModelForSequenceClassification` 的区别：前者输出 hidden states，后者输出 logits。
- 明确了最后一个 Transformer layer 输出的是最终 hidden states。
- 明确了后处理的关键：把模型原始输出转成人能看懂的结果。
- 修正了“两个字符串参数就是多个句子 batch”的误解：两个字符串参数表示句子对，多个独立句子需要传 list。
- 明确了 padding 后 `attention_mask=0` 的位置是不该被模型关注的 padding。

## 今日知识总结

### pipeline 是封装版推理流程

`pipeline()` 把 tokenizer、model、后处理封装在一起。它看起来只接收文本并输出标签，但内部实际经过了文本转 token id、模型计算 logits、softmax 转概率、标签映射等步骤。学习拆开 pipeline 的意义，是知道每一步输入输出是什么。

### checkpoint 是模型文件地址

checkpoint 不是模型对象，而是一个已经保存好的模型包地址。`from_pretrained(checkpoint)` 会根据这个地址加载配置、权重、tokenizer 和词表。加载完成后，Python 里的 `model` 才是可以执行前向计算的模型对象。

### tokenizer 负责文本到数字

Tokenizer 把原始文本拆成 token，再映射成 `input_ids`。同时它会生成 `attention_mask`、`token_type_ids` 等辅助输入。模型不能直接处理字符串，只能处理这些数字张量。

### AutoModel 输出 hidden states

`AutoModel` 只加载 Transformer 主体。它输出的是每个 token 的上下文化向量，也就是 hidden states。hidden states 通常是三维张量：batch size、sequence length、hidden size。它不是最终分类结果。

### Head 决定具体任务输出

不同任务需要不同 head。分类任务使用 sequence classification head，输出每个类别的 logits。问答任务使用 QA head，输出答案起止位置。语言模型使用 LM head，输出下一个 token 的分数。

### logits 需要后处理

logits 是模型输出的原始分数，不是概率。分类任务中，需要用 softmax 把 logits 转成概率，再取最大概率对应的类别 ID，最后用 `id2label` 映射成 `POSITIVE`、`NEGATIVE` 这类标签。

### padding 和 truncation 解决长度问题

多个句子组成 batch 时，长度必须一致，所以短句要 padding。padding token 不属于真实文本，因此 `attention_mask` 要标为 0。长文本超过模型最大长度时，需要 truncation，否则模型无法处理。

### 句子对不是 batch

`tokenizer("A", "B")` 表示一个样本里的句子 A 和句子 B，常用于句子对任务。`tokenizer(["A", "B"])` 才表示两个独立样本组成 batch。两者输出形状和 `token_type_ids` 含义不同，不能混淆。

## 下一步

- 继续学习 Week 1 的 Day 04：Tokenizer 基础。
- 重点练习 `input_ids`、`attention_mask`、`token_type_ids`、padding、truncation。
- 把 tokenizer 稳定知识同步整理到 `03-NLP-Basic/tokenizer-basics.md`。

