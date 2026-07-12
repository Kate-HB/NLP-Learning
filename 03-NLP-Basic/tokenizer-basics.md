# Tokenizer 基础

Tokenizer 的作用：把原始文本转换成模型能处理的数字输入。

## 核心流程

```text
原始文本
-> tokenize（切分成 token 字符串）
-> convert_tokens_to_ids（映射为词表 ID）
-> 添加特殊 token（[CLS]、[SEP] 等）
-> padding / truncation（统一长度）
-> attention_mask（标记真实 token 和 padding）
-> tensor
-> model
```

## 三种分词粒度

### 基于单词（Word-based）

按空格和标点拆分文本，规则简单。例如 `"Jim Henson was a puppeteer".split()` 得到 `['Jim', 'Henson', 'was', 'a', 'puppeteer']`。

问题：词表巨大（英语超 50 万词），每个词需要独立 ID；"dog" 和 "dogs" 被当作无关词汇，无法共享语义；遇到词表外词汇只能标记为 `[UNK]`，信息丢失。

### 基于字符（Character-based）

将文本拆分为单个字符，词表极小（几十个字符），几乎不会出现未知词。

问题：单个字符语义信息很少（中文单字信息量相对更高）；一个单词被切成十几个 token，序列变长、计算量增大。

### 基于子词（Subword-based）——工业界主流

常用词保持完整不拆分，罕见词拆成有意义的子词片段。例如 "annoyingly" → "annoying" + "ly"，"tokenization" → "token" + "ization"。词表小、未知词少、语义保留充分。

常见子词算法：

| 算法 | 使用模型 |
|---|---|
| Byte-level BPE（Byte Pair Encoding） | GPT-2 |
| WordPiece | BERT |
| SentencePiece / Unigram | T5、多语言模型 |

## Tokenizer 的三种原子操作

### tokenize() —— 文本到 token 字符串

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
```

`##` 前缀表示该 token 是前一个词的续接部分，不是独立词。"Trans" + "##former" 合并为 "Transformer"。

### convert_tokens_to_ids() —— token 字符串到数字 ID

```python
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [7993, 170, 13809, 23763, 2443, 1110, 3014]
```

每个 token 在词表中对应唯一的整数 ID，模型只能处理这些数字。

### decode() —— 数字 ID 还原为文本

```python
decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoded_string)
# 'Using a Transformer network is simple'
```

`decode()` 不仅把 ID 转回 token，还会自动把带 `##` 的子词片段合并回完整单词。

## BertTokenizer 与 AutoTokenizer

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

`BertTokenizer` 硬编码只支持 BERT 模型。换用其他模型（如 GPT-2）需要改类名。

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

`AutoTokenizer` 是通用接口，根据 checkpoint 名称自动推断对应的 tokenizer 类。传入 "bert-base-cased" 会自动映射到 `BertTokenizer`，传入 "gpt2" 则映射到 `GPT2Tokenizer`。切换模型时无需改代码。

## 模型必须接收批处理输入

Transformer 模型要求输入形状为 `(batch_size, sequence_length)`，即使只有一句话也需要 batch 维度。

```python
import torch

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor(ids)        # 形状 (seq_len,)，缺少 batch 维
# model(input_ids)                   # 报错！

input_ids = torch.tensor([ids])      # 形状 (1, seq_len)，加了 batch 维
output = model(input_ids)            # 正确
```

`torch.tensor([ids])` 的外层 `[]` 让列表变成 `[[101, 7993, ...]]`（二维），PyTorch 据此生成 `(1, seq_len)` 形状。等价写法：`torch.tensor(ids).unsqueeze(0)`。

直接调用 `tokenizer(sequence, return_tensors="pt")` 一步到位，分词、转 ID、加 batch 维、加 attention_mask 全部完成。

## 批处理与 Padding

多句话组成 batch 时，PyTorch 要求张量是矩形，但句子长度往往不同。Padding 在短句末尾添加 `pad_token_id`（通常为 0），使所有句子长度对齐。

```python
# 不等长 list 无法直接转 tensor
batched_ids = [
    [200, 200, 200],
    [200, 200],
]

# padding 后
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
```

## 为什么需要 Attention Mask

仅做 padding 不够——将 padding 后的 batch 输入模型，padding 句子的 logits 与单独输入时不一致。

**原因**：Transformer 的自注意力会关注序列中所有 token，包括 padding token。padding 位置参与了上下文计算，影响了有效 token 的表示。

**解决**：使用 attention mask，形状与 input_ids 相同：

```text
1 = 真实 token（需要关注）
0 = padding token（需要忽略）
```

```python
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(
    torch.tensor(batched_ids),
    attention_mask=torch.tensor(attention_mask),
)
```

加上 attention mask 后，mask 为 0 的位置在注意力计算中被置为极小值（负无穷），softmax 后权重趋近于 0，模型不再关注 padding 位置。此时 padding 句子的 logits 恢复为单独输入时的值。

**实验验证**：

```python
# 不加 attention mask → 第二行 logits 与单独输入时不同
print(model(torch.tensor(batched_ids)).logits)
# tensor([[ 1.5694, -1.3895],
#         [ 1.3374, -1.2163]])  # ← 不等于 [ 0.5803, -0.4125]

# 加 attention mask → 第二行 logits 恢复
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
# tensor([[ 1.5694, -1.3895],
#         [ 0.5803, -0.4125]])  # ← 恢复！
```

## 特殊 Token：[CLS] 和 [SEP]

`tokenizer(sequence)` 返回的 `input_ids` 比手动 `tokenize + convert_tokens_to_ids` 多了两个 ID：

```python
tokenizer(sequence)["input_ids"]
# [101, 1045, ..., 102]  ← 首尾多了 101 和 102

tokenizer.tokenize(sequence) + convert_tokens_to_ids
# [1045, ...]             ← 没有特殊 token
```

解码后可以看出：

```python
print(tokenizer.decode([101, 1045, ..., 102]))
# "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"

print(tokenizer.decode([1045, ...]))
# "i've been waiting for a huggingface course my whole life."
```

- `[CLS]`（ID=101）：位于句首，其最终隐藏状态用于汇总整句信息（分类任务的关键）
- `[SEP]`（ID=102）：位于句尾，用于分隔两个句子

这些特殊 token 是 BERT 预训练时使用的，推理时必须同样添加，否则输入分布与训练时不一致。不同模型的特殊 token 不同，有的只加开头，有的加其他类型的 token。

## 句子对与 Batch

```python
# 句子对——一个样本里的两个句子
tokenizer("How are you?", "I'm fine.")
# → [CLS] How are you ? [SEP] I'm fine . [SEP]
# token_type_ids 区分：0=句子A，1=句子B

# Batch——多个独立样本
tokenizer(["How are you?", "I'm fine."], padding=True)
# → 两个独立样本，各自有 [CLS] 和 [SEP]
```

## Padding 策略

```python
# 对齐到 batch 内最长句子的长度
tokenizer(sequences, padding="longest")

# 对齐到模型最大长度（BERT 为 512）
tokenizer(sequences, padding="max_length")

# 对齐到指定长度
tokenizer(sequences, padding="max_length", max_length=8)
```

## Truncation

模型有最大序列长度限制（BERT 为 512 token）。过长输入需要截断：

```python
# 截断到模型最大长度
tokenizer(sequences, truncation=True)

# 截断到指定长度
tokenizer(sequences, max_length=8, truncation=True)
```

## return_tensors

控制输出格式：

```python
# PyTorch tensor
tokenizer(sequences, padding=True, return_tensors="pt")

# TensorFlow tensor
tokenizer(sequences, padding=True, return_tensors="tf")

# NumPy array
tokenizer(sequences, padding=True, return_tensors="np")
```

加 `return_tensors="pt"` 后会自动添加 batch 维度：`(seq_len,)` 变为 `(1, seq_len)`。

## Tokenizer 和 Model 必须匹配

Tokenizer 的词汇表、特殊 token 定义、分词算法都是在模型预训练时确定的。用 BERT 的 tokenizer 切词后喂给 GPT-2，token ID 语义完全错乱——BERT 中 ID 1045 可能表示 "i"，GPT-2 中同一个 ID 可能表示完全不同的词，特殊 token 也不一致。因此 `AutoTokenizer.from_pretrained()` 和 `AutoModel.from_pretrained()` 必须使用相同的 checkpoint 名称。
