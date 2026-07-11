# Tokenizer 基础

Tokenizer 的作用：把原始文本转换成模型能处理的数字输入。

## 核心流程

```text
原始文本
-> token
-> input_ids
-> attention_mask / token_type_ids
-> tensor
-> model
```

## input_ids

`input_ids` 是 token 在词表中的数字编号。

例如：

```text
[CLS] Hello ! [SEP]
```

会被转换成类似：

```text
[101, 8667, 106, 102]
```

模型不能直接处理文本，只能处理这些数字。

## attention_mask

`attention_mask` 标记哪些位置是真实文本，哪些位置是 padding。

```text
1 = 真实 token
0 = padding token
```

它防止模型把补齐用的 padding 当成真实内容。

## token_type_ids

`token_type_ids` 用于区分句子对中的句子 A 和句子 B。

```text
0 = 第一句
1 = 第二句
```

常见于 BERT 的句子对任务。

## 句子对和 batch

句子对：

```python
tokenizer("How are you?", "I'm fine.")
```

表示一个样本里有两个句子。

batch：

```python
tokenizer(["How are you?", "I'm fine."], padding=True)
```

表示两个独立样本。

## return_tensors

不加 `return_tensors`，返回 Python list。

加：

```python
return_tensors="pt"
```

返回 PyTorch tensor，可以直接输入模型。

## padding

多个句子长度不同，不能直接组成规则 tensor。

`padding=True` 会把短句补齐到同一长度。

## truncation

模型有最大输入长度。BERT 通常最多处理 512 个 token。

`truncation=True` 会截断过长输入。

