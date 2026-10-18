# 36 注意力直观含义与 QKV

> 一句话：注意力让模型在处理当前位置时，为其他位置分配不同权重、加权汇总重要信息；Q（查什么）、K（凭什么被查）、V（被查到后给什么）是这套检索机制的三个角色。


## 前置知识

- 先读 [19-标量-向量-矩阵-张量-与点积](19-标量-向量-矩阵-张量-与点积.md)。
- 先读 [19-标量-向量-矩阵-张量-与点积](19-scalar-vector-matrix-tensor-dot-product.md)。
- 公式层面的计算见 [37-Scaled-Dot-Product-Attention](37-Scaled-Dot-Product-Attention.md)，本文只讲"它想干什么"。
- 公式层面的计算见 [37-Scaled-Dot-Product-Attention](37-scaled-dot-product-attention.md)，本文只讲"它想干什么"。

## 一、直观含义

读"他拿起书"时，理解"他"到底指谁，需要回头看前文哪个人名——这就是注意力：**处理当前位置时，有选择地参考其他位置**。

可以把它类比成图书馆检索：

- **Query（查询）**：你手里的搜索条件——"当前词想找什么信息"。
- **Key（键）**：每本书的索引标签——"每个词能拿什么特征被别人匹配"。
- **Value（值）**：书的正文——"被关注后实际提供的内容"。

匹配 Query 和 Key 得出"谁相关"，再用相关性加权汇总 Value，就是注意力输出。

## 二、具体例子

理解 "John gave the book to Mary because **she** liked it" 中的 "she"：

- "she" 的 Query 可能想找"近处的人名"。
- 序列里每个词的 Key 提供自己的特征；"Mary" 的 Key 和 "she" 的 Query 匹配度高。
- 于是模型给 "Mary" 的 Value 很高权重，把 "Mary" 的信息汇总进 "she" 的表示。

## 三、技术要点（严谨部分）

### 1. 注意力的定义

- **注意力（Attention）**：处理当前位置时，为其他位置分配不同权重，并把重要位置的信息**加权汇总**。

### 2. 三个注意点

1. 注意力不是简单地"寻找最近的词"，而是根据**当前层学到的表示**动态计算相关性。
2. 注意力权重描述"当前计算如何分配信息"，**不一定等于人类可解释的"原因"**。
3. 不能单凭一张注意力图，就证明模型做出预测的因果依据。

### 3. Query、Key、Value（QKV）

- 在自注意力中，同一序列的每个输入向量会经过**三个不同的线性变换**，得到：

| 角色 | 作用 | 类比 |
|---|---|---|
| Query（Q） | 当前 Token 想寻找什么信息 | 搜索条件 |
| Key（K） | 当前 Token 可以用什么特征被其他位置匹配 | 索引标签 |
| Value（V） | 如果当前 Token 被关注，实际提供什么内容 | 取回的正文 |

- Q、K、V 都是**模型学习出来的连续向量**，不是人工设定的关键词。
- ⚠️ Value 不直接参与"匹配"（那是 Query 与 Key 的事），只参与"汇总"。

### 4. Self-Attention 与 Cross-Attention 的差别（预览）

- **Self-Attention**：Q、K、V 来自**同一序列**（见 [39-Self-Attention-与-Multi-Head-Attention](39-Self-Attention-与-Multi-Head-Attention.md)）。
- **Self-Attention**：Q、K、V 来自**同一序列**（见 [39-Self-Attention-与-Multi-Head-Attention](39-self-attention-multi-head-attention.md)）。
- **Cross-Attention**：Query 来自 Decoder，Key 和 Value 来自 Encoder 输出（见 [45-Decoder-的完整结构](45-Decoder-的完整结构.md)）。
- **Cross-Attention**：Query 来自 Decoder，Key 和 Value 来自 Encoder 输出（见 [45-Decoder-的完整结构](45-decoder-architecture.md)）。

## 四、一句话总结

注意力 = 按相关性给其他位置分配权重并加权汇总；Q 问"我想要啥"、K 答"我有啥可被匹配"、V 给"被选中后提供的内容"——三个角色都由模型自己学出来。
