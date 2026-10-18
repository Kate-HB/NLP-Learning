# 44 Encoder 的完整结构

> 一句话：Encoder = 多个"双向自注意力 + FFN + Add & Norm"Block 堆叠，输入带位置的 Token 向量，输出每个 Token 的上下文化隐藏状态。


## 前置知识

- 先读 [32-Transformer-完整架构总览](32-Transformer-完整架构总览.md)、[39-Self-Attention-与-Multi-Head-Attention](39-Self-Attention-与-Multi-Head-Attention.md)、[43-Residual-LayerNorm-Dropout-与-Add-Norm](43-Residual-LayerNorm-Dropout-与-Add-Norm.md)。
- 先读 [32-Transformer-完整架构总览](32-transformer-overview.md)、[39-Self-Attention-与-Multi-Head-Attention](39-self-attention-multi-head-attention.md)、[43-Residual-LayerNorm-Dropout-与-Add-Norm](43-residual-layernorm-dropout-add-norm.md)。

## 一、直观含义

Encoder 是"通读全文的理解者"：它拿到整句话的所有 Token，每个 Token 都能同时看左右两侧，最后输出"每个词在整句话语境下的含义"。

它的每一层 Block 都做同一件事：**让每个词重新看看整句话**（自注意力），再**自己消化**（FFN），外面套 Add & Norm 保稳定。叠很多层，理解就越来越深。

## 二、具体例子

"苹果发布了新系统"：

- 第 1 层：每个词开始关注相邻词——"新"看"系统"，"发布"看"了"。
- 中间层：逐步建立更远关系——"苹果"关注"发布"和"系统"，从而知道这里"苹果"不是水果。
- 最后层：`[CLS]` 的向量已经汇聚了整句信息，可作为整段表示。

## 三、技术要点（严谨部分）

### 1. Encoder Block 的组成

一个经典 Encoder Block 包含：

1. 多头**双向自注意力**。
2. 残差连接与层归一化（Add & Norm）。
3. 逐位置**前馈网络（FFN）**。
4. 再一次残差连接与层归一化。

多个 Block 堆叠后，每一层都在上一层表示基础上重新计算 Token 之间的关系。

### 2. 输入与输出

- **输入**：带位置和结构信息的 Token 向量序列（见 [34-Token-Embedding-与输入表示相加](34-Token-Embedding-与输入表示相加.md)）。
- **输入**：带位置和结构信息的 Token 向量序列（见 [34-Token-Embedding-与输入表示相加](34-token-embedding-input-adding.md)）。
- **输出**：每个 Token 的**上下文化隐藏状态**。
  - 输出序列长度通常与输入 Token 数相同。
  - 每个位置的向量已融合其他位置的信息。

### 3. 分层的工作方式

- 浅层可能更偏**局部形式**（相邻词、短语）。
- 深层通常形成更**抽象的上下文表示**。
- ⚠️ 这种分工不是人工硬编码的，是训练自然涌现的。

### 4. 双向上下文（Bidirectional Context）

- 当前位置能同时利用**左侧和右侧**文本。
- BERT 在编码完整句子时具有这一特点。
- ⚠️ **双向 ≠ 把序列正向跑一次再反向跑一次**。在 Transformer Encoder 里，它主要意味着自注意力没有因果限制，每个位置能直接关注两侧。

### 5. 为什么 Encoder 适合理解类任务

因为"每个词能同时看两边"，模型能充分利用完整上下文来判断类别、抽取实体、预测答案跨度（详见 [47-三类架构-BERT-GPT-T5](47-三类架构-BERT-GPT-T5.md)）。

## 四、一句话总结

Encoder 是"通读型理解者"：Block = 双向自注意力 + FFN + Add & Norm，层层堆叠让每个 Token 变成"融合了整句话信息"的上下文化向量；双向 = 每个位置能直接看两侧。
