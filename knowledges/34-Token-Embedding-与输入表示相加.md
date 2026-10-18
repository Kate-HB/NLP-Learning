# 34 Token Embedding 与输入表示相加

> 一句话：Token Embedding 把 Token ID 查表成稠密向量，再与位置表示（和句段嵌入）相加，就得到模型第一层的输入。


## 前置知识

- 先读 [19-标量-向量-矩阵-张量-与点积](19-标量-向量-矩阵-张量-与点积.md)、[13-词表-Token-ID-与-OOV](13-词表-Token-ID-与-OOV.md)。
- 先读 [19-标量-向量-矩阵-张量-与点积](19-scalar-vector-matrix-tensor-dot-product.md)、[13-词表-Token-ID-与-OOV](13-vocabulary-token-id-oov.md)。
- 位置表示具体有哪些，见 [35-位置编码-绝对-正弦-相对-RoPE](35-位置编码-绝对-正弦-相对-RoPE.md)。
- 位置表示具体有哪些，见 [35-位置编码-绝对-正弦-相对-RoPE](35-positional-encoding-abs-sinusoidal-relative-rope.md)。

## 一、直观含义

模型第一件事：把"词表里查到的编号"变成一个向量。

- **Token Embedding**：一张大查表（Embedding 矩阵），每一行是某个 Token 的可学习向量。编号 5 → 拿第 5 行。
- **位置表示**：单独一套"第几个位置"的向量，告诉模型先后顺序。
- **相加**：`Token向量 + 位置向量`——这样每个位置的输入里既装"是什么词"，也装"在哪一位"。

## 二、具体例子

- "我爱NLP" → Token IDs `[5, 12, 30]`。
- 查表：`token_vec(5)`、`token_vec(12)`、`token_vec(30)`，假设都是 768 维。
- 位置：`pos_vec(0)`、`pos_vec(1)`、`pos_vec(2)`。
- 相加：`x_0 = token_vec(5) + pos_vec(0)`，以此类推，三个向量组成输入矩阵 `(3, 768)`。

## 三、技术要点（严谨部分）

### 1. Token Embedding

- 把每个 Token ID 映射为一个**稠密向量**。
- Embedding 矩阵的**每一行对应一个 Token** 的可学习表示。
- 关键区分（见 [72-易混淆概念辨析-上](72-易混淆概念辨析-上.md)）：
- 关键区分（见 [72-易混淆概念辨析-上](72-confusing-concepts-part-1.md)）：
  - **初始 Token Embedding** 只由 Token ID 决定——同一个 Token 在任何句子里初始向量相同。
  - 经过多层 Transformer 后，才变成**结合上下文的隐藏状态**。

### 2. 位置表示（详见 35）

- 注意力本身若不加位置信息，只知道"有哪些 Token"，不知道"先后顺序"——所以必须引入位置表示。

### 3. 句段嵌入（Segment Embedding）

- BERT 还可能加入句段嵌入，用来**区分句子 A 与句子 B**（Token Type ID 就是查这个表，见 [17-Attention-Mask-与-Token-Type-ID](17-Attention-Mask-与-Token-Type-ID.md)）。
- BERT 还可能加入句段嵌入，用来**区分句子 A 与句子 B**（Token Type ID 就是查这个表，见 [17-Attention-Mask-与-Token-Type-ID](17-attention-mask-token-type-id.md)）。

### 4. 相加的含义

> 相加后的每个位置既包含"是什么 Token"，也包含"位于哪里"和"属于哪个句段"等信息，然后进入第一层 Transformer Block。

### 5. 矩阵视角

- 一批输入 = `(batch, seq_len, hidden)` 的三维张量。
- 逐位置相加是**向量加法**，不同信息在同一个向量里"叠加共存"。

## 四、一句话总结

Token Embedding 回答"这个词是什么"，位置表示回答"它在哪"，句段嵌入回答"它属于哪句"，三者加在一起构成模型真正看到的第一个输入。
