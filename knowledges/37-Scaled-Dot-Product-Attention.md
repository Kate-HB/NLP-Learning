# 37 Scaled Dot-Product Attention

> 一句话：注意力就是"Query 和所有 Key 点积打分 → 缩放 → Softmax 变权重 → 加权求和 Value"；那个缩放（除以 √dₖ）防止点积过大让 Softmax 饱和、梯度过小。


## 前置知识

- 先读 [36-注意力直观含义与-QKV](36-注意力直观含义与-QKV.md)。
- 先读 [36-注意力直观含义与-QKV](36-attention-intuition-qkv.md)。
- 需要点积、矩阵乘法、Softmax（见 [19](19-标量-向量-矩阵-张量-与点积.md)、[22](22-Hidden-State-Logits-与-Softmax.md)）。
- 需要点积、矩阵乘法、Softmax（见 [19](19-scalar-vector-matrix-tensor-dot-product.md)、[22](22-hidden-state-logits-softmax.md)）。

## 一、直观含义

完整的注意力是一个三步流程：

1. **打分**：当前 Query 和所有 Key 点积——分高 = 相关。
2. **归一**：Softmax 把所有分数变成总和为 1 的权重。
3. **汇总**：按权重加权求和所有 Value，得到当前位置的注意力输出。

公式只有一行：

```
Attention(Q, K, V) = Softmax(QKᵀ / √dₖ) V
```

## 二、具体例子

序列有 3 个 Token，处理第 1 个 Token 时：

- Q₁ 与 K₁、K₂、K₃ 点积 → 分数 `[2, 0.5, -1]`。
- 除以 √dₖ（比如 √64=8）→ `[0.25, 0.06, -0.125]`。
- Softmax → `[0.43, 0.37, 0.20]`。
- 输出 = `0.43·V₁ + 0.37·V₂ + 0.20·V₃`。

看，第 1 个 Token 的信息占大头——它"更关注自己"，这是它自己的选择。

## 三、技术要点（严谨部分）

### 1. 各组成部分的含义

- **QKᵀ**：得到每个 Query 对每个 Key 的**原始相关性分数**。转置 K 让所有 Query 能同时与所有 Key 做点积（矩阵乘法，见 [19](19-标量-向量-矩阵-张量-与点积.md)）。
- **QKᵀ**：得到每个 Query 对每个 Key 的**原始相关性分数**。转置 K 让所有 Query 能同时与所有 Key 做点积（矩阵乘法，见 [19](19-scalar-vector-matrix-tensor-dot-product.md)）。
- **dₖ**：Key 向量维度。
- **÷√dₖ（缩放）**：防止维度较大时点积绝对值过大，使 Softmax **过早饱和**（概率推向 0 或 1）、**梯度过小**（学不动）。
- **Softmax**：把分数转成权重，和为 1。
- **加权求和 V**：得到该 Query 聚合后的**上下文表示**。

### 2. 为什么需要缩放

- 点积对向量长度敏感。维度 dₖ 越大，点积的期望绝对值越大。
- 当分数绝对值很大时，Softmax 的输出会极度接近 one-hot——梯度区域非常平坦，反向传播学不到东西。
- 除以 √dₖ 把分数拉回到温和的尺度。

### 3. 三个阶段的名称

- **Attention Score**：Softmax 之前的 `QKᵀ/√dₖ`。
- **Attention Weight**：经过掩码和 Softmax 后的概率式权重。
- **Attention Output**：按权重加权汇总 Value 后得到的向量。

（分数、权重、输出是三个不同阶段，详见 [38-注意力分数-权重-输出-与-Attention-Matrix](38-注意力分数-权重-输出-与-Attention-Matrix.md)。）

## 四、一句话总结

缩放点积注意力 = 点积打分（QKᵀ）→ 除 √dₖ 防止饱和 → Softmax 归一 → 加权汇总 V；一行公式，三步含义。
