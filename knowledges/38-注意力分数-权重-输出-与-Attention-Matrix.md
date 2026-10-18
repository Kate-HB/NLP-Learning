# 38 注意力分数、权重、输出与 Attention Matrix

> 一句话：注意力分三个阶段——原始分数（Score）、Softmax 后的权重（Weight）、加权汇总的输出（Output）；整张"谁看谁"的权重表就是 Attention Matrix，它的大小随序列长度二次增长。


## 前置知识

- 先读 [37-Scaled-Dot-Product-Attention](37-Scaled-Dot-Product-Attention.md)。
- 先读 [37-Scaled-Dot-Product-Attention](37-scaled-dot-product-attention.md)。

## 一、直观含义

- **Score（分数）**：还没"归一"的原始匹配分——可能正可能负，绝对值可大可小。
- **Weight（权重）**：分数经过掩码和 Softmax 之后，变成"总和为 1、每个都在 0~1"的权重。
- **Output（输出）**：用这些权重去"加权求和"所有 Value，得到当前词的最终注意力向量。
- **Attention Matrix（注意力矩阵）**：把"每个位置看每个位置"的权重铺成一张表——行是"看的人"（Query），列是"被看的人"（Key）。

## 二、具体例子

序列长度 4，Attention Matrix 大概是：

```
         被看→  K₁   K₂   K₃   K₄
看的人↓ Q₁    0.5  0.2  0.2  0.1
       Q₂    0.1  0.6  0.2  0.1
       Q₃    0.1  0.2  0.6  0.1
       Q₄    0.1  0.1  0.2  0.6
```

每行和为 1。Q₁ 主要看 K₁（0.5）——自己权重最大。

## 三、技术要点（严谨部分）

### 1. 三阶段必须分清

| 阶段 | 公式/位置 | 性质 |
|---|---|---|
| Attention Score | `QKᵀ/√dₖ`（Softmax 前） | 未归一，可正可负 |
| Attention Weight | Softmax 后 | 每行和为 1，0~1 |
| Attention Output | 权重 × Value 求和 | 上下文化向量 |

> ⚠️ 分数、权重、输出是三个不同阶段，不能混为一谈。

### 2. Attention Matrix

- 记录每个 Query 位置对每个 Key 位置的权重。
- 若序列长度为 L，标准自注意力矩阵大小约为 **L × L**。
- 因此标准注意力的**时间和显存成本随序列长度近似二次增长**——这就是长上下文昂贵的主要原因之一（详见 [41-全-稀疏-局部-全局-LSH-注意力](41-全-稀疏-局部-全局-LSH-注意力.md)）。
- 因此标准注意力的**时间和显存成本随序列长度近似二次增长**——这就是长上下文昂贵的主要原因之一（详见 [41-全-稀疏-局部-全局-LSH-注意力](41-full-sparse-local-global-lsh-attention.md)）。

### 3. 行归一的意义

- 每行（每个 Query）的权重之和为 1，表示"这个 Query 把 100% 的关注度分配给了哪些位置"。
- 权重值大 ≠ 因果关系成立，只能说明"计算时给了它更多信息份额"。

## 四、一句话总结

Score 是打分、Weight 是归一后的权重、Output 是加权汇总的结果，Attention Matrix 把它们铺成"谁看谁"的表；矩阵是 L×L，所以长序列的计算量是二次增长。
