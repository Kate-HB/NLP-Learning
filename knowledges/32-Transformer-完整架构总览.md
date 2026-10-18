# 32 Transformer 完整架构总览

> 一句话：Transformer 不是"一个注意力层"，而是一整套"表示 → 信息交换 → 非线性变换 → 稳定训练"的流水线；一条数据流贯穿从文本到结果的全程。


## 前置知识

- 先读 [31-Transformer-为什么出现](31-Transformer-为什么出现.md)。
- 先读 [31-Transformer-为什么出现](31-why-transformer.md)。
- 需要知道：向量、矩阵、线性层、激活函数、Softmax（见 [19](19-标量-向量-矩阵-张量-与点积.md)、[21](21-线性层-激活函数-与网络深度.md)、[22](22-Hidden-State-Logits-与-Softmax.md)）。
- 需要知道：向量、矩阵、线性层、激活函数、Softmax（见 [19](19-scalar-vector-matrix-tensor-dot-product.md)、[21](21-linear-layer-activation-network-depth.md)、[22](22-hidden-state-logits-softmax.md)）。

## 一、直观含义

把 Transformer 想成一条"流水线工厂"，原料是文字，产品是预测结果：

```
原始文本 → Token 序列 → Token ID → Token Embedding + 位置表示
→ 多层 Transformer Block → 上下文隐藏状态 → 任务 Head → Logits → 任务结果
```

中间那几块"Transformer Block"是工厂的核心车间，被堆叠很多层；每个车间里又分"注意力"和"FFN"两条生产线。

## 二、具体例子

拿 BERT 分类一句话"这家店很好吃"：

1. Tokenizer：`["这", "家", "店", "很", "好", "吃"]` → 变成 ID。
2. Embedding：每个 ID 查成一个向量，加上位置信息。
3. 过 12 层 Block：每层都让每个词"看看"整句话，再各自做非线性变换。
4. 取出 `[CLS]` 的最终隐藏状态 → 分类 Head → 3 个 Logits → Softmax → "正面"。

## 三、技术要点（严谨部分）

### 1. 完整数据流

> 原始文本 → Token 序列 → Token ID → Token Embedding 与位置表示 → 多层 Transformer Block → 上下文隐藏状态 → 任务 Head → Logits → 任务结果

### 2. 关键组件定义

- **Transformer Block**：重复堆叠的基本层。
- **注意力头（Attention Head）**：一次独立的注意力关系计算。
- **前馈神经网络（FFN）**：逐位置进行非线性变换的子层（见 [42-Feed-Forward-Network](42-Feed-Forward-Network.md)）。
- **前馈神经网络（FFN）**：逐位置进行非线性变换的子层（见 [42-Feed-Forward-Network](42-feed-forward-network.md)）。
- **任务头（Task Head）**：把最终隐藏状态变成特定任务的输出（见 [48-Backbone-Task-Head-与-Weight-Tying](48-Backbone-Task-Head-与-Weight-Tying.md)）。
- **任务头（Task Head）**：把最终隐藏状态变成特定任务的输出（见 [48-Backbone-Task-Head-与-Weight-Tying](48-backbone-task-head-weight-tying.md)）。

### 3. Block 内部的构成

- **Encoder Block** 通常包含：自注意力 + FFN。
- **Decoder Block** 通常包含：带因果掩码的自注意力 + 可选的交叉注意力 + FFN。

### 4. 每个 Block 的通用骨架

每个 Block 内部都遵循同一套"公式"：

```
输入
 → 注意力（信息交换：不同 Token 之间）
 → 残差 + LayerNorm（Add & Norm）
 → FFN（位置内部变换）
 → 残差 + LayerNorm（Add & Norm）
 → 输出
```

（残差和 LayerNorm 见 [43-Residual-LayerNorm-Dropout-与-Add-Norm](43-Residual-LayerNorm-Dropout-与-Add-Norm.md)。）

### 5. 三层分工

- **注意力**：负责"不同 Token 之间"的信息交换。
- **FFN**：负责"每个 Token 内部"的特征变换。
- **Add & Norm**：负责让深层网络"不塌"。

## 四、一句话总结

Transformer = Tokenizer 做入口 + Embedding 做表示 + 多层 Block 反复"交换信息并变换特征" + Head 做出口；注意力管词间、FFN 管词内、残差归一层保稳定。
