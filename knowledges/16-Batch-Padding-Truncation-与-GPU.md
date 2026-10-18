# 16 Batch、Padding、Truncation 与 GPU

> 一句话：GPU 靠大规模并行矩阵运算吃饭，所以要把多条样本凑成一个"批次"（Batch）；但句子长短不一，必须用 Padding 补齐或 Truncation 截断来获得规则形状。


## 前置知识

- 先读 [08-语料-样本-序列与上下文](08-语料-样本-序列与上下文.md)。
- 先读 [08-语料-样本-序列与上下文](08-corpus-sample-sequence-context.md)。
- 本文铺垫 GPU 硬件背景，后面 [37-Scaled-Dot-Product-Attention](37-Scaled-Dot-Product-Attention.md) 的矩阵运算会反复用到。
- 本文铺垫 GPU 硬件背景，后面 [37-Scaled-Dot-Product-Attention](37-scaled-dot-product-attention.md) 的矩阵运算会反复用到。

## 一、直观含义

GPU 像一间"流水线工厂"，一次能同时加工一整批零件。如果一次只喂一条句子，工厂大部分产能都闲置了。所以把很多条句子打包成一个**批次**一起送进去。

但工厂要求"所有零件尺寸一样"（张量要规则形状），而句子有长有短。于是：

- **Padding（填充）**：给短句子补上 PAD，拉到批内最长。
- **Truncation（截断）**：把超长的句子砍到限制长度。

## 二、具体例子

Batch 里三句："你好"（2 token）、"我爱自然语言处理"（8 token）、"很好"（2 token）→ 全部补到 8 个 token：

```
[你, 好, PAD, PAD, PAD, PAD, PAD, PAD]
[我, 爱, 自, 然, 语, 言, 处, 理]
[很, 好, PAD, PAD, PAD, PAD, PAD, PAD]
```

若设定 `max_length=5`，"我爱自然语言处理"会被截成"我爱自然语"，丢失后面信息。

## 三、技术要点（严谨部分）

### 1. Batch 与 Batch Size

- **批次（Batch）**：一次并行送入模型的多个样本。
- **批次大小（Batch Size）**：一个 Batch 包含的样本数。它影响：
  - 显存占用；
  - 梯度稳定性；
  - 每次参数更新所依据的数据量。

### 2. GPU 与显存

- **GPU（图形处理器）**：擅长大规模并行数值运算的硬件。神经网络的矩阵运算量大，GPU 通常比 CPU 高效得多。
- **显存（VRAM）**：GPU 的高速工作内存。模型参数、隐藏状态、梯度和 Batch 都会占用显存——Batch 越大，显存压力越大。

### 3. 为什么需要 Padding / Truncation

> 同一 Batch 中的张量需要**规则形状**，但文本长度通常不同。

### 4. 填充（Padding）与截断（Truncation）

- **填充**：在短序列末尾（或开头）补 PAD Token，使批内序列长度一致。
- **截断**：删除超出最大长度的 Token。能控制计算量，但**可能丢失关键信息**。

### 5. 动态填充（Dynamic Padding）

- 只填充到**当前 Batch 的最长序列**，而不是整个数据集的最大长度。
- 通常能减少无效计算——长句少的批次不用白补一堆 PAD。

### 6. 别忘了 Attention Mask

- PAD 位置对模型没有意义，必须用 **Attention Mask** 告诉模型"这些位置是填充、请忽略"（见 [17-Attention-Mask-与-Token-Type-ID](17-Attention-Mask-与-Token-Type-ID.md)）。
- PAD 位置对模型没有意义，必须用 **Attention Mask** 告诉模型"这些位置是填充、请忽略"（见 [17-Attention-Mask-与-Token-Type-ID](17-attention-mask-token-type-id.md)）。

## 四、一句话总结

GPU 要并行所以要 Batch，Batch 要规则形状所以要 Padding 或 Truncation；补了 PAD 还得靠 Attention Mask 提醒模型别把它们当真。
