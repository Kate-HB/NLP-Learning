# 17 Attention Mask 与 Token Type ID

> 一句话：Attention Mask 标记"哪些位置是有效内容、哪些是填充"，让注意力忽略 PAD；Token Type ID 标记"这个 Token 属于句子 A 还是 B"。


## 前置知识

- 先读 [16-Batch-Padding-Truncation-与-GPU](16-Batch-Padding-Truncation-与-GPU.md)。
- 先读 [16-Batch-Padding-Truncation-与-GPU](16-batch-padding-truncation-gpu.md)。
- 若"注意力分数""Softmax"陌生，可先读 [37-Scaled-Dot-Product-Attention](37-Scaled-Dot-Product-Attention.md) 再回来看。
- 若"注意力分数""Softmax"陌生，可先读 [37-Scaled-Dot-Product-Attention](37-scaled-dot-product-attention.md) 再回来看。

## 一、直观含义

补了 PAD 之后，模型不知道哪些是真实内容。Attention Mask 就是一张"有效位置名单"：

- 有效 Token 标 **1**。
- PAD 位置标 **0**。

被标 0 的位置，在注意力计算时会被赋予**极小值**，经过 Softmax 后权重趋近 0——等于说"这些位置当没看见"。

Token Type ID 则是给句子对用的"门牌号"：第一句全标 0，第二句全标 1，帮模型区分"哪半句是哪句"。

## 二、具体例子

句子"我爱猫"+"我也爱狗"，补 PAD 后：

```
Token 序列:   [CLS] 我 爱 猫 [SEP] 我 也 爱 狗 [SEP] PAD PAD
Attention Mask:  1   1  1  1    1   1  1  1  1    1   0   0
Token Type ID:   0   0  0  0    0   1  1  1  1    1   1   1
```

- Attention Mask：后两个 PAD 标 0，注意力忽略它们。
- Token Type ID：第一句（含 CLS）为 0，第二句（含 SEP）为 1。

## 三、技术要点（严谨部分）

### 1. Attention Mask（注意力掩码）

- 告诉模型哪些位置有效、哪些应被忽略。
- 常见约定：有效 Token = 1，Padding = 0。
- 实现：被忽略位置在注意力分数中被赋予**极小值**；Softmax 把一组分数归一化为总和为 1 的权重，这些位置的权重接近 0。

### 2. 三个容易混淆的"掩码"

> Attention Mask 不负责遮盖 MLM 的答案，也不等于 Decoder 的因果掩码。

- **MLM**：恢复被随机遮盖 Token 的**训练目标**。
- **Decoder**：负责按可见前文生成后续内容的结构。
- **因果掩码**：禁止 Decoder 读取未来位置的**规则**。
- **Attention Mask**：描述当前输入中**哪些位置属于有效内容**（即本页讲的这个）。

四者控制的对象分别是"训练目标""结构""可见方向""输入有效性"，详见 [40-四种掩码辨析](40-四种掩码辨析.md)。

### 3. Token Type ID（Token 类型 / 句段 ID）

- 标记 Token 属于**句子 A 还是句子 B**。
- BERT 的句子对输入常使用它（源自预训练的"下一句预测"任务，见 [50-CLM-与-NSP](50-CLM-与-NSP.md)）。
- BERT 的句子对输入常使用它（源自预训练的"下一句预测"任务，见 [50-CLM-与-NSP](50-clm-nsp.md)）。
- **不是所有 Transformer 都有这一输入**。

## 四、一句话总结

Attention Mask 用 1/0 告诉注意力"哪些位置是真话"，Token Type ID 用 0/1 告诉模型"哪半句是哪句"；它们和"遮盖答案的 MASK Token""因果掩码"是四码事。
