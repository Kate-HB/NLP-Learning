# Week 4：Transformer

本周目标：理解 Transformer 核心结构，重点掌握 Attention、Self-Attention、Multi-Head Attention，并实现 Mini Transformer 的关键模块。

---

## 1. 本周学习什么

### 1.1 Attention

- Query
- Key
- Value
- 点积相似度
- scaled dot-product attention
- softmax 权重
- 加权求和

### 1.2 Transformer 结构

- Multi-Head Attention
- Positional Encoding
- Feed Forward Network
- Residual Connection
- LayerNorm
- Encoder Block
- Decoder Block
- Causal Mask

### 1.3 代码实现

- PyTorch 矩阵乘法
- mask 处理
- attention score
- attention weight
- encoder block
- 简单分类模型

---

## 2. 本周学习资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| The Illustrated Transformer | 图解 Transformer | https://jalammar.github.io/illustrated-transformer/ |
| Attention Is All You Need | Transformer 原论文 | https://arxiv.org/abs/1706.03762 |
| NLP-LOVE/ML-NLP | Attention 和 Transformer | https://github.com/NLP-LOVE/ML-NLP |
| PyTorch nn.Transformer | 官方实现参考 | https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html |

---

## 3. 本周最终产出

```text
04-Transformer/
├── attention_notes.md
├── self_attention.py
├── multi_head_attention.py
├── positional_encoding.py
└── mini_transformer.py

notes/
├── day16.md
├── day17.md
├── day18.md
├── day19.md
├── day20.md
└── week04_summary.md
```

---

## 4. 每日计划

## Day 16：Attention 原理

### 学习目标

- 理解 Q、K、V
- 理解 Attention 的计算流程

### 任务

1. 阅读 The Illustrated Transformer 的 Attention 部分。
2. 手写 Attention 计算步骤。
3. 用小矩阵模拟 Q、K、V。
4. 计算 attention score。
5. 写原理笔记。

### 输出文件

`04-Transformer/attention_notes.md`

### 验收标准

- 能解释 Q、K、V
- 能解释为什么要 softmax

---

## Day 17：Self-Attention 代码

### 学习目标

- 用 PyTorch 实现 Self-Attention

### 任务

1. 创建输入矩阵。
2. 生成 Q、K、V。
3. 计算 score。
4. 除以 `sqrt(d_k)`。
5. softmax。
6. 加权求和。

### 代码文件

`04-Transformer/self_attention.py`

### 验收标准

- 能跑通 Self-Attention
- 能解释输出 shape

---

## Day 18：Multi-Head Attention

### 学习目标

- 理解多头注意力为什么有用
- 实现简化版 Multi-Head Attention

### 任务

1. 将 hidden size 拆成多个 head。
2. 每个 head 计算 attention。
3. 拼接多个 head。
4. 通过线性层输出。

### 代码文件

`04-Transformer/multi_head_attention.py`

### 验收标准

- 能解释 head 的作用
- 能解释 concat 的过程

---

## Day 19：位置编码与 Encoder Block

### 学习目标

- 理解位置编码
- 理解 Transformer 为什么需要位置信息
- 组装 Encoder Block

### 任务

1. 实现 sinusoidal positional encoding。
2. 实现 FFN。
3. 加入 residual connection。
4. 加入 LayerNorm。

### 代码文件

`04-Transformer/positional_encoding.py`

### 验收标准

- 能解释位置编码
- 能解释 residual connection
- 能解释 LayerNorm

---

## Day 20：Mini Transformer

### 学习目标

- 组装一个最小 Transformer Encoder
- 用于简单分类任务

### 任务

1. 复用前几天模块。
2. 组装 Mini Transformer。
3. 输入随机 token embedding。
4. 输出分类 logits。
5. 写 Week 4 总结。

### 代码文件

`04-Transformer/mini_transformer.py`

### 验收标准

- 能说明 Transformer Encoder 流程
- 能解释输入输出 shape
- 代码能运行

---

## 5. 本周必须掌握的问题

1. Attention 解决什么问题？
2. Q、K、V 分别是什么？
3. 为什么要除以 `sqrt(d_k)`？
4. Multi-Head Attention 的作用是什么？
5. Transformer 为什么需要位置编码？
6. Residual Connection 有什么作用？
7. LayerNorm 有什么作用？
8. Encoder 和 Decoder 有什么区别？
9. Causal Mask 用在哪里？
10. BERT 和 GPT 分别对应哪种结构？

---

## 6. Week 4 完成标准

- 能解释 Attention
- 能实现 Self-Attention
- 能实现 Multi-Head Attention
- 能实现位置编码
- 能组装 Mini Transformer
