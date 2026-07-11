# Week 3：深度学习概念与 Transformer

目标：补足理解 Transformer 必须的数学和结构，不从零造完整大模型。

## 本周资源

| 资源 | 链接 |
|---|---|
| NumPy Absolute Beginners | https://numpy.org/doc/stable/user/absolute_beginners.html |
| NumPy Quickstart | https://numpy.org/doc/stable/user/quickstart.html |
| D2L Softmax Regression | https://d2l.ai/chapter_linear-classification/index.html |
| D2L Attention | https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html |
| D2L Transformer | https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |

## 每日计划

| Day | 学习内容 | 代码/实验 | 笔记重点 |
|---|---|---|---|
| Day 15 | NumPy shape、axis、广播 | 写广播示例 | 矩阵维度怎么读 |
| Day 16 | softmax、cross entropy | 手写 softmax | 分类损失为什么这样算 |
| Day 17 | Embedding | 查表式 embedding 示例 | token id 到向量 |
| Day 18 | Attention 与 QKV | 小矩阵手算 attention | Q、K、V 各自含义 |
| Day 19 | Multi-Head Attention | 拆 head、拼接 head | 多头为什么有用 |
| Day 20 | Transformer Encoder | 梳理 MHA、FFN、LayerNorm | Encoder 数据流 |
| Day 21 | 复盘 | 画 Transformer Encoder | mask、残差、位置编码 |

## 本周产出

```text
01-Python/
├── numpy_shape_demo.py
└── numpy_broadcast_demo.py

04-Transformer/
├── attention_notes.md
├── self_attention.py
├── multi_head_attention.py
└── transformer_structure.md
```

## 必须掌握

1. softmax 和 cross entropy 的关系。
2. embedding 是什么。
3. Q、K、V 分别是什么。
4. attention 为什么需要 mask。
5. Transformer Encoder 包含哪些模块。

