# 39 Self-Attention 与 Multi-Head Attention

> 一句话：Self-Attention 的 Q、K、V 都来自同一序列；Multi-Head 把表示投影到多个子空间并行算多组注意力，再拼接融合——让模型能同时关注"不同种类的关�系"。


## 前置知识

- 先读 [36-注意力直观含义与-QKV](36-注意力直观含义与-QKV.md)、[37-Scaled-Dot-Product-Attention](37-Scaled-Dot-Product-Attention.md)。
- 先读 [36-注意力直观含义与-QKV](36-attention-intuition-qkv.md)、[37-Scaled-Dot-Product-Attention](37-scaled-dot-product-attention.md)。

## 一、直观含义

- **Self-Attention（自注意力）**：每个词和"同一句话里的所有词"互相看。Q、K、V 都来自同一个序列——自己看自己所在的整句话。
- **Multi-Head（多头）**：不多算一遍，而是把表示分成几个"子空间"并行算。像一个团队分几个小组，每个小组带着不同偏好去读句子——有的偏爱看相邻词（局部搭配）、有的偏爱看指代（"她"→"Mary"）、有的看句法结构。

## 二、具体例子

- Encoder 双向自注意力："苹果发布了新系统"里的"苹果"，能同时看左边（无）和右边（"发布了"）——所以能判断它是公司。
- Decoder 因果自注意力：生成"苹果"时只能看它自己及之前已生成的内容，不能看后面（见 [40-四种掩码辨析](40-四种掩码辨析.md)）。
- Decoder 因果自注意力：生成"苹果"时只能看它自己及之前已生成的内容，不能看后面（见 [40-四种掩码辨析](40-four-masks-distinguished.md)）。
- 多头（BERT-base 是 12 头）：12 个小组并行读句子，各自关注不同关系，最后拼接融合成一个 768 维向量。

## 三、技术要点（严谨部分）

### 1. Self-Attention

- Q、K、V 都来自**同一序列**。
- 每个位置都可以读取这条序列中**允许访问**的位置。
- Encoder 的双向自注意力：允许每个有效 Token 同时查看左侧和右侧，适合理解完整输入。
- Decoder 的自注意力：通常加入因果掩码，只允许读取当前位置和之前位置，适合自回归生成。

### 2. Multi-Head Attention

- 不是只做一次注意力，而是：
  1. 把表示投影到**多个较小的子空间**；
  2. 多个 Head **并行**计算注意力；
  3. **拼接（Concatenation）** 各头结果；
  4. 再做一次线性变换（输出投影）融合回模型隐藏维度。

### 3. 注意力头（Attention Head）

- 一组独立的 Q、K、V 投影。
- 不同 Head 可以学习不同关系：局部搭配、指代关系、句法结构、长距离依赖等。
- ⚠️ 多头的意义**不是保证每个 Head 都对应清晰的人类语言规律**，而是让模型拥有多个并行的信息交互子空间，增强表达能力。

### 4. 拼接（Concatenation）

- 把多个向量沿**特征维度**首尾组合成更长向量。
- 多头输出拼接后，通过输出投影融合回模型隐藏维度。

### 5. 和 Task Head 的区别

- Attention Head 在 Transformer Block **内部**，负责关系计算。
- Task Head 在骨干网络**之后**，负责任务输出（见 [48-Backbone-Task-Head-与-Weight-Tying](48-Backbone-Task-Head-与-Weight-Tying.md)）。两者名字都带 Head，但位置和职责完全不同。
- Task Head 在骨干网络**之后**，负责任务输出（见 [48-Backbone-Task-Head-与-Weight-Tying](48-backbone-task-head-weight-tying.md)）。两者名字都带 Head，但位置和职责完全不同。

## 四、一句话总结

Self-Attention 让每个词看同一序列的其他词；Multi-Head 用多组 QKV 并行看多种关系再融合——一个管"看谁"，一个管"怎么看得多"。
