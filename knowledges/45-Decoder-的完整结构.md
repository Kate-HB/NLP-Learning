# 45 Decoder 的完整结构

> 一句话：Decoder 比 Encoder 多两样东西——带因果掩码的自注意力（只能看前文）和交叉注意力（读取 Encoder 的源表示）；没有独立 Encoder 的 Decoder-only 模型则只有前者。


## 前置知识

- 先读 [33-三类架构的数据流](33-三类架构的数据流.md)、[40-四种掩码辨析](40-四种掩码辨析.md)。
- 先读 [33-三类架构的数据流](33-data-flow-of-three-architectures.md)、[40-四种掩码辨析](40-four-masks-distinguished.md)。
- 若"Teacher Forcing/目标右移"还没看，不阻塞本页，见 [51-Seq2Seq-Teacher-Forcing-与目标右移](51-Seq2Seq-Teacher-Forcing-与目标右移.md)。
- 若"Teacher Forcing/目标右移"还没看，不阻塞本页，见 [51-Seq2Seq-Teacher-Forcing-与目标右移](51-seq2seq-teacher-forcing-label-shift.md)。

## 一、直观含义

Decoder 是"一边看着稿子、一边写答案"的人：

1. **先回顾自己已经写了什么**（因果自注意力）——只能看"已经写出的部分"，不能看还没写的内容。
2. **再看一眼源稿**（交叉注意力）——对照原文（比如翻译的源句），决定下一步写什么。
3. **消化一下**（FFN）。

每一步只吐出一个 Token，吐完再写下一个。

## 二、具体例子

翻译 "I love NLP" → 生成中文，生成到第 2 个词"爱"时：

- 因果自注意力：能看"我"（已生成），不能看后面还没生成的"NLP"。
- 交叉注意力：以"爱"为 Query，去源句 "I love NLP" 的表示里找信息——重点关注 "love"。
- FFN 消化后，通过 LM Head 给出下一个词的概率。

## 三、技术要点（严谨部分）

### 1. Decoder Block 的组成

原始 Encoder-Decoder Transformer 的 Decoder Block 包含：

1. 带**因果掩码**的多头自注意力。
2. 残差连接与层归一化。
3. 对 Encoder 输出的多头**交叉注意力**。
4. 残差连接与层归一化。
5. 逐位置前馈网络（FFN）。
6. 再一次残差连接与层归一化。

### 2. Masked Self-Attention（带掩码的自注意力）

- 在 Decoder 中通常指**带因果掩码的自注意力**。
- 当前位置只能利用**已经出现的目标前文**，不能访问未来目标 Token。
- ⚠️ 这里的"Masked"是**注意力可见性限制**，不是把输入替换为 MASK Token（见 [40-四种掩码辨析](40-四种掩码辨析.md)）。
- ⚠️ 这里的"Masked"是**注意力可见性限制**，不是把输入替换为 MASK Token（见 [40-四种掩码辨析](40-four-masks-distinguished.md)）。

### 3. Cross-Attention（交叉注意力）

- 让 Decoder **读取 Encoder 输出**。
- Query 来自 **Decoder 当前隐藏状态**，Key 和 Value 来自 **Encoder 输出**。
- 表达的意思："根据当前已经生成的目标前文，应从源文本中读取哪些信息？"
- 在翻译中，Decoder 预测每个目标词时都可以**重新关注源句不同位置**；源语言和目标语言词序不同也不妨碍这种动态对齐。

### 4. Decoder-only 与 Encoder-Decoder 的区别

- **Decoder-only 模型**没有独立 Encoder，因此**通常没有对 Encoder 输出的交叉注意力**，只保留因果自注意力和 FFN。
- 只有 Encoder-Decoder 模型里的 Decoder 才有交叉注意力。

### 5. 为什么叫"解码器"

- 它把"源序列的表示"一步步"解"成目标 Token 序列（自回归生成见 [46-自回归生成](46-自回归生成.md)）。
- 它把"源序列的表示"一步步"解"成目标 Token 序列（自回归生成见 [46-自回归生成](46-autoregressive-generation.md)）。
- 注意和"Tokenizer 解码""生成解码"区分（见 [73-易混淆概念辨析-下](73-易混淆概念辨析-下.md)）。
- 注意和"Tokenizer 解码""生成解码"区分（见 [73-易混淆概念辨析-下](73-confusing-concepts-part-2.md)）。

## 四、一句话总结

Decoder Block = 因果自注意力（看前文）＋ 交叉注意力（看源稿）＋ FFN（消化）；没有源稿的 Decoder-only 只有前两样里的因果自注意力。
