# 48 Backbone、Task Head 与 Weight Tying

> 一句话：Backbone 是产生通用表示的 Transformer 主体，Task Head 是接在后面的任务专用输出层，Weight Tying 让输入 Embedding 和输出 LM Head 共享参数。


## 前置知识

- 先读 [22-Hidden-State-Logits-与-Softmax](22-Hidden-State-Logits-与-Softmax.md)。
- 先读 [22-Hidden-State-Logits-与-Softmax](22-hidden-state-logits-softmax.md)。
- 若"Embedding 矩阵"不熟，回看 [34-Token-Embedding-与输入表示相加](34-Token-Embedding-与输入表示相加.md)。
- 若"Embedding 矩阵"不熟，回看 [34-Token-Embedding-与输入表示相加](34-token-embedding-input-adding.md)。

## 一、直观含义

- **Backbone（骨干）**：整个 Transformer 主体，学会"通用的语言理解"，但不知道具体要答什么题。
- **Task Head（任务头）**：接在骨干后的"答题器"——根据任务换不同的输出层：
  - 分类 → 输出几个类别的分数。
  - 每个 Token 打标签 → 输出每个 Token 的标签分数。
  - 生成 → 输出下一个词在词表上的分数。
- **Weight Tying**：让"入口查表"和"出口打分"**用同一张表**——省钱又一致。

## 二、具体例子

同一个 BERT 骨干，可以接：

- 序列分类 Head → 情感分类。
- Token 分类 Head → NER。
- QA Head → 抽取式问答。

骨干一模一样，换 Head 就换任务。前提是**新 Head 要训练过才有意义**——随机初始化的 Head 输出毫无任务价值。

## 三、技术要点（严谨部分）

### 1. Backbone（骨干网络）

- 产生通用隐藏状态的 Transformer 主体。
- 负责表示和上下文建模，**不直接规定最终任务格式**。

### 2. Task Head（任务头）

- 接在骨干网络后的任务专用输出层。常见几种：

| Head | 作用 |
|---|---|
| 序列分类头 | 把整段表示映射为类别 Logits |
| Token 分类头 | 把每个 Token 的隐藏状态映射为标签 Logits |
| MLM Head | 把每个位置映射到完整词表，恢复被遮盖 Token |
| 语言模型头（LM Head） | 把隐藏状态映射到词表，预测下一个 Token |
| 问答头（QA Head） | 为每个位置产生答案起点和终点分数 |

- 同一个骨干接不同 Head 可完成不同任务，但**新 Head 若没有经过训练，输出没有实际任务意义**。

### 3. Weight Tying（权重共享）

- 常指**输入 Embedding 矩阵与输出 LM Head 共享参数**：
  - 输入时：用 Embedding 矩阵把 Token ID 映射为向量。
  - 输出时：用**转置方向**把隐藏状态映射回词表分数。
- 好处：减少参数，并让**输入、输出词表表示保持一致**。

### 4. 一个容易混淆的点

- **Attention Head** 在 Transformer Block 内部（见 [39-Self-Attention-与-Multi-Head-Attention](39-Self-Attention-与-Multi-Head-Attention.md)）。
- **Attention Head** 在 Transformer Block 内部（见 [39-Self-Attention-与-Multi-Head-Attention](39-self-attention-multi-head-attention.md)）。
- **Task Head** 在骨干网络之后，面向任务。
- 两者都叫"Head"，但一个是"内部关系计算通道"，一个是"输出适配层"。

## 四、一句话总结

Backbone 是"通用理解者"、Task Head 是"答题器"（分类/标注/生成/问答各换各的）、Weight Tying 让入口出口共享词表参数——同一骨架换不同头，就是不同模型。
