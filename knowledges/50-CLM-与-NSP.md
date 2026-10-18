# 50 CLM 与 NSP

> 一句话：CLM 用前文预测下一个 Token（Decoder-only 的核心目标）；NSP 判断第二句是不是第一句的后续（BERT 的辅助目标）。


## 前置知识

- 先读 [46-自回归生成](46-自回归生成.md)、[49-自监督学习-与-MLM](49-自监督学习-与-MLM.md)。
- 先读 [46-自回归生成](46-autoregressive-generation.md)、[49-自监督学习-与-MLM](49-self-supervised-learning-mlm.md)。

## 一、直观含义

- **CLM（因果语言建模）**：就是"接龙"——看到"我爱自"，预测下一个字"然"。**只能看过去，不能看未来**，所以天然适合 Decoder-only。
- **NSP（下一句预测）**：给两句话，问"第二句是不是紧接第一句？"——比如 A="我去超市了"，B="买了苹果" → 是；B="恐龙灭绝了" → 不是。

## 二、具体例子

- CLM 训练样本：`"自然语言处理很有趣"`，每个位置都用"前文预测后文"：看到"自然语" → 预测"言"，看到"自然语言" → 预测"处"……
- NSP 训练样本：`[CLS] 我去超市了 [SEP] 买了苹果 [SEP]` + 标签 `IsNext`。

## 三、技术要点（严谨部分）

### 1. CLM（因果语言建模）

- 让位置 t 的表示预测**位置 t+1** 的 Token。
- "因果"表示当前预测只能依赖**过去**，不能依赖未来。
- 通过**因果掩码**（看不了未来）和**标签位置对齐**（位置 t 的 Logits 对齐位置 t+1 的标签）实现。
- 是 **Decoder-only 自回归模型的核心预训练目标**。

### 2. NSP（下一句预测）

- 原始 BERT 使用的预训练目标之一。
- 模型看到两个句段，判断第二段是否真的是第一段的后续内容。
- 试图让模型学习**句段之间的关系**，也解释了原始 BERT 为什么使用句段 ID（Token Type ID，见 [17-Attention-Mask-与-Token-Type-ID](17-Attention-Mask-与-Token-Type-ID.md)）。
- 试图让模型学习**句段之间的关系**，也解释了原始 BERT 为什么使用句段 ID（Token Type ID，见 [17-Attention-Mask-与-Token-Type-ID](17-attention-mask-token-type-id.md)）。
- ⚠️ 后续模型并不都保留 NSP；句段 ID 和句子对能力也不能简单归结为这一个目标。

### 3. CLM vs MLM 对照

| | MLM | CLM |
|---|---|---|
| 猜什么 | 被遮盖的 Token | 下一个 Token |
| 能看哪些 | 左右两侧 | 只能看左侧前文 |
| 训练谁 | Encoder（双向理解） | Decoder（自回归生成） |
| 代表 | BERT | GPT |

### 4. 对齐机制的一个提醒

- CLM 的"位置 t 预测位置 t+1"是通过**损失里的对齐**实现的，不是"把 Logits 向右移动"（那是错误表述，详见 [51-Seq2Seq-Teacher-Forcing-与目标右移](51-Seq2Seq-Teacher-Forcing-与目标右移.md)）。
- CLM 的"位置 t 预测位置 t+1"是通过**损失里的对齐**实现的，不是"把 Logits 向右移动"（那是错误表述，详见 [51-Seq2Seq-Teacher-Forcing-与目标右移](51-seq2seq-teacher-forcing-label-shift.md)）。

## 四、一句话总结

CLM 是"看过去猜下一个"，练的是 Decoder 的生成能力；NSP 是"猜下一句"，曾是 BERT 的辅助目标；MLM 双向、CLM 单向，是训练目标的两极。
