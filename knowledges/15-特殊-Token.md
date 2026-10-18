# 15 特殊 Token

> 一句话：特殊 Token 不是普通文本内容，而是给模型提供"结构信号"的标记，例如开头、分隔、补齐、未知、遮盖、结束。


## 前置知识

- 先读 [09-Token-与-Tokenizer](09-Token-与-Tokenizer.md)。
- 先读 [09-Token-与-Tokenizer](09-token-and-tokenizer.md)。

## 一、直观含义

模型只看到一串数字，分不清"这是第几句、这句话到哪结束、哪里是补的空白"。特殊 Token 就是给模型发的"信号灯"：

- 打个招呼说"我开始啦"（BOS）
- 说"这一句结束"（SEP / EOS）
- 说"这里是占位符，别当真"（PAD）
- 说"这词我不认识"（UNK）
- 说"这里被遮住了，你来猜"（MASK）

## 二、具体例子

BERT 处理句子对："我爱猫" + "我也爱狗" → 实际输入：

```
[CLS] 我爱猫 [SEP] 我也爱狗 [SEP]
```

- `[CLS]` 在开头，它的最终表示可代表整段。
- `[SEP]` 分隔两个句子。
- 若句子更长，后面还会补 `[PAD] [PAD] ...` 对齐到批内最长。

## 三、技术要点（严谨部分）

常见的特殊 Token：

| Token | 全称 | 作用 |
|---|---|---|
| CLS | Classification Token | 放在序列开头，BERT 类模型可用其最终隐藏状态表示整段输入，用于分类 |
| SEP | Separator Token | 表示序列结束或两个句子的分隔 |
| PAD | Padding Token | 把不同长度序列补到相同长度 |
| UNK | Unknown Token | 表示词表无法表达的未知内容 |
| MASK | Mask Token | 在掩码语言建模中替代被遮盖的 Token |
| BOS | Beginning of Sequence | 表示目标序列开始 |
| EOS | End of Sequence | 表示序列结束；生成模型学会生成它即可停止 |

### 必须注意的两点

1. **不同模型定义和使用方式不同**。有的模型没有 CLS、有的不用 SEP；同一名称不代表功能完全等价，不能只凭名字假设。
2. **PAD Token 与 Attention Mask 配合**：PAD 不是普通内容，模型必须知道"这些位置无效"——这由 Attention Mask 标记（见 [17-Attention-Mask-与-Token-Type-ID](17-Attention-Mask-与-Token-Type-ID.md)）。

## 四、一句话总结

特殊 Token 是输入里的"结构标点"——告诉模型哪里开始、哪里结束、哪里是补位、哪里被遮盖；不同模型对这些信号的定义各不相同。
