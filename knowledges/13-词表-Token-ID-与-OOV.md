# 13 词表、Token ID 与 OOV

> 一句话：词表是 Tokenizer 认识的全部单词，Token ID 是它们在词表中的整数编号；词表外的词（OOV）要么变 UNK、要么靠子词拆开；模型和 Tokenizer 必须共享同一份词表。


## 前置知识

- 先读 [09-Token-与-Tokenizer](09-Token-与-Tokenizer.md)、[10-三种分词粒度](10-三种分词粒度.md)。
- 先读 [09-Token-与-Tokenizer](09-token-and-tokenizer.md)、[10-三种分词粒度](10-three-tokenization-granularities.md)。

## 一、直观含义

想象一本"单词→编号"的字典：

- **词表**：字典里所有词。
- **Token ID**：每个词在字典里的页码（整数编号）。
- **OOV（未登录词）**：字典里没收录的词。查不到怎么办？要么给个通用符号"查无此词"（UNK），要么用子词把它拆开，让拆出来的碎片都在字典里。

关键点：**ID 本身没有任何语义**。编号 5 在 BERT 词表里是"好"，在别的词表里可能是"bye"。所以"模型 + Tokenizer 必须用同一本字典"，否则牛头不对马嘴。

## 二、具体例子

- 词表里有 `{"[PAD]":0, "[UNK]":1, "[CLS]":2, ... "trans":5200, "##former":5201, ...}`。
- "transformer" → `[5200, 5201]`。
- 一个生僻英文词若词表里拆不开 → 映射成 `[UNK]`（ID=1），信息直接丢失。
- 用同一个词表、但编号方案不同的两个系统互相喂数据 → 模型看到完全无关的向量，结果全错。

## 三、技术要点（严谨部分）

### 1. 词表（Vocabulary）

- Tokenizer 能识别的**全部 Token 集合**。

### 2. Token ID

- Token 在词表中的**整数编号**。
- 神经网络不能直接处理字符串，所以先用 ID 去查对应向量（这一步在 [34-Token-Embedding-与输入表示相加](34-Token-Embedding-与输入表示相加.md) 讲）。
- 神经网络不能直接处理字符串，所以先用 ID 去查对应向量（这一步在 [34-Token-Embedding-与输入表示相加](34-token-embedding-input-adding.md) 讲）。
- **Token ID 本身没有自然语义**——同一个数字在不同词表里代表完全不同的 Token。

### 3. OOV（Out-of-Vocabulary，未登录词）

- 词表中没有的语言单位。
- 不同粒度对 OOV 的处理：
  - **基于词的分词**：常把 OOV 整体映射为未知 Token（UNK），信息丢失。
  - **子词 / 字节级方法**：可把 OOV 拆成已知子词，显著减少 OOV。
  - 字符级：几乎不存在 OOV，但序列太长。

### 4. 一致性的铁律

> 模型与 Tokenizer 必须共享同一套**词表**和**特殊 Token 定义**。

因为 ID 的映射、子词切分算法、特殊 Token 的编号都必须一致，模型的 Embedding 查表才能查到正确的向量。

## 四、一句话总结

词表决定 Tokenizer 认识什么，ID 是给模型的编号，OOV 是词表的边界——而"模型和 Tokenizer 用同一本字典"是所有任务不出错的前提。
