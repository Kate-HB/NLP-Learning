# 60 Token 分类：NER、POS、Chunking 与 IOB2

> 一句话：Token 分类给每个 Token 打标签，常见任务有 NER、POS、Chunking；IOB2 用 B/I/O 三种前缀标注实体边界。


## 前置知识

- 先读 [48-Backbone-Task-Head-与-Weight-Tying](48-Backbone-Task-Head-与-Weight-Tying.md)。
- 先读 [48-Backbone-Task-Head-与-Weight-Tying](48-backbone-task-head-weight-tying.md)。
- 若"子词与词的标签对不上"需要解决，见 [61-子词标签对齐](61-子词标签对齐.md)。
- 若"子词与词的标签对不上"需要解决，见 [61-子词标签对齐](61-subword-label-alignment.md)。

## 一、直观含义

- **Token 分类**：序列分类管"整句一个结论"，Token 分类管"**每个词各自一个标签**"。
- **NER**：找出"人名、地名、机构名"这些实体及边界。
- **POS**：给每个词标"名词、动词、形容词"。
- **Chunking**：标出"名词短语、动词短语"等局部结构。
- **IOB2**：标注格式——B 表示实体开头、I 表示实体内部、O 表示不属于实体。

## 二、具体例子

NER 标注"王小明在北京的谷歌工作"：

```
王小明  北京  的  谷歌  工作
B-PER  B-LOC O  B-ORG  O
```

- "王小明"是人物开头（B-PER）；"北京"是地点（B-LOC）；"谷歌"是机构（B-ORG）。
- 如果"谷歌中国"是一个机构实体，则标 `B-ORG I-ORG`。

## 三、技术要点（严谨部分）

### 1. Token 分类（Token Classification）

- 为**每个 Token** 输出一个标签。
- 输出长度通常与输入 Token 序列长度对应。
- 适合：NER、POS、Chunking。
- 关键难点：必须处理**原始词标签与子词 Token 之间的对齐**（见 [61-子词标签对齐](61-子词标签对齐.md)）。
- 关键难点：必须处理**原始词标签与子词 Token 之间的对齐**（见 [61-子词标签对齐](61-subword-label-alignment.md)）。

### 2. NER（命名实体识别）

- 识别文本中的**实体边界和类别**：人名、组织、地点、日期、时间、金额、产品等。
- 识别的对象由**数据集的实体类型体系**决定，不只限于专名。
- 仅知道某个 Token 属于"人名"还不够，还要知道多个 Token 是否属于**同一个实体**——所以需要**边界标注**。

### 3. POS（词性标注）

- 为词标注名词、动词、形容词等**语法类别**。

### 4. Chunking（句法分块）

- 识别**非递归的短语块**，如名词短语、动词短语。
- 比完整句法树简单，但比单纯词性标注更关注局部结构。

### 5. IOB2 标注

- IOB = **I**nside（内部）、**O**utside（外部）、**B**eginning（开头）。
- B（Begin）：一个实体的**第一个** Token。
- I（Inside）：同一实体**内部**的后续 Token。
- O（Outside）：**不属**于任何实体。
- **即使两个相邻实体类别相同，第二个实体也必须重新从 B 开始**，才能区分边界。例："北京北京"两个相邻地名 → `B-LOC B-LOC`，不是 `B-LOC I-LOC`。
- IOB2 中的"2"用于区别早期 IOB1：IOB2 要求**每个实体无论前面是什么都从 B 开始**，边界更统一、直观。
- 常见数据集 CoNLL-2003 共 9 个标签：O + B/I × 4 类（PER、ORG、LOC、MISC）。

## 四、一句话总结

Token 分类是"一词一标签"：NER 找实体和边界、POS 标词性、Chunking 标短语块；IOB2 用 B 开头、I 内部、O 外部把实体边界标清楚。
