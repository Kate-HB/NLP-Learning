# Week 03 周志：NLP 五大任务实战——从 Token 分类到文本摘要（Day 10–13）

## 概述

本周的核心是理解不同 NLP 任务背后的统一训练骨架。四天覆盖了四类任务：Token 分类（NER）、掩码语言模型微调（MLM）、机器翻译和文本摘要。虽然任务形式不同——有的是给每个 token 贴标签，有的是生成整个序列——但它们共享同一套底层流程：数据准备 → Tokenizer 对齐 → 专用 Collator 组装 batch → 模型前向计算 loss → 任务专用评估指标。

本周最关键的收获不是某个具体 API，而是理解了**标签构造**才是任务之间的本质差异。MLM 的标签是被遮盖的原 token，翻译的标签是目标语言序列，摘要的标签是参考摘要，NER 的标签是每个 token 的 BIO 类别——但一旦 labels 构造正确，剩下的 `Trainer` + `compute_metrics` 流程高度统一。

## Token 分类：子词对齐是核心（Day 10）

### NER 任务与 IOB2 标注

Day 10 使用 CoNLL-2003 数据集完成了 BERT NER 微调。Token 分类的本质是"序列进，序列出"：输入是一串 token，输出是每个 token 的类别标签。NER 使用 IOB2 格式——B- 标记实体开头，I- 标记内部，O 标记非实体。CoNLL-2003 有 9 个标签：O + B/I × 4 类实体（PER/ORG/LOC/MISC）。B/I 前缀不是冗余设计：相邻两个 PER 实体（如 "John Smith and Mary Jones"）依赖 B-PER 区分边界，全标 PER 则无法拆分。

### 子词分词与标签对齐

这是 Day 10 最具迁移价值的技术点。BERT tokenizer 把单词切成子词后（如 "lamb" → "la" + "##mb"），原始按单词给的标签会和 token 序列长度不一致。解决方案：`tokenizer(word_list, is_split_into_words=True)` 处理预分词输入，用 `word_ids()` 获取每个 token 对应的原词索引，再通过 `align_labels_with_tokens()` 完成对齐。三条规则：特殊 token（[CLS], [SEP]）标 -100 被 loss 忽略；同一单词后续子词的 B- 标签改为 I-；所有 padding 位置也标 -100。

### 评估：seqeval 实体级指标

NER 不能用 token 级 accuracy——O 标签占绝大多数，全标 O 就有 90%+ 准确率但毫无意义。seqeval 按实体边界+类别双重匹配计算 precision/recall/F1，更贴近实际需求。`DataCollatorForTokenClassification` 用 -100 填充标签，与之前 `align_labels_with_tokens` 中的 -100 保持一致，共同保证 padding 不计入 loss。

## 掩码语言模型：领域适应的利器（Day 11）

### MLM 的自监督本质

Day 11 在 IMDB 影评上微调 DistilBERT 做 MLM 领域适应。MLM 没有人工标签——输入是随机遮盖后的文本，监督目标就是被遮盖位置的原 token。这属于自监督学习，只需大量无标注领域文本即可训练。

### 数据处理：拼接后分块

与分类任务逐条 tokenize 不同，MLM 先 tokenize 所有文本（不截断），再拼接成一条超长序列，最后按 `chunk_size`（如 512）切成等长块。这种做法最大化 token 利用率，避免截断浪费，并让模型学习跨句子的长程依赖。缺点是同一个 chunk 可能包含来自不同文档的内容，`[SEP]` 标记了边界但上下文本身不连续。

### 动态遮盖与 80-10-10 拆分

`DataCollatorForLanguageModeling(mlm_probability=0.15)` 在每个 batch 组装时动态随机选择 15% 的 token 进行遮蔽。被选中的 token 按 80-10-10 分配：80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持原词。后两种处理减少预训练-微调的分布差异——真实下游输入没有 `[MASK]`。动态遮盖让同一文本在不同 epoch 产生不同训练目标，相当于免费的数据增强。

### 全词掩码与 remove_unused_columns 陷阱

普通 token 级遮盖可能只遮 "##ing" 而留 "play"，模型利用词内线索就能猜出答案。全词掩码（WWM）将同词全部子词一起遮蔽，迫使模型依赖上下文语义。实现依赖 `word_ids()` 建立 word→token 映射。一个棘手的问题：`TrainingArguments` 默认 `remove_unused_columns=True`，Trainer 会删除模型 `forward()` 不接受的列，导致 `word_ids` 在 collator 执行前被丢弃，抛出 KeyError。必须设置 `remove_unused_columns=False`。

### 困惑度

`Perplexity = exp(cross_entropy_loss)` 是语言模型的核心评估指标，值越低越好。IMDB 微调后 perplexity 从 62.55 降到 24.60，说明模型学到了电影评论领域的语言模式。微调后用 `pipeline("fill-mask")` 推理，"The movie is so [MASK]" 预测 "bad"/"great" 而非通用语料的 "deal"/"success"，验证了领域适应有效。

## 机器翻译：Seq2Seq 的完整周期（Day 12）

### Encoder-Decoder 与 Teacher Forcing

Day 12 延续 Token 分类和 MLM 后，进入 Seq2Seq 任务。使用 Helsinki-NLP/opus-mt-en-zh（MarianMT）做英译中。Encoder 双向理解源语言，Decoder 自回归生成目标语言。`text_target` 参数是关键——告诉 tokenizer 目标语言用不同分词规则（英语按空格、中文按字），不传则中文按英语规则处理导致全错。

### decoder_input_ids 的移位机制

`DataCollatorForSeq2Seq` 同时处理输入和标签的填充，并自动从 labels 生成 `decoder_input_ids`（右移一位，开头补 `<s>`）。训练时 decoder 看到的是正确前文（teacher forcing），预测的是下一个 token，所有位置可以并行计算。因果掩码阻止当前位置读取后面的 decoder 输入，二者共同实现"已知前文猜下一步"。

### 评估：SacreBLEU 的中文处理

BLEU 通过 n-gram 匹配衡量翻译质量，但中文没有空格，整句被当作一个 token 导致 n-gram 全为 0。SacreBLEU 的 `tokenize="zh"` 调用 jieba 分词解决此问题。`predict_with_generate=True` 确保评估时调用 `generate()` 模拟真实推理，而非用 teacher forcing 的 labels 直接产生评估结果。

### 微调效果有限的背后

opus-mt-en-zh 预训练已大量使用 OPUS 系列英中数据，再用 opus100 微调，数据分布高度重叠，BLEU 仅从 48.06 提升到 49.52。这说明：想看到明显的微调效果，需要选择预训练模型没见过的新领域数据（法律、医疗、特定行业术语）。

## 文本摘要：双语数据与 baseline 意识（Day 13）

### 从 Hub 脚本到本地加载

Day 13 将 Seq2Seq 骨架从翻译迁移到摘要，使用 mT5 模型。遇到的第一个障碍：`buruzaemon/amazon_reviews_multi` 使用旧版 `.py` 数据集脚本，datasets 5.0.0 拒绝加载（`RuntimeError: Dataset scripts are no longer supported`）。解决方案：通过 `huggingface-cli download` 直接下载原始 `.jsonl.gz` 文件，再用 `load_dataset("json", data_files={...})` 加载。Windows 上的符号链接问题（`huggingface-cli download` 创建的相对路径链接可能断裂）需要改用 `cat` 写入实际文件内容。

### 双语数据的不同处理策略

中英文数据不能套用同一规则。过滤短标题：英文 `split()` 按空格分词数判断（`> 2`），中文没空格，改 `len()` 字符数判断（`> 5`）。句子分割：英文 `nltk.sent_tokenize()` 处理缩写，中文 `re.split(r"[。！？；]")` 按标点切分。合并中英文数据后用 `concatenate_datasets()` + `shuffle()` 防止模型偏向单一语言，并保留 `language` 字段用于后续分叉处理。

### Lead-3 Baseline 的意义

在跑模型之前，先用"取前三句作为摘要"这种简单规则建立 baseline。如果复杂模型连这个都打不过，说明训练有问题。这是 NLP 项目的标准流程：先 baseline，再尝试复杂方案，确保投入有回报。

### ROUGE 评估指标

ROUGE 通过 n-gram 重叠衡量生成摘要与参考摘要的匹配度。rouge1 测单词级重叠，rouge2 测二元词组重叠，rougeL 测最长公共子序列。同时报告精确度、召回率和 F1。课程使用旧版 `AggregateScore` 对象（含 `.mid` 属性），当前环境中 ROUGE 返回值直接是数值，无 `.mid`。

### Seq2Seq 的通用骨架

从 Day 12 翻译到 Day 13 摘要，代码骨架几乎相同：`AutoModelForSeq2SeqLM` → `DataCollatorForSeq2Seq` → `Seq2SeqTrainingArguments(predict_with_generate=True)` → `Seq2SeqTrainer` → 自定义 `compute_metrics()` 解码后算指标。差异仅在三处：数据、max_input_length/max_target_length、评估指标（BLEU vs ROUGE）。`processing_class=tokenizer` 取代已废弃的 `tokenizer=` 参数。

## 已解决问题

本周解决的主要技术问题：

- **datasets 5.0.0 脚本拒绝**（Day 10/13）：CoNLL-2003 用 `revision="refs/convert/parquet"` 解决；amazon_reviews_multi 无 Parquet 分支，改用 `huggingface-cli download` + `load_dataset("json")` 绕过。
- **Windows 符号链接断裂**（Day 13）：`huggingface-cli download` 创建的相对路径链接在 Windows 上指向错误位置，用 `cat` 复制实际文件内容替代。
- **KDE4 数据集不可用**（Day 12）：改用 `opus100`（完全 Parquet 格式，1M 英中平行语料）。
- **SacreBLEU 中文全 0**（Day 12）：`tokenize="zh"` 调用 jieba 分词。
- **ROUGE AggregateScore 变化**（Day 13）：新版 evaluate 返回数值，无 `.mid` 属性，`value.mid.fmeasure` → `value`。
- **NLTK punkt_tab 缺失**（Day 13）：`nltk.download("punkt_tab")`。
- **`evaluation_strategy` 重命名**（Day 10/11/12/13）：全周统一使用 `eval_strategy`（Transformers 4.57.3）。
- **`tokenizer=` 废弃**（Day 13）：`Trainer` 改用 `processing_class=tokenizer`。
- **`remove_unused_columns` 导致 `word_ids` 丢失**（Day 11）：WWMM collator 需设置 `remove_unused_columns=False`。
- **Windows 路径与 GBK 编码**（Day 13）：`huggingface-cli download` 在系统 Python 下因 GBK 编码 emoji 崩溃，改用 Conda Python + `PYTHONIOENCODING=utf-8`。

## 下一步

Week 04 进入 GPT 与生成：Decoder-only 架构、自回归生成、贪婪搜索 vs beam search vs top-k/top-p 采样策略、温度参数、KV 缓存机制。这些是理解 ChatGPT 类模型如何"说话"的基础，也是后续 Week 08 LLM 应用整合的前置知识。

下周预计产出：

```text
notes/day14.md — day18.md
notebooks/day14.ipynb — day18.ipynb
06-GPT/ （GPT 生成与解码主题文档）
```
