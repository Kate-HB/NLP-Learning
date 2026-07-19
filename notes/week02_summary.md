# Week 02 周志：BERT 微调实战与 Tokenizers 库深度剖析（Day 05–09）

## 概述

本周的核心是从"调用 API"过渡到"理解每一行训练代码在做什么"。前三天（Day 05–07）围绕 BERT 微调的完整流程展开——从 Trainer 一键训练到手动拆解 PyTorch 训练循环，再到 Datasets 库的数据处理与语义搜索实践。后两天（Day 08–09）深入 Tokenizers 库，依次复现了 NER、QA 管道的完整内部流程，并手写了 BPE、WordPiece、Unigram 三种子词算法的训练与推理代码。本周的实践密度显著高于 Week 01，四天之间完成了 8 个 Notebook 实验和 4 篇主题文档。

## BERT 微调：从 Trainer 到手动训练循环（Day 05–06）

### Trainer 全流程

Day 05 使用 GLUE/MRPC 数据集完成了 BERT 微调的完整流程。MRPC 是句子对分类任务（判断两个句子是否语义等价），包含 3668/408/1725 三集，标签为 0（不等价）和 1（等价）。BERT 处理句子对的方式是将两句拼接为 `[CLS] A [SEP] B [SEP]`，并用 `token_type_ids` 区分两句（0 表示第一句，1 表示第二句），这一设计源于 BERT 预训练的"下一句预测"任务。DistilBERT 等蒸馏模型没有 NSP 预训练，因此没有 `token_type_ids` 层。

数据预处理的关键是 `Dataset.map()` 配合 `batched=True`：Rust 实现的快速 tokenizer 能在批处理模式下大幅加速，Apache Arrow 格式的惰性加载避免了一次性读入内存，`num_proc` 参数可启用多进程并行。动态填充由 `DataCollatorWithPadding` 实现——tokenize 时不填充到全局最大长度，而是在每个 batch 内部填充到该 batch 的最长序列，显著节省计算量。Trainer 的组装顺序是 `TrainingArguments`（必传 `output_dir`，`eval_strategy="epoch"` 是 4.57 后的新参数名）→ `AutoModelForSequenceClassification`（随机初始化分类头，`num_labels=2`）→ `Trainer` → `train()` → `predict()`。评估时 `np.argmax(logits, axis=-1)` 取最大 logit 的索引作为预测标签，再用 `evaluate.load("glue", "mrpc")` 计算 accuracy 和 F1。bert-base-uncased 训练 3 个 epoch 后达到 accuracy 0.833、F1 0.885。

一个容易踩的坑：训练完后重新执行 `AutoModelForSequenceClassification.from_pretrained(checkpoint)` 会导致分类头回到随机初始化状态，之前训练的权重全部丢失。正确的做法是 train 后立刻 evaluate，或从保存的 checkpoint 目录加载。

### 手写训练循环

Day 06 拆开了 Trainer 的黑箱，逐行还原 PyTorch 训练循环。五步标准流程为：`loss = model(**batch).loss` → `loss.backward()` → `optimizer.step()` → `scheduler.step()` → `optimizer.zero_grad()`。数据准备方面，先用 `remove_columns` 和 `rename_column` 整理数据集，再用 `set_format("torch")` 将列转为 PyTorch tensor，最后通过 `DataLoader` + `collate_fn` 生成 batch。AdamW 是 Adam 加权重衰减正则化的变体，是 BERT 微调的事实标准优化器。学习率调度器采用线性衰减策略，总步数 = epochs × `len(train_dataloader)`，学习率从初始值（如 5e-5）线性衰减到 0。batch size 和学习率存在线性缩放关系：batch size 翻倍时学习率通常也应翻倍，以保持有效步长一致。

评估循环的模式是：`model.eval()` + `torch.no_grad()` 关闭 dropout 和梯度计算，然后逐 batch 用 `metric.add_batch()` 累积预测结果，最后 `metric.compute()` 汇总。Accelerate 库的 `accelerator.backward()` 和 `prepare()` 将分布式训练的细节（梯度平均、设备分发）封装起来，而 Trainer 内部已经集成了 Accelerate，这是其能自动适配多卡的原因。分布式训练的两种范式：数据并行（每张卡持有完整模型副本，梯度在卡间同步后平均）适合 BERT-base 这种单卡装得下的模型；模型并行（将模型切分到不同卡上）适用于参数超出单卡显存的大模型。

## Datasets 库与语义搜索（Day 07）

Day 07 系统学习了 HuggingFace Datasets 库的数据处理能力，并用 GitHub Issues 数据集完成了一个端到端的语义搜索实践。

### 数据加载与处理

`load_dataset()` 支持三种加载方式：本地文件（CSV/JSON/JSONL/Parquet 等）、远程 URL 和 HuggingFace Hub 数据集。`data_files` 参数非常灵活，可以传入单文件路径、文件列表或 `{"train": "train.json", "test": "test.json"}` 字典来指定不同 split 的数据源。即使相同结构的 JSON 文件，手动指定 `field` 参数也能避免 schema 推断冲突。数据处理的四大操作：`filter()` 按条件筛选行，`map()` 对每行应用函数（`batched=True` 配合 Rust tokenizer 可提速数十倍），`rename_column` 和 `remove_columns` 整理列结构，`add_column` 追加新列。

大数据场景下，Datasets 使用 Apache Arrow 内存映射机制——数据不全部加载到 RAM，磁盘上的文件直接映射到虚拟内存，操作系统按需换入换出。流式处理（`streaming=True`）返回 `IterableDataset`，适合远超内存的数据集，代价是无法随机访问和 shuffle。

### 语义搜索

语​​义搜索的完整流程是：用预训练模型将文本编码为向量（embedding），在向量空间中用相似度搜索找到语义相近的文档。具体实现中，使用 BERT 的 CLS 位置输出作为整句向量，通过 FAISS 建立高效索引（`dataset.add_faiss_index("embeddings")`），再调用 `get_nearest_examples()` 返回最相似的文档。FAISS 的核心优势在于近邻搜索的高效实现，不依赖 GPU（`faiss-gpu` 在 Windows 上不可用时 fallback 到 CPU 版本）。批量获取 embedding 比逐条编码快数十倍。

实践过程中解决了一个典型的工程问题：GitHub Issues API 未认证时请求频率极低，通过 `python-dotenv` 加载 `.env` 文件中的 Bearer token 后恢复正常的请求频率。从零创建数据集的标准流程是：`requests.get()` 获取数据 → 写入 JSONL 文件 → `Dataset.from_pandas()` 或 `load_dataset("json")` 加载为 Dataset → `push_to_hub()` 上传到 Hub。

## Tokenizers 库深度剖析（Day 08–09）

### 快速 Tokenizer 的内部机制

Day 08 聚焦于快速 tokenizer（Rust 实现）相比慢速 tokenizer（纯 Python）的额外能力。`AutoTokenizer.train_new_from_iterator()` 可以从自定义语料库训练新 tokenizer，它保留原 tokenizer 的算法（如 WordPiece）但替换词表。tokenizer 训练与模型训练有本质区别：前者是统计过程（确定性的），反复对语料库统计词频和合并规则；后者是随机梯度下降（随机性的），每次训练结果可能不同。

BatchEncoding 对象（`tokenizer()` 的返回值）是字典的子类，提供了一系列辅助方法：`tokens()` 返回分词后的 token 列表，`word_ids()` 将每个 token 映射到它所属的原始单词索引（用于序列标注任务中把 token 级预测对齐到 word 级），`word_to_chars()` 和 `token_to_chars()` 返回单词或 token 在原始文本中的起止偏移量。偏移映射中有一个反直觉的细节：偏移量始终是原始文本的绝对位置，空格字符占据一个字符位但不一定被任何 token 覆盖，这是正常现象而非 bug。

NER（命名实体识别）管道的完整复现过程揭示了 pipeline 的内部逻辑：模型使用 `AutoModelForTokenClassification`，输出每个 token 在 9 个标签上的 logits（CoNLL-2003 使用 IOB2 格式：O 表示非实体，B/I 分别标记实体的开始和内部，PER/ORG/LOC/MISC 四类实体共 9 个标签）。后处理步骤是所有 token 级的预测需要通过偏移量分组成完整的实体——例如相邻的 B-PER 和 I-PER 合并为一个 "PER" 实体，并关联到对应的原始文本片段。

QA（问答）管道的逻辑则更复杂一层：模型输出 start logits 和 end logits（每个 token 位置的两个分数），通过 `sequence_ids()` 将 question 部分的 token 屏蔽（因为答案不可能出现在问题中），然后在 passage 部分对每个可能的起止组合打分，取总分最高的片段。对于超出模型最大长度的文本，使用 `return_overflowing_tokens=True` + `stride` 参数进行滑动窗口分块处理，每块独立预测后再合并结果。

### 子词算法的预分词差异

标准化（Normalization）和预分词（Pre-tokenization）是在 BPE/WordPiece 等子词算法之前必须完成的两步。标准化处理 Unicode 规范化（如 NFD）、大小写转换、去重音等；预分词确定"初始边界"——即哪些字符组是"不可再分割的初步单元"。BERT、GPT-2、T5 三者的预分词策略截然不同：BERT 的 `BertPreTokenizer` 以空格和标点为分割边界（`Let's` → `Let` + `'` + `s`），GPT-2 的字节级预分词器不按标点分（`Let's` 保持为一个单元），T5 的 `Unigram` 预分词器同样较为松散。预分词策略直接影响最终词表的形态和分词习惯，是选择 tokenizer 时需要了解的关键设计决策。

### BPE 算法原理与手写实现

BPE（Byte-Pair Encoding）最初是一种文本压缩算法，后来被 OpenAI 用于 GPT 系列的 tokenization。其核心思想是从字符级词汇表出发，反复找出语料库中出现频率最高的相邻 token 对，将该对合并为一个新 token 加入词表，并在所有单词的切分中执行这一合并。训练过程是确定性的贪心算法，每一步选择的都是当前统计下的全局最优对。

手写实现中的核心数据结构是四个：`word_freqs`（`defaultdict(int)`，每个单词在语料库中的出现次数）、`splits`（每个单词当前被切分成的 token 列表，初始为字符列表）、`pair_freqs`（所有相邻 token 对的频率统计，同样用 `defaultdict(int)` 避免 KeyError）、`merges`（学到的合并规则字典，key 为 pair 元组，value 为合并后的字符串）。训练循环每步调用 `compute_pair_freqs()` 统计频率 → 选出 `best_pair` → 调用 `merge_pair()` 全局执行合并 → 记录 `merges[best_pair] = best_pair[0] + best_pair[1]` → 将新 token 追加到 `vocab`。推理时对输入文本逐字符拆分，按 `merges` 中记录的顺序依次应用合并规则，最后通过 `sum(splits, [])` 将嵌套列表拍平返回最终的 token 列表。GPT-2 的特殊之处在于使用字节级 BPE——初始词表只有 256 个字节值，理论上能编码任意 Unicode 字符而不出现 UNK。

### WordPiece 算法

WordPiece 是 BERT 采用的分词算法，与 BPE 形式相似但选择合并的标准不同：BPE 选频率最高的对，WordPiece 选能使训练语料似然度提升最大的对。从公式角度看，WordPiece 比较的是 `P(token_a + token_b) / (P(token_a) × P(token_b))`，这个比值越大说明两个 token 的共现不是独立事件，合并它们能更好地建模语言。BERT 还沿用了 `##` 前缀标记续接子词——BERT tokenizer 的 `tokenize()` 方法会在非词首的子词前添加 `##`，而 `decode()` 会自动去除这些标记并还原连续文本。直接调用 `tokenizer(text)` 时不会出现 `##`，因为内部调用了 `convert_tokens_to_ids()` 和 `decode()` 的完整链路。

### Unigram 算法与 Viterbi 解码

Unigram 的思想与 BPE/WordPiece 完全相反：它从一个大词表出发，假设每个 token 独立出现，计算每种可能分词方式的概率（各子词概率的乘积），选择概率最高的分词作为结果。在"pug"的例子中，`["pu", "g"]` 和 `["p", "ug"]` 的概率分别是 `P("pu") × P("g")` 和 `P("p") × P("ug")`，实际计算中分词数较少的方案通常概率更高，这符合直觉。

Viterbi 算法通过动态规划高效求解最优分词。数据结构 `best_segmentations` 是一个长度为 `len(word)+1` 的列表，索引 `i` 处存储前缀 `word[0:i]` 的最优分词信息，包含两个关键字段：`start`（最优分词中最后一个 token 的起始位置）和 `score`（该前缀的最优累积得分，使用负对数概率以加法替代乘法）。填充过程使用双重循环：外层 `start_idx` 遍历每个可能的起始位置，内层 `end_idx` 从 `start_idx+1` 扫描到末尾，对每个在词表中的子串 `word[start_idx:end_idx]` 计算 `model[token] + best_score_at_start`，如果优于 `best_segmentations[end_idx]` 的现有记录则更新。全部填完后，从 `len(word)` 位置出发，利用每个位置的 `start` 字段一步步往前跳，每次提取 `word[start:end]` 并更新 `end = start`，直到回到位置 0，完成完整分词的回溯重建。

Unigram 训练的剪枝策略同样基于 loss：初始词表远大于目标大小，每轮计算删除每个 token 对 loss 的影响（`compute_loss(model_without_token) - compute_loss(model)`），逐批删除影响最小的 token 直到词表缩至目标大小。一个 token 的 loss 影响取决于它是否被语料库中任何词的最优分词所使用——例如 "ll" 被 "Hopefully" 的分词 `["H","o","p","e","f","u","ll","y"]` 使用，删除后只能退化为两个 `"l"`，loss 上升；而 "his" 虽在词表中，但 "This" 的最优分词是直接作为完整 token `["This"]`，删除 "his" 的 loss 变化为零。

### Tokenizers 库的模块化流水线

Day 09 最后一部分使用 HuggingFace 的 `tokenizers` 库（Rust 实现，Python 绑定）从零构建了一个完整的 BERT tokenizer。库将分词过程拆分为五个可独立替换和组合的组件：

- **Normalizer**：Unicode 规范化（NFD/NFC/NFKD/NFKC）、大小写转换、去重音。BertNormalizer 内置了这些功能。必要时还可以用 `Sequence` 组合多个正规化步骤。
- **PreTokenizer**：确定初始切分边界。`Whitespace()` 的命名有误导性——它在空白符和所有非字母/数字/下划线的字符处都分割，功能上是 `WhitespaceSplit() + Punctuation()` 的超集。如果只想按空白分割，应使用 `WhitespaceSplit()`。BERT 使用 `BertPreTokenizer`，它将标点也作为分割边界。
- **Model**：核心分词算法与词表。注意这里"Model"不是 Transformer 神经网络，而是 `models.BPE()`、`models.WordPiece()`、`models.Unigram()` 这类分词模型，负责存储词表和执行 token ↔ ID 转换。
- **PostProcessor**：添加特殊 token，如 BERT 的 `[CLS]` 和 `[SEP]`。使用 `TemplateProcessing` 定义拼接模板。
- **Decoder**：将 token ID 序列还原为文本字符串。

Trainer 配置（如 `WordPieceTrainer`）需要传入 `vocab_size`、`special_tokens`（必须传入，否则训练语料中没有的特殊 token 不会被加入词表）、`min_frequency` 等参数。训练通过 `tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)` 完成，语料通过生成器函数提供以支持多次遍历。训练完成后，用 `PreTrainedTokenizerFast` 或 `BertTokenizerFast` 包装 tokenizer 对象，即可获得完整的 Transformers 兼容 tokenizer，支持 `save_pretrained()` 和 `push_to_hub()`。

## 已解决问题

本周解决的主要技术问题：`evaluation_strategy` 在 Transformers 4.57 中重命名为 `eval_strategy`；Accelerate 未安装导致 Trainer 报错（`pip install accelerate>=0.26.0`）；训练后重新执行模型定义 cell 导致分类头权重丢失；GitHub API 未认证限流（`python-dotenv` + `.env` + Bearer token）；Windows `num_proc` 的 spawn 问题（`if __name__ == "__main__"` 保护）；JSONL schema 冲突（指定 `field` 参数）；`faiss-gpu` 在 Windows 不可用（fallback 到 CPU 版）；交叉环境 pip 安装问题（`D:/Soft/Conda/python.exe -m pip install` 明确指定路径）；以及 padding 后 logits 不一致的根因分析（attention mask 缺失）。

## 下一步

Week 03 进入 PyTorch 最小训练基础：从 Tensor 操作、autograd 自动求导、nn.Module、Dataset/DataLoader 到手写完整训练循环的逐项击破。这些是后续所有模型训练（无论 BERT 微调还是 LLM 应用）不可绕过的代码基础。

下周预计产出：

```text
notes/day10.md — day14.md
notebooks/day10.ipynb — day14.ipynb
02-PyTorch/ （PyTorch 主题文档）
```
