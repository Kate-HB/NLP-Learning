# 项目记忆

> 用于后续继续学习时快速恢复上下文。只记录稳定信息，不保存账号、Token、密码等敏感数据。

## 学习者

- NLP 初学者
- 目标：通过 Hugging Face 和项目实践系统学习 NLP
- 学习方式：概念学习、Notebook 实验、Markdown 笔记、项目产出结合
- 解释要求：先讲直观含义，再给示例，最后说明技术原理
- 回复要求：中文、直接、简洁；不提供无关方案

## 项目

```text
C:\Users\27729\Desktop\NLP-Learning
```

主学习路线为 8 周：系统理解 NLP，并用项目验证每个阶段。

1. Hugging Face 入门与 Tokenizer
2. PyTorch 最小训练基础
3. 深度学习概念与 Transformer
4. BERT 微调与中文文本分类项目
5. 传统 NLP、BERT 与序列标注
6. GPT、生成与解码策略
7. Embedding、语义检索与 RAG 项目
8. LLM 应用整合项目

详细周计划位于：

```text
resources/weeks/week01.md
...
resources/weeks/week08.md
```

总路线位于：

```text
resources/roadmap.md
```

## 当前环境

项目使用 D 盘 Conda `base` 环境：

```text
Python: D:\Soft\Conda\python.exe
Python version: 3.13.9
PyTorch: 2.11.0+cu128
CUDA Runtime: 12.8
CUDA available: True
Transformers: 4.57.3
Datasets: 5.0.0
Accelerate: 1.14.0
```

关键约束：

- VS Code 解释器固定为 `D:\Soft\Conda\python.exe`
- Notebook Kernel 也必须选择该解释器
- Transformers 固定为 `4.57.3`，用于兼容课程 Pipeline 示例
- 安装或切换包版本后必须重启 Notebook Kernel
- 当前 `pip check` 无依赖冲突

环境文档：

```text
00-Environment/env.md
```

## 当前进度

Day 01 已完成：

- Conda、VS Code、PyTorch CUDA 环境验证
- Hugging Face Hub 登录
- Jupyter Notebook 基础使用
- `pipeline()` 工作流程
- 情感分析 Pipeline
- 文本生成、掩码填充、Token 分类和问答任务体验
- 预训练、微调和迁移学习概念
- Transformer 架构入门
- 已学习 Hugging Face LLM Course Chapter 1/4

Day 02 已推进：

- 学习 Encoder-only、Decoder-only、Encoder-Decoder 三类模型
- 理解 BERT 的 MLM：完整句子先存在，再随机遮盖部分 token，用原词作为监督答案
- 区分 BERT 的 `[MASK]` token 和 Decoder 的 attention mask
- 理解因果注意力掩码：未来位置 logits 被设为极小值，softmax 后注意力权重接近 0
- 理解隐藏状态：模型内部对 token 的上下文化向量表示
- 理解 logits：模型对词表中每个候选 token 的原始分数，不是概率
- 修正 GPT-2 训练表述：不是 “logits 向右移动”，而是用当前位置 logits 预测下一个 token
- 学习语言建模头：把隐藏状态线性变换为词表 logits
- 学习摘要、翻译、Whisper、ViT、LLM 推理、采样策略、KV 缓存等概念
- 理解图片 `summary.png`：Encoder 双向理解，Decoder 自回归生成
- 理解图片 `注意力掩码示例.png`：绿色表示可见位置，白色表示被 mask 的位置
- 遇到并处理音频任务依赖：`ffmpeg was not found`，原因是音频解码需要系统 ffmpeg
- 已整理 `04-Transformer/transformer-basics.md`
- 已整理 `notes/day02.md`，并在末尾追加“今日知识总结”

Day 03 已推进：

- 学习 Hugging Face Chapter 2.2 到 2.5 的核心内容
- 拆解 `pipeline()`：tokenizer 预处理、model 计算、后处理
- 理解 `checkpoint`：模型文件地址，不是模型对象
- 理解 `AutoTokenizer.from_pretrained(checkpoint)` 和 `AutoModel.from_pretrained(checkpoint)` 的区别
- 理解 `AutoModel` 输出 hidden states，`AutoModelForSequenceClassification` 输出 logits
- 理解 hidden states 形状：`batch_size, sequence_length, hidden_size`
- 理解模型 head：把 hidden states 转成具体任务输出
- 理解 logits、softmax、概率和 `id2label`
- 理解 `return_tensors="pt"`：把 list 输出改成 PyTorch tensor，并增加 batch 维度
- 区分句子对输入 `tokenizer("A", "B")` 和 batch 输入 `tokenizer(["A", "B"])`
- 理解 padding、truncation、`attention_mask`
- 已整理 `notes/day03.md`
- 已新增 `03-NLP-Basic/tokenizer-basics.md`

Day 04 已推进：

- 学习三种分词粒度：基于单词、基于字符、基于子词（BPE、WordPiece、SentencePiece、Unigram）
- 理解 `BertTokenizer` 和 `AutoTokenizer` 的区别：前者硬编码只支持 BERT，后者根据 checkpoint 自动推断
- 理解 `##` 前缀：表示子词续接部分，decoder 会自动合并。例如 "transformer" → "Trans" + "##former"
- 理解 tokenizer 三个原子操作：`tokenize()` 分词、`convert_tokens_to_ids()` 编码、`decode()` 解码
- 理解 batch 维度：`torch.tensor(ids)` 形状 `(seq_len,)` 缺 batch 维，`torch.tensor([ids])` → `(1, seq_len)`
- 理解 `tokenizer(sequence, return_tensors="pt")` 一步完成分词、转 ID、加 batch 维、加 attention_mask
- 理解 padding：多句话组成 batch 时对齐到最长句，短句末尾填充 `pad_token_id`
- 理解 attention mask：padding 位置标 0，告诉注意力层忽略这些位置
- 通过实验验证：不加 attention mask 时 padding 句子 logits 会变化；加上后与单独输入结果一致
- 理解 `[CLS]` 和 `[SEP]`：BERT 预训练时使用的特殊 token，`tokenizer()` 直调会自动添加
- 掌握 padding 策略：`padding="longest"`、`padding="max_length"`、指定 `max_length`
- 掌握 truncation：`truncation=True`、`max_length` 截断
- 理解 `return_tensors`：`"pt"` 返回 PyTorch tensor，`"tf"` 返回 TensorFlow tensor，`"np"` 返回 NumPy array
- 理解 tokenizer 和 model 必须使用相同 checkpoint：词表、特殊 token、分词算法必须一致
- 已整理 `notes/day04.md`（含今日知识总结）
- 已完成 `notebooks/day04.ipynb` 实验
- 已产出 `notes/week01_summary.md` 周志文档

Day 05 已推进：

- 掌握训练一步：`loss = model(**batch).loss` → `loss.backward()` → `optimizer.step()`，`AdamW` 为 BERT 微调标配优化器
- 学习 GLUE/MRPC 数据集：`load_dataset("glue", "mrpc")`，3668/408/1725 三集，`ClassLabel` 映射 0=not_equivalent, 1=equivalent
- 理解 BERT 句子对输入：`[CLS] A [SEP] B [SEP]`，`token_type_ids` 区分两句（0/1），源自预训练的"下一句预测"任务（DistilBERT 无此层）
- 掌握 `Dataset.map()` + `batched=True`：Rust tokenizer 加速批预处理，Apache Arrow 格式惰性加载，num_proc 可多进程并行
- 掌握动态填充：`DataCollatorWithPadding` 对齐到 batch 内最长而非全局最大，tokenize 时省略 padding 节省算力
- 掌握 Trainer 全流程：`TrainingArguments`（必传 output_dir）→ `AutoModelForSequenceClassification`（随机初始化 classification head）→ `Trainer` 组装 → `train()` → `predict()`
- 掌握评估：`Trainer.predict()` 返回 predictions/logits + label_ids；`np.argmax(logits, axis=-1)` 转标签；`evaluate.load("glue", "mrpc")` 计算 accuracy/F1
- 掌握 `compute_metrics`：每 epoch 自动调用，配合 `eval_strategy="epoch"`（非旧版 `evaluation_strategy`）
- 训练实践：bert-base-uncased + MRPC 3 epochs → accuracy 0.833, F1 0.885
- 解决三个问题：`evaluation_strategy` 重命名、accelerate 依赖、训练后重跑 model 定义 cell 导致权重丢失
- 已整理 `notes/day05.md`（含今日知识总结）、`05-BERT/bert-finetuning.md`
- 已完成 `notebooks/day05-test.ipynb`、`notebooks/day05-train.ipynb`

Day 06 已推进：

- 拆开 Trainer 黑盒，手写 PyTorch 训练循环（forward → backward → optimizer.step → scheduler.step → zero_grad）
- 掌握数据准备：`remove_columns`、`rename_column`、`set_format("torch")`、`DataLoader` + `collate_fn`
- 理解优化器 AdamW：Adam + 权重衰减正则化，BERT 微调标配
- 理解学习率调度器：线性衰减从 5e-5 到 0，总步数 = epochs × len(train_dataloader)
- 掌握 epoch、batch size、学习率三者的关系：总步数决定衰减斜率，batch size 翻倍学习率通常也翻倍
- 理解微调本质：不是"加数据跑"，而是以预训练权重为起点，反向传播更新所有权重，把通用能力转化为任务能力
- 掌握评估循环：`model.eval()` + `torch.no_grad()` + `metric.add_batch()` + `metric.compute()`
- 掌握 Accelerate：`accelerator.backward()` 替代 `loss.backward()`，`prepare()` 自动设备分发，Trainer 内部已集成
- 理解分布式训练：数据并行（每卡完整模型，梯度平均）vs 模型并行（切分模型），BERT-base 单卡即够
- 完成 MRPC 手动训练循环实验
- 产出 SST-2 Trainer 训练脚本 `day06-sst-train.py`
- 已整理 `notes/day06.md`（含今日知识总结）
- 已完成 `notebooks/day06.ipynb` 实验

Day 07 已推进：

- 学习训练诊断：损失曲线、准确率曲线、收敛、过拟合、欠拟合、不稳定曲线的特征与解决方案
- 掌握模型卡片概念：预训练模型的用途、局限性、偏见信息应在其模型卡片上展示
- 掌握 Datasets 库三大加载方式：本地文件、远程 URL、Hugging Face Hub（`load_dataset()` + `data_files` + `field`）
- 理解 `data_files` 灵活用法：单文件路径、文件列表、`{"train": ..., "test": ...}` 字典
- 掌握数据处理操作：`filter()`、`map()`（含 `batched=True` 加速）、`rename_column`、`remove_columns`、`add_column`
- 理解 `batched=True` 原理：Rust tokenizer 批量处理可快 30 倍，`num_proc` 对快速 tokenizer 帮助有限
- 掌握大数据处理：内存映射（Apache Arrow）、流式处理（`streaming=True`、`IterableDataset`）
- 掌握 GitHub Issues API 认证：python-dotenv + .env + Bearer token，解决未认证限流问题
- 掌握从零创建数据集：`requests.get()` → JSONL → `Dataset.from_pandas()` / `load_dataset("json")` → `push_to_hub()`
- 掌握语义搜索流程：文本嵌入（CLS pooling）→ FAISS 索引 → `get_nearest_examples()` 相似文档检索
- 理解 FAISS：高效向量相似度搜索库，`add_faiss_index()` 建索引
- 解决多个实际问题：GitHub API 限流认证、Windows num_proc spawn 问题、JSONL schema 冲突、faiss-gpu 不可用、get_embeddings 批处理加速
- 已整理 `notes/day07.md`（含今日知识总结）
- 已完成 `notebooks/day07.ipynb` 实验（GitHub Issues 数据集获取与清洗）
- 已完成 `notebooks/day07-2.ipynb` 实验（语义搜索与 FAISS）

Day 08 已推进：

- 掌握 `AutoTokenizer.train_new_from_iterator()`：从语料库训练新 tokenizer，保留旧 tokenizer 算法只换词表
- 理解 tokenizer 训练 vs 模型训练：前者是统计过程（确定性），后者是随机梯度下降（随机性）
- 掌握语料库生成器模式：函数返回生成器可多次使用，纯生成器只能用一次
- 理解快速 tokenizer（Rust）vs 慢速 tokenizer（Python）：快速版支持偏移映射、并行批处理
- 掌握 BatchEncoding 对象：字典子类，提供 `tokens()`、`word_ids()`、`word_to_chars()`、`token_to_chars()` 等方法
- 理解 char/token/word 三层概念和偏移映射机制：偏移始终是原始文本绝对位置，空格占位但不被 token 覆盖
- 理解 `AutoXxx` 类：自动选架构类（非自动选任务），checkpoint 决定模型，Auto 根据 config 加载对应实现
- 复现 NER 管道：`AutoModelForTokenClassification`，9 标签（CoNLL-2003: O + B/I-PER/ORG/LOC/MISC），偏移量实体分组
- 复现 QA 管道：start/end logits，`sequence_ids()` 屏蔽 question，argmax 展平还原，偏移量提取答案原文
- 掌握标准化（normalization）和预分词（pre-tokenization）：BERT/GPT-2/T5 三种预分词规则差异
- 了解三种子词算法：BPE（合并常见对）、WordPiece（得分制）、Unigram（大词表删除低频）
- 理解 B/I 标签的 IOB2 格式：B 标记实体开头，I 标记内部，O 标记非实体
- 掌握 QA 长文本处理：`return_overflowing_tokens=True` + `stride` 滑动窗口分块
- 已整理 `notes/day08.md`（含今日知识总结）
- 已完成 `notebooks/day08-train_newtokenizer.ipynb` 实验

Day 09 已推进：

- 手写 BPE 算法：`compute_pair_freqs()` + `merge_pair()` 训练循环，`tokenize()` 推理
- 理解 WordPiece vs BPE：前者选似然度提升最大的对，后者选频率最高的对
- 手写 Unigram + Viterbi：`encode_word()` 动态规划分词，`compute_loss()` + `compute_scores()` 剪枝训练
- 理解 Viterbi 回溯：`best_segmentations` 每个位置保存最后一个 token 的起始索引，从末尾往前跳
- 理解 `tokenizers` 库模块化流水线：Normalizer → PreTokenizer → Model → PostProcessor → Decoder
- 区分"Tokenizer 模型"（词表+分词规则）和"Transformer 模型"（神经网络）
- 理解 `Whitespace()` 不只在空白处分割，而是所有非单词字符处
- 从零构建 BERT WordPiece tokenizer：``BertNormalizer`` + ``Whitespace`` pre-tokenizer + ``WordPieceTrainer``
- 掌握 `PreTrainedTokenizerFast` 包装 → `save_pretrained()` / `push_to_hub()`
- 已整理 `notes/day09.md`（含今日知识总结）
- 已完成 `notebooks/day09-BPE.ipynb`、`day09-wordpiece.ipynb`、`day09-unigram.ipynb`、`day09-bulidberttokenizer.ipynb` 实验

Day 10 已推进：

- 学习 Token 分类任务：NER（命名实体识别）、POS（词性标注）、Chunking（分块）三者的共性与区别
- 理解 IOB2 标注格式：B-（实体开头）、I-（实体内部）、O（非实体），CoNLL-2003 共 9 标签（O + B/I × 4 类）
- 掌握子词分词后的标签对齐问题：`is_split_into_words=True` + `word_ids()` + `align_labels_with_tokens()`
- 理解标签对齐三条规则：特殊 token 标 -100、同词后续子词 B- 转 I-、-100 被交叉熵损失忽略
- 掌握 `DataCollatorForTokenClassification`：用 -100 填充标签，确保 padding 位置不计入损失
- 掌握 seqeval 实体级评估：按实体边界+类别匹配计算 F1，而非 token 级 accuracy
- 完成 BERT NER 微调全流程：CoNLL-2003 加载 → 标签对齐 → DataCollator → seqeval → AutoModelForTokenClassification → Trainer
- 解决 datasets 5.0.0 脚本不支持问题：`load_dataset("conll2003", revision="refs/convert/parquet")`
- 再次确认 `evaluation_strategy` → `eval_strategy` 重命名（Transformers 4.57）
- 已整理 `notes/day10.md`（含今日知识总结）
- 已完成 `notebooks/day10-tokenclassification.ipynb` 实验

Day 11 已推进：

- 学习 MLM（掩码语言建模）微调：在 IMDB 电影评论上微调 DistilBERT 做领域适应
- 理解 MLM 数据预处理：拼接所有文本 + 等长分块（chunk），与分类任务的逐条 tokenize 不同
- 掌握 `DataCollatorForLanguageModeling`：每 batch 动态随机遮蔽，80-10-10 拆分（80% MASK、10% 随机词、10% 原词）
- 手写全词掩码（WWM）collator：`word_ids()` 获取 token→word 映射，整词一起遮蔽，避免 subword 泄漏
- 解决 `word_ids` KeyError：`TrainingArguments` 默认 `remove_unused_columns=True` 会删除模型 `forward()` 不接受的列，需显式设为 `False`
- 掌握语言模型评估：困惑度 `Perplexity = exp(cross_entropy_loss)`，微调后从 62.55 降到 24.60
- 使用微调模型推理：`pipeline("fill-mask", model="./output_dir")` 加载本地模型
- 已整理 `notes/day11.md`（含今日知识总结）
- 已完成 `notebooks/day11-masktoken.ipynb` 实验

Day 12 已推进：

- 学习 Seq2Seq 翻译任务：用 Helsinki-NLP/opus-mt-en-zh（MarianMT）做英译中
- 掌握 `text_target` 参数：tokenizer 据此用不同规则处理源语言和目标语言（英语空格分词、中文按字切分）
- 理解 `DataCollatorForSeq2Seq`：动态填充 + labels 用 -100 忽略 + 从 labels 移位生成 `decoder_input_ids`
- 理解 `decoder_input_ids` 移位原理：labels 右移一位，开头补 `<s>`，实现 teacher forcing
- 掌握 `predict_with_generate=True`：评估时用 `generate()` 逐 token 生成，模拟真实推理
- 掌握 SacreBLEU 中文评估：`tokenize="zh"` 调用 jieba 分词，解决中文 n-gram 全为 0 的问题
- 理解训练/验证/测试集关系：按用法区分而非名称，验证集反复用、测试集只跑一次
- 解决 kde4 RuntimeError → opus100 替代（datasets 5.0.0 兼容）
- 解决 sentencepiece/sacremoses 安装、HF 网络镜像、验证集未 tokenize 等问题
- 微调后 BLEU 48.06 → 49.52（同域数据提升有限，预训练已覆盖）
- 已整理 `notes/day12.md`（含今日知识总结）
- 已完成 `notebooks/day12-translation.ipynb` 实验

Day 01 笔记：

```text
notes/day01.md
```

实验 Notebook：

```text
notebooks/day01.ipynb
```

## 已掌握的 Transformer 概念

- Encoder：理解完整输入并生成上下文化表示
- Decoder：根据目标前文生成下一个 Token
- Encoder-only：适合理解任务
- Decoder-only：适合生成任务
- Encoder-Decoder：适合翻译、摘要等条件生成任务
- Masked Self-Attention：只能查看目标序列当前位置及之前允许的信息
- Cross-Attention：解码器可以读取编码器的完整源句表示
- 因果掩码：阻止当前位置看到未来 Token
- 目标右移：防止当前位置直接看到当前正确答案
- Teacher Forcing：训练时后续位置使用真实前文，不使用前一位置的预测结果
- 训练可以并行：完整目标已知，所有位置的输入可提前构造
- 自回归推理通常串行：下一步依赖上一步实际生成的 Token

详细文档：

```text
04-Transformer/transformer-basics.md
```

## 已解决问题

### Torchaudio WinError 127

根因：PyTorch、Torchvision、Torchaudio CUDA 版本不一致。

结果：统一安装为 PyTorch CUDA 12.8 版本后解决。

### SymPy 版本冲突

根因：PyTorch 对 SymPy 有精确版本要求。

结果：依赖已修复，`pip check` 通过。

### Hugging Face 未认证警告

通过以下命令登录：

```powershell
hf auth login
hf auth whoami
```

### Windows 符号链接警告

不是运行错误。开启 Windows 开发者模式可减少 Hugging Face 缓存中的重复文件。

### `No mask_token found`

`fill-mask` 输入必须包含 `tokenizer.mask_token`。若其值为 `None`，模型不支持掩码填充。

### `model_type` 缺失

Hugging Face Hub 中的模型可能属于 GGUF、Flair 等其他格式，不能直接传给 Transformers `pipeline()`。

### `evaluation_strategy` 参数重命名

根因：Transformers 4.57 将 `evaluation_strategy` 重命名为 `eval_strategy`。

解决：使用 `TrainingArguments(output_dir, eval_strategy="epoch")`。

### Accelerate 未安装导致 Trainer 报错

根因：`Trainer` 依赖 `accelerate>=0.26.0`，Conda 环境未预装。

解决：`D:/Soft/Conda/python.exe -m pip install "accelerate>=0.26.0"` 后重启 Kernel。

### 训练后评估指标骤降

根因：`trainer.train()` 后重新执行 model 定义 cell（`AutoModelForSequenceClassification.from_pretrained(checkpoint)`）导致 classification head 回到随机初始化。

解决：train 后立刻 evaluate，不重新创建 model；或从保存的 checkpoint 加载已训练权重。

### `Unknown task question-answering`

Transformers 5.13.0 未注册该 Pipeline，而课程示例仍依赖它。已降级并固定为 `4.57.3`。

### `code_search_net` 数据集不可用

根因：数据集在 HuggingFace Hub 路径已更新，且国内直连不可达。

解决：
- 路径：`"code_search_net"` → `"code-search-net/code_search_net"`
- 国内需在 `load_dataset` 前设置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`

### `conll2003` 数据集脚本不支持

根因：datasets 5.0.0 不再支持旧的脚本式数据集加载。

解决：指定 Parquet 格式分支 `load_dataset("conll2003", revision="refs/convert/parquet")`。

### 交叉环境 pip 安装

教训：Notebook Kernel 使用 `D:\Soft\Conda\python.exe`，与系统 Python（C 盘）是不同环境。`pip install` 需显式指定 `D:/Soft/Conda/python.exe -m pip install <pkg>` 才能装到正确环境。

## 模型使用原则

使用模型前检查：

1. 模型任务是否匹配当前 Pipeline。
2. 模型框架是否为 Transformers。
3. 仓库是否包含有效的 `config.json`。
4. Tokenizer 是否包含任务需要的特殊 Token。
5. 模型文件大小是否适合本机。
6. 模型页是否提供官方调用示例。

不要因为模型位于 Hugging Face Hub，就默认它一定支持 Transformers `pipeline()`。

## 文档约定

- 每天学习记录放入 `notes/dayXX.md`
- 每天实验放入 `notebooks/dayXX.ipynb`
- 环境说明放入 `00-Environment`
- PyTorch 练习放入 `02-PyTorch`
- NLP 基础放入 `03-NLP-Basic`
- Transformer 原理与实现放入 `04-Transformer`
- 每周计划放入 `resources/weeks/weekXX.md`
- 周志放入 `notes/weekXX_summary.md`
- 项目放入 `projects/p1-text-classification`、`projects/p2-rag-qa`、`projects/p3-llm-app`
- 主 README 只保留路线、索引、目标和阶段产出
- 新内容优先写入对应主题文档，避免所有内容堆积在每日笔记


## 每日学习工作流

- 学习过程中，用户会随时向 Codex 提问概念、代码、报错、图片和文档内容。
- 回答问题时，优先用用户当前正在看的文件和选中文本作为上下文。
- 概念解释遵循：先讲直观含义，再给简单例子，最后说明技术原理。
- 遇到环境或代码报错时，先解释错误含义，再给最小修复命令，不主动扩大范围。
- 每天学习结束时，需要完成以下三项工作：

### 1. 整理笔记正文

- 笔记格式参考 `notes/day06.md`：`# Day XX：简短主题` → 一句话概述 → `##` 大节（代码精简、有注释、节末关键点总结、表格辅助对比）→ `## 今日知识总结`
- 补充缺失的概念解释、代码示例和关键注意事项
- 修正错误表述、错别字、不准确的术语
- 修复图片链接、Markdown 格式和路径问题
- 确保代码片段可运行、参数名与实际使用一致
- 删除重复内容和过时信息

### 2. 添加今日知识总结

- 在 `notes/dayXX.md` 末尾追加 `## 今日知识总结`
- 格式参考 `notes/day06.md`：使用多个 `### 小标题`，每个小标题下写**段落式解释**（非列表项）
- 覆盖：今天学了什么、关键概念是什么、容易混淆点是什么、今天解决了什么问题、下一步应该学什么

### 3. 同步更新

- 更新 `MEMORY.md` 中的学习进度
- 如果当天内容属于某个主题目录，同步整理到对应主题文档（如 Transformer → `04-Transformer/`，语义搜索 → `08-RAG/`）
- 写入文件后验证：无乱码、总结只有一处、关键旧错误表述已清除
## 下一步

继续学习建议：

- 当前路线：`resources/roadmap.md` → `resources/weeks/week06.md`
- 下一步：GPT、生成与解码策略（Week 6），或补 Week 5 的 TF-IDF baseline 对比

## 恢复上下文顺序

后续继续学习时，依次读取：

1. `MEMORY.md`
2. `README.md`
3. `resources/roadmap.md`
4. 当前周的 `resources/weeks/weekXX.md`
5. 最近一天的 `notes/dayXX.md`
6. 对应主题目录中的文档

