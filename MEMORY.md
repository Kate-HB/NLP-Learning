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
- 项目放入 `projects/p1-text-classification`、`projects/p2-rag-qa`、`projects/p3-llm-app`
- 主 README 只保留路线、索引、目标和阶段产出
- 新内容优先写入对应主题文档，避免所有内容堆积在每日笔记


## 每日学习工作流

- 学习过程中，用户会随时向 Codex 提问概念、代码、报错、图片和文档内容。
- 回答问题时，优先用用户当前正在看的文件和选中文本作为上下文。
- 概念解释遵循：先讲直观含义，再给简单例子，最后说明技术原理。
- 遇到环境或代码报错时，先解释错误含义，再给最小修复命令，不主动扩大范围。
- 每天学习结束时，需要整理当天学习内容到 `notes/dayXX.md`。
- 每天笔记末尾必须添加 `## 今日知识总结`，格式参考 `notes/day01.md`：使用多个 `### 小标题`，每个小标题下写段落式解释。
- 每天总结要覆盖：今天学了什么、关键概念是什么、容易混淆点是什么、今天解决了什么问题、下一步应该学什么。
- 如果当天内容属于某个主题目录，也要同步整理到对应主题文档。例如 Transformer 相关内容整理到 `04-Transformer/transformer-basics.md`。
- 每日整理时要修正明显错误表述、图片链接、Markdown 格式和路径问题。
- 写入文件后要验证：无乱码、总结只有一处、关键旧错误表述已清除。
- 更新学习进度后，需要同步更新 `MEMORY.md`，方便后续继续学习。
## 下一步

继续学习建议：

- 当前路线：`resources/roadmap.md` → `resources/weeks/week01.md`
- Day 06 重点：手动推理（tokenizer + model + softmax），完整走通 logits 到 label 的过程
- Day 07：Week 1 复盘，重跑 Day 02 到 Day 06，画出完整推理流程图
- Week 2：PyTorch 最小训练基础（Tensor、Dataset、nn.Module、autograd、训练循环）

预期下一批产出：

```text
notes/day06.md
notebooks/day06.ipynb
notes/week01_summary.md  （已产出）
```

## 恢复上下文顺序

后续继续学习时，依次读取：

1. `MEMORY.md`
2. `README.md`
3. `resources/roadmap.md`
4. 当前周的 `resources/weeks/weekXX.md`
5. 最近一天的 `notes/dayXX.md`
6. 对应主题目录中的文档

