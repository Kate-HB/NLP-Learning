# Hugging Face 主要 NLP 任务：从数据到评估的完整知识总结
[图示](03-NLP-Basic\pictures\NLP任务框架.png)
[图示](03-NLP-Basic\pictures\NLP数据到标签.png)
[图示](03-NLP-Basic\pictures\NLP五大任务.png)
[图示](03-NLP-Basic\pictures\NLP掩码机制.png)
> 范围：Hugging Face LLM Course Chapter 7.3–7.7，并结合本项目 Day 01–10 已完成的 Tokenizer、Datasets、Trainer、Accelerate、BERT、NER、QA 管道与训练循环实验。

## 1. 这一阶段到底在学什么

前面的学习已经解决了三个基础问题：

1. 文本如何通过 Tokenizer 变成 `input_ids`、`attention_mask` 等模型输入。
2. Transformer 主体如何把 token 转成上下文化的 hidden states。
3. 如何在 hidden states 后接任务 Head，得到 logits，再用标签计算 loss。

Chapter 7.3–7.7 进一步回答：**面对不同 NLP 任务，如何把原始数据加工成正确的监督信号，如何选择模型与数据整理器，如何把模型输出还原为可评估、可使用的结果。**

五个任务虽然表面不同，但都遵循同一主线：

```text
明确任务输出
→ 检查原始数据结构
→ 选择匹配的模型架构与 Tokenizer
→ Tokenize 并构造 labels
→ Data Collator 组装 batch
→ 模型前向传播并计算 loss
→ 反向传播更新参数
→ 将 logits 后处理为任务结果
→ 用任务指标评估
→ 保存并通过 pipeline 推理
```

真正变化最大的不是 `Trainer`，而是下面三件事：

- **标签是什么**：被遮盖 token、目标语言 token、摘要 token、下一个 token，还是答案起止位置。
- **标签如何与输入对齐**：直接复制、右移、字符位置转 token 位置，还是跨滑动窗口聚合。
- **如何评估**：困惑度、BLEU、ROUGE、Exact Match/F1 不能互换。

## 2. 五类任务总览

| 任务 | 典型架构 | 输入 | 监督目标 | 模型输出 | 关键指标 |
|---|---|---|---|---|---|
| 掩码语言模型 MLM | Encoder-only | 被随机遮盖的文本 | 被遮盖位置的原 token | 每个位置的词表 logits | loss、perplexity |
| 翻译 | Encoder-Decoder | 源语言文本 | 目标语言文本 | 逐 token 生成的译文 | BLEU |
| 文本摘要 | Encoder-Decoder | 长文档 | 参考摘要 | 逐 token 生成的摘要 | ROUGE |
| 因果语言模型 CLM | Decoder-only | token 前缀 | 每个位置的下一个 token | 每个位置的词表 logits | loss、perplexity、人工/任务评估 |
| 抽取式问答 | Encoder-only | 问题 + 上下文 | 答案起止 token 索引 | start/end logits | Exact Match、F1 |

这张表的意义是：看到新任务时，不要先背 API，而要先回答四个问题：

1. 模型最终要预测什么？
2. 原始标注如何变成模型能计算 loss 的 labels？
3. 哪些位置不应计入 loss？
4. logits 如何还原为人能理解的结果？

---

## 3. 通用训练框架：所有任务共享的骨架

### 3.1 数据层

`datasets` 负责加载、检查、过滤和批量变换数据：

```python
raw_datasets = load_dataset(...)
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

`batched=True` 不只是速度优化。某些任务会让输入行数发生变化：

- 一篇长文本切成多个固定长度块。
- 一个长上下文切成多个重叠窗口。
- 多个短文本拼接后重新切块。

`Dataset.map()` 允许一个输入 batch 产生更多或更少的输出样本，正适合这类预处理。

### 3.2 Tokenizer 层

Tokenizer 不只做“切词”，还承担：

- 文本规范化、预分词、子词切分、token 到 ID 映射。
- 添加 `[CLS]`、`[SEP]`、`</s>` 等特殊 token。
- 生成 `attention_mask`、`token_type_ids`。
- 提供 `word_ids()`、`offset_mapping`、`sequence_ids()` 等对齐信息。

本项目已经手写过 BPE、WordPiece、Unigram，并从零构建过 BERT Tokenizer。这里最重要的迁移是：**分词不只是输入预处理，它直接决定标签能否正确对齐。** NER 依赖 `word_ids()`，QA 依赖 `offset_mapping`，MLM 全词遮盖同时依赖二者背后的词—token 映射。

### 3.3 Data Collator 层

Data Collator 在取出一个 batch 时动态完成组装。不同任务使用不同 Collator，是因为需要填充或动态构造的内容不同：

| Collator | 任务 | 额外职责 |
|---|---|---|
| `DataCollatorWithPadding` | 序列分类 | 动态填充输入 |
| `DataCollatorForTokenClassification` | NER | 同时用 `-100` 填充 token 标签 |
| `DataCollatorForLanguageModeling` | MLM/CLM | 动态遮盖或复制语言模型标签 |
| `DataCollatorForSeq2Seq` | 翻译/摘要 | 填充输入和标签，并准备 `decoder_input_ids` |

Collator 的意义是把“每个样本的逻辑”与“一个 batch 如何形成”分开。动态随机遮盖放在 Collator 中后，同一文本在不同 epoch 可以产生不同训练目标，相当于廉价的数据增强。

### 3.4 模型层

模型类必须与预测目标匹配：

```text
AutoModelForMaskedLM          → 每个位置预测词表 token
AutoModelForSeq2SeqLM         → 条件生成目标序列
GPT2LMHeadModel               → 自回归预测下一个 token
AutoModelForQuestionAnswering → 预测答案起止位置
```

`AutoModel` 只输出 hidden states；`AutoModelForXxx` 在主体后附带任务 Head。加载基础 checkpoint 时，如果新 Head 是随机初始化的，警告是正常的，但这也表示模型必须经过微调才能完成任务。

### 3.5 训练层

无论使用 `Trainer` 还是手写循环，底层都相同：

```text
batch → model(**batch) → loss
→ backward
→ optimizer.step
→ scheduler.step
→ optimizer.zero_grad
```

`Trainer` 适合标准流程；`Accelerate` 适合自定义 loss、固定评估随机性、特殊采样、梯度累积或自定义保存逻辑。两者不是两种模型训练原理，而是不同抽象层级。

> **本项目版本兼容提醒：** 当前环境为 Transformers 4.57.3，`TrainingArguments` 使用 `eval_strategy`，`Trainer` 使用 `processing_class=tokenizer`。课程旧代码中的 `evaluation_strategy` 和 `Trainer(tokenizer=tokenizer)` 不能直接照抄。API 变化不改变任务原理，但会决定代码能否运行。

本项目版本的标准组装骨架是：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

需要自定义循环时，Accelerate 的关键闭环是：

```python
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, eval_loader, scheduler
)

for batch in train_loader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# 评估时用 gather_for_metrics 聚合各进程结果；保存前恢复原模型。
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

---

## 4. 任务一：微调掩码语言模型（MLM）

课程原文：[微调掩码语言模型](https://huggingface.co/learn/llm-course/zh-CN/chapter7/3)

### 4.1 是什么

掩码语言模型先把完整文本中的部分 token 替换为 `[MASK]`，再让 Encoder 根据左右两侧上下文恢复原 token。

```text
原文：This is a great movie.
输入：This is a great [MASK].
标签：只要求被遮盖位置预测 movie
```

BERT、RoBERTa、DistilBERT 属于典型 MLM。它们使用双向注意力，适合学习上下文理解能力，不以自由续写为主要目标。

### 4.2 为什么做领域适应

通用预训练模型学到的是预训练语料中的语言分布。法律合同、医学论文、金融报告、电影评论或企业内部文档往往有不同术语和表达方式。直接做下游分类时，模型虽然能工作，但对领域词和领域语义的表示可能不够好。

先在无标注领域语料上继续做 MLM，再微调具体任务 Head，称为**领域适应**：

```text
通用预训练模型
→ 领域无标注文本上的 MLM
→ 领域适应后的基础模型
→ 分类、NER、检索等下游任务
```

它的意义是：领域无标注文本通常远多于人工标注数据，而且领域适应得到的基础模型可被多个下游任务复用。

### 4.3 完整数据流程

#### 第一步：选择支持 MLM 的 checkpoint

模型和 Tokenizer 必须来自同一 checkpoint：

```python
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
```

先用 `fill-mask` 或直接查看 `[MASK]` 位置 logits，可以建立微调前的基线。

#### 第二步：加载并检查领域语料

MLM 是自监督任务，不需要人工类别标签。课程使用 IMDb 评论，真正有用的是 `text`；原有情感 `label` 对 MLM 不参与训练。

必须先抽样检查：文本是否为空、语言是否一致、标签列是否只是无关元数据、是否包含大量 HTML 或异常字符。数据错误不会因模型强大而自动消失。

#### 第三步：不截断地 Tokenize

```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result
```

这里不立即 `truncation=True`，因为直接按原样截断每篇文章会永久丢掉尾部文本。语言模型希望尽量利用全部语料。

#### 第四步：拼接后按固定长度切块

```text
多篇文本 tokenize
→ 将 input_ids 等字段分别拼接
→ 按 chunk_size 切成等长块
→ 丢弃或填充最后一个短块
```

这样能提高 token 利用率。`chunk_size` 受模型最大上下文和显存限制：块太短会丢失长距离信息，块太长会增加注意力计算量与显存占用。

需要注意：简单拼接可能让一个块跨越两篇文档，特殊 token 会保留边界，但相邻上下文本身并不连续。更严格的实现可以逐文档切块，或明确插入 EOS/SEP 后再拼接。

#### 第五步：创建 labels

课程为了同时支持后续自定义全词遮盖，切块时先令：

```python
result["labels"] = result["input_ids"].copy()
```

这不代表所有位置都计算 loss。真正送入 batch 时，Collator 会随机选择待预测位置；未被选中的标签设为 `-100`，交叉熵会忽略它们。标准 `DataCollatorForLanguageModeling` 本身也能从 `input_ids` 生成 labels，因此预先复制不是所有实现的必需步骤；自定义 Collator 是否需要原始 labels，要以其输入契约为准。

#### 第六步：动态随机遮盖

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)
```

动态遮盖每次组 batch 时重新采样，因此同一文本可以在不同轮次训练不同 token。相比预先固定遮盖，模型能看到更多“完形填空”组合。

经典 BERT 风格的被选位置并非全部替换为 `[MASK]`：典型策略是 80% 替换为 `[MASK]`、10% 替换为随机 token、10% 保持原 token。后两种处理能减小预训练中总见 `[MASK]`、实际下游输入却没有 `[MASK]` 的分布差异。

### 4.4 普通遮盖与全词遮盖

WordPiece 可能把一个词切成多个子词。普通 token 遮盖可能只遮掉其中一个：

```text
playing → play + ##ing
普通遮盖：play + [MASK]
全词遮盖：[MASK] + [MASK]
```

全词遮盖先找到属于同一个词的全部 token，再以“词”为单位采样。它减少模型利用未遮盖子词直接猜答案的机会，任务更难，也更符合词级语义学习。

课程通过 `word_ids` 手写 Collator；本项目 Transformers 4.57.3 的 `DataCollatorForLanguageModeling` 已支持全词遮盖。当前版本应显式配置：

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    whole_word_mask=True,
    mlm_probability=0.15,
    mask_replace_prob=1.0,
    random_replace_prob=0.0,
)
```

预处理时还要让 Fast Tokenizer 返回并保留 `offset_mapping`，同时设置 `remove_unused_columns=False`，否则 Trainer 会在 Collator 使用前删除该列。显式把替换概率设为 1/0 还能避开 4.57.3 在全词遮盖自动改参警告中的类型错误。最后应在小 batch 上确认整词的所有子词是否同时被选中。

若自定义全词遮盖 Collator 需要 `word_ids`，`TrainingArguments` 应设置 `remove_unused_columns=False`，否则 Trainer 会因模型 `forward()` 不接收该列而提前删除它。

### 4.5 训练、评估与使用

训练主体与文本分类相同，只是模型类和 Collator 不同。语言模型常用困惑度：

```text
perplexity = exp(eval_loss)
```

困惑度越低，表示模型对当前遮盖任务中的真实 token 越不“意外”。对 MLM 而言，`exp(masked-token loss)` 只基于被抽样遮盖的位置，不是 CLM 对完整序列联合似然定义的困惑度；它只能在相同 Tokenizer、预处理和遮盖规则下公平比较。

动态遮盖会让每次评估的 mask 不同，导致 perplexity 波动。若要稳定比较模型，应先对验证集只遮盖一次并保存输入与 labels，评估时使用只负责堆叠/填充、不会再次随机遮盖的 Collator；不能把随机波动误认为模型变化。

训练完成后用 `pipeline("fill-mask")` 检查领域适应效果。IMDb 适应后的模型在 “This is a great [MASK].” 中更可能预测 `movie`、`film`、`story`，说明权重已向电影评论语域移动。

### 4.6 难点与易混淆点

> **`[MASK]` token 不等于 attention mask。** `[MASK]` 是词表中的真实输入 token；`attention_mask` 是 0/1 可见性标记，通常用于忽略 padding。

> **MLM 的 labels 不是人工标签。** 原文自己提供答案，因此属于自监督学习。

> **`labels = input_ids.copy()` 不是让模型逐位置复制输入。** 绝大多数计分位置的原 token 已被遮盖或替换；少量位置会按 BERT 策略保持不变，但整体目标仍是依靠双向上下文恢复被抽样位置。

> **Perplexity 不能跨不同词表随意比较。** Tokenizer 会改变序列长度和每步预测空间，数值尺度也会改变。

---

## 5. 任务二：机器翻译

课程原文：[翻译](https://huggingface.co/learn/llm-course/zh-CN/chapter7/4)

### 5.1 是什么

翻译是序列到序列任务：输入一个源语言序列，生成语义对应的目标语言序列。

```text
源句：I love NLP.
目标：我喜欢自然语言处理。
```

Encoder 一次读取完整源句；Decoder 通过 masked self-attention 读取已知目标前文，通过 cross-attention 读取完整源句表示，再预测下一个目标 token。

### 5.2 为什么通常选择微调

从头训练翻译模型需要大量高质量平行语料和计算资源。更常见的做法是：

- 从专用语言对模型继续微调，如 Marian 英法模型。
- 从 mT5、mBART 等多语言模型微调到特定语言对或领域。

领域微调不仅改变语言，还改变术语风格。通用模型可能保留 `plugin`、`thread` 等英语技术词；技术文档平行语料可能要求正式目标语言表达。微调让输出符合特定组织、行业和语料的术语规范。

### 5.3 完整数据流程

#### 第一步：准备平行语料

每个样本必须成对：

```python
{"translation": {"en": "source text", "fr": "target text"}}
```

训练前要检查语言方向、空值、错位样本、重复句、极端长度和术语一致性。平行数据一旦错位，模型会学习“输入与输出无关”。

#### 第二步：同时编码源文本和目标文本

```python
def preprocess_function(examples):
    sources = [item["en"] for item in examples["translation"]]
    targets = [item["fr"] for item in examples["translation"]]
    return tokenizer(
        sources,
        text_target=targets,
        max_length=max_input_length,
        truncation=True,
    )
```

`text_target` 明确告诉 Tokenizer：这部分是目标序列。对 mBART、M2M100 等多语言 Tokenizer，还要设置 `src_lang` 和 `tgt_lang`，因为语言代码可能决定特殊 token 和生成起始行为。

#### 第三步：加载 Seq2Seq 模型

```python
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

与文本分类不同，翻译 checkpoint 通常已经有可用的 Seq2Seq LM Head，因此不会像新建分类 Head 那样完全随机初始化任务输出层。

#### 第四步：Seq2Seq Collator 动态组 batch

```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

它同时完成：

1. 动态填充 Encoder 输入。
2. 用 `-100` 填充 labels，使标签 padding 不参与 loss。
3. 若模型实现了 `prepare_decoder_input_ids_from_labels()`，按该架构规则把 labels 右移，构造 `decoder_input_ids`。

右移后的逻辑是：

```text
decoder_input_ids：<BOS> 我 喜欢
labels：              我 喜欢 NLP
```

训练时使用真实目标前文，称为 Teacher Forcing。目标右移让预测位置拿不到当前答案；因果掩码再阻止它读取更后面的 decoder 输入。两者共同防止答案泄漏，同时允许所有目标位置并行训练。

#### 第五步：用真实生成方式评估

训练时有真实目标前文，推理时没有。若评估也把真实目标喂给 Decoder，会高估模型能力。应使用：

```python
Seq2SeqTrainingArguments(..., predict_with_generate=True)
```

这样评估调用 `generate()` 自回归生成译文，条件与真实使用一致。

训练完成后应从保存的 checkpoint 重新加载模型与 Tokenizer，再用 `pipeline("translation", model=checkpoint_path)` 或 `model.generate()` 验证真实推理闭环。生成时还要固定并记录 `max_new_tokens`、beam 数、长度惩罚等参数；否则同一模型也可能得到不同 BLEU 和输出风格。

### 5.4 BLEU 指标

BLEU 比较候选译文与一个或多个参考译文的 n-gram 重合，并对过短输出施加长度惩罚。分数越高通常越好。

评估流程：

```text
生成 token IDs
→ batch_decode(skip_special_tokens=True)
→ labels 中 -100 替换为 pad_token_id
→ decode 参考译文
→ 清理空格
→ 计算 BLEU
```

BLEU 的局限是：同一语义可能有多种正确译法，字面重合低不一定翻译差；术语正确、事实忠实、语法自然也不能完全由一个分数表达。BLEU 通常应在整个语料库上统计，而且分词、大小写、平滑方法和实现会影响数值，跨实验比较必须保持配置一致。因此实际系统还需人工评估或更强的语义指标。

### 5.5 难点与易混淆点

> **源文本的 padding 与目标标签的 padding 不同。** 输入 padding 用 `pad_token_id` 并由 `attention_mask=0` 忽略；标签 padding 用 `-100`，由 loss 忽略。

> **目标右移通常由模型或 Collator 完成。** 不要再手工把 labels 右移一次，否则会发生双重错位。

> **训练 logits 评估不等于生成质量评估。** 真实部署要逐 token 生成，因此验证阶段应使用 `generate()`。

> **翻译方向不能只靠数据列名猜。** checkpoint、源语言、目标语言、语言代码必须一致。

---

## 6. 任务三：文本摘要

课程原文：[文本摘要](https://huggingface.co/learn/llm-course/zh-CN/chapter7/5)

### 6.1 是什么

文本摘要把长文本压缩为更短、保留核心信息的文本。它同时要求：

- 理解原文主题与重要事实。
- 舍弃次要内容。
- 生成连贯、简洁、事实一致的语言。

摘要分两类：

- **抽取式摘要**：从原文选择句子或片段。
- **生成式摘要**：重新组织语言生成新文本。

课程使用 mT5 做生成式摘要。这里的“提取文本摘要”是课程中文标题，不应误解为模型只复制原句。

### 6.2 为什么有意义

摘要把高阅读成本的长文本转为可快速消费的信息，可用于新闻聚合、会议纪要、客服工单、医学记录和检索结果压缩。它也能作为 RAG 的中间环节：先压缩候选文档，再把关键上下文交给生成模型。但压缩意味着信息损失，因此摘要系统的核心不是“写得像人”，而是**在长度约束下最大化关键信息保留，并避免生成原文没有的事实**。

### 6.3 与翻译的相同点和不同点

两者的训练机械结构几乎相同：

```text
Encoder 输入源序列
→ Decoder 根据源表示和目标前文生成目标序列
→ DataCollatorForSeq2Seq 负责目标右移与标签填充
```

区别在任务约束：

- 翻译强调跨语言语义等价。
- 摘要强调压缩、覆盖关键信息、减少冗余和事实忠实。
- 翻译源与目标长度可能相近；摘要目标通常远短于输入。
- 翻译常用 BLEU；摘要常用 ROUGE。

### 6.4 完整数据流程

#### 第一步：构造“正文—摘要”数据对

课程把商品评论正文作为输入、评论标题作为目标摘要，并组合英语和西班牙语书评训练双语模型。这个设计展示了监督信号不一定来自专门标注：已有标题可以充当弱监督摘要。

但必须检查标题是否真的概括正文。营销词、空标题、仅含星级情绪词的标题会让模型学到错误目标。

#### 第二步：分别限制输入和目标长度

```python
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=512,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["review_title"],
        max_length=30,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

输入与目标必须分别设置长度：输入长度影响可读到多少原文，目标长度影响摘要能表达多少内容。输入截断过强会丢失关键信息；目标上限过短会强迫摘要不完整，过长又可能增加重复和跑题。

#### 第三步：使用 Seq2Seq 模型与 Collator

```python
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

`mT5` 使用 SentencePiece/Unigram 风格 Tokenizer，`▁` 表示词前空格边界，`</s>` 表示序列结束。多语言模型的优势是共享参数与子词空间，但不保证所有语言表现相同，数据比例仍很重要。

#### 第四步：先建立 baseline

课程使用 `lead-3`：直接取原文前三句作为摘要。baseline 的意义不是追求先进，而是回答：**模型是否真的比廉价规则更好？**

若复杂模型没有超过合理 baseline，问题可能在数据、指标、训练或任务定义，而不是模型规模不够。

#### 第五步：训练并生成摘要

使用 `Seq2SeqTrainer` 时开启 `predict_with_generate=True`，评估阶段通过 `generate()` 得到真实摘要。手写 Accelerate 循环时，也应在验证阶段调用 `accelerator.unwrap_model(model).generate(...)`，再聚合多进程结果。

训练后要从保存的 checkpoint 重新加载，并用真实长文本做端到端推理。摘要的 `max_new_tokens`、`min_new_tokens`、beam 数、长度惩罚和重复惩罚会直接影响压缩率、覆盖率与重复，应与评估结果一起记录。

### 6.5 ROUGE 指标

ROUGE 衡量生成摘要与参考摘要的重叠：

- `ROUGE-1`：单词/一元组重叠。
- `ROUGE-2`：二元词组重叠，更关注局部顺序。
- `ROUGE-L`：基于最长公共子序列。
- `ROUGE-Lsum`：面向整个摘要的句子级处理。

Precision 表示生成内容中有多少与参考相关；Recall 表示参考内容中有多少被生成摘要覆盖；F1 平衡两者。

ROUGE 的局限同样明显：高重叠不保证事实正确，低重叠也可能是优秀改写；分词、词干化和句子边界处理也会改变分数，跨实验必须使用相同配置。摘要系统还应检查：

- 是否遗漏关键事实。
- 是否编造原文没有的信息。
- 是否指代不清。
- 是否冗余或重复。
- 长度是否符合业务要求。

### 6.6 难点与易混淆点

> **生成式摘要不是抽取式问答。** 摘要可以生成原文中没有连续出现的新表述；抽取式 QA 的答案必须是 context 中的连续字符片段。

> **ROUGE 高不等于事实可靠。** 它主要衡量词面重叠，不直接验证事实一致性。

> **多语言模型不等于自动完成任意语言。** Tokenizer 覆盖、训练语料比例和任务数据质量共同决定结果。

> **摘要与翻译可共用代码骨架，但指标不能直接互换解释。** 指标并非由模型架构强制绑定，而是应匹配任务的成功标准。

---

## 7. 任务四：从头训练因果语言模型（CLM）

课程原文：[从头开始训练因果语言模型](https://huggingface.co/learn/llm-course/zh-CN/chapter7/6)

### 7.1 是什么

因果语言模型根据已有前文预测下一个 token：

```text
输入位置：<BOS>  import  pandas
监督目标：import  pandas  as
```

Decoder 的因果掩码使位置 `t` 只能看到 `≤t` 的输入，不能看到未来 token。训练时完整序列已知，所以各位置可以并行；推理时下一个输入依赖上一步实际生成结果，所以通常逐 token 串行。

### 7.2 为什么以及何时从头训练

微调通常更划算。只有以下条件较强时，才考虑从头训练：

- 有足够大且高质量的语料。
- 数据分布与现有模型差异很大，如代码、DNA、音乐符号、特殊协议。
- 需要完全控制 Tokenizer、词表、模型大小或许可。
- 计算预算允许长时间预训练和反复评估。

从头训练的意义是让整个词表和全部模型权重都针对新分布形成，而不受原预训练语料的强先验限制；代价是数据、算力和训练风险显著增加。

### 7.3 完整数据流程

#### 第一步：收集并过滤领域数据

课程从代码语料中筛选包含 `pandas`、`sklearn`、`matplotlib`、`seaborn` 的 Python 文件，形成数据科学代码模型。大数据可用 `streaming=True` 逐条处理，避免全部下载到内存。

过滤不是越窄越好。过度过滤会让模型只会重复少数库的模板，失去通用 Python 语法；过滤太宽又会稀释目标领域。应先明确模型最终用途，再决定数据分布。

训练集与验证集应先按文件、仓库或其他自然文档边界划分，再进行切块；若先切块后随机划分，同一文件的相邻甚至重复片段可能同时出现在两侧，造成数据泄漏和虚高的验证指标。

#### 第二步：选择或训练领域 Tokenizer

本项目 Day 08 已用 `train_new_from_iterator()` 在 CodeSearchNet 上训练新词表，Day 09 又从底层实现了 BPE、WordPiece、Unigram。这些工作在 CLM 中直接发挥作用：代码专用 Tokenizer 可以减少常见标识符、缩进模式和 API 名称被过度切碎。

Tokenizer 与模型配置必须匹配：

```text
config.vocab_size == len(tokenizer)
bos_token_id / eos_token_id 与 tokenizer 一致
context_length 与训练切块长度一致
```

还要显式定义 BOS、EOS、PAD、UNK 等特殊 token 的用途，抽样检查代码缩进、换行、常见 API 和罕见标识符如何切分，再冻结 Tokenizer 版本。训练过程中更换词表会使已有 Embedding 与 LM Head 失效。

#### 第三步：切成固定上下文块

```python
outputs = tokenizer(
    examples["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)
```

每篇长文档可产生多个样本。课程丢弃不足 `context_length` 的尾块，简单但会浪费数据。更高效的做法是：完整 tokenize，多文档之间插入 `eos_token_id`，拼接后统一切块。

文档边界必须有 EOS，否则模型可能把两个不相关文件误当作连续文本。

#### 第四步：只加载配置，不加载预训练权重

```python
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    n_positions=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
```

`AutoConfig.from_pretrained("gpt2")` 只借用 GPT-2 的结构配置；`GPT2LMHeadModel(config)` 随机初始化权重。若调用 `from_pretrained()`，就不是从头训练。

#### 第五步：构造 CLM batch

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

`mlm=False` 表示因果语言建模。Collator 通常令 `labels` 与 `input_ids` 相同；**真正的 shift 在模型内部计算 loss 时完成**：最后一个 logit 没有下一 token 标签，第一个 token 也没有前一位置预测它。

这与项目中已纠正的概念一致：不是“把 logits 向右移动”，而是用位置 `t` 的 logits 预测输入位置 `t+1` 的 token。

#### 第六步：训练与资源控制

从头训练比微调更依赖训练工程：

- 梯度累积：多个小 batch 累积后再更新，模拟更大有效 batch。
- Warmup：初期逐步升高学习率，避免随机初始化模型被大步更新破坏。
- 学习率调度：如 cosine decay。
- 混合精度：降低显存并提升 GPU 吞吐。
- 定期验证、保存 checkpoint、记录 loss/perplexity。
- 保留数据、Tokenizer、配置和代码版本，保证可复现。

有效 batch size 不能只看 `per_device_train_batch_size`：

```text
有效 batch size
= 单卡 batch × 梯度累积步数 × GPU 数量
```

#### 第七步：生成与任务评估

训练 loss 下降只表示模型更会预测验证语料中的下一个 token，不保证生成代码能运行。应使用 `pipeline("text-generation")` 对真实提示生成，再做：

- 语法解析或编译检查。
- 单元测试与执行正确率。
- API 使用正确性。
- 安全性检查。
- 重复、截断、幻觉分析。

最终验证必须从磁盘 checkpoint 重新加载模型和 Tokenizer，确认 `config.vocab_size`、特殊 token ID 与生成配置一致；仅测试训练进程内存中的对象，无法证明保存产物可复用。

### 7.4 自定义加权 loss 的逻辑

课程用 `plt`、`pd`、`fit`、`predict` 等关键词识别数据科学相关样本，提高这些样本的 loss 权重：

```text
逐 token 交叉熵
→ 汇总为每个样本 loss
→ 统计每个样本的关键 token 数
→ 权重 = 关键 token 数 + 1
→ 对 batch 做加权平均
```

“加 1”保证不含关键词的样本仍参与训练。此方法的意义是把业务优先级注入优化目标，但风险是模型可能过度偏向关键词表。关键词还必须确认是单 token；若被切成多个 token，简单 ID 匹配会漏检。

Accelerate 适合这种自定义循环，因为它保留 PyTorch 训练结构，同时处理设备放置、多卡同步、梯度累积与模型保存。

### 7.5 难点与易混淆点

> **从头训练模型不等于从头训练 Tokenizer。** 二者是独立决策，但不能只检查词表大小。若新 Tokenizer 重排了 token ID，旧模型的 Embedding 与 LM Head 语义会错位；若保留旧 ID、只追加新 token，还必须调用 `model.resize_token_embeddings(len(tokenizer))` 扩展 Embedding/LM Head，并训练新增权重。否则应重新初始化相关权重，不能把它当成普通微调。

> **`labels == input_ids` 不表示预测自己。** 因果掩码与模型内部 shift 共同形成“当前位置预测下一位置”。

> **不要轻易令 `pad_token = eos_token`。** `DataCollatorForLanguageModeling(mlm=False)` 会按 `pad_token_id` 把标签改为 `-100`；若 PAD 与 EOS 共用 ID，真实文档 EOS 也可能失去监督，模型学不到正确停止。本流程为 GPT-2 Tokenizer 新增独立 PAD；另一种做法是依据 `attention_mask` 精确构造 labels，只忽略真正的 padding。

> **模型变大不能补救语料错误。** 数据泄漏、重复、许可证、隐私、恶意代码都会进入权重。

---

## 8. 任务五：抽取式问答

课程原文：[抽取式问答](https://huggingface.co/learn/llm-course/zh-CN/chapter7/7)

### 8.1 是什么

抽取式问答给定问题和上下文，答案必须是上下文中的一段连续文本：

```text
问题：谁提出了 Transformer？
上下文：……Transformer 架构由 Vaswani 等人在 2017 年提出……
答案：Vaswani 等人
```

BERT 类 Encoder 将问题与上下文一起编码，QA Head 为每个 token 输出两组 logits：

```text
start_logits：该 token 是答案起点的分数
end_logits：该 token 是答案终点的分数
```

训练标签不是答案字符串，而是答案在当前 token 序列中的 `start_positions` 和 `end_positions`。

### 8.2 意义与边界

抽取式 QA 擅长证据已明确存在于文档中的事实问题，结果天然可追溯到原文片段。它不能自由综合多个段落，也不能可靠回答上下文没有提供的开放问题。

这与本项目 RAG 路线紧密相关：

```text
语义检索/FAISS 找到候选文档
→ 抽取式 QA 在候选上下文中定位答案
→ 返回答案及原文字符范围
```

RAG 的生成器可以重写和综合，抽取式 QA 则更强调答案必须有原文依据。

### 8.3 原始数据结构

SQuAD 样本包含：

```python
{
    "id": "...",
    "question": "...",
    "context": "...",
    "answers": {
        "text": ["answer text"],
        "answer_start": [515],
    },
}
```

`answer_start` 是答案在原始 context 中的字符起点，不是 token 索引。训练集通常只保留一个答案；验证集可能包含多个可接受答案，评估时取与预测匹配最好的参考答案。

### 8.4 训练数据预处理

#### 第一步：问题和上下文组成句子对

```python
tokenizer(
    questions,
    contexts,
    truncation="only_second",
    max_length=384,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```

- `only_second`：只截断 context，不截断问题。
- `stride`：相邻窗口保留重叠，降低答案被切断的概率。
- `return_overflowing_tokens`：保留长上下文产生的所有窗口。
- `return_offsets_mapping`：记录每个 token 对应原文字符范围。
- `overflow_to_sample_mapping`：记录每个窗口来自哪个原始样本。

这套流程依赖支持 `offset_mapping` 和 `sequence_ids()` 的 Fast Tokenizer。还要确保问题本身短于窗口可用容量，并让 `max_length` 与 `stride` 足以使每个训练答案完整落入至少一个窗口；否则所有窗口都会被标成“当前窗口无答案”。

#### 第二步：字符答案转成 token 标签

转换逻辑：

1. 从 `answer_start` 与答案长度得到 `[start_char, end_char)`。
2. 用 `sequence_ids()` 找出当前窗口中 context 的 token 范围；问题、特殊 token、padding 不能成为答案。
3. 检查答案是否完整落在当前窗口。
4. 若不在，令标签为 `(cls_index, cls_index)`；课程中的 BERT 右侧填充使 `cls_index=0`，其他架构不能硬编码为 0。
5. 若在，用 `offset_mapping` 找到覆盖答案字符范围的最小 start/end token。

```text
字符位置（原始标注）
→ offset_mapping
→ 当前窗口内的 token 起止位置
→ start_positions / end_positions
```

这是整个 QA 训练最难的部分。一个 off-by-one 错误就会让模型学习错误边界，因此必须把 token 区间 decode 回文本，与理论答案逐样本核对。

### 8.5 为什么验证预处理与训练不同

训练只需给模型 token 起止标签；验证还要把模型预测还原回原始文本，并把同一原始样本的多个窗口重新聚合。

因此验证特征要保留：

- `example_id`：窗口属于哪个原始问题。
- `offset_mapping`：token 如何映射回原始 context。

具体做法是遍历 `overflow_to_sample_mapping`，把原始数据中的 `examples["id"][sample_idx]` 写入每个窗口的 `example_id`。同时把问题、特殊 token、padding 的 offset 设为 `None`，后处理时便可直接排除非 context 位置。

### 8.6 模型训练

```python
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
```

模型接收 `start_positions`、`end_positions` 后，分别计算起点和终点交叉熵并组合为 loss。QA Head 通常是新初始化的，必须微调。

标准 Trainer 能完成训练，但完整 SQuAD 指标依赖窗口到原样本的映射和原始 context，超出普通 `compute_metrics(eval_preds)` 的简单输入。可以训练后统一后处理，或用自定义 Trainer/Accelerate 在验证循环中收集全部 logits。

### 8.7 后处理：从 logits 得到答案文本

每个原始样本可能对应多个窗口。完整逻辑如下：

```text
收集所有窗口的 start_logits / end_logits
→ 按 example_id 找到该问题的全部窗口
→ 每个窗口分别取 start、end 的 top n_best 候选
→ 组合候选起止对
→ 排除非 context、end < start、答案过长等无效组合
→ 分数 = start_logit + end_logit
→ 在所有窗口中选择最高分组合
→ 用 offset_mapping 从原 context 切出字符串
```

课程直接使用 `start_logit + end_logit` 作为候选跨度的常见启发式分数，因此不必先做 softmax。要注意：logits 不是已经归一化的 log 概率，不同窗口的 softmax 分母也不同，不能用“softmax 单调”证明跨窗口分数严格可比。更严谨的系统应校准跨窗口分数，并可使用 max-context 规则减少重叠窗口边界上的重复和偏置。

不能只对 start、end 分别 `argmax`：独立最大值可能组成 `end < start`、跨越超长范围或落在不同语义片段中的无效答案。

### 8.8 Exact Match 与 F1

- **Exact Match（EM）**：规范化后预测字符串与某个参考答案完全一致。
- **F1**：按规范化后 token 的出现次数计算多重集重叠，再求 Precision/Recall 的调和平均，允许部分重叠。

EM 严格，F1 更能反映“答案大体正确但边界略有差异”。验证集有多个参考答案时，应与全部参考答案比较并取最佳值。

SQuAD v1 默认问题都有答案；SQuAD v2 包含无答案问题。处理 v2 时还需比较“无答案分数”和最佳文本答案分数，并调节阈值，不能机械沿用 v1 流程。

### 8.9 难点与易混淆点

> **`answer_start` 是字符位置，不是 token 位置。** 必须经 offset mapping 转换。

> **一个原始样本可能变成多个训练/验证特征。** `overflow_to_sample_mapping` 负责从窗口回到样本，`example_id` 负责评估聚合。

> **`(cls_index, cls_index)` 在课程流程中表示当前窗口没有答案。** 它不一定表示原始问题无答案；答案可能在同一问题的另一个窗口。

> **抽取式 QA 不调用 `generate()`。** 它选择输入中的起止位置；生成式 QA 才逐 token 生成答案。

---

## 9. 最容易混淆的统一辨析

### 9.1 四种“mask”不是一回事

| 名称 | 形式 | 作用 |
|---|---|---|
| MLM `[MASK]` | 输入中的特殊 token ID | 隐去某个词，让模型恢复 |
| `attention_mask` | 通常为 0/1 张量 | 让模型忽略 padding 等位置 |
| causal mask | 对 query 位置 `i` 屏蔽所有 `j > i` 的 key | 禁止 Decoder 看未来 token |
| loss mask（`-100`） | labels 中的忽略值 | 让交叉熵不计算该位置 |

它们可能同时出现。例如 Seq2Seq 训练既有 Encoder 的 `attention_mask`，又有 Decoder 的 causal mask，labels padding 还会被写成 `-100`。常见 Transformers 接口约定 `attention_mask` 中 1 表示有效、0 表示忽略，但应以具体模型 API 为准；`-100` 只影响 loss，不改变注意力可见性。

### 9.2 三种 labels 对齐

```text
MLM：labels 先复制 input_ids，动态遮盖后只在被选位置保留原 token
CLM：labels 等于 input_ids，模型内部用位置 t 的 logits 对齐位置 t+1
Seq2Seq：labels 是目标文本，decoder_input_ids 是 labels 右移后的真实前文
QA：labels 是 start_positions 和 end_positions 两个索引
```

不要把“CLM 内部 shift”和“Seq2Seq 构造 decoder 输入的右移”写成同一个 API 动作。概念目标相同——不让当前位置直接看到答案——但实现位置由具体模型决定。

### 9.3 三种 padding 忽略机制

```text
输入 padding：input_ids 填 pad_token_id，attention_mask 填 0
分类/NER/Seq2Seq 标签 padding：labels 填 -100
GPT-2 无 pad token：可把 pad_token 指向 eos_token，但仍要正确提供 attention_mask
```

### 9.4 训练、验证、推理的差异

| 阶段 | 是否有真实 labels | Decoder 前文来源 | 是否更新参数 |
|---|---|---|---|
| 训练 | 有 | 真实目标前文（Teacher Forcing） | 是 |
| 验证 | 有，仅用于评分 | 应模拟推理并由模型生成 | 否 |
| 推理 | 无 | 模型已生成 token | 否 |

生成任务若在验证时继续使用真实目标前文，得到的 loss 可用于观察拟合，但不能替代真实生成指标。

### 9.5 Token、word、char 三种坐标系

- MLM 全词遮盖：`word_ids()` 把 token 归到词。
- NER 标签对齐：词级标签通过 `word_ids()` 扩展到子词。
- QA：字符级答案通过 `offset_mapping` 转为 token 区间，再从 token 区间还原字符文本。

对齐错误通常不是模型问题，而是坐标系混用。

### 9.6 loss 与业务指标不能互相替代

| 任务 | 训练 loss 反映 | 仍需关注 |
|---|---|---|
| MLM/CLM | token 预测交叉熵 | perplexity、领域效果、生成质量 |
| 翻译 | Teacher Forcing 下的目标 token 预测 | BLEU、术语、忠实度、流畅度 |
| 摘要 | 参考摘要 token 预测 | ROUGE、事实一致性、覆盖与压缩 |
| QA | 起止位置分类 | EM、F1、无答案与长文档表现 |

loss 下降说明优化目标在改善，不等于最终业务体验一定改善。

---

## 10. 从一个新任务开始时的标准执行清单

### 10.1 任务定义

1. 明确输入、输出和是否必须忠实于原文。
2. 判断是理解、条件生成、自回归生成还是位置抽取。
3. 选择 Encoder-only、Decoder-only 或 Encoder-Decoder。
4. 明确最终业务指标，不先用 accuracy 代替一切。

### 10.2 数据检查

1. 打印 `DatasetDict`、列名、特征类型和 split 大小。
2. 随机抽样，不只看第一条。
3. 检查空值、异常长度、重复、语言、标签范围和数据泄漏。
4. 确认训练/验证划分不会让同源文本跨 split 泄漏。
5. 先做小样本端到端实验，再扩大数据。

### 10.3 预处理验证

1. decode 一个 tokenized 输入，确认特殊 token 与截断方向正确。
2. 检查 batch 中每个张量的 shape。
3. 检查 `-100` 只出现在应忽略的位置。
4. Seq2Seq 检查 `decoder_input_ids` 是否为目标右移结果。
5. QA 将 start/end token decode 回文本，确认与原答案一致。
6. 长文本检查窗口数、stride 和尾块处理。

### 10.4 训练验证

1. 用极小数据尝试过拟合，验证数据—标签—loss 链路是否正确。
2. 同时记录 train loss、eval loss 和任务指标。
3. 不在训练后重新实例化随机 Head。
4. 保存模型、Tokenizer、配置和训练参数。
5. 用固定种子和稳定验证预处理做可比实验。

### 10.5 推理验证

1. 从保存的 checkpoint 重新加载，而不是继续使用内存对象。
2. 用 `pipeline` 或显式 Tokenizer + Model 跑真实样本。
3. 生成任务检查长度、重复、事实和解码参数。
4. QA 检查答案文本、字符范围和上下文来源。
5. 记录失败样本并按错误类型分析，而非只看平均分。

---

## 11. 与本项目已有知识的衔接

本项目 Day 01–10 已经为这五类任务准备了完整前置能力：

- Day 01–04：`pipeline`、Tokenizer、特殊 token、padding、truncation、attention mask。
- Day 05–06：Trainer 与手写训练循环、AdamW、scheduler、评估、Accelerate。
- Day 07：Datasets 的 `filter/map`、流式处理、语义搜索与 FAISS。
- Day 08：快速 Tokenizer、`word_ids()`、`offset_mapping`、NER/QA 管道内部流程、滑动窗口。
- Day 09：BPE、WordPiece、Unigram、Viterbi、自定义 Tokenizer。
- Day 10：token classification 的标签对齐、`-100`、专用 Collator、实体级评估。

对应的项目主题文档包括：[Tokenizer 基础](../03-NLP-Basic/tokenizer-basics.md)、[Transformer 基础](../04-Transformer/transformer-basics.md)、[BERT 微调基础](../05-BERT/bert-finetuning.md)、[BERT NER](../05-BERT/ner-notes.md) 与 [语义搜索](../08-RAG/semantic-search.md)。

Chapter 7.3–7.7 不是五套互不相关的新代码，而是把这些能力重新组合：

```text
Tokenizer 对齐能力
+ Datasets 批量变换
+ 任务专用 Collator
+ 任务专用 Model Head
+ 通用训练循环
+ 任务专用后处理与指标
= 完整 NLP 任务
```

最值得掌握的不是函数名，而是这条判断链：

```text
原始标注处于哪种坐标系？
→ 模型输出处于哪种坐标系？
→ 中间如何对齐？
→ 哪些位置必须忽略？
→ 怎样还原成最终业务结果？
```

当这五个问题能独立回答时，换数据集、换 checkpoint、换 Trainer 或 Accelerate，都只是工程实现变化。

## 12. 建议实践顺序

1. **MLM 领域适应**：复用现有 BERT/Trainer 基础，重点练习拼接切块、动态遮盖和 perplexity。
2. **翻译**：理解 `DataCollatorForSeq2Seq`、Teacher Forcing、`generate()` 与 BLEU。
3. **摘要**：复用 Seq2Seq 骨架，重点比较 lead-3、ROUGE 与事实一致性。
4. **抽取式 QA**：集中攻克字符—token—窗口三层映射和后处理。
5. **从头训练 CLM**：最后整合新 Tokenizer、流式数据、随机初始化、梯度累积和自定义 Accelerate 循环。

这个顺序从“改动最少的微调”逐步走到“数据与训练工程最重的预训练”，能最大限度复用已掌握知识，也更容易定位错误。
