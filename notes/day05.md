# 微调一个预训练模型

## 预处理数据
用模型中心的数据在 PyTorch 上训练句子分类器的一个例子
[代码](notebooks\day05-test.ipynb)
仅用两句话训练模型不会产生很好的效果，需要准备一个更大的数据集才能得到更好的训练结果。
在本节中，我们以 MRPC（微软研究院释义语料库）数据集为例，该数据集由 5801 对句子组成，每个句子对带有一个标签来指示它们是否为同义（即两个句子的意思相同）。它的数据体量小，容易对其进行训练。

### 从模型中心（Hub）加载数据集

使用 MRPC 数据集中的 GLUE 基准测试数据集 作为我们训练所使用的数据集，它是构成 MRPC 数据集的 10 个数据集之一，作为一个用于衡量机器学习模型在 10 个不同文本分类任务中性能的学术基准。
[代码](notebooks\day05-test.ipynb)

`load_dataset("glue", "mrpc")` 返回 `DatasetDict`，含训练集(3668 对)、验证集(408 对)、测试集(1725 对)。每行四个字段：`sentence1`、`sentence2`、`label`、`idx`。数据默认缓存至 `~/.cache/huggingface/datasets`，可通过 `HF_HOME` 环境变量自定义路径。底层以 Apache Arrow 格式存储，每次只加载需要使用的数据到内存，不需要一次性加载整个数据集。

label 是 `ClassLabel` 类型，内部用整数 0/1 映射到标签名：`0=not_equivalent`（不同义）、`1=equivalent`（同义）。标签已是整数，不需要额外预处理。

### 预处理数据集
将文本转换为数字。通过一个 Tokenizer 完成的，我们可以向 Tokenizer 输入一个句子或一个句子列表
以下代码表示对每对句子中的所有第一句和所有第二句进行 tokenize
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
不过在将两句话传递给模型，预测这两句话是否是同义之前，我们需要给这两句话依次进行适当的预处理。Tokenizer 不仅仅可以输入单个句子，还可以输入一组句子，并按照 BERT 模型所需要的输入进行处理：
[代码](notebooks\day05-test.ipynb)

### GLUE 通用预处理函数

GLUE 各任务列名不一致——CoLA/SST-2 用 `sentence`，MRPC/STS-B/RTE/WNLI 用 `sentence1` + `sentence2`，MNLI 用 `premise` + `hypothesis`，QQP 用 `question1` + `question2`，QNLI 用 `question` + `sentence`。下面这个函数自动适配所有场景：

```python
def preprocess_glue(examples, tokenizer, task_keys=None):
    """适用于任何 GLUE 任务的预处理函数。

    参数
    ----
    examples : dict
        数据集的一个 batch，由 Dataset.map(batched=True) 传入。
    tokenizer : AutoTokenizer
        分词器。
    task_keys : tuple[str] | None
        传给 tokenizer 的参数名，如 ("sentence",) 或 ("sentence1", "sentence2")。
        传 None 时自动从 examples 的 key 推断（过滤掉 label、idx 等非文本列）。

    返回
    ----
    dict
        tokenizer 的输出（input_ids、attention_mask、token_type_ids 等），
        同时保留原始 label 列以便 Trainer 使用。
    """
    if task_keys is None:
        # 自动检测文本列：取所有 value 为字符串的列名，保留常见顺序
        text_keys = [k for k, v in examples.items()
                     if k not in ("label", "idx", "index") and isinstance(v[0], str)]
    else:
        text_keys = list(task_keys)

    if len(text_keys) == 1:
        # 单句任务：CoLA, SST-2
        return tokenizer(examples[text_keys[0]], truncation=True)
    elif len(text_keys) == 2:
        # 句子对任务：MRPC, MNLI, QQP 等
        return tokenizer(
            examples[text_keys[0]], examples[text_keys[1]], truncation=True
        )
    else:
        raise ValueError(f"无法确定文本列，检测到: {text_keys}")


# 用法示例
tokenized_datasets = raw_datasets.map(
    preprocess_glue,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=raw_datasets["train"].column_names,
)
```

这样切到 SST-2、CoLA、MNLI 都不需要改预处理逻辑，函数自动适配。

### 动态填充

将数据整理为一个 batch 的函数称为 collate 函数。默认 collate 函数直接将列表转为 PyTorch 张量，但变长输入无法直接拼接。解决方法是推迟填充：tokenize 时不传 `padding`，只在每个 batch 构建时填充到该 batch 内的最大长度——这就是 `DataCollatorWithPadding` 的职责。

[代码](notebooks\day05-test.ipynb)
DataCollatorWithPadding：动态批处理填充器。把每个 batch 内不同长度的句子对齐到该 batch 最长那条，而不是对齐到整个数据集最大长度。

[代码](notebooks\day05-test.ipynb)
例如从训练集取 8 条样本，长度从 32 到 67 不等，经 data_collator 后统一填充为 67（该 batch 最大长度）。没有动态填充则需全局填充到 512 或全数据集最大长度。



## 使用 Trainer API 微调模型

### training
Transformers 提供了一个 Trainer 类，可以帮助你在数据集上微调任何预训练模型。在上一节中完成所有数据预处理工作后，你只需完成几个步骤来定义 Trainer 。最困难的部分可能是准备运行 Trainer.train() 所需的环境，因为在 CPU 上运行速度非常慢。如果你没有设置 GPU，可以使用 Google Colab （国内网络无法使用） 上获得免费的 GPU 或 TPU。
[代码](notebooks\day05-train.ipynb)
定义 Trainer 之前第一步要定义一个 TrainingArguments 类，它包含 Trainer 在训练和评估中使用的所有超参数。你只需要提供的参数是一个用于保存训练后的模型以及训练过程中的 checkpoint 的目录。对于其余的参数你可以保留默认值，这对于简单的微调应该效果就很好了。

定义模型。使用 AutoModelForSequenceClassification 类，它有两个参数：
[代码](notebooks\day05-train.ipynb)
在实例化此预训练模型后会收到警告。这是因为 BERT 没有在句子对分类方面进行过预训练，所以预训练模型的 head 已经被丢弃，而是添加了一个适合句子序列分类的新头部。这些警告表明一些权重没有使用（对应于被放弃的预训练头的权重），而有些权重被随机初始化（对应于新 head 的权重）。

关于 `token_type_ids`：BERT 预训练时使用了"下一句预测"任务——给模型输入成对的句子（其中一半连续、一半随机来自不同文档），模型判断第二个句子是否紧接第一个。因此 BERT 依赖 `token_type_ids`（0=第一句，1=第二句）来区分 A/B 句子。但并非所有模型都需要这一层：例如 DistilBERT 没有做下一句预测，因此不会返回 `token_type_ids`。只要 tokenizer 和 model 使用同一个 checkpoint，tokenizer 就知道向模型提供什么，无需手动处理。

定义一个 Trainer 把到目前为止构建的所有对象 —— model ，training_args，训练和验证数据集， data_collator 和 tokenizer 传递给 Trainer ：
[代码](notebooks\day05-train.ipynb)
在这里传递 tokenizer 时， Trainer 默认使用的 data_collator 是之前预定义的 DataCollatorWithPadding。所以你可以在本例中可以跳过 data_collator=data_collator 一行。

要在我们的数据集上微调模型，我们只需调用 Trainer 的 train() 方法：
[代码](notebooks\day05-train.ipynb)
开始微调（在 GPU 上应该需要几分钟），每 500 步报告一次训练损失。然而它不会告诉你模型的性能（或质量）如何。这是因为：
我们没有告诉 Trainer 在训练过程中进行评估，比如将 `eval_strategy` 设置为 `"step"`（在每个 eval_steps 步骤评估一次）或 `"epoch"`（在每个 epoch 结束时评估）。
我们没有为 Trainer 提供一个 compute_metrics() 函数来计算上述评估过程的指标（否则评估将只会输出 loss，但这不是一个非常直观的数字）。

### 评估
构建一个有用的 compute_metrics() 函数，并在下次训练时使用它。该函数必须接收一个 EvalPrediction 对象（它是一个带有 predictions 和 label_ids 字段的参数元组），并将返回一个字符串映射到浮点数的字典（字符串是返回的指标名称，而浮点数是其值）。为了从我们的模型中获得预测结果，可以使用 Trainer.predict() 命令：
[代码](notebooks\day05-train.ipynb)

predict() 方法的输出另一个带有三个字段的命名元组: predictions label_ids 和 metrics 
metrics 字段将只包含所传递的数据集的损失,以及一些时间指标(总共花费的时间和平均预测时间)

predictions：模型的 logits 输出，形状 (样本数, 类别数)，比如 (408, 2)，还不是最终标签。
label_ids：真实标签，形状 (样本数,)，就是数据集里的 label 列。
metrics：评估指标字典，比如 {'test_loss': 0.68, 'test_runtime': 5.2, ...}。但如果是分类任务，loss 以外的指标需要你提供 compute_metrics 函数才会有，否则只返回 loss。

为了将它们转化为可以与我们的标签进行比较的预测值,我们需要找出第二个维度上取值最大的索引:
predictions 形状是 (样本数, 类别数)，每一行是一个样本，列是各类别的 logit 分数。取第二维的最大值索引，就是把每个样本判给分数最高的那个类：
predictions = [[-2.7,  2.8],   # 第 1 列更高 → 判为类别 1
               [ 3.1, -2.6]]   # 第 0 列更高 → 判为类别 0
np.argmax(predictions, axis=-1)  # → [1, 0]
不取第二维的话，argmax 默认在展平后的整张矩阵上找，输出一个数字而不是每个样本的预测。
[代码](notebooks\day05-train.ipynb)


将这些 preds 与标签进行比较。为了构建我们的 compute_metric() 函数，我们将使用Evaluate 库中的指标。我们可以像加载数据集一样轻松地加载与 MRPC 数据集关联的指标，这次是使用 evaluate.load() 函数。返回的对象有一个 compute() 方法，我们可以用它来进行指标的计算：
[代码](notebooks\day05-train.ipynb)

得到的确切结果可能会有所不同，因为模型头部的随机初始化可能会改变指标。本实验实际结果：`{'accuracy': 0.833, 'f1': 0.885}`（bert-base-uncased，3 epochs，从 checkpoint-1377 加载评估）。BERT 论文中基础模型报告 F1=88.9，我们的 uncased 模型结果与论文接近。注意：如果直接使用未训练的模型评估（classification head 随机初始化），F1 会降至 0.81。


最后把所有东西打包在一起，我们就得到了 compute_metrics() 函数：

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

为了查看模型在每个训练周期结束时的好坏，下面是我们如何使用 compute_metrics() 函数定义一个新的 Trainer

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
请注意，我们设置了一个新的 TrainingArguments ，其 eval_strategy 设置为 epoch 并且创建了一个新模型。如果不创建新的模型就直接训练，就只会继续训练我们已经训练过的模型。注意参数名是 `eval_strategy`，旧版 `evaluation_strategy` 在 Transformers 4.57 后已废弃。

trainer.train()
这一次，它将在每个 epoch 结束时在训练损失的基础上报告验证损失和指标。同样，由于模型的随机头部初始化，达到的准确率/F1 分数可能与我们发现的略有不同，这是由于模型头部的随机初始化造成的，但应该相差不多。 Trainer 可以在多个 GPU 或 TPU 上运行，并提供许多选项，例如混合精度训练（在训练的参数中使用 fp16 = True ）。

---

## 今日知识总结

### 最小训练一步：前向算 loss → 反向算梯度 → 更新参数

PyTorch 里一个训练 step 的核心流程只有三行：`loss = model(**batch).loss` 计算损失，`loss.backward()` 反向传播计算梯度，`optimizer.step()` 根据梯度更新参数。`AutoModelForSequenceClassification` 在传入 `labels` 时会自动计算交叉熵损失，不需要手动写 loss 函数。`AdamW` 是 Adam 的改进版，将权重衰减从 L2 正则化解耦出来，是 BERT 微调的标配优化器。实际训练循环中必须在前加上 `optimizer.zero_grad()`，否则梯度会累积。

### GLUE 与 MRPC 数据集

GLUE 是包含 10 个文本分类任务的 NLP 基准测试集，MRPC（微软释义对语料库）是其中之一，任务是判断两个句子是否同义（二分类）。训练集 3668 对、验证集 408 对、测试集 1725 对，数据量小、适合快速微调实验。`load_dataset("glue", "mrpc")` 返回 `DatasetDict`，包含 `sentence1`、`sentence2`、`label`、`idx` 四个字段。label 是 `ClassLabel` 类型，内部用整数 0/1 映射到标签名 `not_equivalent`/`equivalent`，不需要额外做标签预处理。

### 句子对预处理与 map() 的高效批处理

MRPC 需要同时传入两个句子，tokenizer 会拼接成 `[CLS] A [SEP] B [SEP]` 并用 `token_type_ids`（全 0 的第一句 vs 全 1 的第二句）区分。`Dataset.map()` 比直接 tokenize 全量数据更高效：它接收一个处理函数，用 `batched=True` 一次传入一批数据，结合 Rust 实现的快速 tokenizer 大幅提升预处理速度；返回的仍是 Dataset 对象，支持后续的 Trainer 流程。可以在 `map()` 中通过 `remove_columns` 去掉原始文本列，只保留模型需要的数字列。另一个常用参数 `num_proc` 可利用多进程进一步加速，但使用快速 tokenizer 时通常不需要。

### 动态填充：在 batch 级别对齐而非全局对齐

`DataCollatorWithPadding` 是动态批处理填充器。它的逻辑是：每个 batch 内对齐到该 batch 最长的那条样本，而不是全局对齐到整个数据集最大长度或模型最大长度（512）。Tokenize 时不传 `padding=True`，让每条数据保留原始长度；等 DataLoader 取出一个 batch 后，再由 collator 补齐到该 batch 的最大长度。这样短句子不会浪费算力在无意义的 padding token 上，TPU 场景除外（TPU 偏好固定形状）。

### Trainer API 封装了训练全流程

Hugging Face 的 `Trainer` 将模型、训练参数、数据集、collator 和评估函数封装为一个统一的训练接口。`TrainingArguments` 配置所有超参数——唯一必传的是 `output_dir`（模型保存目录），其余如 `per_device_train_batch_size=8`、`num_train_epochs=3`、`learning_rate=5e-5` 均有默认值。`Trainer.train()` 启动训练，自动处理 batch 迭代、梯度累积、checkpoint 保存等细节。训练完成后 `Trainer.predict()` 对验证集推理，返回一个包含 `predictions`（logits）、`label_ids`（真实标签）、`metrics`（损失和时间）的命名元组。logits 经过 `np.argmax(predictions, axis=-1)` 转换为预测标签。

### Trainer 评估与 compute_metrics

默认训练不会自动评估验证集，需要在 `TrainingArguments` 中设置 `eval_strategy="epoch"`（每个 epoch 评估一次）或 `eval_strategy="steps"`（每 N 步评估一次）。`compute_metrics` 函数接收 `EvalPrediction` 对象（含 `predictions` 和 `label_ids`），返回指标字典。使用 `evaluate.load("glue", "mrpc")` 加载 MRPC 专用评估器（内置 accuracy 和 F1 计算规则），切换其他 GLUE 任务只需改任务名（如 `"glue", "sst2"`）。

### 两个常见坑

第一个坑：`evaluation_strategy` 在 Transformers 4.57 后已重命名为 `eval_strategy`，使用旧名会报 `TypeError: unexpected keyword argument`。第二个坑：训练完成后如果重新执行 model 定义 cell（`AutoModelForSequenceClassification.from_pretrained(checkpoint)`），model 对象会回到未训练状态（classification head 被重新随机初始化），导致评估指标骤降（F1 从 0.88 跌到 0.81）。避免方法是 train 后立刻 evaluate，或从保存的 checkpoint 直接加载已训练权重进行评估。

