# Day 12：机器翻译（Seq2Seq + MarianMT）

Day 11 学了 MLM 微调。Day 12 学序列到序列（Seq2Seq）任务：用 Helsinki-NLP/opus-mt-en-zh 做英译中，掌握 `text_target` 双语言 tokenize、`DataCollatorForSeq2Seq`、SacreBLEU 评估、`predict_with_generate` 和 `Seq2SeqTrainer`。

翻译是另一个 sequence-to-sequence 任务，输入一个序列输出另一个序列，与文本摘要本质相同。

## 准备数据

### OPUS100 数据集

原课程使用 KDE4，但 datasets 5.0.0 不再支持脚本式加载（`RuntimeError: Dataset scripts are no longer supported`），且 KDE4 无 Parquet 版本。改用 OPUS100 替代：
```Py
from datasets import load_dataset
raw_datasets = load_dataset("opus100", "en-zh")
raw_datasets
```

### 处理数据
将所有文本转换为 token IDs 的集合，这样模型才可以理解它们。对于这个任务，我们需要同时对原始文本和翻译后的文本同时进行 tokenize。首先，我们需要创建 tokenizer 对象。如果你使用下面的代码微调另一对语言，请记得更改下面代码中的 checkpoint。 Helsinki-NLP 组织提供了超过一千个多语言模型。
```Py
from transformers import AutoTokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
```

我们的数据准备相当简单。只有一点要记住；你需要确保 tokenizer 处理的目标是输出语言（在这里是汉语）。你可以通过将目标语言传递给 tokenizer 的 __call__ 方法的 text_targets 参数来完成此操作。
```Py
max_length = 128
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["zh"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs
```

我们不需要对待遇测的目标设置注意力掩码，因为模型序列到序列的不会需要它。不过，我们应该将填充（padding） token 对应的标签设置为 -100 ，以便在 loss 计算中忽略它们。由于我们正在使用动态填充，这将在稍后由我们的数据整理器完成，但是如果你在此处就打算进行填充，你应该调整预处理函数，将所有填充（padding） token 对应的标签设置为 -100 。

我们现在可以一次性使用上述预处理处理数据集的所有数据。
```Py
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

## 使用 Trainer API 微调模型
使用 Trainer 的代码将与以前相同，只是稍作改动：我们在这里将使用 Seq2SeqTrainer ，它是 Trainer 的子类，它使用 generate() 方法来预测输入的输出结果，并且可以正确处理这种序列到序列的评估。

首先，我们需要一个模型来进行微调。我们将使用常用的 AutoModel API：
```Py
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

### 数据整理
在这个任务中，我们需要一个数据整理器来动态批处理填充。因此，我们不能像 第三章 那样直接使用 DataCollatorWithPadding ，因为它只填充输入的部分（inputs ID、注意掩码和 token 类型 ID）。我们的标签也应该被填充到所有标签中最大的长度。而且，如前所述，用于填充标签的填充值应为 -100 ，而不是 tokenizer 默认的的填充 token，这样才可以在确保在损失计算中忽略这些填充值。

上述的这些需求都可以由 DataCollatorForSeq2Seq 完成。它与 DataCollatorWithPadding 一样，它接收用于预处理输入的 tokenizer ，同时它也接收一个 model 参数。这是因为数据整理器还将负责准备解码器 inputs ID，它们是标签偏移之后形成的，开头带有特殊 token 。由于对于不同的模型架构有稍微不同的偏移方式，所以 DataCollatorForSeq2Seq 还需要接收 model 对象：

```Py
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

### 评估指标
Seq2SeqTrainer 是 Trainer 类的一个子类，它的主要增强特性是在评估或预测时使用 generate() 方法。在训练过程中，模型会利用 decoder_input_ids 和一个特殊的注意力掩码来加速训练。这种方法允许模型在预测下一个token时看到部分目标序列，但确保它不会使用预测token之后的信息。这种优化策略显著提高了训练效率。然而，在实际的推理过程中，我们没有真实的标签值，因此无法生成 decoder_input_ids 和相应的注意力掩码。这意味着我们无法在推理时使用这种训练时的优化方法。

为了确保评估结果能够准确反映模型在实际使用中的表现，我们应该在评估阶段模拟真实推理的条件。这意味着我们需要使用Transformers 库中的 generate() 方法。该方法能够逐个生成token，真实地模拟推理过程，而不是依赖于训练时的优化技巧。要启用这个功能，我们需要在训练时添加 predict_with_generate=True 参数。这样做可以确保我们的评估结果更加接近模型在实际应用中的表现。

用于翻译的传统指标是 BLEU 分数，评估翻译与参考翻译的接近程度。它不衡量模型生成输出的可理解性或语法正确性，而是使用统计规则来确保生成输出中的所有单词也出现在参考的输出中。此外，还有一些规则对重复的词进行惩罚，如果这些词在输出中重复出现（模型输出像“the the the the the”这样的句子）；或者输出的句子长度比目标中的短（模型输出像“the”这样的句子）都会被惩罚。

BLEU 的一个缺点是的输入是已分词的文本，这使得比较使用不同分词器的模型之间的分数变得困难。因此，当今用于评估翻译模型的最常用指标是 SacreBLEU ，它通过标准化的分词步骤解决了这个缺点（和其他的一些缺点）。要使用此指标，我们首先需要安装 SacreBLEU 库：
```Py
!pip install sacrebleu

import evaluate
metric = evaluate.load("sacrebleu")
```

SacreBLEU 指标中待评估的预测和参考的目标译文输入的格式都是文本。它的设计是为了支持多个参考翻译，因为同一句话通常有多种可接受的翻译——虽然我们使用的数据集只提供一个，但在 NLP 中找到将多个句子作为标签的数据集是很常见的。因此，预测结果应该是一个句子列表，而参考应该是一个句子列表的列表。


为了将模型的输出转化为评估指标可以使用的文本，我们将使用 tokenizer.batch_decode() 方法。因为 tokenizer 会自动处理填充的 tokens，所以我们只需要清理标签中的所有 -100 token：
```Py
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # 如果模型返回的内容超过了预测的logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 由于我们无法解码 -100,因此将标签中的 -100 替换掉
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 一些简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels,tokenize='zh')
    return {"bleu": result["score"]}
```


### 微调模型
```Py
from huggingface_hub import notebook_login
notebook_login()
或
huggingface-cli login
```

定义我们的 Seq2SeqTrainingArguments
```Py
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-zh",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
```
这里 `eval_strategy="epoch"` 表示每个 epoch 结束后评估，评估采用 `generate()` 逐 token 生成翻译，耗时较长。
我们设置 fp16=True ，这可以加快在支持 fp16 的 GPU 上的训练速度。
和之前我们讨论的一样，我们设置 predict_with_generate=True 。
我们设置了 push_to_hub=True ，在每个 epoch 结束时将模型上传到 Hub。

我们将所有内容传递给 Seq2SeqTrainer ：
```Py
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=small_dataset["train"],
    eval_dataset=small_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

在开始训练之前，我们先查看一下我们的模型目前的 BLEU 分数：
```Py
trainer.evaluate(max_length=max_length)
=>
{'eval_loss': 1.8813917636871338,
 'eval_model_preparation_time': 0.0035,
 'eval_bleu': 48.06258740843155,
 'eval_runtime': 323.5395,
 'eval_samples_per_second': 12.363,
 'eval_steps_per_second': 0.195}
```

接下来是训练，这也需要一些时间：
```Py
trainer.train()
```



训练完成后
```Py
trainer.evaluate(max_length=max_length)
=>
{'eval_loss': 1.7597671747207642,
 'eval_model_preparation_time': 0.0021,
 'eval_bleu': 49.52315702448586,
 'eval_runtime': 452.004,
 'eval_samples_per_second': 8.849,
 'eval_steps_per_second': 1.106}
```

上传
```Py
trainer.push_to_hub(tags="translation", commit_message="Training complete")
```


## 使用微调后的模型
```Py
from transformers import pipeline
model_checkpoint = "./marian-finetuned-kde4-en-to-zh"
translator = pipeline("translation", model=model_checkpoint)
translator("The bug crashed my system.")
=>
```

## 今日知识总结

### Seq2Seq 任务与 Encoder-Decoder 架构

翻译、摘要都属于 Seq2Seq：输入一个序列，输出另一个序列。MarianMT 是标准的 Encoder-Decoder 模型：Encoder 双向理解源语言（英语），Decoder 自回归生成目标语言（中文）。`AutoModelForSeq2SeqLM` 封装了这套架构，预训练时用 teacher forcing 加速训练。

### text_target：双语 tokenize

翻译任务需要同时 tokenize 源语言和目标语言。`tokenizer()` 的 `text_target` 参数告诉 tokenizer"这段文本是目标语言"：MarianMT 对不同语言的分词规则不同（英语按空格、中文按字），不传 `text_target` 会把中文按英语规则处理，结果全错。输出包含 `input_ids`（源语言）、`attention_mask` 和 `labels`（目标语言）。

### DataCollatorForSeq2Seq

`DataCollatorWithPadding` 只填充输入部分，翻译任务还需要填充 `labels`（目标语言）。`DataCollatorForSeq2Seq` 专门处理这个：动态 batch padding，labels 用 -100 填充保证被忽略，同时从 labels 移位生成 `decoder_input_ids`（整体右移一位，开头补起始符）。需要传入 `model` 参数，因为不同架构的移位方式不同。

### decoder_input_ids 与 Teacher Forcing

`decoder_input_ids = labels` 右移一位，开头补 `<s>`。训练时解码器每一步看到的"前文"是正确的前文（来自 labels），而非自己上一步的预测。当前位置的输入和预测目标刚好错位：看到前 i 个词，预测第 i+1 个词。Teacher forcing 让训练可并行化。

### predict_with_generate

训练时用 `decoder_input_ids` + teacher forcing 加速。评估时为模拟真实推理，必须用 `generate()` 逐 token 生成。`predict_with_generate=True` 让 `Seq2SeqTrainer.evaluate()` 自动调用 `generate()`，确保评估结果接近实际使用表现。

### SacreBLEU 与中文分词

BLEU 通过 n-gram 匹配衡量翻译与参考的接近程度。但中文没有空格，整句被视为一个 token，n-gram 全部为 0。SacreBLEU 的 `tokenize="zh"` 会调用 jieba 分词，解决中文评估问题。预测格式为字符串列表，参考格式为字符串列表的列表（支持多参考翻译）。

### opus100 替代 KDE4

datasets 5.0.0 不再支持脚本式加载，KDE4 无 Parquet 版本。OPUS100 是完全的 Parquet 格式，1M 英中平行语料，功能等价。`load_dataset("opus100", "en-zh")`。

### 训练/验证/测试集关系

三者不是按名称区分，而是按用法：训练集用于更新参数，验证集在每个 epoch 评估、监控过拟合、调超参，测试集全程锁死、训练全部结束后只跑一次。验证集被反复使用，模型间接"认识"了它，所以需要全新的测试集验证泛化能力。数据比例：<1 万用 7:3，1 万–10 万用 8:2 或 9:1，百万级用 98:2。

### 微调效果不明显的原因

opus-mt-en-zh 预训练时已大量使用 OPUS 系列英中数据，再用 opus100 微调，数据分布高度重叠，BLEU 从 48.06 仅提升到 49.52。想看到明显效果，需要选择预训练没见过的领域数据（法律、医疗、游戏对话等）。

### 今天解决的问题

kde4 RuntimeError → opus100 替代；sentencepiece/sacremoses 缺失 → pip 安装；HuggingFace 网络问题 → hf-mirror.com 镜像；SacreBLEU 中文 0 分 → `tokenize="zh"`；`evaluation_strategy` 重命名；验证集未 tokenize → 从 `tokenized_datasets` 取；训练/验证/测试集名称混淆 → 按用法区分。

### 下一步

Day 13 预计内容：Embedding 与语义检索基础，为 RAG 项目做准备。```
