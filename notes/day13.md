# 文本摘要（续day12）

## 准备多语言语料库
```Py
from datasets import load_dataset

chinese_dataset = load_dataset("json", data_files={
    "train": "../data/amazon_reviews_multi_zh_raw/zh/train.jsonl.gz",
    "validation": "../data/amazon_reviews_multi_zh_raw/zh/validation.jsonl.gz",
    "test": "../data/amazon_reviews_multi_zh_raw/zh/test.jsonl.gz",
})

english_dataset = load_dataset("json", data_files={
    "train": "../data/amazon_reviews_multi_en_raw/en/train.jsonl.gz",
    "validation": "../data/amazon_reviews_multi_en_raw/en/validation.jsonl.gz",
    "test": "../data/amazon_reviews_multi_en_raw/en/test.jsonl.gz",
})
```

将 english_dataset 转换为 pandas.DataFrame ，并计算每个产品类别的评论数量：
```Py
english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# 显示前 20 个产品的数量
english_df["product_category"].value_counts()[:20]
```

专注于总结书籍的评论！我们可以看到两个符合要求的产品类别（ book 和 digital_ebook_purchase ），所以让我们用这两个产品类别过滤两种语言的数据集。
```Py
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
```
当我们使用这个函数对 english_dataset 和 spanish_dataset 过滤后，结果将只包含涉及书籍类别的那些行。

在使用过滤器之前，让我们将 english_dataset 的格式从 "pandas" 切换回 "arrow" ：
```Py
english_dataset.reset_format()
```

然后我们可以使用过滤器功能，作为一个基本的检查，让我们检查一些评论的样本，看看它们是否确实与书籍有关：
```Py
chinese_books = chinese_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)
```
将英文和中文评论作为单个 DatasetDict 对象组合起来。Datasets 提供了一个方便的 concatenate_datasets() 函数，它将把两个 Dataset 对象堆叠在一起。因此，为了创建我们的双语数据集，我们将遍历数据集的每个部分，并打乱结果以确保我们的模型不会过度拟合单一语言：
```Py
from datasets import concatenate_datasets, DatasetDict

books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], chinese_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)
```

最后要检查的一件事是评论及其标题中单词的分布。这对于摘要任务尤其重要，其中数据中如果出现大量参考摘要过于简短会使模型偏向于生成的摘要中仅有一两个单词。
为了解决这个问题，我们将过滤掉标题非常短的示例，以便我们的模型可以生成更有效的摘要。由于我们正在处理英文和中文
```Py
books_dataset = books_dataset.filter(
    lambda x: x["review_title"] is not None and (
        len(x["review_title"]) > 5 if x["language"] == "zh" 
        else len(x["review_title"].split()) > 2
    )
)
```


## 文本摘要模型
使用 mT5，这是一种基于 T5 的有趣架构，在文本到文本任务中进行了预训练。在 T5 中，每个 NLP 任务都是以任务前缀（如 summarize: ）的形式定义的，模型根据不同的任务生成不同的文本。如下图所示，这让 T5 变得非常通用，因为你可以用一个模型解决很多任务！mT5 不使用前缀，但具有 T5 的大部分功能，并且具有多语言的优势。
[图示](03-NLP-Basic\pictures\t5.svg)


## 预处理数据
```Py
from transformers import AutoTokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

在 NLP 项目的早期阶段，一个好的做法是在小样本数据上训练一类“小”模型。这使你可以更快地调试和迭代端到端工作流。当你对结果有信心之后，你只需要通过简单地更改模型 checkpoint 就可以在较大规模数据上训练模型！

因为我们的输出目标也是文本，所以输入和输出加起来可能超过模型的最大上下文大小。这意味着我们需要对评论及其标题进行截断，以确保我们不会将过长的输入传递给我们的模型。Transformers 中的 tokenizer 提供了一个绝妙的 text_target 参数，允许你将目标文本与输入并行 tokenize。以下是如何为 mT5 处理输入和目标文本的示例：
```Py
max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
```

## 文本摘要的评估指标
最常用的指标之一是ROUGE 分数（Recall-Oriented Understudy for Gisting Evaluation 的缩写）。该指标背后的基本思想是将生成的摘要与一组通常由人类创建的参考摘要进行比较。
更具体地说，假设我们要比较以下两个摘要：
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
比较它们的一种方法是计算重叠单词的数量，在这个例子中为 6。然而，这种方法有些粗糙，因此 ROUGE 是基于计算重叠部分的精确度(Precision) 和 召回率(Recall) 分数来计算的。
```
召回率= 重叠词的数量/参考摘要中的总词数
```

对于上面的那个例子，这个公式给出了 6/6 = 1 的完美召回率；即，参考摘要中的所有单词模型都生成出来了。这听起来可能很棒，但想象一下，如果我们生成的摘要是“我真的很喜欢整晚阅读饥饿游戏”。这也会有完美的 recall，但可以说这是一个更糟糕的总结，因为它很冗长。为了适应于这些场景，我们还计算了精确度，它在 ROUGE 上下文中衡量了生成的摘要中有多少是相关的：
​```
精确度= 重叠词的数量/生成摘要中的总词数
```

在实践中，通常会先计算计算精度和召回率，然后得到 F1 分数（精确度和召回率的调和平均数）。

通过安装 rouge_score 包来实现这些计算：
```Py
!pip install rouge_score
```
然后按如下方式加载 ROUGE 指标：
```Py
import evaluate
rouge_score = evaluate.load("rouge")
```

接着我们可以使用 rouge_score.compute() 函数来一次性计算所有的指标：
```Py
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
scores
=>
{'rouge1': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rouge2': AggregateScore(low=Score(precision=0.67, recall=0.8, fmeasure=0.73), mid=Score(precision=0.67, recall=0.8, fmeasure=0.73), high=Score(precision=0.67, recall=0.8, fmeasure=0.73)),
 'rougeL': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rougeLsum': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92))}
```
首先，Datasets 计算了精度、召回率和 F1 分数的置信区间；也些就是你在这里看到的 low 、 mid 和 high 属性。此外，Datasets 还计算了基于在比较生成摘要和参考摘要时的采用不同文本粒度的各种 ROUGE 得分。 
rouge1 测量的是生成摘要和参考摘要中单个单词的重叠程度。 
rouge2 度量了二元词组（考虑单词对的重叠）之间的重叠
rougeL 和 rougeLsum 通过寻找生成的摘要和参考摘要中最长的公共子串来度量单词的最长匹配序列。 rougeLsum 中的“sum”指的是该指标是在整个摘要上计算的，而 rougeL 是指在各个句子上计算的平均值。

### 创建强大的 baseline
对于文本摘要，一个常见的参考 baseline 是简单地取文章的前三句话作为摘要，通常称为 lead-3 baseline。我们可以使用句号（英文使用．）来跟踪句子边界，但这在“U.S.” or “U.N.”之类的首字母缩略词上会计算错误。所以我们将使用 nltk 库，它包含一个更好的算法来处理这些情况。你可以使用以下方式安装该包：
```Py
!pip install nltk
```

然后下载标点规则：
```Py
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```
接下来，我们从 nltk 导入句子的 tokenizer 并创建一个简单的函数用来提取评论中的前三个句子。文本摘要的默认情况下使用换行符分隔每个摘要，因此我们也按照这样的规则处理，并在训练集的示例上对其进行测试：
```Py
import re
from nltk.tokenize import sent_tokenize

def three_sentence_summary(text, lang):
    if lang == "zh":
        sents = re.split(r"[。！？；]|\.{3}", text)  # 中文句子分割
    else:
        sents = sent_tokenize(text)
    return "\n".join([s.strip() for s in sents if s.strip()][:3])

print(three_sentence_summary(books_dataset["train"][11]["review_body"], books_dataset["train"][11]["language"]))
=>
'I grew up reading Koontz, and years ago, I stopped,convinced i had "outgrown" him.'
'Still,when a friend was looking for something suspenseful too read, I suggested Koontz.'
'She found Strangers.'
```

这似乎有效，接下来让我们现在实现一个函数，从数据集中提取这些“摘要”并计算 baseline 的 ROUGE 分数：
```Py
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text, lang) 
                 for text, lang in zip(dataset["review_body"], dataset["language"])]
    return metric.compute(predictions=summaries, references=dataset["review_title"])
```
然后我们可以使用这个函数来计算验证集上的 ROUGE 分数，并使用 Pandas 对输出的结果进行一些美化：

```Py
import pandas as pd

score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_dict = {}
for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
    rouge_dict[rn] = round(score[rn] * 100, 2)
rouge_dict
```
我们可以看到 rouge2 的分数明显低于其他的rouge；这可能反映了这样一个事实，即评论标题通常很简洁，因此 lead-3 baseline 过于冗长导致得分不高。
​

## 使用 Trainer API 微调 mT5

```Py
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

```Py
from huggingface_hub import notebook_login
notebook_login()

huggingface-cli login
```

为了在训练期间计算 ROUGE 分数，我们需要在训练期间生成文本形式的摘要。幸运的是，Transformers 提供了专用的 Seq2SeqTrainingArguments 和 Seq2SeqTrainer 类，可以自动为我们完成这项工作！

```Py
from transformers import Seq2SeqTrainingArguments
batch_size = 8
num_train_epochs = 3
# 每个训练周期都输出训练损失
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-zh",
    eval_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)
```

在上面的代码中，我们把 predict_with_generate 参数设置为 True ，这样可以在评估期间生成摘要来计算每个 epoch 的 ROUGE 分数。设置 predict_with_generate=True 后，Seq2SeqTrainer 会在评估时使用 generate() 函数。除此之外我们还调整默认的超参数，如学习率、epochs 数和权重衰减，并且设置 save_total_limit 选项， 使训练期间最多只能保存 3 个 checkpoint 的选项。——这是因为即使是 mT5 的“small”版本也使用大约 1 GB 的硬盘空间，我们可以通过限制保存的副本数量来节省一点空间。


为了在训练期间评估模型，我们还需要为 Trainer 提供一个 compute_metrics() 函数。对于摘要模型来说，不能直接调用 rouge_score.compute() 进行评估，因为我们需要将输出和参考摘要解码为文本，然后才能计算 ROUGE 分数。下面的函数就完成了解码和计算分数，除此之外还使用了 nltk 中的 sent_tokenize() 函数将摘要句子用换行符分隔开：

```PY
def split_sents(text):
    """中英文分句"""
    if re.search(r'[\u4e00-\u9fff]', text):  # 含中文
        sents = re.split(r"[。！？；]|\.{3}", text)
    else:
        sents = sent_tokenize(text)
    return [s.strip() for s in sents if s.strip()]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(split_sents(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(split_sents(label.strip())) for label in decoded_labels]
    
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {k: round(v * 100, 4) for k, v in result.items()}
```

接下来，我们需要为我们的序列到序列任务定义一个数据整理器（data collator）。由于 mT5 是一个编码器-解码器的 Transformer 模型，因此在将数据整理成 batch 时有一点需要注意，那就是在解码期间，我们需要将标签向右移动一个单位。这是为了确保解码器只看到之前的参考序列，而不是当前要预测的 token 或之后的参考序列，这样模型就能避免容易记住标签。这类似与在 因果语言模型 这样的任务中使用掩码自注意力的机制类似。

Transformers 提供了一个 DataCollatorForSeq2Seq 整理器，它会动态地填充我们的输入和标签。我们只需要提供 tokenizer 和 model 既可实例化这个整理器：
```PY
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

让我们看看当给这个整理器提供一个小批次的样本时，它的处理过程是怎么样的。首先，我们需要删除带有字符串的列，因为整理器不知道如何对这些元素进行填充（padding）：
```PY
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
```

由于 collator 需要一个 dict 的列表，其中每个 dict 代表数据集中的一个样本，所以我们也需要在将数据传给数据整理器之前，将数据整理成预期的格式：
```PY
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
```

标准参数实例化 Trainer
```PY
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

```PY
trainer.train()
```

在训练期间，应该可以看到训练损失逐渐减小，并且 ROUGE 分数随着 epoch 的增加而增加。训练完成之后，你可以通过运行 Trainer.evaluate() 来查看最后的 ROUGE 分数：
```Py
trainer.evaluate()
{'eval_loss': 3.028524398803711,
 'eval_rouge1': 16.9728,
 'eval_rouge2': 8.2969,
 'eval_rougeL': 16.8366,
 'eval_rougeLsum': 16.851,
 'eval_gen_len': 10.1597,
 'eval_runtime': 6.1054,
 'eval_samples_per_second': 38.982,
 'eval_steps_per_second': 4.914}
```

从分数中我们可以看到，我们的模型轻松超过了我们的 lead-3 baseline——很好！最后要做的是将模型权重推送到 Hub，如下所示：
```Py
trainer.push_to_hub(commit_message="Training complete", tags="summarization")
```

## 使用模型
```Py
from transformers import pipeline
hub_model_id = "Kate-lf/mt5-small-finetuned-amazon-en-zh"
summarizer = pipeline("summarization", model=hub_model_id)
```
我们可以将测试集（模型还没有见过的一些数据）中取一些样本提供给我们的管道，来感受一下生成的摘要的质量。首先让我们实现一个简单的函数，同时显示评论、标题和生成的摘要：
```Py
def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")
```

例子：
```Py
print_summary(1)
print_summary(0)
```


## 今日知识总结

### 多语言数据集的本地加载与处理

从 Hugging Face Hub 下载原始数据到本地后，使用 `load_dataset("json", data_files={...})` 加载 JSONL 文件。注意 `buruzaemon/amazon_reviews_multi` 使用旧版数据集脚本，datasets 5.0.0 不再支持，因此需要通过 `huggingface-cli download` 直接下载原始 `.jsonl.gz` 文件，再用 `load_dataset("json", ...)` 加载。下载到本地时注意 Windows 符号链接问题——`huggingface-cli download` 默认可能创建相对路径符号链接，在 Windows 上容易断裂，用 `cat` 写入实际文件内容可解决。

### 双语数据集的处理逻辑

中英文文本不能套用同一套处理规则。过滤短标题时，英文用 `split()` 按空格分词数判断（`> 2`），中文没有空格分隔，改用 `len()` 字符数（`> 5`）判断。句子分割同理：英文用 `nltk.sent_tokenize()` 处理缩写等边界情况，中文用正则 `r"[。！？；]"` 按标点切分。合并双语数据后，保留 `language` 字段以便后续按语言分叉处理。

### 文本摘要的评估指标 ROUGE

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是摘要任务的核心指标。它计算生成摘要与参考摘要之间的 n-gram 重叠，同时报告精确度（生成内容有多少是相关的）、召回率（参考内容有多少被覆盖）和 F1 分数。rouge1 衡量单词级重叠，rouge2 衡量二元词组重叠，rougeL 衡量最长公共子序列。新版 `evaluate` 库的 ROUGE 返回值不再是 `AggregateScore` 对象（无 `.mid` 属性），而是直接返回 `numpy.float64` 数值。

### Lead-3 Baseline 的意义

在训练摘要模型前，必须先建立一个简单 baseline——取评论前三句话作为"摘要"。如果模型连这个简单策略都打不过，说明训练没学到有用信息。这是 NLP 项目的基本流程：先建立 baseline，再尝试复杂模型，确保投入有回报。

### Seq2Seq 训练的注意事项

`Seq2SeqTrainingArguments` 中 `predict_with_generate=True` 会在评估时调用 `generate()` 生成文本摘要，而非直接输出 logits。需要自定义 `compute_metrics()` 解码预测和标签文本后再算 ROUGE。`DataCollatorForSeq2Seq` 自动处理 decoder_input_ids 的移位（与 Day 12 翻译任务一致）。评估时标签中的 `-100` 需替换为 `pad_token_id` 才能解码。`tokenizer=` 参数已废弃，改用 `processing_class=`。

### 版本兼容性经验

本次遇到多个版本问题：`evaluation_strategy` → `eval_strategy`（4.57 废弃）、ROUGE 返回值结构变化（无 `.mid` 属性）、`sent_tokenize` 需额外下载 `punkt_tab`、旧版数据集脚本被 datasets 5.0.0 拒绝（`RuntimeError: Dataset scripts are no longer supported`）。解决方案：用 `huggingface-cli download` 直接下载原始文件绕过脚本问题，用本地 JSON 加载方式替代 `load_dataset("repo", "config")` 调用。
