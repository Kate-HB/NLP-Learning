# Day 11：掩码语言模型微调（MLM + WWM）

Day 10 学了用 BERT 做 Token Classification（NER）。Day 11 学 MLM 微调：在 IMDB 电影评论上微调 DistilBERT，掌握全词掩码（WWM）、数据拼接分块、困惑度评估，以及 `remove_unused_columns` 等 Trainer 陷阱。

# 掩码语言模型（masked language model）（续day10）
## 选择用于掩码语言建模的预训练模型

我们将使用名为 DistilBERT 的模型
```Py
from transformers import AutoModelForMaskedLM
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

通过调用 num_parameters() 方法查看模型有多少参数：
```Py
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")
=>
'>>> DistilBERT number of parameters: 67M'
'>>> BERT number of parameters: 110M'
```
DistilBERT 大约有 6700 万个参数，大约只有 BERT base 模型的二分之一，这大致意味着训练的速度可以提高两倍 —— 非常棒！现在让我们看看对于下面的一小部分文本，这个模型最有可能预测什么：
text = "This is a great [MASK]."
作为人类，我们可以想象 [MASK] token 有很多可能性，例如 “day”、“ride” 或者 “painting”。对于预训练模型，预测取决于模型所训练的语料库，因为它会学习获取数据中存在的语料统计分布。与 BERT 一样，DistilBERT 在 English Wikipedia 和 BookCorpus 数据集上进行预训练，所以我们猜想模型对 [MASK] 的预测能够反映这些领域。为了预测 [MASK] ，我们需要 DistilBERT 的 tokenizer 来处理输入，所以让我们也从 Hub 下载它：
```Py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

有了 tokenizer 和模型，我们现在可以将我们的示例文本传递给模型，提取 logits，并打印出前 5 个候选词：
```Py
import torch
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# 找到 [MASK] 的位置并提取其 logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# 选择具有最高 logits 的 [MASK] 候选词
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
=>
'>>> This is a great deal.'
'>>> This is a great success.'
'>>> This is a great adventure.'
'>>> This is a great idea.'
'>>> This is a great feat.'
```
我们可以从输出中看到，模型的预测的是日常术语，考虑到模型训练的语料数据主要来源于维基百科，这并不奇怪。现在让我们看看如何将这个领域改变成稍微更加独特——高度两极分化的电影评论！


## 数据集
为了展示领域适应性，我们将使用来自IMDB的 大型电影评论数据集(Large Movie Review Dataset)，这是一个电影评论语料库，通常用于对情感分析模型进行基准测试。通过在这个语料库上对 DistilBERT 进行微调，我们期望语言模型会从其预训练的维基百科的事实性数据，适应到更主观的电影评论的领域。

获取数据：
```Py
from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
imdb_dataset
=>
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```
我们可以看到 train 和 test 分别包含了 25,000 条评论，还有一个没有的标签的 unsupervised（无监督） 部分包含 50,000 条评论。接下来让我们从里面取一些样本，来了解一下我们正在处理的文本的特点。正如我们在本课程的前几章中所做的那样，我们将把 Dataset.shuffle() 函数链接到 Dataset.select() 函数创建随机样本：
```Py
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
=>
#0 代表负面评论， 1 代表正面评论。
```


## 预处理数据

对于自回归和掩码语言建模，常见的预处理步骤是将所有的文本拼接起来，然后再将整个语料库切割为相同大小的块。这与我们之前的做法有很大的不同，我们之前只是对单个的示例进行 tokenize。为什么要将所有的示例连接在一起呢？原因是如果单个示例太长，可能会被截断，这会导致我们失去可能对语言建模任务有用的信息！

因此，我们首先会像往常一样对语料库进行 tokenize 处理，但是不在 tokenizer 中设置 truncation=True 选项。除此之外，我们还需要获取单词的 ID，因为后面我们需要用到它们来进行全词掩码。最后我们将把这个过程封装在一个简单的函数中，并删除 text 和 label 列，因为我们不再需要它们。
```Py
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result
```

#使用 batched=True 来激活快速多线程!
```Py
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
=>
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 50000
    })
})
```
由于 DistilBERT 是一个类似 BERT 的模型，我们可以看到编码后的文本包含了我们在之前章节中看到的 input_ids 和 attention_mask ，以及我们添加的 word_ids 。

现在我们已经对电影评论进行了 tokenize，下一步是将它们全部组合在一起并将结果分割成块。但是，这些块应该有多大呢？这最终将取决于你可以使用的显存大小，但一个好的起点是查看模型的最大上下文大小。这可以在 tokenizer 的 model_max_length 属性中找到：
```Py
tokenizer.model_max_length
=>
512
```
该值来自于与 checkpoint 相关联的 tokenizer_config.json 文件；在我们的例子中，我们可以看到上下文大小是 512 个 tokens 与 BERT 模型一样。


选择一个稍小一点、可以放入内存中的分块大小：
chunk_size = 128

注意，在实际应用场景中，使用小的块可能会有丢失长句子之间的语义信息从而对最终模型的性能产生不利的影响，所以如果显存条件允许的话，你应该选择一个与你将要使用模型的相匹配的大小。

现在来到了最有趣的部分。为了展示如何把这些示例连接在一，我们从分词后的训练集中取出几个评论，并打印出每个评论的 token 数量：
```Py
#切片会为每个特征生成一个列表的列表
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
Copied
'>>> Review 0 length: 200'
'>>> Review 1 length: 559'
'>>> Review 2 length: 192'
```

然后，我们可以用一个简单的字典推导式将所有这些示例连接在一起，如下所示：
```Py
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
=>
'>>> Concatenated reviews length: 951'
```

总长度计算出来了 —— 现在，让我们将连接的评论拆分为大小为 chunk_size 的块。为此，我们迭代了 concatenated_examples 中的特征，并使用列表推导式为每个特征分块。结果是一个字典，键是特征的名称，值是对应值经过分块的列表：
```Py
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
=>
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 55'
```
正如你在这个例子中看到的，最后一个块通常会小于所设置的分块的大小。有两种常见的策略来处理这个问题：

如果最后一个块小于 chunk_size ，就丢弃。
填充最后一个块，直到其长度等于 chunk_size 。
我们将在这里采用第一种方法，最后让我们将上述所有逻辑包装在一个函数中，以便我们可以将其应用于我们的已分词数据集上：
```Py
def group_texts(examples):
    # 拼接所有的文本
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算拼接文本的长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果最后一个块小于 chunk_size,我们将其丢弃
    total_length = (total_length // chunk_size) * chunk_size
    # 按最大长度分块
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # 创建一个新的 labels 列
    result["labels"] = result["input_ids"].copy()
    return result
```
注意，在 group_texts() 的最后一步，我们创建了一个新的 labels 列，它是通过复制 input_ids 列形成的。这是因为在掩码语言模型的目标是预测输入中随机遮住(Masked)的 token，我们保存了让我们的语言模型从中学习 [Mask] 的答案。

现在，让我们使用我们强大的 Dataset.map() 函数将 group_texts() 应用到我们的已分词数据集上：
```Py
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets
=>
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 61289
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 59905
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 122963
    })
})
```
通过对文本进行分块，我们得到了比原来的训练集和测试集的 25000 个例子多得多的评论数据。这是因为我们现在有了涉及跨越原始语料库中多个例子的连续标记的例子。你可以通过在其中一个块中查找特殊的 [SEP] 和 [CLS] tokens 来清晰地看到这一点：
```Py
tokenizer.decode(lm_datasets["train"][1]["input_ids"])
=>
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```
在这个例子中，你可以看到两个重叠的电影评论，一个关于高中电影，另一个关于无家可归的问题。让我们也检查一下掩码语言模型待预测的标签是什么样的：
```Py
tokenizer.decode(lm_datasets["train"][1]["labels"])
=>
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```
正如我们上面的 group_texts() 函数所预期的那样，这看起来与解码的 input_ids 完全相同 —— 但是要怎么样才能让我们的的模型可以学习到一些东西呢？我们缺少一个关键的步骤：在输入中随机插入 [MASK] token！

## 使用 Trainer API 微调 DistilBERT
微调掩码语言模型几乎与微调序列分类模型相同。唯一的区别是我们需要一个特殊的数据整理器，它可以随机屏蔽每批文本中的一些 tokens。Transformers 为这项任务准备了专用的 DataCollatorForLanguageModeling 。我们只需要将 tokenizer 和一个 mlm_probability 参数（掩盖 tokens 的比例）传递给它。在这里我们将 mlm_probability 参数设置为 15%，这是 BERT 默认的数量，也是文献中最常见的选择。
```Py
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
为了了解随机掩码数据整理器的工作原理，让我们把一些例子输入到数据整理器。由于数据整理器期望接收一个字典列表，其中每个字典中存储一段连续文本的块，所以我们首先遍历数据集取出来一些样本数据，然后将样本数据输入到整理器。在输入到数据整理器之间，我们删除了 word_ids 这个键，因为它不需要这个键。
```Py
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
=>
'>>> [CLS] bromwell [MASK] is a cartoon comedy. it ran at the same [MASK] as some other [MASK] about school life, [MASK] as " teachers ". [MASK] [MASK] [MASK] in the teaching [MASK] lead [MASK] to believe that bromwell high\'[MASK] satire is much closer to reality than is " teachers ". the scramble [MASK] [MASK] financially, the [MASK]ful students whogn [MASK] right through [MASK] pathetic teachers\'pomp, the pettiness of the whole situation, distinction remind me of the schools i knew and their students. when i saw [MASK] episode in [MASK] a student repeatedly tried to burn down the school, [MASK] immediately recalled. [MASK]...'

'>>> .... at.. [MASK]... [MASK]... high. a classic line plucked inspector : i\'[MASK] here to [MASK] one of your [MASK]. student : welcome to bromwell [MASK]. i expect that many adults of my age think that [MASK]mwell [MASK] is [MASK] fetched. what a pity that it isn\'t! [SEP] [CLS] [MASK]ness ( or [MASK]lessness as george 宇in stated )公 been an issue for years but never [MASK] plan to help those on the street that were once considered human [MASK] did everything from going to school, [MASK], [MASK] vote for the matter. most people think [MASK] the homeless'
```
很棒，成功了！我们可以看到， [MASK] tokens 已随机插入我们文本中的不同位置。这些将是我们的模型在训练期间必须预测的 tokens —— 数据整理器的美妙之处在于，它会在每个 batch 中随机插入 [MASK] ！


随机掩码的一个缺点是，当使用 Trainer 时，每次计算出来的评估结果会有些许不同，即使我们会对训练集和测试集使用相同的数据整理器。稍后，我们在学习使用Accelerate 进行微调时， 就会看到如何利用灵活的自定义评估循环的来冻结随机性。

在为掩码语言建模训练模型时，不仅仅可以遮蔽单个 token，还可以一次遮蔽整个单词的所有 token，这种方法被称为全词屏蔽（whole word masking）。如果我们想使用全词屏蔽（whole word masking），我们就需要自己构建一个数据整理器。数据整理器的核心是一个函数，它接受一个样本列表并将它们转换为一个 batch，所以现在让我们这样做吧！我们将使用先前计算的word ID，构建一个单词索引和相应 token 之间的映射，然后随机决定遮蔽哪些单词，并使用这种方法对输入进行遮蔽。请注意，除了与掩码对应的标签外，所有其他的标签均应该设置为 -100 。
```Py
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # 创建一个单词与对应 token 索引之间的映射
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # 随机遮蔽单词
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
```

在之前的样本上试试它：
```Py
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
=>
'>>> [CLS] bromwell high is a cartoon comedy [MASK] it ran at the same time as some other programs about school life, such as " teachers ". my 35 years in the teaching profession lead me to believe that bromwell high\'s satire is much closer to reality than is " teachers ". the scramble to survive financially, the insightful students who can see right through their pathetic teachers\'pomp, the pettiness of the whole situation, all remind me of the schools i knew and their students. when i saw the episode in which a student repeatedly tried to burn down the school, i immediately recalled.....'

'>>> .... [MASK] [MASK] [MASK] [MASK]....... high. a classic line : inspector : i\'m here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn\'t! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless'
```

现在我们有了两个数据整理器，剩下的微调步骤与其他任务类似都是标准的。练可能会需要一些时间，所以我们首先将训练集的大小减小到几千个例子。不用担心，我们仍然可以得到一个相当不错的语言模型！在 Datasets 中快速筛选数据集的方法是使用Dataset.train_test_split() 函数：
```Py
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset
=>
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 1000
    })
})
```

运行上述代码会自动创建新的 train 和 test 数据集，训练集大小为 10,000 个示例，验证的大小是训练集的 10％ —— 如果你有一个强大的 GPU，可以自行增加这个比例！我们接下来要做的事情是登录 Hugging Face Hub。如果你在 Notebook 中运行这段代码：
```Py
from huggingface_hub import notebook_login
notebook_login()
```
或终端中输入指令：
```Py
huggingface-cli login
```

登陆后，我们可以指定 Trainer 的参数：
```Py
from transformers import TrainingArguments
batch_size = 64
# 在每个 epoch 输出训练的 loss
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)
```
在这里，我们调整了一些默认选项，包括 logging_steps ，以确保我们可以跟踪每个 epoch 的训练损失。我们还使用了 fp16=True 来实现混合精度训练，从而进一步提高训练速度。默认情况下， Trainer 将删除模型的 forward() 方法中未使用的列。这意味着，如果你使用全词屏蔽（whole word masking）数据整理器，你还需要设置 remove_unused_columns=False ，以确保我们不会在训练期间丢失 word_ids 列。

请注意，你可以使用 hub_model_id 参数指定你想推送到的仓库的名称（如果你想把它推送到一个组织，就必须使用这个参数）。例如，当我们将模型推送到 huggingface-course 组织 时，就在 TrainingArguments 中添加了 hub_model_id="huggingface-course/distilbert-finetuned-imdb" 。默认情况下，使用的仓库将保存在你的账户中并以你设置的输出目录命名，因此在我们的示例中，它将是 "lewtun/distilbert-finetuned-imdb" 。

现在，我们拥有了初始化 Trainer 所需的所有要素。这里我们只使用了标准的 data_collator ，但你可以尝试使用全词屏蔽作为数据整理器的一个练习，并对比一下不同屏蔽方式的结果有什么不同：
```Py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=whole_word_masking_data_collator,#data_collator
    tokenizer=tokenizer,
)
```


### 语言模型的困惑度（perplexity）
语言建模与文本分类或问答等其他任务有所不同，在其他任务中，我们会得到一个带标签的语料库进行训练，而语言建模则没有任何明确的标签。那么我们如何确定什么是好的语言模型呢？就像手机中的自动更正功能一样，一个好的语言模型会较高概率输出一个语法正确的句子，较低概率输出无意义的句子。

如果测试集主要由语法正确的句子组成，那么衡量语言模型质量的一种方式就是计算它给测试集中所有句子的下一个词的概率。高概率表示模型对未见过的例子不感到“惊讶”或“困惑”，这表明它已经学习了语言的基本语法模式。困惑度有很多种数学定义，我们将使用的定义是交叉熵损失的指数。具体方法是使用 Trainer.evaluate()方法计算测试集上的交叉熵损失，取结果的指数来计算预训练模型的困惑度。

```Py
import math
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
=>
Perplexity: 21.75
```
较低的困惑度分数意味着更好的语言模型，我们可以看到，我们的初始模型的困惑度相当地高。让我们看看我们是否可以通过微调来降低它！为此，我们首先运行训练循环：
```Py
trainer.train()
```
然后像之前那样计算测试集上的结果困惑度：
```Py
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
=>
Perplexity: 11.32
```
太棒了——困惑度显著降低，这告诉我们模型已经学习到了电影评论领域的一些知识！

一旦训练完成，可以将带有训练信息的模型卡片推送到 Hub：
```Py
trainer.push_to_hub()
```

## 使用我们微调的模型
使用 Hub 上的模型部件或者在本地使用🤗 Transformers 的 pipeline 加载微调模型预测文本。让我们使用后者通过 fill-mask pipeline 下载我们的模型：
```Py
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="Kate-lf/distilbert-base-uncased-finetuned-imdb"
)

text1='This is a great [MASK]'
preds1 = mask_filler(text1)
for pred in preds1:
    print(f">>> {pred['sequence']}")

text2='The movie is so [MASK]'
preds2 = mask_filler(text2)
for pred in preds2:
    print(f">>> {pred['sequence']}")
=>
'>>> this is a great movie.'
'>>> this is a great film.'
'>>> this is a great story.'
'>>> this is a great movies.'
'>>> this is a great character.'
```


## 今日知识总结

### MLM 微调目标

掩码语言建模没有显式标签，而是通过随机遮蔽输入中的 token，让模型预测被遮蔽的原始 token。损失函数是交叉熵，只在被遮蔽位置计算（`labels=-100` 的位置被忽略）。这与文本分类的监督学习本质不同——MLM 是自监督学习。

### 数据预处理：拼接 + 分块

自回归和 MLM 训练通常不按单条样本 tokenize，而是将所有文本拼接后再切成等长块（chunk）。原因是：单条样本长度不一，截断浪费信息；拼接分块能最大化 GPU 显存利用率，且让模型学习跨句子的长程依赖。chunk_size 通常设为模型最大上下文长度（DistilBERT 512），显存受限时可减小（如 128）。

### DataCollatorForLanguageModeling

Transformers 内置的 MLM 数据整理器，每个 batch 动态随机选择 15% 的 token 进行遮蔽：其中 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持原词。这种 80-10-10 拆分让模型学会：有时依赖上下文（MASK）、有时纠正错误（随机词）、有时确认正确（原词）。

### 全词掩码 Whole Word Masking

BERT 原始做法随机遮蔽单个 subword token，可能只遮 "##ing" 而留 "play"，模型靠词内线索就能猜出答案。WWM 将同一个单词的所有 subword token 一起遮蔽，迫使模型依赖上下文语义而非词内模式。实现关键：tokenizer 的 `word_ids()` 方法提供 token→word 映射，collator 据此整词遮蔽。

### remove_unused_columns 陷阱

`TrainingArguments` 默认 `remove_unused_columns=True`，Trainer 初始化时会检查模型 `forward()` 签名，删除不在参数列表中的列。`word_ids` 不是模型参数，会被静默删除，导致 WWM collator 中 `feature.pop("word_ids")` 抛出 KeyError。修复：设置 `remove_unused_columns=False`。

### 困惑度 Perplexity

`Perplexity = exp(cross_entropy_loss)`。它是语言模型的核心评估指标：值越低模型越好。随机猜测 30522 词表的 perplexity ≈ 30522；经过微调，perplexity 从 62.55 降到 24.60，说明模型学到了电影评论领域的语言模式。

### 微调后推理

微调后的 MLM 模型可用 `pipeline("fill-mask", model="./output_dir")` 加载。模型学会了领域知识：输入 "The movie is so [MASK]"，预测 "bad"/"great"/"good"/"awful" 等情感词，而非通用语料的 "deal"/"success"。

### 今天解决的问题

`evaluation_strategy` → `eval_strategy`（Transformers 4.57 重命名）；`word_ids` KeyError → `remove_unused_columns=False`；`tokenizer` 参数 deprecated → 将来改用 `processing_class`。

### 下一步

Week 5 计划：传统 NLP baseline（TF-IDF + 分类器）对比 BERT，理解 baseline 的价值。
