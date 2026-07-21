# 各种NLP任务

## Token 分类
这个通用任务涵盖了所有可以表述为“给句子中的词或字贴上标签”的问题，例如：
实体命名识别 （NER）：找出句子中的实体（如人物、地点或组织）。这可以通过为每个实体指定一个类别的标签，如果没有实体则会输出无实体的标签。
词性标注 （POS）：将句子中的每个单词标记为对应于特定的词性（如名词、动词、形容词等）。
分块（chunking）：找出属于同一实体的 tokens 这个任务（可以与词性标注或命名实体识别结合）可以被描述为将位于块开头的 token 赋予一个标签（通常是 “ B- ” （Begin），代表该token位于实体的开头），将位于块内的 tokens 赋予另一个标签（通常是 “ I- ”（inner）代表该token位于实体的内部），将不属于任何块的 tokens 赋予第三个标签（通常是 “ O ” （outer）代表该token不属于任何实体）。

在本节中，我们将在 NER 任务上微调模型 （BERT）

### 准备数据

只要你的数据集由分词文本和对应的标签组成，就能够将这里描述的数据处理过程应用到自己的数据集中。

#### CoNLL-2003 数据集
加载 CoNLL-2003 数据集
```Py
from datasets import load_dataset
raw_datasets = load_dataset("conll2003", revision="refs/convert/parquet")
```

查看 raw_datasets 对象可以让我们看到数据集中存在哪些列，以及训练集、验证集和测试集之间是如何划分的：
```Py
raw_datasets
=>
DatasetDict({
    train: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3453
    })
})
```
可以看到数据集包含了我们之前提到的三项任务的标签：命名实体识别（NER）、词性标注（POS）以及分块（chunking）。这个数据集与其他数据集的一个显著区别在于，输入文本的那一列并非以句子或整片的文本的形式存储，而是以单词列表的形式（最后一列被称为 tokens ，不过 tokens 列保存的还是单词，也就是说，这些预先分词的输入仍需要经过 tokenizer 进行子词分词处理）。

我们来看看训练集的第一个元素：
```Py
raw_datasets["train"][0]["tokens"]
=>
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```
由于我们要进行命名实体识别，让我们查看一下 NER 标签：
```Py
raw_datasets["train"][0]["ner_tags"]
=>
[3, 0, 7, 0, 0, 0, 7, 0, 0]
```

这些是可以直接训练的整数类别标签，当我们想要检查数据时，直接看整数的标签不是很直观。就像在文本分类中，我们可以通过查看数据集的 features 属性来找到这些整数和标签名称之间的对应关系：
```Py
ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature
=>
Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)
```
因此，我们得到了 ClassLabels 类型的序列。序列中元素的类型存储在 ner_feature 的 feature 中，我们可以通过查看该 feature 的 names 属性来访问标签名称的列表：
```Py
label_names = ner_feature.feature.names
label_names
=>
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

接下来使用这些标签名对数据集的 ner_tags 进行解码，将得到以下输出结果:
```Py
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
=>
'EU    rejects German call to boycott British lamb .'
'B-ORG O       B-MISC O    O  O       B-MISC  O    O'
```


#### 处理数据

像往常一样，我们的文本需要转换为 Token ID，然后模型才能理解它们。正如我们在 第六章 所学的那样。不过与 tokens 分类任务不同的是数据集已经完成了预分词，我们在处理的过程中不需要再次分词。

首先，创建 tokenizer 对象。
```Py
from transformers import AutoTokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```
你可以更换把 model_checkpoint 更换为 Hub 上你喜欢的任何其他模型，或使用本地保存的预训练模型和 tokenizer。唯一的限制是 tokenizer 需要由Tokenizers 库支持，并且有一个“快速”版本可用。你可以在 Huggingface 模型后端支持表 上看到所有带有快速版本的 tokenizer 架构，或者你也可以通过查看它 is_fast 属性来检测正在使用的 tokenizer 对象是否由 Tokenizers 支持：
```Py
tokenizer.is_fast
=>
True
```
我们可以像往常一样使用我们的 tokenizer 对预先分词的输入进行 tokenize ，只需额外添加 is_split_into_words=True 参数：
```Py
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()
=>
['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
```
如我们所见，tokenizer 在结果中添加了模型需要使用的特殊 tokens（在开头的 [CLS] ，在结尾的 [SEP] ），并且大部分单词保持不变。不过，单词 lamb 被分词为两个子词， la 和 ##mb 。这导致了输入和标签之间的不匹配：标签列表只有 9 个元素，而我们的输入现在有 12 个 tokens。解决特殊 tokens 的问题很容易（我们已经知道了每个 token 开始和结束的位置），我们需要确保的是我们将所有的标签与正确的词对齐。

幸运的是，由于我们使用的是快速 tokens 因此我们可以使用Tokenizers 超能力，这意味着我们可以轻松地将每个 token 映射到其相应的单词：
```Py
inputs.word_ids()
=>
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
```

只需要一点点工作，我们就可以扩展我们的标签列表。我们将添加的使其与 token 列表相匹配。 添加的第一条规则是，特殊 tokens 的标签设置为 -100 。这是因为默认情况下， -100 会被我们的损失函数（交叉熵）忽略。然后将每个 token 的标签设置为这个 token 所在单词开头的 token 的标签，因为同一个单词一定是同一个实体的一部分。最后将单词的内部 tokens 标签中的 B- 替换为 I-，因为该 token 不在实体的开头，B- 在每个实体中只应该出现一次：

```Py
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新单词的开始!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊的token
            new_labels.append(-100)
        else:
            # 与前一个 tokens 类型相同的单词
            label = labels[word_id]
            # 如果标签是 B-XXX 我们将其更改为 I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
```

让我们在数据集的第一个元素上试一试：
```Py
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
=>
[3, 0, 7, 0, 0, 0, 7, 0, 0]
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
```
正如我们所看到的，我们的函数为开头和结尾添加了两个特殊 tokens ：-100 ，并为切分成两个 tokens 的单词添加了一个新的 0 标签。


为了预处理我们的整个数据集，我们需要对所有输入进行 tokenize，并使用 align_labels_with_tokens() 函数处理所有标签。为了充分利用快速 tokenizer 的优势，最好是同时对大量文本一起进行 tokenize，所以我们需要编写一个处理一组示例的函数，并使用带有 batched=True 参数的 Dataset.map() 方法。与我们之前的示例唯一不同的是，当 tokenizer 的输入是文本列表（或示例中单词的列表的列表）时， word_ids() 函数需要根据列表的索引获取 token 的 ID，所以我们也在下面的函数中添加了这个功能：
```Py
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```
注意，我们还没有填充我们的输入，我们将在稍后使用数据整理器创建 batch 时进行。

我们现在可以一次性使用预处理函数处理整个数据集：
```Py
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

### 使用 Trainer API 微调模型

使用 Trainer 的代码会和以前一样，唯一的变化是数据如何整理成 batch 以及计算评估的分数。

#### 整理数据
我们不能像 第三章 那样直接使用 DataCollatorWithPadding ，因为那样只会填充输入（inputs ID、注意掩码和 tokens 类型 ID）。除了输入部分，在这里我们还需要对标签也使用与输入完全相同的方式填充，以保证它与输入的大小相同。我们将使用 -100 进行填充，以便在损失计算中忽略填充的内容。

以上的这些过程可以由 DataCollatorForTokenClassification 实现。它是一个带有填充功能的数据整理器，使用时只需要传入用于预处理输入的 tokenizer ：
```Py
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

为了在几个样本上测试这个数据整理器，我们可以先在训练集中的几个示例上调用它：
```Py
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]
=>
tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
        [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
```
将数据整理器的结果与数据集中未经处理的第一个和第二个元素的标签进行比较。

```Py
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])
=>
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
[-100, 1, 2, -100]
```
正如我们所看到的，已经使用 -100 将第二组标签填充到了与第一组标签相同的长度。


#### 评估指标
要让 Trainer 在每个周期计算一个指标，我们需要定义一个 compute_metrics() 函数，该函数的输入是预测值和标签的数组，并返回带有指标名称和评估结果的字典。

用于评估 Token 分类预测的经典框架是 seqeval 。要使用此指标，我们首先需要安装 seqeval 库：
!pip install seqeval
然后我们可以通过 evaluate.load() 函数加载它：

```Py
import evaluate
metric = evaluate.load("seqeval")
```

该指标和常规的评测指标有些区别：它需要字符串形式的标签列表而不是整数，所以我们需要在将它们传递给指标之前将预测值和标签由数字解码为字符串。让我们先用一些测试数据看看它是如何工作的：
```Py
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels
=>
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

然后我们可以通过更改索引 2 处的值来为这些标签创建假的预测值来测试该指标：
```Py
predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])
#请注意，该指标的输入是预测列表（不是一个）和标签列表，输出结果如下：
=>
{'MISC': {'precision': 1.0, 'recall': 0.5, 'f1': 0.67, 'number': 2},
 'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
 'overall_precision': 1.0,
 'overall_recall': 0.67,
 'overall_f1': 0.8,
 'overall_accuracy': 0.89}
```
结果返回很多信息！包括每个单独实体及整体的准确率、召回率和 F1 分数。在这里我们将只保留总分，但是你可以自由地调整 compute_metrics() 函数返回的所需要指标。这个函数中我们首先会先取预测 logits 的 argmax，并将其转换为预测值。 compute_metrics() 函数首先取 logits 的 argmax，将它们转换为预测值（通常情况下，logits 和概率的顺序是相同，所以我们不需要使用 softmax）。然后我们需要将标签和预测值都从整数转换为字符串。我们删除所有标签为 -100 的值，最后将结果传递给 metric.compute() 方法：
```Py
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # 删除忽略的索引(特殊 tokens )并转换为标签
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

现在已经完成了，我们下面就可以开始定义我们的 Trainer 了。接下来我们只需要一个 model 并对其微调！

#### 定义模型
由于我们正在研究 Token 分类问题，因此我们将使用 AutoModelForTokenClassification 类。定义此模型时要记得传递我们标签的数量，最简单方法是将该数字传递给 num_labels 参数，但是如果我们想要一个就像我们在本节开头看到的那样的推理小部件，就需要用最标准的方法正确设置标签的对应关系。

最标准的方式是用两个字典 id2label 和 label2id 来设置标签，这两个字典包含从 ID 到标签的映射以及反向的映射：
```Py
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
```
现在我们只需将它们传递给 AutoModelForTokenClassification.from_pretrained() 方法，它们就会被保存在模型的配置中，然后被正确地保存和上传到 Hub：
```Py
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```
确认一下我们的模型是否具有正确的标签数量：
```Py
model.config.num_labels
=>
9
```
⚠️ 如果你的模型的标签数量有错误，那么在后面调用 Trainer.train() 时，你会得到一个晦涩的错误（类似于“CUDA error：device-side assert triggered”）。这可能会令人烦恼，所以确保你做了这个检查，确认你的标签数量是正确。

#### 微调模型
在定义 Trainer 之前，我们只需要做最后两件事：登录 Hugging Face 并设置我们的训练参数。如果你在 notebook 上工作，有一个方便的小工具可以帮助你：
```Py
from huggingface_hub import notebook_login
notebook_login()
```
登录后，我们就可以设置我们的 TrainingArguments ：

```Py
from transformers import TrainingArguments
args = TrainingArguments(
    "bert-finetuned-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
```
你已经对大多数内容有所了解了：我们设置了一些超参数（如学习率、训练的轮数和权重衰减），并设定 push_to_hub=True ，表示我们希望在每个训练轮次结束时保存并评估模型，然后将结果上传到模型中心。注意，你可以通过 hub_model_id 参数指定你想推送的仓库的名称（特别需要注意的是，如果你需要推送给某个组织，就必须使用这个参数）。例如，当我们将模型推送到 huggingface-course 组织 时，我们在 TrainingArguments 中添加了 hub_model_id="huggingface-course/bert-finetuned-ner" 。默认情况下，使用的仓库将保存在你的账户之内，并以你设置的输出目录命名，所以在我们的例子中，仓库的地址是 "sgugger/bert-finetuned-ner" 。


最后，我们将所有内容传递给 Trainer 并启动训练：
```Py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()
```

训练完成后，我们使用 push_to_hub() 上传模型的最新版本
```Py
trainer.push_to_hub(commit_message="Training complete")
#如果你想检查一下是否上传成功，这个命令会返回刚刚执行的提交的 URL：
=>
'https://huggingface.co/sgugger/bert-finetuned-ner/commit/26ab21e5b1568f9afeccdaed2d8715f571d786ed'
```

### 使用微调模型

from transformers import pipeline
#将此替换为你自己的 checkpoint
model_checkpoint = "Kate-lf/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Kate McGill and I study in Hubei University of Technology.")


## 今日知识总结

### Token 分类任务的本质

Token 分类是对输入序列中的每个 token 做分类，不像文本分类那样对整个句子只输出一个标签。三大典型任务：命名实体识别（NER）标注人名/地名/组织名，词性标注（POS）标注名词/动词/形容词，分块（Chunking）标注句法块的边界。理解了 "序列进，序列出" 这个核心区别，就能区分 token classification 和 sequence classification。

### IOB2 标注格式

NER 标签不只有实体类别（PER/ORG/LOC/MISC），还有位置前缀。B-（Begin）标记实体开头，I-（Inside）标记实体内部，O（Outside）标记非实体。这是因为相邻两个同类型实体需要边界来区分——比如 "John Smith and Mary Jones" 中，B-PER I-PER O B-PER I-PER 能正确拆成两个人，全标 PER 则无法区分。CoNLL-2003 的 9 个标签就是：O + B/I × 4 类实体。

### 子词分词导致的标签对齐问题

这是今日最核心的技术点。BERT tokenizer 会把一个单词切成多个子词（如 "lamb" → "la" + "##mb"），但原始标签是按单词给的。直接用会错位。解决方案：用 `tokenizer(word_list, is_split_into_words=True)` 处理预分词输入，再通过 `word_ids()` 获取每个 token 对应的原词索引。特殊 token（[CLS], [SEP]）对应 None，标 -100 让损失函数忽略。同一单词的后续子词若原标签是 B-XXX，改为 I-XXX（因为只有第一个子词算实体开头）。这个对齐逻辑封装在 `align_labels_with_tokens()` 函数中，是整个 NER 预处理的核心。

### DataCollatorForTokenClassification 的作用

不同于文本分类用 `DataCollatorWithPadding` 只填充输入，NER 的标签也需要填充到相同长度。这个整理器用 -100 填充标签，确保 padding 位置不被计入损失。这就是为什么前面标签对齐时也用 -100——与 padding 值保持一致，避免混淆。

### seqeval：实体级评估而非 token 级

NER 不能用 accuracy 直接评估，因为 O 标签占绝大多数，全标 O 也能拿高 accuracy 但毫无意义。seqeval 按实体级别计算 precision/recall/F1——只有实体的边界和类别都完全匹配才算正确。这更贴近实际需求：我们要找对"人"，而不是找对"每个字"。注意 seqeval 要求输入是字符串标签列表而非整数 ID。

### 解决的两个环境问题

datasets 5.0.0 不再支持旧的脚本式数据集加载。`load_dataset("conll2003")` 报 "Dataset scripts are no longer supported"。解决：指定 `revision="refs/convert/parquet"` 使用已转换为 Parquet 格式的分支。`TrainingArguments` 的 `evaluation_strategy` 参数在 Transformers 4.57 中已重命名为 `eval_strategy`，这是第二次遇到此问题。

### 完整 NER 微调流程

数据加载（CoNLL-2003, parquet 分支）→ tokenizer 预处理（`is_split_into_words=True`）→ 标签对齐（`word_ids()` + `align_labels_with_tokens()`）→ `Dataset.map()` 批处理 → `DataCollatorForTokenClassification` 动态填充 → `seqeval` 评估指标 → `AutoModelForTokenClassification` + `id2label`/`label2id` → `TrainingArguments(eval_strategy="epoch")` → `Trainer` 组装 → `trainer.train()`。整个流程和文本分类的 Trainer 模式一致，差异仅在数据处理和评估两个环节。
