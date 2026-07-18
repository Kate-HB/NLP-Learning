# Day 08：TOKENIZERS库

深入 Tokenizers 库：训练新 tokenizer、快速 tokenizer 的高级能力（偏移映射、实体分组），以及复现 token-classification 和 QA 管道的完整内部流程。

## 基于已有的 tokenizer 训练新的 tokenizer
如果你感兴趣的语言中没有可用的语言模型，或者你的语料库与语言模型训练时所使用的语料库差异很大，你可能需要从零开始重新训练一个适应你的数据的 tokenizer 模型。训练一个新的 tokenizer 意思是为了找到语料库中的常见子词，tokenizer 需要深入统计语料库中的所有文本——这个过程我们称之为 训练 （training） 。具体的训练规则取决于使用的 tokenizer 类型。

训练 tokenizer 与训练模型不同！模型训练使用随机梯度下降使每个 batch 的 loss 小一点。它本质上是随机的（这意味着在即使两次训练的参数和算法完全相同，你也必须设置一些随机数种子才能获得相同的结果）。训练 tokenizer 是一个统计过程，它试图确定哪些子词最适合为给定的语料库选择，确定的过程取决于分词算法。它是确定性的，这意味着在相同的语料库上使用相同的算法进行训练时，得到的结果总是相同的。

### 准备语料库
Transformers 中，有一个非常简单的 API 可以让你从旧的 tokenizer 训练一个新的 tokenizer 且新的 tokenizer 具有和旧 tokenizer 相同的特性，它就是： AutoTokenizer.train_new_from_iterator() 。为了演示这个功能，我们将尝试从零开始训练 GPT-2 模型，但是在非英语的语言上。我们首先需要做的就是在训练语料库中收集大量的目标语言数据。为了让每个人都能理解，我们不会使用俄语或汉语这样的语言，而是使用一种特殊的英语语言：Python 代码。

Datasets 库可以帮助我们下载一个 Python 源代码语料库。我们将使用 load_dataset() 功能下载和缓存 CodeSearchNet 数据集。该数据集包含了 GitHub 上开源库中的数百万个函数，涵盖了多种编程语言。在这里，我们将加载这个数据集的 Python 部分：

```python
from datasets import load_dataset
raw_datasets = load_dataset("code-search-net/code_search_net", "python")
```

我们可以查看训练集部分来看我们可以使用哪些列：
```python
raw_datasets["train"]
=>
Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})
```

我们可以看到数据集把函数的文档说明（func_documentation_string）与代码（func_code_string）分开保存，并提供了一个可以参考的分词后的结果（func_code_tokens）。在这里，我们仅使用 whole_func_string 列来训练我们的 tokenizer 我们可以通过索引来查看其中一个函数的示例：
```python
print(raw_datasets["train"][123456]["whole_func_string"])
=>
def handle_simple_responses(
      self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
    """Accepts normal responses from the device.

    Args:
      timeout_ms: Timeout in milliseconds to wait for each response.
      info_cb: Optional callback for text sent from the bootloader.

    Returns:
      OKAY packet's message.
    """
    return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)
```

我们首先需要做的是将数据集转换为一个文本列表的 迭代器 例如，文本列表的列表。使用文本列表会使我们的 tokenizer 运行得更快（这样可以以文本批次为单位进行训练，而不是一次处理一个文本），并且使用迭代器可以不把所有内容都加载到内存中。如果你的语料库很大，你可能会想利用Datasets 将数据集的元素存储在磁盘上分批加载，而不是将所有内容加载到 RAM 的特性。

下面的操作会创建一个由每个列包含 1000 个文本组成的文本列表，但会将所有内容加载到内存中：

> ⚠️ 除非你的数据集很小，否则不要直接运行下面的代码！

```python
training_corpus = [
    raw_datasets["train"][i: i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
]
```


通过使用 Python 生成器，我们可以使 Python 只将正在使用的数据加载到内存中。要创建这样一个生成器，你只需要将方括号替换为圆括号：
```python
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```
这行代码不会加载数据集的任何元素；它只创建了一个你可以在 Python for 循环中使用的对象。只有当你使用它们（即，当你在 for 循环尝试访问他们）时，文本才会被加载，而且一次只会加载 1000 个文本。这样，即使你在处理大型数据集，也不会耗尽所有内存。

生成器对象的问题是它只能被使用一次。
这就是为什么我们需要定义一个返回生成器的函数。通过每次调用函数生成一个新的生成器对象，我们可以多次使用生成器而不会遇到只能使用一次的限制。
```python
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

training_corpus = get_training_corpus()
```


### 训练一个新的 tokenizer
现在我们已经将文本转化为迭代器形式准备好了我们的语料库，我们就可以开始训练新的 tokenizer 了。首先，我们需要加载我们想要与我们的模型匹配的 tokenizer （这我们这个例子中是 GPT-2）：
```python
from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
尽管我们要训练一个新的 tokenizer，但从旧的 tokenizer 开始初始化依然是个不错的主意，这样，我们就不必指定具体的 tokenization 算法或设置我们想要使用的特殊 tokens；我们新的 tokenizer 将与 GPT-2 完全相同，唯一的区别是词汇表，这将由我们的语料库通过训练来重新确定。

首先让我们看看旧的 tokenizer 将如何处理示例的数据：
```python
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokens
=>
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo',
 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', ]
```
这个 tokenizer 输出了一些特殊的符号，比如 Ċ 和 Ġ ，分别表示空格和换行符。正如我们所看到的，这并不是非常高效：tokenizer 将每个空格视作为单独的 token，其实它可以将缩进级别组合在一起时（因为在代码中经常出现相邻在一起的四个或八个空格）。它也有点奇怪地拆分了函数名称，对使用 _ 命名方法的函数并不友好。

让我们训练一个新的 tokenizer 看看它是否能解决这些问题。为此，我们将使用 train_new_from_iterator() 方法：
```python
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```
注意 AutoTokenizer.train_new_from_iterator() 只有你使用的 tokenizer 是“快速（fast）” tokenizer 时才有效。 Transformers 库包含两种类型的 tokenizer.（慢速的）完全用 Python 编写，而（快速的）由Tokenizers 库支持，该库用 Rust 编程语言编写。Python 是最常用于数据科学和深度学习应用程序的语言，但是当需要并行化以提高速度时，就需要用另一种语言来编写。例如，模型计算核心的矩阵乘法是用 CUDA 编写的，这是一个针对 GPU 优化的 C 语言库。

用纯 Python 训练一个全新的 tokenizer 会非常缓慢，这就是我们开发  Tokenizers 库的原因。正如你无需学习 CUDA 语言即可在 GPU 上训练你的模型一样，你也无需学习 Rust 即可使用快速 tokenizer。Tokenizers 库为许多内部调用 Rust 代码的方法提供 Python 语言绑定；例如，并行化训练新的 tokenizer 或者对一批输入进行 tokenize。大多数 Transformer 模型都有可用的快速 tokenizer ，如果 AutoTokenizer 可用，API 默认为你选择快速 tokenizer 

使用我们的全新 tokenizer
```python
tokens = tokenizer.tokenize(example)
=>
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`','a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

在这里我们再次看到了表示空格和换行符的特殊符号 Ċ 和 Ġ ，但我们也可以看到我们的 tokenizer 学习了一些专属于 Python 函数语料库的 token：例如，有一个 ĊĠĠĠtoken 表示缩进，以及 Ġ token 表示开始文档字符串的三个引号。tokenizer 也正确地在 _ 上拆分了函数名称。这是一个非常紧凑的表示；相比之下，使用简单的英语 tokenizer 会得到一个更长的句子：
```python
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))
=>
27
36
```

### 保存 tokenizer
像模型一样，是通过 save_pretrained() 方法进行保存：
```python
tokenizer.save_pretrained("code-search-net-tokenizer")
```
这将创建一个名为的 code-search-net-tokenizer 的新文件夹，它将包含重新加载 tokenizer 所需要的所有文件。

上传到 Hub。
```python
from huggingface_hub import notebook_login
notebook_login()
```
输入你的 Hugging Face 账号密码。

如果你不是在 notebook 上工作，只需在终端中输入以下行：
huggingface-cli login

登录后，你可以通过执行以下命令来推送你的 tokenizer
tokenizer.push_to_hub("code-search-net-tokenizer")

这将在你的账户中创建一个名为 code-search-net-tokenizer 的新仓库，其中将包含 tokenizer 文件。然后，你可以使用 tokenizer 的 from_pretrained() 方法从任何地方加载 tokenizer 。
```python
tokenizer = AutoTokenizer.from_pretrained("Kate-lf/code-search-net-tokenizer")
```



## 快速 tokenizer 的特殊能力
到目前为止，我们只使用它们来对文本进行 tokenize 或将token ID 解码回文本，但是 tokenizer —— 特别是由Tokenizers 库支持的 tokenizer —— 能够做的事情还有很多。为了说明这些附加功能，我们将探讨如何复现token-classification （我们称之为 ner ） 和 question-answering 管道的结果。

在接下来的讨论中，我们会经常区分“慢速”和“快速” tokenizer 。慢速 tokenizer 是在 Transformers 库中用 Python 编写的，而快速版本是由 Tokenizers 提供的，它们是用 Rust 编写的。

对单个句子进行 tokenize 时，你不总是能看到同一个 tokenizer 的慢速和快速版本之间的速度差异。事实上，快速版本可能更慢！只有同时对大量文本进行 tokenize 时，你才能清楚地看到差异。

### 批量编码

tokenizer 的输出不是简单的 Python 字典；我们得到的实际上是一个特殊的 BatchEncoding 对象。它是字典的子类（这就是为什么我们之前能够直接使用索引获取结果的原因），但是它还提供了一些主要由快速 tokenizer 提供的附加方法。

除了它们的并行化能力之外，快速 tokenizer 的关键功能是它们始终跟踪最终 token 相对于的原始文本的映射——我们称之为 偏移映射（offset mapping） 。这反过来又解锁了如将每个词映射到它生成的 token，或者将原始文本的每个字符映射到它所在的 token 等功能。

一个例子：
```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
=>
<class 'transformers.tokenization_utils_base.BatchEncoding'>
```
由于 AutoTokenizer 类默认选择快速 tokenizer 因此我们可以使用 BatchEncoding 对象提供的附加方法。
两种方法来检查我们的 tokenizer 是快速的还是慢速的。
检查 tokenizer 的 is_fast 属性：
```Python
tokenizer.is_fast
=>
True
```

检查我们 encoding 的 is_fast 属性：
```Python
encoding.is_fast
=>
True
```

快速 tokenizer 能为我们提供更多功能：

1. 直接得到 tokenization 之前的单词而无需将 ID 转换回单词：
```Python
encoding.tokens()
=>
['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in','Brooklyn', '.', '[SEP]']
```

在这种情况下，索引 5 处的 token 是 ##yl ，它是原始句子中“Sylvain”一词的一部分。我们也可以使用 word_ids() 方法来获取每个 token 原始单词的索引：
```Python
encoding.word_ids()
=>
[None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]
```

我们可以看到 tokenizer 的特殊 token [CLS] 和 [SEP] 被映射到 None ，然后每个 token 都映射到它来源的单词。这对于确定一个 token 是否在单词的开头或两个 token 是否在同一个单词中特别有用。对于 BERT 类型（BERT-like）的 tokenizer 我们也可以依靠 ## 前缀来实现这个功能；不过只要是快速 tokenizer 它所提供的 word_ids() 方法适用于任何类型的 tokenizer 。

利用这种能力，可将每个词正确地对应到词汇任务中的标签，如命名实体识别（NER）和词性标注（POS）。也可以使用它在掩码语言建模（masked language modeling）中来遮盖来自同一词的所有 token（一种称为 全词掩码（whole word masking） 的技术）。

2 sentence_ids() 方法，可以用它把一个 token 映射到它原始的句子（尽管在这种情况下，tokenizer 返回的 token_type_ids也可以为我们提供相同的信息）。

3 通过 word_to_chars() 或 token_to_chars() 和 char_to_word() 或 char_to_token() 方法，将任何词或 token 映射到原始文本中的字符，反之亦然。例如， word_ids() 方法告诉我们 ##yl 是索引 3 处单词的一部分，但它是句子中的哪个单词？我们可以这样找出来：
```Python
start, end = encoding.word_to_chars(3)
example[start:end]
=>
Sylvain
```
以 "My name is Sylvain" 为例：
char（字符）：原始字符串中的每个字符位置。Sylvain 在原始文本的第 11~18 个字符。
token：分词后的子词单元。Sylvain 被 BERT 切成 S, ##yl, ##va, ##in 四个 token。
word（词）：分词前的完整单词。word_ids() 返回 [0, 1, 2, 3, 3, 3, 3]，即 token 3~6 都属于 word 3（Sylvain）。
映射关系：word_to_chars(3) → (11, 18)，切出原文 example[11:18] = "Sylvain"。token_to_chars(5) → ##yl 在原文中的字符范围。

如前所述，这一切都是由于快速分词器跟踪每个 token 来自的文本范围的一组偏移。为了阐明它们的作用，接下来复现 token-classification 管道的结果。

### token-classification 管道内部流程
命名实体识别（NER）——该任务是确定文本的哪些部分对应于人名、地名或组织名等实体——当时是使用Transformers 的 pipeline() 函数实现的。管道如何将获取原始文本到预测结果的三个阶段整合在一起：tokenize、通过模型处理输入和后处理。 token-classification 管道中的前两步与其他任何管道中的步骤相同，但后处理稍有复杂——让我们看看具体情况！


#### 使用管道获得基本结果
首先，让我们获取一个 token 分类管道，以便我们可以手动比较一些结果。这次我们选用的模型是 dbmdz/bert-large-cased-finetuned-conll03-english ；我们使用它对句子进行 NER：
```Python
from transformers import pipeline
token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
=>
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```
模型正确地识别出：“Sylvain”是一个人，“Hugging Face”是一个组织，以及“Brooklyn”是一个地点。
我们也可以让管道将同一实体的 token 组合在一起：
```Python
from transformers import pipeline
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
=>
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```
选择不同的 aggregation_strategy 可以更改每个分组实体计算的策略。对于 simple 策略，最终的分数就是给定实体中每个 token 的分数的平均值：例如，“Sylvain”的分数是我们在前一个例子中看到的 token S ， ##yl ， ##va ，和 ##in 的分数的平均值。

其他可用的策略包括：
“first”，其中每个实体的分数是该实体的第一个 token 的分数
“max”，其中每个实体的分数是该实体中 token 的最大分数
“average”，其中每个实体的分数是组成该实体的单词分数的平均值

#### 从输入到预测
首先，我们需要将我们的输入进行 tokenization 并将其传递给模型。我们使用 AutoXxx 类实例化 tokenizer 和模型，然后将我们的示例传递给它们：
```Python
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

print(inputs["input_ids"].shape)
print(outputs.logits.shape)
=>
torch.Size([1, 19])
torch.Size([1, 19, 9])
```


> **Auto 不是"自动选任务"，是"自动选架构类"。** checkpoint 决定"用哪个模型"，Auto 自动查 config 找到对应的加载类。
>
> `AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")` → 查 config 发现是 BERT → 自动用 `BertTokenizer` 加载。
>
> `AutoModelForTokenClassification.from_pretrained(checkpoint)` → 查 config 发现是 BERT → 自动用 `BertForTokenClassification` 加载。
>
> 不用 Auto 的写法：`BertTokenizer.from_pretrained(...)` / `BertForTokenClassification.from_pretrained(...)`。
>
> Auto 的好处：换 checkpoint（如换 RoBERTa）只改 checkpoint 名，Auto 自动切换到 `RobertaTokenizer`。

我们有一个包含 19 个 token 序列的 batch 和有 9 个不同的标签类型，所以模型的输出形状为 1 x 19 x 9。像文本分类管道一样，我们使用 softmax 函数将这些 logits 转化为概率，并取 argmax 来得到预测（请注意，我们可以在 logits 上直接算取 argmax，因为 softmax 不会改变顺序）：
```Python
import torch
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
=>
[0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
```

model.config.id2label 属性包含索引到标签的映射，我们可以用它来将预测转化为标签：
```Python
model.config.id2label
=>
0: O       (非实体)
1: B-MISC  (杂项-开头)
2: I-MISC  (杂项-内部)
3: B-PER   (人名-开头)
4: I-PER   (人名-内部)
5: B-ORG   (组织-开头)
6: I-ORG   (组织-内部)
7: B-LOC   (地点-开头)
8: I-LOC   (地点-内部)
```
4 × 2 + 1 = 9。B/I 的区分是因为一个实体可能占多个 token（如 S, ##yl, ##va, ##in），B 标记实体的第一个 token，I 标记后续。

对于 B- 和 I- 标签，实际上有两种格式：IOB1 和 IOB2。我们介绍的是 IOB2 格式，而在 IOB1 格式（蓝色）中，以 B- 开头的标签只用于分隔同一类型的两个相邻实体。
[图示](../03-NLP-Basic/pictures/IOB_versions.svg)


有了这个映射字典，我们就可以几乎完全复现管道的结果 —— 我们只需要获取每个没有被分类为 O 的 token 的得分和标签：
```Python
results = []
tokens = inputs.tokens()
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

print(results)
=>
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S'},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl'},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va'},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in'},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu'},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging'},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face'},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn'}]
```

这与我们之前的结果非常相似，但有一点不同：pipeline 还给我们提供了每个实体在原始句子中的 start 和 end 的信息。如果要复现这个特性，这就是我们的偏移映射要发挥作用的地方。要获得偏移量，我们只需要在使用 tokenizer 器时设置 return_offsets_mapping=True ：
```Python
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
=>
[(0, 0), (0, 2), (3, 7), (8, 10), (11, 12), (12, 14), (14, 16), (16, 18), (19, 22), (23, 24), (25, 29), (30, 32),
 (33, 35), (35, 40), (41, 45), (46, 48), (49, 57), (57, 58), (0, 0)]
```
每个元组都是每个 token 对应的文本范围，其中 (0, 0) 是为特殊 token 保留的。我们之前看到索引 5 的 token 是 ##yl ，它所对应的偏移量是 (12, 14) 。

使用这个，我们现在可以完成之前的想法：
```Python
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "index": idx,
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)
=>
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

#### 实体分组
使用偏移来确定每个实体的开始和结束的索引很方便，但这并不是它唯一的用法。当我们希望将实体分组在一起时，偏移映射将为我们省去很多复杂的代码。例如，如果我们想将 Hu 、 ##gging 和 Face token 分组在一起，我们可以制定特殊规则，比如说前两个应该在去除 ## 同时连在一起， Face 应该在不以 ## 开头的情况下增加空格 —— 但这些规则只适用于这种特定类型的分词器。当我们使用 SentencePiece 或 Byte-Pair-Encoding 分词器（在本章后面讨论）时就要重新写另外一套规则。

有了偏移量，就可以免去为特定分词器定制特殊的分组规则：我们只需要取原始文本中以第一个 token 开始和最后一个 token 结束的范围。因此，假如说我们有 Hu 、 ##gging 和 Face token，我们只需要从字符 33（ Hu 的开始）截取到字符 45（ Face 的结束）：

example[33:45]
=>
Hugging Face

为了编写处理预测结果并分组实体的代码，我们将对连续标记为 I-XXX 的实体进行分组，因为只有实体的第一个 token 可以被标记为 B-XXX 或 I-XXX ，因此，当我们遇到实体 O 、新类型的实体或 B-XXX 时，我们就可以停止聚合同一类型实体。
```Python
import numpy as np
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # 删除 B- 或者 I-
        label = label[2:]
        start, _ = offsets[idx]

        # 获取所有标有 I 标签的token
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # 分数是该分组实体中所有token分数的平均值
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
=>
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```



## 在 QA 管道中使用快速 tokenizer
我们现在将深入研究 question-answering 管道，看看如何利用偏移量从上下文（context）中获取当前问题的答案，这与我们在上一节中处理分组实体的方式有些相似。我们会看到如何处理那些因为过长而最终被截断的上下文（context）。

### 使用 question-answering 管道
```Python
from transformers import pipeline
question_answerer = pipeline("question-answering")
context = """
Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back Transformers?"
question_answerer(question=question, context=context)
=>
{'score': 0.97773,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}
```
与其他不能处理超过模型接受的最大长度的文本的管道不同，这个管道可以处理非常长的上下文（context），并且即使答案在末尾也能返回问题的答案

### 使用模型进行问答
与任何其他管道一样，我们首先对输入进行 tokenize，然后将其传入模型。 question-answering 管道默认情况下用于的 checkpoint 是 distilbert-base-cased-distilled-squad （名字中的”squad”源自模型微调所用的数据集:
```Python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```
请注意在这里，我们将问题放在前面,上下文放后面，一起作为一对进行tokenization。
[图示](../03-NLP-Basic/pictures/question_tokens.svg)
问答模型的工作方式与我们迄今为止看到的模型略有不同。以上图为例，模型训练的目标是来预测答案开始的 token 的索引（这里是 21）和答案结束的 token 的索引（这里是 24）。这就是为什么这些模型不返回一个 logits 的张量，而是返回两个：一个对应于答案的开始 token 的 logits，另一个对应于答案的结束 token 的 logits。在这个例子中，我们的输入包含了 66 个 token ，因此我们得到：
```Python
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
=>
torch.Size([1, 66]) torch.Size([1, 66])
```

为了将这些 logits 转换为概率，我们将使用一个 softmax 函数——但在此之前，我们需要确保我们屏蔽了不属于上下文的索引。我们的输入格式是 [CLS] question [SEP] context [SEP] ，所以我们需要屏蔽 question 的 tokens 以及 [SEP] token 。不过，我们将保留 [CLS] ，因为某些模型使用它来表示答案不在上下文中。

由于我们将在之后使用 softmax，我们只需要将我们想要屏蔽的 logits 替换为一个大的负数就可以在计算 softmax 的时候屏蔽他们。在这里，我们使用 -10000 ：
```Python
import torch
sequence_ids = inputs.sequence_ids()
#屏蔽除 context 之外的所有内容
mask = [i != 1 for i in sequence_ids]
#不屏蔽 [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000
```

> 输入结构是 `[CLS] question [SEP] context [SEP]`。答案只应该从 context 部分找。
>
> `sequence_ids()` 返回每个 token 属于哪段：`None`=特殊 token, `0`=question, `1`=context：
>
> ```
> token:  [CLS]  What  is  ...  [SEP]  Hugging  Face  ...  [SEP]
> seq_id:  None    0    0   0   None      1      1    1   None
> mask:    True   True True True True   False  False False  True   ← i != 1
> mask:   False   True True True True   False  False False  True   ← mask[0]=False 放行[CLS]
> ```
>
> 然后 `start_logits[mask] = -10000` 把 question 和 `[SEP]` 位置的分数压到极小值，softmax 后这些位置概率≈0。
>
> 保留 `[CLS]` 是因为：如果答案不在 context 中，模型应该预测 `[CLS]` 作为"无答案"——这是 SQuAD 数据集的设计。

现在我们已经屏蔽了与我们不想预测的位置相对应的 logits，接下来我们可以使用 softmax：
```Python
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
```

在这个阶段，我们可以取开始和结束概率的 argmax —— 但是我们可能会得到一个比结束索引大的开始索引，因此我们需要采取一些更多的措施来处理这些特殊情况。我们将在满足 start_index <= end_index 的前提下计算每个可能的 start_index 和 end_index 的概率，然后取概率最高的 (start_index, end_index) 元组。

假设事件”答案开始于 start_index “和”答案结束于 end_index “是独立的，答案在 start_index 开始并在 end_index 结束的概率是：
start_probabilities[start_index]×end_probabilities[end_index]

所以，要计算所有的分数，我们只需要计算所有的 start_index <= end_index 的 start_probabilities[start_index]×end_probabilities[end_index]

首先让我们计算所有可能的乘积：
```Python
scores = start_probabilities[:, None] * end_probabilities[None, :]
```

然后我们将 start_index > end_index 的值设置为 0 来屏蔽他们（其他概率都是正数）。 torch.triu() 函数返回传入的 2D 张量的上三角部分，所以我们可以使用它来完成屏蔽：
```Python
import numpy as np
scores = torch.triu(scores)
```

现在我们只需要得到最大值的索引。由于 PyTorch 将返回展平（flattened）后张量中的索引，因此我们需要使用向下取整的除法 // 和取模 % 操作来获得 start_index 和 end_index ：
```Python
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
=>
0.97773
```

> `argmax()` 默认把多维张量**展平成一维**再找最大值的位置。`scores` 形状 `[66, 66]` 展平为 `[4356]`，返回值是展平后的位置，不是 `(行, 列)`。所以要还原：
>
> ```python
> start_index = 2478 // 66   # = 37（行号，// 是除以列数取整）
> end_index   = 2478 % 66    # = 36（列号，% 是取余）
> ```

我们有了答案的 start_index 和 end_index ，所以现在我们只需要将他们转换为上下文中的字符索引。这就是偏移量将会非常有用的地方。我们可以像我们在 token 分类任务中那样获取偏移量并使用它们：
```Python
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]
```

现在我们只需要格式化所有内容，获取我们的结果：
```Python
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)
=>
{'answer': 'Jax, PyTorch and TensorFlow',
 'start': 78,
 'end': 105,
 'score': 0.97773}
 ```


### 处理长文本
如果我们尝试将我们之前使用的长问题和长上下文进行 tokenize，我们将得到一个比 question-answering pipeline 中使用的最大长度（384）更大的 tokens 数量

所以，我们需要将我们的输入截断到模型允许输入的最大长度。我们可以用几种方式做到这一点，但我们不想截断问题部分，只想截断上下文部分，并且由于上下文部分是第二项，因此我们将使用 "only_second" 截断策略。然后又出现了新的问题：问题的答案可能在截断后被丢弃了，并没有在截断后保留下来的上下文文本中。

这意味着模型将很难找到正确的答案。为了解决这个问题， question-answering 管道允许我们将上下文分成更小的块，指定最大长度。为了确保我们不在刚好可能找到答案的地方将上下文分割，它还在各块之间包含了一些重叠。

我们可以通过添加 return_overflowing_tokens=True 参数，并可以用 stride 参数指定我们想要的重叠长度来让 tokenizer为我们做这个工作。
```Python
sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
=>
'[CLS] This sentence is not [SEP]'
'[CLS] is not too long [SEP]'
'[CLS] too long but we [SEP]'
'[CLS] but we are going [SEP]'
'[CLS] are going to split [SEP]'
'[CLS] to split it anyway [SEP]'
'[CLS] it anyway. [SEP]'
```
正如我们所看到的，句子已被分成多个块，使得每个条目 inputs["input_ids"] 最多有 6 个 token （我们需要添加填充以使分割后的最后一个条目与其他条目的大小相同）并且每个条目之间有 2 个 token 的重叠。

tokenization的结果：
```Python
print(inputs.keys())
=>
dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
```
最后一个键，overflow_to_sample_mapping，是一个映射，告诉我们每个结果对应哪个句子——在这里，我们有 7 个结果，它们都来自我们传递给 tokenizer 的（唯一的）句子：
```Python
print(inputs["overflow_to_sample_mapping"])
=>
[0, 0, 0, 0, 0, 0, 0]
```

## 标准化和预分词
Transformer 模型常用的三种分词算法（字节对编码[BPE]、WordPiece 和 Unigram）
tokenization过程：
[图示](../03-NLP-Basic/pictures/tokenization_pipeline.svg)
在分词（根据其模型）之前，tokenizer 需要进行两个步骤： 标准化（normalization） 和 预分词（pre-tokenization） 。

### 标准化（normalization）
标准化步骤涉及一些常规清理，例如删除不必要的空格、小写和“/”或删除重音符号。

Transformers 的 tokenizer 具有一个名为 backend_tokenizer 的属性，该属性可以访问来自Tokenizers 库的底层 tokenizer 。
```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
=>
<class 'tokenizers.Tokenizer'>
```

tokenizer 对象的 normalizer 属性具有一个 normalize_str() 方法，我们可以使用该方法查看如何进行标准化：
```Python
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
=>
'hello how are u?'
```

### 预分词(pre-tokenization)
tokenizer 一般不会在原始文本上进行训练。因此，我们首先需要将文本拆分为更小的实体，例如单词。这就是预分词步骤的作用。

要查看快速 tokenizer 如何执行预分词，我们可以使用 tokenizer 对象的 pre_tokenizer 属性的 pre_tokenize_str() 方法：
```Python
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
=>
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```
请注意 tokenizer 记录了偏移量，这是就是我们在前一节中使用的偏移映射。在这里 tokenizer 将两个空格并将它们替换为一个，从 are 和 you 之间的偏移量跳跃可以看出来这一点。

由于我们使用的是 BERT tokenizer 所以预分词会涉及到在空白和标点上进行分割。其他的 tokenizer 可能会对这一步有不同的规则。例如，如果我们使用 GPT-2 的 tokenizer
```Python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
=>
[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
 ('?', (19, 20))]
```
它也会在空格和标点符号上拆分，但它会保留空格并将它们替换为 Ġ 符号，不会忽略双空格。

最后一个例子，基于 SentencePiece 算法的 T5 tokenizer
```Python
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
=>
[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```
与 GPT-2 的 tokenizer 类似，这个 tokenizer 保留空格并用特定 token 替换它们（ _ ），但 T5 tokenizer 只在空格上拆分，不考虑标点符号。另外，它会在句子开头（在 Hello 之前）默认添加一个空格，并忽略 are 和 you 之间的双空格。

### 算法概述
我们将深入研究三种主要的子词 tokenization 算法：BPE（由 GPT-2 等使用）、WordPiece（由 BERT 使用）和 Unigram（由 T5 等使用）。先来快速了解它们各自的工作方式。

模型：BPE	
训练：从小型词汇表开始，学习合并 token 的规则
训练步骤：合并对应最常见的 token 对
学习：合并规则和词汇表
编码：将一个单词分割成字符并使用在训练过程中学到的合并

模型：WordPiece	
训练：从小型词汇表开始，学习合并 token 的规则
训练步骤：合并对应得分最高的 token 对，优先考虑每个独立 token 出现频率较低的对
学习：仅词汇表
编码：从开始处找到词汇表中的最长子词，然后对其余部分做同样的事

模型：Unigram	
训练：从大型词汇表开始，学习删除 token 的规则
训练步骤：删除会在整个语料库上最小化损失的词汇表中的所有 token
学习：含有每个 token 分数的词汇表
编码：使用在训练中学到找到最可能的 token 分割方式


## 今日知识总结

### 训练新 tokenizer

`AutoTokenizer.train_new_from_iterator()` 可以从语料库训练新 tokenizer，保留旧 tokenizer 的算法和特殊 token，只换词表。训练是统计过程（确定性），不同于模型训练（随机梯度下降）。语料库要用生成器避免全部加载到内存。注意到 `code_search_net` 数据集路径已更新为 `code-search-net/code_search_net`，国内需设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 快速 vs 慢速 tokenizer

慢速 tokenizer 纯 Python 实现，快速 tokenizer 由 Rust 的 Tokenizers 库支持。快速版的关键能力是**偏移映射**（offset mapping）：始终跟踪每个 token 在原始文本中的字符范围 `(start, end)`。`tokenizer()` 返回的 `BatchEncoding` 对象是字典子类，提供 `tokens()`、`word_ids()`、`word_to_chars()`、`token_to_chars()` 等附加方法。

### char、token、word 三层概念

- **char**：原始字符串的字符位置，最细粒度
- **token**：分词后的子词单元（如 `S`, `##yl`, `##va`, `##in`）
- **word**：分词前的完整单词（如 "Sylvain"）

`word_ids()` 将每个 token 映射到所属单词的索引，`word_to_chars(n)` 返回第 n 个词在原文中的字符范围。偏移映射始终是原始字符串的绝对位置，空格仍占字符位，但被 token 跳过不覆盖。

### NER 管道与实体分组

`AutoModelForTokenClassification` 输出 logits 形状 `[batch, seq_len, 9]`，9 个标签来自 CoNLL-2003：O + B/I-PER/ORG/LOC/MISC（4×2+1=9）。实体分组核心逻辑：用 `offset_mapping` 追踪连续 I-XXX token，`start` 取第一个 token 的起始偏移，`end` 循环更新为最后一个 token 的结束偏移，最后用 `example[start:end]` 提取完整实体原文。聚合策略有 `simple`（均值）、`first`（首 token）、`max`（最大值）。

### QA 管道

QA 模型输出 `start_logits` 和 `end_logits` 两组 logits，分别预测答案起始和结束 token。关键步骤：用 `sequence_ids()` 区分 question（0）和 context（1），将非 context 位置的 logits 设为 -10000 屏蔽，仅保留 `[CLS]` 用于"无答案"预测。`argmax()` 在 2D 张量上返回展平索引，需用 `//` 和 `%` 还原为 `(start_index, end_index)`。最终用偏移量将 token 索引转为原文字符位置。

### 标准化与预分词

**标准化**：常规清理（小写、去重音、归一化空格等）。**预分词**：将文本拆分为更小单元，不同 tokenizer 规则不同：

| tokenizer | 空格处理 | 标点处理 |
|-----------|---------|---------|
| BERT | 忽略双空格 | 在标点上拆分 |
| GPT-2 | 保留空格为 Ġ | 在标点上拆分 |
| T5 | 保留空格为 ▁，忽略双空格 | 只在空格上拆分，标点不断开 |

### 三种子词算法

- **BPE**：从小词表开始，合并最常见 token 对（GPT-2）
- **WordPiece**：从小词表开始，按得分合并 token 对，优先低频独立 token（BERT）
- **Unigram**：从大词表开始，删除最小化损失的 token（T5）
