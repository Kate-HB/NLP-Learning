# tokenizer 
NLP 管道的核心组件之一。模型只能处理数字，因此 tokenizer 将文本输入转换为数字。

## 基于单词（Word-based）的 tokenization

容易配置和使用，只需几条规则，并且通常会产生不错的结果。例如，在下图中，目标是将原始文本拆分为单词并为每个单词找到一个数字表示：
[图示](03-NLP-Basic\pictures\word_based_tokenization.svg)

有多种方法可以拆分文本。例如，我们可以通过使用 Python 的 split() 函数，使用空格将文本分割为单词：
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
['Jim', 'Henson', 'was', 'a', 'puppeteer']

此外，还有一些基于单词的 tokenizer 的变体，对标点符号有额外的规则。使用这类 tokenizer，我们最终可以得到一些非常大的“词汇表（vocabulary）”，其中词汇表的大小由我们在语料库中拥有的独立 tokens 的总数确定。

每个单词都分配了一个 ID，从 0 开始一直到词汇表的大小。模型使用这些 ID 来识别每个词。

如果我们想用基于单词的 tokenizer 完全覆盖一种语言，我们需要为语言中的每个单词设置一个标识符，这将生成大量的 tokens。例如，英语中有超过 500,000 个单词，因此要构建从每个单词到 ID 的映射，我们需要跟踪这么多 ID。此外，像“dog”这样的词与“dogs”这样的词的表示方式不同，模型最初无法知道“dog”和“dogs”是相似的：它会将这两个词识别为不相关。

最后，我们需要一个自定义 token 来表示不在我们词汇表中的单词。这被称为“unknown” token，通常表示为“[UNK]”或“<unk>”。如果你看到 tokenizer 产生了很多这样的 token 这通常是一个不好的迹象，因为它无法检索到一个词的合理表示，并且你会在转化过程中丢失信息。制作词汇表时的其中一个目标是 tokenizer 将尽可能少的单词标记为未知 tokens。

减少未知 tokens 数量的一种方法是使用更深一层的 tokenizer 即基于字符（character-based）的 tokenizer

## 基于字符（Character-based）的 tokenization

将文本拆分为字符，而不是单词。

好处：
词汇量要小得多。
unknown tokens 要少得多，因为每个单词都可以由字符构建。

问题：关于空格和标点符号：
[图示](03-NLP-Basic\pictures\character_based_tokenization.svg)

问题：每个字符本身并没有多大意义，但是单词则不然。然而，这又因语言而异；例如，在中文中，每个字符比拉丁语言中的字符包含更多的信息。

问题：这样做会导致我们的模型需要处理大量的 tokens：虽然一个单词在基于单词的 tokenizer 中只是一个 token，但当它被转换为字符时，很可能就变成了 10 个或更多的 tokens

为了两全其美，我们可以使用结合这两种方法的第三种技术：基于子词（subword）的 tokenization。

## 基于子词（subword）的 tokenization

基于子词（subword）的 tokenization 算法依赖于这样一个原则：常用词不应被分解为更小的子词，但罕见词应被分解为有意义的子词。

例如，“annoyingly”可能被视为一个罕见的词，可以分解为“annoying”和“ly”。这两者都可能作为独立的子词并且出现得更频繁，同时“annoyingly”的含义通过“annoying”和“ly”的复合含义得以保留。

下图展示了基于子词的 tokenization 算法如何将序列“Let’s do tokenization!”分词：
[图示](03-NLP-Basic\pictures\bpe_subword.svg)
这些子词最终提供了大量的语义信息：例如，在上面的例子中，“tokenization”被分割成“token”和“ization”，这两个 tokens 在保持空间效率的同时具有语义意义（只需要两个 tokens 就能表示一个长词）。这让我们能够在词汇量小的情况下获得相对良好的覆盖率，并且几乎没有未知的 token。

这种方法在土耳其语等粘着型语言（agglutinative languages）中特别有用，可以通过将子词串在一起来形成（几乎）任意长的复杂词。

## 其他分词方法
Byte-level BPE，用于 GPT-2
WordPiece，用于 BERT
SentencePiece or Unigram，用于多个多语言模型


## 加载和保存tokenizer
它基于相同的两种方法： from_pretrained() 和 save_pretrained() 。这些方法会加载或保存分词器使用的算法（有点像模型的架构（architecture））以及其词汇表（有点像模型的权重（weights））。

加载使用与 BERT 相同的 checkpoint 训练的 BERT tokenizer 与加载模型的方式相同，只是换成了 Bert tokenizer 类：
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

如同 AutoModel ， AutoTokenizer 类将根据 checkpoint 名称在库中获取正确的 tokenizer 类，并且可以直接与任何 checkpoint 一起使用：
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

使用 tokenizer：
tokenizer("Using a Transformer network is simple")
=>
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

保存 tokenizer 与保存模型完全相同：
tokenizer.save_pretrained("directory_on_my_computer")


## 编码

将文本翻译成数字被称为编码（encoding）。编码分两步完成：分词，然后转换为 inputs ID。

正如我们所见，第一步是将文本拆分为单词（或部分单词、标点符号等），通常称为 tokens 不同的分词器使用的算法也不一样，这就是为什么我们需要使用模型名称来实例化 tokenizer，以确保我们使用模型预训练时使用的相同的算法。

第二步是将这些 tokens 转换为数字，这样我们就可以用它们构建一个张量并将它们提供给模型。为此，tokenizer 有一个词汇表（vocabulary），这是我们在使用 from_pretrained() 方法实例化它时下载的部分。同样，我们需要使用与预训练模型时相同的词汇表。

### tokenization
tokenization 过程由 tokenizer 的 tokenize() 方法实现：

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
这个方法的输出是一个字符串列表，或者说 tokens
=>
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
这个 tokenizer 是一个基于子词的 tokenizer：它对词进行拆分，直到获得可以用其词汇表表示的 tokens。以 transformer 为例，它分为两个 tokens 'Trans', '##former' 。
##表示该 token 是词中续接部分，不是独立词。

### 从 tokens 到 inputs ID
inputs ID 的转换由 tokenizer 的 convert_tokens_to_ids() 方法实现：

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
=>
[7993, 170, 13809, 23763, 2443, 1110, 3014]
这些输出，一旦转换为适当的框架张量，就可以用作模型的输入。

## 解码
解码（Decoding） 正好相反：从 inputs ID 到一个字符串。这以通过 decode() 方法实现：
decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoded_string)
=>
'Using a Transformer network is simple'
decode 方法不仅将索引转换回 tokens，还将属于相同单词的 tokens 组合在一起以生成可读的句子。

tokenizer 可以处理的原子操作：分词、转换为 ID 以及将 ID 转换回字符串。


# 处理多个序列
要解决的问题：
如何处理多个句子？
如何处理不同长度的多个句子？
词汇索引是唯一可以让模型运行的输入吗？
是否存在句子太长的问题？

## 模型需要一批输入
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)

model(input_ids)# 这一行会运行失败
=>
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

问题是我们向模型发送了一个单独的句子，而Transformers 模型默认情况下需要一个句子列表。在这里，当我们试图重现 tokenizer 在输入 sequence 后在其内部进行的所有操作。但如果你仔细观察，你会发现 tokenizer 不仅仅是将 inputs ID 的列表转换为张量，它还在其上添加了一个维度：

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
返回的 tensor 形状是 (1, seq_len)，不是 (seq_len,)。
多出来的那个维度是 batch 维度。

添加一个新的维度：
input_ids = torch.tensor([ids]) #变为了二维张量(1,14)

批处理（Batching）是一次性通过模型发送多个句子的行为。如果你只有一句话，你可以构建一个只有一个句子的 batch：
batched_ids = [ids, ids]
这就是一个包含两个相同句子的 batch

批处理支持模型在输入多个句子时工作。
但存在问题。当你试图将两个（或更多）句子组合在一起时，它们的长度可能不同。张量必须是矩形，因此无法将 inputs ID 列表直接转换为张量。为了解决这个问题，我们通常填充输入（Padding）。

## 填充输入（Padding）
以下列表不能转换为张量：
batched_ids = [
    [200, 200, 200],
    [200, 200]]
为了解决这个问题，我们将使用填充使张量成为标准的矩形。Padding 通过在值较少的句子中添加一个名为 padding_id 的特殊单词来确保我们所有的句子长度相同。
padding_id = 100
填充后=>
batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],]
我们可以在 tokenizer.pad_token_id 中找到填充 token 的 ID。

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
=>
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
批处理预测中的 logits 值有点问题：第二行应该与第二句的 logits 相同，但我们得到了完全不同的值！

这是因为 Transformer 模型的关键特性：注意力层，它考虑了每个 token 的上下文信息。具体来说，每个 token 的含义并非单独存在的，它的含义还取决于它在句子中的位置以及周围的其他 tokens。当我们使用填充（padding）来处理长度不同的句子时，我们会添加特殊的“填充 token”来使所有句子达到相同的长度。但是，注意力层会将这些填充 token 也纳入考虑，因为它们会关注序列中的所有 tokens。这就导致了：尽管填充 token 本身并没有实际的含义，但它们的存在会影响模型对句子的理解。

我们需要告诉这些注意层忽略填充 token。这是通过使用注意力掩码（attention mask）层来实现的。

## 注意力掩码（attention mask）层

注意力掩码（attention mask）是与 inputs ID 张量形状完全相同的张量，用 0 和 1 填充：1 表示应关注相应的 tokens，0 表示应忽略相应的 tokens

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
=>
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
现在我们得到了批处理中第二句话相同的 logits 值。


## 更长的句子
对于 Transformers 模型，我们可以通过模型的序列长度是有限的。大多数模型处理多达 512 或 1024 个 的 tokens 序列，当使用模型处理更长的序列时，会崩溃。此问题有两种解决方案：

使用支持更长序列长度的模型。
截断你的序列。
建议你通过设定 max_sequence_length 参数来截断序列：
sequence = sequence[:max_sequence_length]


# 直接使用tokenizer

## 使用多种不同的方法对目标进行填充：
将句子序列填充到最长句子的长度
model_inputs = tokenizer(sequences, padding="longest")

将句子序列填充到模型的最大长度
(512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

将句子序列填充到指定的最大长度
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

## 对序列进行截断：
将截断比模型最大长度长的句子序列
(512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

将截断长于指定最大长度的句子序列
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

## 返回不同框架的张量 
"pt" 返回 PyTorch 张量， "tf" 返回 TensorFlow 张量，而 "np" 则返回 NumPy 数组：

返回 PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

返回 TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

返回 NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")


## 特殊的 tokens 开头和结尾

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
=>
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
=>
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

句子的开始和结束分别增加了一个 inputs ID。

print(tokenizer.decode(model_inputs["input_ids"]))
=>
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
print(tokenizer.decode(ids))
=>
"i've been waiting for a huggingface course my whole life."

tokenizer 在开头添加了特殊单词 [CLS] ，在结尾添加了特殊单词 [SEP] 。这是因为模型在预训练时使用了这些字词，所以为了得到相同的推断结果，我们也需要添加它们。

有些模型不添加特殊单词，或者添加不同的特殊单词；模型也可能只在开头或结尾添加这些特殊单词。

问答：
Transformer 模型的输出有的张量多少个维度，每个维度分别是什么？
 3 个维度，分别是：序列长度(Sequence Length)、批次大小(Batch Size)和隐藏层大小(Hidden Size)

下列哪一个是子词分词的例子（从分词的颗粒度来划分）？
 WordPiece
 BPE(Byte Pair Encoding)
 Unigram

什么是 AutoModel？
 一个根据 checkpoint(检查点)返回模型体系结构的对象

使用 SoftMax 激活函数对序列分类(Sequence Classification)模型的 logits 输出进行处理有什么意义？
 它限定了上下界，使模型的输出结果可以被解释。
 输出的和是 1，从而产生一个可能的概率解释。

下面的代码有什么错误吗？
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")
encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
 Tokenizer 和模型应该来自相同的 checkpoint。

---

## 今日知识总结

### 三种分词粒度各有取舍

基于单词的分词最直观，按空格和标点切分即可，但词表巨大——英语有超过 50 万个单词，每个词都需要独立 ID，且 "dog" 和 "dogs" 被当作无关词汇，无法共享语义。未知词问题严重，一旦遇到词表外的词就只能标记为 `[UNK]`。基于字符的分词词表极小（几十个字符），几乎不会出现未知词，但单个字符语义信息很少，且一个单词会被切成十几个 token，序列变长、计算量增大。基于子词的分词是折中方案：常用词保持完整不拆分，罕见词拆成有意义的子词片段，如 "annoyingly" → "annoying" + "ly"，"transformer" → "trans" + "##former"。这样词表小、未知词少、语义信息保留充分。BERT 使用的 WordPiece 和 GPT-2 使用的 BPE 都属于子词分词算法。

### Tokenizer 的原子操作：分词、编码、解码

Tokenizer 提供三个基础操作。`tokenize()` 把文本拆成 token 字符串列表，例如 "Using a Transformer network is simple" 被拆成 `['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']`，其中 `##` 前缀表示该 token 是前一个词的续接部分，不是独立词。`convert_tokens_to_ids()` 把 token 字符串映射为词表中对应的整数 ID。反过来，`decode()` 把 ID 序列还原为可读文本，它会自动把带 `##` 的子词片段合并回完整单词。这三个操作构成了 tokenizer 最底层的原子能力，`tokenizer(text)` 一键调用不过是它们的封装。

### 模型必须接收批处理输入

Transformer 模型默认要求输入形状为 `(batch_size, sequence_length)`，即使只有一句话也需要 batch 维度。手动用 `torch.tensor(ids)` 得到的形状是 `(seq_len,)`，缺少 batch 维度，直接传入模型会报错。解决办法是在外层再套一层列表 `torch.tensor([ids])`，形状变为 `(1, seq_len)`，或使用 `unsqueeze(0)` 在第 0 维添加一个维度。直接调用 `tokenizer(sequence, return_tensors="pt")` 则一步到位：分词、转 ID、添加 batch 维度和 attention_mask 全部完成，返回的 `input_ids` 形状直接是 `(1, seq_len)`。

### 批处理需要 Padding 和 Attention Mask

多句话组成 batch 时，PyTorch 要求张量是矩形，但不同句子长度往往不同。Padding 通过向短句末尾添加 `pad_token_id`（通常为 0）使所有句子长度对齐。然而仅做 padding 还不够——Transformer 的自注意力会关注序列中的所有 token，包括 padding token，导致 padding 的句子 logits 发生变化（与单独输入时的结果不同）。这是因为注意力层把 padding 位置也纳入了上下文计算。解决办法是使用 attention mask：形状与 input_ids 完全相同，真实 token 位置标记为 1，padding 位置标记为 0。模型在计算注意力时会将 mask 为 0 的位置的注意力分数置为极小值，softmax 后这些位置的权重趋近于 0，从而忽略 padding 的影响。加上正确的 attention mask 后，padding 句子的 logits 与单独输入时一致。

### 特殊 Token 是模型预训练的遗产

直接调用 `tokenizer(sequence)` 返回的 `input_ids` 比手动 `tokenize` + `convert_tokens_to_ids` 多了两个 ID：开头是 `[CLS]` 对应的 101，结尾是 `[SEP]` 对应的 102。这是因为 BERT 在预训练时使用了这些特殊 token——`[CLS]` 用于汇总整句信息做分类，`[SEP]` 用于分隔两个句子。推理时必须同样添加，否则输入分布与训练时不一致，模型表现会下降。不同模型使用的特殊 token 不同，有些模型只加开头或只加结尾，因此 tokenizer 和 model 必须来自同一个 checkpoint，否则特殊 token 的语义完全对不上。

### Tokenizer 和 Model 必须匹配

Tokenizer 的词汇表、特殊 token 定义、分词算法都是在模型预训练时确定的。如果用 BERT 的 tokenizer 去切词然后喂给 GPT-2 的模型，token ID 的语义完全错乱——BERT 中 ID 1045 可能表示 "i"，GPT-2 中同一个 ID 可能表示完全不同的词。此外特殊 token 也不一致，BERT 有 `[CLS]` 和 `[SEP]`，GPT-2 没有。因此 `AutoTokenizer.from_pretrained()` 和 `AutoModel.from_pretrained()` 必须使用相同的 checkpoint 名称。