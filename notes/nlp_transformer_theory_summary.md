# NLP 与 Transformer 完整理论知识总结

> 本文只讲知识本身，不涉及代码、框架接口、环境配置或工程命令。内容从自然语言处理的基本问题出发，完整解释文本如何进入 Transformer、信息如何在模型中流动、模型如何学习，以及不同 NLP 任务如何建立在同一架构之上。

## 1. NLP 到底是什么

### 1.1 自然语言处理

**自然语言处理（Natural Language Processing，NLP）**是研究如何让计算机处理、理解和生成人类语言的领域。这里的“自然语言”是人类日常形成的语言，如中文、英语和法语，用来区别按照严格规则设计的编程语言。

NLP 不等于“让机器真正像人一样理解语言”。在当前主流方法中，模型主要通过大量数据学习语言形式、语义关系和任务规律，再根据输入计算最可能的输出。模型能表现出很强的语言能力，但这种能力受到训练数据、模型结构、上下文长度和任务目标的限制。

NLP 的核心问题可以概括为四类：

1. **表示**：怎样把文字变成可计算的数字。
2. **理解**：怎样判断文本表达的类别、实体、关系、意图和语义。
3. **生成**：怎样根据已有文本产生后续文本或新的目标文本。
4. **检索与应用**：怎样从大量文本中找到相关知识，并将模型能力组合成可使用的系统。

**自然语言理解（Natural Language Understanding，NLU）**是 NLP 中偏向识别意义、意图、实体和关系的一组任务。

**自然语言生成（Natural Language Generation，NLG）**是 NLP 中根据条件产生连贯语言的一组任务。

NLU 与 NLG 是方便理解的任务分区，不是绝对分开的技术体系。生成高质量文本也需要理解输入，理解模型也通过语言建模目标学习表示。

### 1.2 语言的多个层次

自然语言不是简单的字符排列。一个句子同时包含多个层次的信息：

- **字符（Character）**：书写系统中的基本符号，如汉字、字母和标点。
- **词（Word）**：具有相对独立意义或语法功能的语言单位。
- **词法（Morphology）**：研究词的内部构成，如词根、前缀、后缀和词形变化。
- **句法（Syntax）**：研究词怎样按照语法规则组成短语和句子。
- **语义（Semantics）**：研究语言表达的字面意义以及词句之间的意义关系。
- **语用（Pragmatics）**：研究语言在具体情境中的实际含义，包括说话者目的、常识和上下文暗示。
- **篇章（Discourse）**：研究多个句子如何组成连贯文本，包括指代、主题延续和逻辑关系。

例如“苹果发布了新系统”中的“苹果”不是水果。判断它指公司，需要结合“发布”“系统”等上下文，这属于语义消歧；若前文已经出现“这家公司”，还涉及篇章指代。

### 1.3 NLP 的主要任务

**文本分类（Text Classification）**为整段文本预测一个或多个类别，例如情感正负、新闻主题和用户意图。

**序列标注（Sequence Labeling）**为序列中的每个词或 Token 预测标签。Token 是模型实际处理的文本单位，后文第 3 节会详细解释。典型任务包括命名实体识别（NER——Named Entity Recognition，识别人名、地名等实体）、词性标注（POS——Part-of-Speech Tagging，判断名词、动词等语法类别）和句法分块（Chunking，识别名词短语等局部结构）。

**信息抽取（Information Extraction，IE）**从非结构化文本中识别实体、关系、事件等结构化信息。NER 是信息抽取的基础任务之一。

**非结构化文本（Unstructured Text）**没有固定表格字段和严格模式；**结构化信息（Structured Information）**按明确字段、类别和关系组织，便于数据库存储和计算。

**机器翻译（Machine Translation）**把一种语言的文本转换为另一种语言，同时尽量保持语义、语气和术语一致。

**文本摘要（Text Summarization）**把长文本压缩成较短文本，保留重要信息并减少冗余。

**问答（Question Answering，QA）**根据问题和已有知识给出答案。答案可以从原文抽取，也可以由模型生成。

**语言建模（Language Modeling）**学习语言序列的概率规律，例如根据前文预测下一个 Token，或根据双向上下文恢复被遮盖的 Token。

**嵌入（Embedding）**的上位含义是把离散对象映射为连续数值向量。**文本嵌入（Text Embedding）**把文本表示为固定长度向量；向量是有顺序的一组数值，向量空间是这些向量所在的多维坐标空间。理想情况下，语义接近的文本在选定距离规则下也更接近。

**检索（Retrieval）**是从候选集合中找出与查询相关的内容。**语义搜索（Semantic Search）**根据含义而不是只根据相同关键词检索文本。

**知识库（Knowledge Base）**是供系统查询的一组外部事实、文档或结构化记录。**检索增强生成（Retrieval-Augmented Generation，RAG）**先从知识库检索相关内容，再让生成模型依据检索结果回答问题。

### 1.4 AI、机器学习、深度学习与模型

**人工智能（Artificial Intelligence，AI）**是让机器表现出感知、推理、学习、决策或生成等智能行为的广泛领域。

**机器学习（Machine Learning，ML）**是 AI 的一类方法：不把所有规则直接写死，而是让系统从数据中学习规律。

**深度学习（Deep Learning，DL）**是以多层神经网络为核心的机器学习方法。这里的“深度”主要指网络由多层可学习变换组成。

**模型（Model）**是从数据中学习出的数学映射。它接收输入，依据内部参数计算输出。模型架构规定计算结构，模型参数记录训练得到的具体数值，两者共同决定模型行为。

**训练（Training）**是利用样本和训练目标反复调整参数的过程；**训练目标（Training Objective）**规定希望模型学会的预测关系；**推理（Inference）**是参数固定后用模型处理新输入。

**模型对齐（Model Alignment）**是让模型行为更符合人类意图、价值约束和使用要求的过程。这里的“对齐”不同于词标签与子词位置对齐，也不同于翻译中的句子对应。

NLP 是应用问题领域；机器学习和深度学习是解决这些问题的方法；Transformer 是一种深度学习模型架构；BERT、GPT 和 T5 是基于该架构形成的模型家族。

## 2. NLP 方法的发展脉络

### 2.1 规则方法

**规则方法（Rule-based Method）**由人手工编写词典、模板和语法规则。例如规定出现某些词就判断为负面情感。

它的优点是行为清楚、容易解释；缺点是覆盖范围有限，维护成本高，对语言歧义、变体和新表达适应性差。

### 2.2 传统统计方法

**词袋模型（Bag of Words，BoW）**只统计文本中各词是否出现或出现多少次，不保留词序。它把文本转成高维稀疏向量，适合简单分类，但无法表达“人咬狗”和“狗咬人”的顺序差异。

**N-gram**是由连续 N 个语言单位组成的片段。例如二元组关注相邻两个词，三元组关注连续三个词。N-gram 能保留局部顺序，但 N 越大，可能组合越多，数据稀疏越严重。

**词频—逆文档频率（Term Frequency–Inverse Document Frequency，TF-IDF）**同时考虑一个词在当前文档中的频率和它在全部文档中的稀有程度。高频但到处出现的常用词权重较低；在某篇文档中频繁出现、在其他文档中少见的词权重较高。

**稀疏向量（Sparse Vector）**是大部分维度为零的向量。传统词袋和 TF-IDF 的每个维度通常对应一个词，因此维度很高但单篇文本只激活少量维度。

传统统计方法训练快、可解释性较强，是重要基线；但它们通常难以表达上下文、多义词和长距离依赖。

### 2.3 神经网络方法

**神经网络（Neural Network）**由多层可学习的数学变换组成。它不依赖人工设计全部规则，而是通过数据和损失函数自动调整参数。

**词嵌入（Word Embedding）**把离散词映射为低维稠密向量。稠密向量的大部分维度都有非零值，能够通过距离表达一定语义关系。

**循环神经网络（Recurrent Neural Network，RNN）**按顺序读取 Token，并把前面信息压缩进隐藏状态。这里的隐藏状态是随序列步骤更新、用于携带历史信息的内部向量。RNN 适合序列，但难以并行，长序列中的早期信息容易逐渐衰减。

**长短期记忆网络（Long Short-Term Memory，LSTM）**是改进的 RNN。**门控机制（Gating Mechanism）**用取值接近零到一的控制量决定多少信息应保留、写入或遗忘，从而缓解长期依赖问题，但 LSTM 仍受串行计算限制。

**卷积神经网络（Convolutional Neural Network，CNN）**用可学习的小型过滤器在序列上滑动，反复检测局部模式。**局部窗口（Local Window）**是每次计算直接覆盖的一小段相邻位置。CNN 在文本分类中能捕捉局部短语，但直接建模任意远位置之间的关系需要堆叠很多层。

### 2.4 预训练模型与大语言模型

**预训练（Pretraining）**是在大规模通用语料上先学习语言规律。预训练阶段不针对单一业务任务，而是学习可迁移的通用表示和生成能力。

**微调（Fine-tuning）**是在预训练权重基础上，用较小规模的任务或领域数据继续训练，使模型适应具体目标。

**迁移学习（Transfer Learning）**是把一个阶段学到的能力迁移到另一个任务。预训练加微调就是 NLP 中最典型的迁移学习。

**大语言模型（Large Language Model，LLM）**通常指参数规模、训练数据和能力都较大的语言模型。它们大多基于 Transformer，通过下一个 Token 预测或相关目标学习语言规律。

“大”不是严格的参数阈值。LLM 的关键不只是参数数量，还包括数据规模、训练目标、上下文能力、对齐方法和推理方式。

## 3. 文本怎样变成模型输入

### 3.1 语料、样本与序列

**语料（Corpus）**是用于训练、验证或研究的一组语言材料。

**文档（Document）**是一段相对完整的文本单位，例如一篇文章、一条评论或一个网页。

**样本（Example 或 Sample）**是一次训练或评估处理的基本数据单位。一个样本可能是一句话、一对句子、一个问题和上下文，或一篇文章与摘要。

**序列（Sequence）**是按顺序排列的符号集合。在 NLP 中通常指 Token 序列。

**上下文（Context）**是模型在当前预测时可利用的信息。它可以指一句话的其他部分，也可以指提示词、历史对话或检索到的文档。

### 3.2 Token 与 Tokenizer

**Token**是模型实际处理的离散文本单位。它可能是一个字、一个完整单词、一个子词、一个标点，甚至一个字节片段。

**分词（Tokenization）**是把原始文本切分为 Token 的过程。

**Tokenizer**是执行完整文本编码和解码流程的组件。它不仅切分文本，还负责规范化、词表查找、特殊 Token 添加以及 Token 与字符位置的对应。

Token 不等于词。例如一个罕见英文单词可能被切成多个子词；中文常见词也可能由单字或更小片段组成。模型看到的是 Token 序列，不是人类预先认定的词序列。

### 3.3 三种基本分词粒度

**基于词的分词（Word-based Tokenization）**把完整词作为单位。它直观、序列较短，但词表非常大，罕见词和新词容易变成未知词。

**基于字符的分词（Character-based Tokenization）**把字符作为单位。它词表小、未知字符少，但序列更长，单个字符携带的语义通常较弱。

**基于子词的分词（Subword Tokenization）**让常见词保持完整，把罕见词拆成较小片段。它在词表大小、序列长度和未知词之间取得平衡，是现代 Transformer 的主流方案。

### 3.4 BPE、WordPiece、Unigram 与 SentencePiece

**字节对编码（Byte Pair Encoding，BPE）**从较小单位开始，反复合并语料中最常见的相邻单位。合并规则记录了从字符或字节逐渐组成常见子词的过程。

**WordPiece**也从小单位逐步合并，但选择标准更关注合并后对语料概率的改善，而不只看相邻对的绝对频率。BERT 系列常使用 WordPiece。

**Unigram 语言模型分词**从一个较大的候选词表开始，为每个子词估计概率，再逐步删除对整体损失影响较小的子词。对一个单词分词时，会寻找总体概率最高的子词组合。

**动态规划（Dynamic Programming）**把大问题拆成相互重叠的子问题，保存每个子问题的最优结果，避免重复计算。

**Viterbi 算法**是一种动态规划方法。这里的“路径”是一连串分词选择，“累计得分”是沿这串选择相加的代价或对数概率。Viterbi 在每个字符位置保存到达该位置的最佳结果，最后从终点回溯出总体最优子词切分。

**SentencePiece**是一套直接把原始文本当作字符流处理的分词方法框架。它不要求先按空格切词，常用特殊符号表示空格边界，因此适合中文、日文等不依赖空格分词的语言，也可以承载 BPE 或 Unigram 模型。

### 3.5 词表、Token ID 与未知词

**词表（Vocabulary）**是 Tokenizer 能识别的全部 Token 集合。

**Token ID**是 Token 在词表中的整数编号。神经网络不能直接处理字符串，因此先用 ID 查找对应向量。

**未登录词（Out-of-Vocabulary，OOV）**是词表中没有的语言单位。基于词的分词常把它整体映射为未知 Token；子词或字节级方法可把它拆开，从而显著减少 OOV。

Token ID 本身没有自然语义。同一个数字在不同词表中可以代表完全不同的 Token。因此模型与 Tokenizer 必须共享同一套词表和特殊 Token 定义。

### 3.6 文本规范化与预分词

**规范化（Normalization）**是在正式切分前统一文本形式，例如大小写转换、Unicode 形式统一、去除重音或整理空白。

**Unicode**是为世界文字分配统一编码的标准。看起来相同的字符可能有不同底层组合，规范化可以减少这种差异。

**预分词（Pre-tokenization）**先按空格、标点或其他边界把文本分成初步片段，之后再由子词算法进一步切分。

规范化会影响信息保留。例如全部转小写可减少词表规模，却会丢失专有名词和大小写差异。不存在对所有任务都最好的统一规则。

### 3.7 特殊 Token

**特殊 Token（Special Token）**不是普通文本内容，而是为模型提供结构信息的 Token。

- **CLS Token（Classification Token）**：常放在序列开头。BERT 类模型可利用其最终隐藏状态表示整段输入，用于分类。
- **SEP Token（Separator Token）**：表示序列结束或两个句子的分隔。
- **PAD Token（Padding Token）**：用于把不同长度序列补到相同长度。
- **UNK Token（Unknown Token）**：表示无法由词表表达的未知内容。
- **MASK Token（Mask Token）**：在掩码语言建模中替代被遮盖的 Token。
- **BOS Token（Beginning of Sequence）**：表示目标序列开始。
- **EOS Token（End of Sequence）**：表示序列结束，生成模型学会生成它后可停止。

不同模型对特殊 Token 的定义和使用方式不同，不能只凭名称假设它们完全等价。

### 3.8 Batch、Padding 与 Truncation

**批次（Batch）**是一次并行送入模型的多个样本。批处理能更充分利用 GPU 的矩阵计算能力。

**批次大小（Batch Size）**是一个 Batch 包含的样本数。它影响显存占用、梯度稳定性和每次参数更新所依据的数据量。

**图形处理器（Graphics Processing Unit，GPU）**是擅长大规模并行数值运算的硬件。神经网络训练包含大量矩阵运算，因此 GPU 通常比通用中央处理器更高效。GPU 的高速工作内存称为**显存（Video Memory，VRAM）**，模型参数、隐藏状态、梯度和 Batch 都会占用显存。

同一 Batch 中的张量需要规则形状，但文本长度通常不同，因此要使用 Padding 或 Truncation。

**填充（Padding）**是在短序列末尾或开头补 PAD Token，使同一 Batch 中序列长度一致。

**截断（Truncation）**是删除超出最大长度的 Token。它能控制计算量，但可能丢失关键信息。

**动态填充（Dynamic Padding）**只填充到当前 Batch 的最长序列，而不是整个数据集的最大长度，通常能减少无效计算。

### 3.9 Attention Mask 与 Token Type ID

**注意力掩码（Attention Mask）**告诉模型哪些位置有效、哪些位置应被忽略。常见约定是有效 Token 为 1，Padding 为 0。被忽略位置在注意力分数中会被赋予极小值；Softmax 会把一组分数归一化为总和为一的权重，因此这些位置的权重接近零。

Attention Mask 不负责遮盖 MLM 的答案，也不等于 Decoder 的因果掩码。MLM 是恢复被随机遮盖 Token 的训练目标；Decoder 是负责按可见前文生成后续内容的结构；因果掩码是禁止它读取未来位置的规则。Attention Mask 主要描述当前输入中哪些位置属于有效内容。

**Token Type ID**也称句段 ID，用来标记 Token 属于句子 A 还是句子 B。BERT 的句子对输入常使用它；不是所有 Transformer 都有这一输入。

### 3.10 字符、词与 Token 的坐标映射

**偏移映射（Offset Mapping）**记录每个 Token 对应原始文本中的字符起止位置。

**词 ID 映射（Word ID Mapping）**记录每个子词 Token 属于原始的哪个词。

这些映射解决不同坐标系之间的对齐问题：NER 的标签往往按词标注，但模型按 Token 预测；抽取式问答的答案往往按字符位置标注，但模型预测 Token 起止位置。

## 4. 理解神经网络所需的基础术语

### 4.1 标量、向量、矩阵与张量

**标量（Scalar）**是单个数值。

**向量（Vector）**是一维有序数值集合，可表示一个 Token、句子或样本的特征。

**特征（Feature）**是模型用于描述输入的可计算属性。在深度学习中，许多特征不是人工命名的，而是由隐藏向量的多个维度共同学习出来。

**矩阵（Matrix）**是二维数值数组，可表示一组向量或线性变换。

**张量（Tensor）**是更一般的多维数值数组。一个文本 Batch 常表示为“批次数 × 序列长度”的 ID 张量；隐藏状态常表示为“批次数 × 序列长度 ×隐藏维度”的三维张量。

**形状（Shape）**描述张量每个维度的大小。

**维度（Dimension）**既可能指张量的轴数，也可能指向量包含多少个数。阅读上下文时需要区分。

**矩阵乘法（Matrix Multiplication）**按照行与列的对应乘积求和，把一组向量映射到新的表示空间，是线性层和注意力计算的核心操作。

**转置（Transpose）**交换矩阵的行与列，常用上标 T 表示。注意力中的 Kᵀ 使所有 Query 能同时与所有 Key 做点积。

**线性变换（Linear Transformation）**用矩阵乘法把输入向量投影到新的坐标空间。加入偏置后严格说是仿射变换，但神经网络中通常仍简称线性层。

**点积（Dot Product）**把两个等长向量的对应元素相乘后求和，其数值同时受向量方向和长度影响，也是注意力相关性分数的基础。只有先把向量归一化，点积才只反映方向并等同于余弦相似度。

### 4.2 参数、超参数与权重

**参数（Parameter）**是模型从数据中学习的数值，如线性层权重和偏置。

**权重（Weight）**通常指参数中的乘法系数。广义讨论中，“模型权重”常泛指全部可学习参数。

**偏置（Bias）**是线性变换中额外加上的可学习常数，使输出不必经过原点。

“Bias”还可以表示**数据或模型偏见**，即训练数据中的不平衡、刻板印象或系统性误差被模型学习并反映在输出中。它与线性层中的偏置参数同名，但含义完全不同。

**超参数（Hyperparameter）**是训练前由人设定的配置，如学习率、层数、隐藏维度、批次大小和训练轮数。它们不是模型通过普通反向传播直接学习的参数。

### 4.3 线性层、激活函数与深度

**线性层（Linear Layer）**对输入执行矩阵乘法并加偏置，用于改变表示维度或产生任务分数。

**激活函数（Activation Function）**为网络加入非线性。若所有层都只是线性变换，无论叠多少层都等价于一层线性变换，难以学习复杂关系。

**修正线性单元（Rectified Linear Unit，ReLU）**把负数变为零、正数保持不变，是经典激活函数。

**GELU（Gaussian Error Linear Unit）**以更平滑的方式控制输入通过程度，常用于 BERT 和许多 Transformer。

**网络深度（Depth）**通常指模型堆叠的层数。更深的层可以逐步形成更抽象的表示，但训练和计算成本也更高。

### 4.4 Hidden State、Logits 与概率

**隐藏状态（Hidden State）**是模型内部对输入的中间向量表示。它之所以叫“隐藏”，是因为它不是最终任务答案，而是后续层使用的内部状态。

Transformer 的隐藏状态是上下文化的：同一个 Token 在不同句子中会得到不同向量，因为它融合了周围信息。

**隐藏维度（Hidden Size）**是每个 Token 隐藏向量的长度。

**Logits**是模型输出的未经归一化的原始分数。Logits 可以为负数，不要求位于零到一之间，也不要求总和为一。

**Softmax**把一组 Logits 转成总和为一的概率分布。某个 Logit 相对其他 Logit 越大，其概率通常越高。

**概率分布（Probability Distribution）**描述一组可能结果各自被赋予的概率。在离散分类中，每项概率通常非负，全部结果概率之和为一。

**Argmax**是选出最大值所在位置的操作。分类任务常用它选出分数最高的类别。

### 4.5 标签、目标、损失与优化

**标签（Label）**是训练时希望模型预测的正确答案。它可能是类别、每个 Token 的实体标签、目标文本 Token，或答案起止位置。

**训练目标（Objective）**描述模型究竟被要求学会什么，例如恢复被遮盖 Token 或预测下一个 Token。

**损失函数（Loss Function）**把模型预测与正确答案之间的差异压缩成一个可优化数值。

**交叉熵（Cross Entropy）**是分类和语言模型常用损失。正确类别的预测概率越低，损失越大。

**梯度（Gradient）**是各个参数偏导数组成的向量。以 L(x, y) = 2x + y 为例，∂L/∂x = 2（x 方向变化率），∂L/∂y = 1（y 方向变化率），梯度 = (2, 1)。

梯度为何是最速上升方向？移动方向与梯度的内积 = 实际上升量。比较沿不同方向走单位长度一步：沿 x 走 → 上升 2，沿 y 走 → 上升 1，沿梯度方向走 → 内积最大 → 上升约 2.24。因为内积 = |A|·|B|·cos(夹角)，两方向一致时 cos=1，内积最大。所以梯度向量天然指向损失增长最快的方向。

要降低损失，就沿反方向走，每一步下降最快：`w_new = w - 学习率 × 梯度`，减号就是取反方向。反复迭代走到损失最低点，即为梯度下降。

**反向传播（Backpropagation）**利用链式法则从损失向前面各层计算梯度。

**链式法则（Chain Rule）**说明复合函数的导数可以由各层局部导数相乘得到。神经网络由许多连续变换组成，反向传播正是用它把最终损失的影响逐层传回参数。

**优化器（Optimizer）**根据梯度更新参数。**Adam（Adaptive Moment Estimation）**维护近期梯度的移动平均和梯度平方的移动平均；前者反映大致更新方向，后者反映梯度尺度，再据此自适应调整不同参数的更新幅度。**AdamW**把权重衰减与梯度更新更合理地分离，是 Transformer 微调中的常见优化器。

**学习率（Learning Rate）**控制每次更新步长。过大可能震荡或发散，过小则收敛缓慢。

**权重衰减（Weight Decay）**通过抑制参数无限增大来帮助正则化。

**学习率调度（Learning Rate Schedule）**让学习率随训练过程变化，例如先预热再逐步衰减。

**预热（Warmup）**是在训练初期从较小学习率逐步增加，降低开始阶段更新不稳定的风险。

### 4.6 Epoch、Step 与梯度累积

**训练轮次（Epoch）**表示模型完整遍历一次训练集。

**训练步（Step）**通常指处理一个 Batch 并完成一次或一次累计后的参数更新。

**梯度累积（Gradient Accumulation）**先处理多个小 Batch 并累积梯度，再更新一次参数，用较小显存近似更大的有效 Batch。

### 4.7 训练、验证、测试与泛化

**训练集（Training Set）**用于更新模型参数。

**验证集（Validation Set）**用于选择超参数、观察训练状态和比较方案，不直接参与参数更新。

**测试集（Test Set）**用于最终评估，原则上不应反复用于调参。

**泛化（Generalization）**是模型在未见数据上仍能正确工作的能力。

**过拟合（Overfitting）**是模型过度记住训练数据，训练表现很好但验证表现变差。

**欠拟合（Underfitting）**是模型连训练数据中的主要规律都没有学好，训练和验证表现都较差。

**收敛（Convergence）**是训练过程中损失和参数变化逐渐稳定。

**数据泄漏（Data Leakage）**是验证或测试信息不恰当地进入训练过程，导致评估虚高。

**正则化（Regularization）**是一类抑制过拟合的方法，包括权重衰减、Dropout、数据增强和提前停止。

**数据增强（Data Augmentation）**通过保持标签语义基本不变的变换产生更多训练样本，如同义改写或随机遮盖。

**提前停止（Early Stopping）**在验证指标不再改善时停止训练，避免继续拟合训练数据噪声。

### 4.8 混合精度与并行训练

**数值精度（Numerical Precision）**描述数值用多少位表示。位数越少通常越节省显存、运算越快，但可表示范围和精确程度也会降低。

**混合精度训练（Mixed-Precision Training）**同时使用较低精度和较高精度计算，在尽量保持训练稳定的同时提高速度、降低显存占用。

**分布式训练（Distributed Training）**使用多个计算设备共同训练模型。

**数据并行（Data Parallelism）**让每个设备保存完整模型、处理不同数据子集，再同步梯度。

**模型并行（Model Parallelism）**把模型的不同部分分布到不同设备，适合单个设备放不下完整模型的情况。

数据并行切分数据，模型并行切分模型；二者可以组合使用。

## 5. Transformer 为什么出现

RNN 和 LSTM 按序列顺序处理信息。位置较后的计算依赖位置较前的结果，因此训练难以在序列维度完全并行。长文本还要求把大量历史信息逐步压缩进有限状态，早期信息容易衰减。

Transformer 的核心改变是：**不再依赖循环结构传递信息，而是让每个位置直接通过注意力读取其他相关位置。**

它带来三个重要结果：

1. 任意两个位置之间的信息路径更短。
2. 训练时可对序列中的多个位置并行计算。
3. 每个位置能根据当前内容动态决定关注谁，而不是使用固定窗口。

Transformer 仍有代价。标准全注意力需要构造长度乘长度的注意力矩阵，序列越长，计算量和显存增长越快。

## 6. Transformer 的完整架构总览

Transformer 不是单独一个注意力层，而是一套完整的表示、信息交换、非线性变换和稳定训练结构。

完整数据流可以概括为：

原始文本 → Token 序列 → Token ID → Token Embedding 与位置表示 → 多层 Transformer Block → 上下文化隐藏状态 → 任务 Head → Logits → 任务结果

**Transformer Block**是重复堆叠的基本层。**注意力头（Attention Head）**是一次独立的注意力关系计算；**前馈神经网络（Feed-Forward Network，FFN）**是逐位置进行非线性变换的子层；**任务头（Task Head）**则把最终隐藏状态变成特定任务的输出。Encoder Block 通常包含自注意力和 FFN；Decoder Block 通常包含带因果掩码的自注意力、可选的交叉注意力和 FFN。

### 6.1 三类架构的数据流不是同一条线

Encoder-only 只有一条输入路径：输入序列经过 Token 与位置表示、多层 Encoder，再按任务进行整段池化、逐 Token 分类或答案跨度预测。

Decoder-only 也只有一条序列路径，但使用因果约束：位置 i 的输入表示经过多层 Decoder，得到的 LM Logits 用来预测位置 i+1 的 Token。

Encoder-Decoder 有两条输入路径：

1. 源序列经过自己的 Token Embedding 与位置表示，进入 Encoder，形成源序列隐藏状态。
2. 目标侧输入也经过 Token Embedding 与位置表示，进入 Decoder。训练时它是按目标位置错开的正确目标；推理时它是起始 ID 与模型已经生成的 Token 前缀。
3. Decoder 先对目标前文做因果自注意力，再以当前 Decoder 状态为 Query、以 Encoder 输出为 Key 和 Value 做交叉注意力。
4. Decoder 最终隐藏状态通过 LM Head 产生目标词表 Logits。

因此，Encoder-Decoder 不是“Encoder 输出直接变成文字”，而是 Decoder 在源序列条件下逐步生成目标序列。

## 7. 输入表示层

### 7.1 Token Embedding

**Token Embedding**把每个 Token ID 映射为一个稠密向量。Embedding 矩阵的每一行对应一个 Token 的可学习表示。

初始 Token Embedding 只由 Token ID 决定，同一个 Token 在任何句子中的初始向量相同。经过多层 Transformer 后，它才变成结合上下文的隐藏状态。

### 7.2 Position Encoding 与 Position Embedding

注意力本身若不加入位置信息，只知道有哪些 Token，不知道它们的先后顺序。因此必须引入位置表示。

**位置编码（Positional Encoding）**泛指把顺序信息加入模型的方法。

**位置嵌入（Position Embedding）**通常指每个位置对应一个可学习向量，再与 Token Embedding 相加。

**正弦位置编码（Sinusoidal Positional Encoding）**使用不同频率的正弦和余弦函数生成固定位置向量，不需要学习。

**绝对位置（Absolute Position）**直接表示“这是第几个位置”。

**相对位置（Relative Position）**更关注两个 Token 相距多远、方向如何。

**旋转位置编码（Rotary Position Embedding，RoPE）**通过旋转 Query 和 Key 的向量分量，把相对位置信息融入注意力计算，常用于现代 Decoder-only 模型。

位置方法会影响模型对长文本和超出训练长度位置的适应能力。

### 7.3 输入表示的相加

经典 Transformer 通常把 Token Embedding 与位置表示相加。BERT 还可能加入句段嵌入，用来区分句子 A 与句子 B。

相加后的每个位置既包含“是什么 Token”，也包含“位于哪里”和“属于哪个句段”等信息，然后进入第一层 Transformer Block。

## 8. 注意力机制：Transformer 的核心

### 8.1 注意力的直观含义

**注意力（Attention）**让模型在处理当前位置时，为其他位置分配不同权重，并把重要位置的信息加权汇总。

例如理解“他”时，模型可能对前文的人名分配较高权重。注意力不是简单地“寻找最近的词”，而是根据当前层学到的表示动态计算相关性。

注意力权重描述当前计算如何分配信息，不一定等于人类可解释的“原因”，也不能单凭一张注意力图证明模型作出预测的因果依据。

### 8.2 Query、Key 与 Value

在自注意力中，同一序列的每个输入向量会经过三个不同的线性变换，得到：

- **Query（查询，Q）**：当前 Token 想寻找什么信息。
- **Key（键，K）**：当前 Token 可以用什么特征被其他位置匹配。
- **Value（值，V）**：如果当前 Token 被关注，实际提供什么内容。

可以把它类比为检索：Query 是搜索条件，Key 是索引描述，Value 是最终取回的信息。但 Q、K、V 都是模型学习出来的连续向量，不是人工设定的关键词。

### 8.3 Scaled Dot-Product Attention

**点积（Dot Product）**把两个等长向量对应元素相乘后求和，可作为匹配分数，但同时受方向和向量长度影响。

注意力先计算 Query 与所有 Key 的点积，再除以 Key 维度平方根：

Attention(Q, K, V) = Softmax(QKᵀ / √dₖ) V

其中：

- QKᵀ 得到每个 Query 对每个 Key 的原始相关性分数。
- dₖ 是 Key 向量维度。
- 除以 √dₖ 称为缩放，可防止维度较大时点积绝对值过大，使 Softmax 过早饱和、梯度过小。
- Softmax 把分数转成权重。
- 权重与 V 加权求和，得到该 Query 聚合后的上下文表示。

这称为**缩放点积注意力（Scaled Dot-Product Attention）**。

### 8.4 Attention Score、Attention Weight 与 Attention Output

**注意力分数（Attention Score）**通常指 Softmax 之前的 QKᵀ/√dₖ 数值。

**注意力权重（Attention Weight）**是分数经过掩码和 Softmax 后的概率式权重。

**注意力输出（Attention Output）**是按权重加权汇总 Value 后得到的向量。

分数、权重和输出是三个不同阶段，不能混为一谈。

### 8.5 Self-Attention

**自注意力（Self-Attention）**中的 Q、K、V 都来自同一序列。每个位置都可以读取这条序列中允许访问的位置。

Encoder 的双向自注意力通常允许每个有效 Token 同时查看左侧和右侧，因此适合理解完整输入。

Decoder 的自注意力通常加入因果掩码，只允许读取当前位置和之前位置，因此适合自回归生成。

### 8.6 Multi-Head Attention

**多头注意力（Multi-Head Attention）**不是只做一次注意力，而是把表示投影到多个较小的子空间，多个 Head 并行计算注意力，再拼接结果并做一次线性变换。

**拼接（Concatenation）**是把多个向量沿特征维度首尾组合成更长向量。多头输出拼接后，再通过输出投影融合回模型隐藏维度。

**注意力头（Attention Head）**是一组独立的 Q、K、V 投影。不同 Head 可以学习不同关系，例如局部搭配、指代关系、句法结构或长距离依赖。

多头的意义不是保证每个 Head 都对应清晰的人类语言规律，而是让模型拥有多个并行的信息交互子空间，增强表达能力。

### 8.7 Attention Matrix

**注意力矩阵（Attention Matrix）**记录每个 Query 位置对每个 Key 位置的权重。若序列长度为 L，标准自注意力矩阵大小约为 L × L。

因此标准注意力的时间和显存成本随序列长度近似二次增长，这就是长上下文昂贵的主要原因之一。

### 8.8 三种易混淆机制

**Padding Mask**屏蔽补齐位置，避免模型把 PAD 当作真实内容。

**因果掩码（Causal Mask）**屏蔽未来位置。对位置 i，所有 j > i 的位置不可见。它保证模型预测下一个 Token 时不能偷看答案。

**MASK Token**是 MLM 输入中的真实特殊 Token，用于替换内容。它不是注意力矩阵上的屏蔽规则。

掩码通常通过把禁止位置的注意力分数设为极小值实现。经过 Softmax 后，这些位置的权重接近零。

Decoder 自注意力通常同时组合目标侧 Padding Mask 与 Causal Mask：既不能读取目标 Padding，也不能读取未来位置。Cross-Attention 则使用源序列的 Padding Mask，避免 Decoder 关注 Encoder 输出中的源 Padding。

Padding 位置作为 Query 时仍可能产生隐藏状态，因此只做注意力屏蔽并不等于自动排除训练误差。训练通常还要使用**损失掩码（Loss Mask）**，让 Padding 等无效标签位置不参与损失计算。

### 8.9 Full、Sparse、Local 与 LSH Attention

**全注意力（Full Attention）**允许每个 Query 与所有可见 Key 计算关系。它表达能力直接，但注意力矩阵随序列长度二次增长。

**稀疏注意力（Sparse Attention）**只计算预先选择的一部分位置关系，以降低长序列成本。它不是把注意力权重训练得很小，而是在结构上省略部分连接。

**局部注意力（Local Attention）**让每个 Token 主要关注附近窗口。单层视野有限，但多层堆叠后信息可以逐层传播到更远位置。

**全局注意力（Global Attention）**让少数选定 Token 与全序列交互。Longformer 把局部窗口和少量全局 Token 结合，使长文档既保留局部效率，又能传播全局信息。

**局部敏感哈希注意力（Locality-Sensitive Hashing Attention，LSH Attention）**用哈希方法把可能相似的 Query 和 Key 分到相近桶中，只在候选桶内计算注意力。**哈希（Hashing）**是把对象映射到较小标识空间的方法；局部敏感哈希特别希望相似对象有较高概率落入同一桶。Reformer 使用这类思想减少长序列计算。

**轴向位置编码（Axial Positional Encoding）**把一个很大的位置表示分解到多个轴上，再组合各轴表示，以减少超长序列位置参数的存储成本。它优化的是位置表示，不是注意力匹配本身。

## 9. Feed-Forward Network

**前馈神经网络（Feed-Forward Network，FFN）**是每个 Transformer Block 中对每个位置独立应用的非线性网络。

它通常先把隐藏维度扩展到更大的中间维度，经过激活函数，再投影回隐藏维度。

注意力负责不同 Token 之间的信息交换；FFN 负责在每个 Token 内部对聚合后的特征进行更复杂的变换。经典 Transformer Block 通常同时包含二者，它们承担不同作用。

FFN 对所有位置使用同一组参数，但每个位置输入的隐藏状态不同，因此输出也不同。

## 10. Residual、LayerNorm、Dropout 与 Add & Norm

### 10.1 Residual Connection

**残差连接（Residual Connection）**把子层输入直接加到子层输出上。

它提供一条更短的信息和梯度通路，使深层网络更容易保留原始信息并稳定训练。如果子层暂时学不到有效变换，模型仍可通过残差路径传递输入。

### 10.2 Layer Normalization

**层归一化（Layer Normalization，LayerNorm）**对单个样本某个位置的隐藏维度进行标准化，再使用可学习缩放和平移参数调整。

它减少不同层数值分布变化，改善训练稳定性。LayerNorm 与 Batch Normalization 不同：后者依赖 Batch 统计，前者主要在特征维度上工作，更适合可变长度序列。

### 10.3 Pre-Norm 与 Post-Norm

**Post-Norm**先执行子层和残差相加，再做 LayerNorm，是原始 Transformer 的形式。

**Pre-Norm**先做 LayerNorm，再执行子层并与残差相加。许多深层现代 Transformer 使用 Pre-Norm，因为梯度传播通常更稳定。

若输入为 x、子层为 Sublayer，则两种顺序可写为：Post-Norm = LayerNorm(x + Sublayer(x))；Pre-Norm = x + Sublayer(LayerNorm(x))。它们使用相同组件，但归一化位置不同。

### 10.4 Dropout

**Dropout**在训练时随机把部分神经元输出置零，减少模型过度依赖固定特征组合，是一种正则化方法。推理时通常关闭 Dropout，并按训练规则保持期望值一致。

### 10.5 Add & Norm

**Add & Norm**是“残差相加加归一化”的简称。注意力子层和 FFN 子层外通常都包有这套结构。

## 11. Encoder 的完整结构

### 11.1 Encoder Block

一个经典 Encoder Block 包含：

1. 多头双向自注意力。
2. 残差连接与层归一化。
3. 逐位置前馈网络。
4. 再一次残差连接与层归一化。

多个 Encoder Block 堆叠后，每一层都在上一层表示基础上重新计算 Token 之间的关系。浅层可能更偏局部形式，深层通常形成更抽象的上下文表示，但这种分工不是人工硬编码的。

### 11.2 Encoder 的输入与输出

Encoder 输入是带位置和结构信息的 Token 向量序列。

Encoder 输出是每个 Token 的上下文化隐藏状态。输出序列长度通常与输入 Token 数相同，但每个位置的向量已经融合其他位置的信息。

### 11.3 Bidirectional Context

**双向上下文（Bidirectional Context）**表示当前位置能同时利用左侧和右侧文本。BERT 在编码完整句子时具有这一特点。

双向不等于模型把序列正向跑一次再反向跑一次。在 Transformer Encoder 中，它主要意味着自注意力没有因果限制，每个位置能直接关注两侧。

## 12. Decoder 的完整结构

### 12.1 Decoder Block

原始 Encoder-Decoder Transformer 的 Decoder Block 包含：

1. 带因果掩码的多头自注意力。
2. 残差连接与层归一化。
3. 对 Encoder 输出的多头交叉注意力。
4. 残差连接与层归一化。
5. 逐位置前馈网络。
6. 再一次残差连接与层归一化。

Decoder-only 模型没有独立 Encoder，因此通常没有对 Encoder 输出的交叉注意力，只保留因果自注意力和 FFN。

### 12.2 Masked Self-Attention

**带掩码的自注意力（Masked Self-Attention）**在 Decoder 中通常指带因果掩码的自注意力。当前位置只能利用已经出现的目标前文，不能访问未来目标 Token。

这里的“Masked”是注意力可见性限制，不是把输入替换为 MASK Token。

### 12.3 Cross-Attention

**交叉注意力（Cross-Attention）**让 Decoder 读取 Encoder 输出。

其中 Query 来自 Decoder 当前隐藏状态，Key 和 Value 来自 Encoder 输出。它表达的是：“根据当前已经生成的目标前文，应从源文本中读取哪些信息？”

在翻译中，Decoder 预测每个目标词时都可以重新关注源句不同位置。源语言和目标语言词序不同也不妨碍这种动态对齐。

### 12.4 Autoregressive Generation

**自回归生成（Autoregressive Generation）**是每次根据已有前文预测一个新 Token，再把新 Token 加回输入继续预测。

它形成条件概率分解：整段输出的概率由每一步“在已有前文条件下生成当前 Token”的概率连乘得到。

自回归生成具有误差累积风险：某一步生成错误后，后续步骤会以错误内容作为真实上下文继续生成。

## 13. Encoder-only、Decoder-only 与 Encoder-Decoder

### 13.1 Encoder-only

**Encoder-only 架构**只使用 Encoder Block，代表模型包括 BERT、RoBERTa 和 DistilBERT。

**BERT（Bidirectional Encoder Representations from Transformers）**通过双向 Encoder 预训练上下文表示。**RoBERTa（Robustly Optimized BERT Pretraining Approach）**调整了 BERT 的预训练策略。**DistilBERT**通过知识蒸馏压缩 BERT，即让较大的教师模型指导较小的学生模型，以更少参数换取更快推理。

它通常利用双向上下文，擅长理解类任务：文本分类、句子对判断、NER、抽取式问答、Embedding 和掩码填充。

它不天然适合长篇自回归生成，因为训练结构不是逐步生成目标序列。

### 13.2 Decoder-only

**Decoder-only 架构**只使用带因果掩码的 Decoder Block，代表模型包括 GPT 系列和许多现代 LLM。

**GPT（Generative Pre-trained Transformer）**使用因果语言建模预训练，通过前文预测后续 Token。“Generative”表示生成式，“Pre-trained”表示先经过大规模预训练。

它把各种任务统一成“根据前文继续生成”。分类、问答和翻译也可以通过提示词转成文本生成问题。

它擅长开放式生成、对话、续写和代码生成，但每个新 Token 的推理依赖前面已生成内容。

### 13.3 Encoder-Decoder

**Encoder-Decoder 架构**先由 Encoder 理解完整输入，再由 Decoder 条件生成输出，代表模型包括 T5、BART 和许多翻译模型。

**T5（Text-to-Text Transfer Transformer）**把不同 NLP 任务统一表达为文本输入到文本输出。**BART（Bidirectional and Auto-Regressive Transformers）**结合双向 Encoder 与自回归 Decoder，通过破坏并重建文本学习 Seq2Seq 表示。

它适合输入和输出都是序列、但二者作用不同的任务，如翻译、摘要、语音识别和条件文本生成。

### 13.4 三类架构的本质差异

三类架构最核心的区别不是模型名字，而是信息可见范围和输出方式：

| 架构 | 输入可见性 | 输出方式 | 典型优势 |
|---|---|---|---|
| Encoder-only | 通常双向读取完整输入 | 输出表示或任务标签 | 理解与抽取 |
| Decoder-only | 位置 i 读取输入中的位置 ≤ i，并预测位置 i+1 | 自回归生成 | 开放式生成 |
| Encoder-Decoder | Encoder 双向理解，Decoder 因果生成并读取 Encoder | 条件生成 | 输入到输出转换 |

## 14. Transformer 的 Head 与最终输出

### 14.1 Backbone

**骨干网络（Backbone）**是产生通用隐藏状态的 Transformer 主体。它负责表示和上下文建模，但不直接规定最终任务格式。

### 14.2 Task Head

**任务头（Task Head）**是接在骨干网络后的任务专用输出层。

- **序列分类头**把整段表示映射为类别 Logits。
- **Token 分类头**把每个 Token 的隐藏状态映射为标签 Logits。
- **MLM Head**把每个位置映射到完整词表，用于恢复被遮盖 Token。
- **语言模型头（LM Head）**把隐藏状态映射到词表，用于预测下一个 Token。
- **问答头（QA Head）**为每个位置产生答案起点和终点分数。

同一个骨干网络接不同 Head 可以完成不同任务，但新 Head 若没有经过训练，输出没有实际任务意义。

### 14.3 Weight Tying

**权重共享（Weight Tying）**常指输入 Embedding 矩阵与输出 LM Head 共享参数。输入时用它把 Token ID 映射为向量，输出时用转置方向把隐藏状态映射回词表分数。

这样可以减少参数，并让输入、输出词表表示保持一致。

## 15. 预训练目标

### 15.1 自监督学习

**自监督学习（Self-Supervised Learning）**从原始数据本身构造监督信号，不需要人工逐条标注。

语言模型中，原文本同时提供输入和答案：遮盖部分后恢复原词，或用前文预测后文。

自监督不等于无监督。它有明确标签和损失，只是标签由数据自动产生。

### 15.2 Masked Language Modeling

**掩码语言建模（Masked Language Modeling，MLM）**随机选择部分 Token，对输入进行遮盖或扰动，再预测其原始内容。

MLM 允许模型使用左右两侧上下文，因此特别适合训练 Encoder 的双向理解能力。

只有被选中的预测位置参与 MLM 损失，其余位置的标签会被忽略。训练时常采用**动态遮盖（Dynamic Masking）**：同一文本在不同批次或训练轮次可遮盖不同位置，使模型看到更多预测组合。

经典 BERT 对被选中的位置通常采用 80% 替换为 MASK Token、10% 替换为随机 Token、10% 保持原 Token 的策略。后两种情况让模型不能只依赖 MASK 标记，也减小“预训练时存在 MASK、实际使用时通常不存在 MASK”的输入差异。这里的比例是经典做法，不是所有 MLM 的硬性规则。

**全词遮盖（Whole Word Masking）**对被拆成多个子词的完整词一起遮盖，防止模型仅凭未遮盖子词轻易恢复答案。

### 15.3 Causal Language Modeling

**因果语言建模（Causal Language Modeling，CLM）**让位置 t 的表示预测位置 t+1 的 Token。

“因果”表示当前预测只能依赖过去，不能依赖未来。它通过因果掩码和标签位置对齐实现。

CLM 是 Decoder-only 自回归模型的核心预训练目标。

### 15.4 Next Sentence Prediction

**下一句预测（Next Sentence Prediction，NSP）**是原始 BERT 使用的预训练目标之一。模型看到两个句段，判断第二段是否真的是第一段的后续内容。

NSP 试图让模型学习句段之间的关系，也解释了原始 BERT 为什么使用句段 ID。不过后续模型并不都保留 NSP；句段 ID 和句子对能力也不能简单归结为这一个目标。

### 15.5 Sequence-to-Sequence Learning

**序列到序列学习（Sequence-to-Sequence，Seq2Seq）**把输入序列映射为输出序列。

训练时 Encoder 读取源序列，Decoder 根据源表示和正确目标前文预测目标序列的下一个 Token。

### 15.6 Teacher Forcing

**教师强制（Teacher Forcing）**是在训练 Decoder 时，每个位置使用真实目标前文，而不是前一步模型自己的预测。

它使所有目标位置的输入都能提前准备，从而并行训练，也能避免早期错误在一次训练样本中不断累积。

它带来的问题叫**暴露偏差（Exposure Bias）**：训练时模型总看到正确前文，推理时却必须面对自己可能错误的输出，两种分布不完全一致。

### 15.7 目标右移

**目标右移（Target Shifting）**是让 Decoder 输入表示“此前的正确目标”，让同一位置的监督标签表示“当前要预测的目标”。其本质是 Decoder 输入与标签错开一个目标位置。

序列开头使用什么起始 ID 由模型决定：可能是 BOS，也可能是专门的 Decoder Start Token；某些模型还会复用 PAD 作为起始 ID。因此“目标右移”等于“固定加入 BOS”并不准确。

Encoder-Decoder 训练通常根据标签构造右移后的 Decoder 输入。CLM 则常让输入 ID 与标签表面相同，并在损失计算中用位置 i 的 Logits 对齐位置 i+1 的标签。二者都在学习下一 Token，但具体对齐机制不能混为一谈；也不能表述成“把 Logits 向右移动”。

## 16. 预训练、领域适应、微调与知识蒸馏

### 16.1 Pretraining 与 Fine-tuning

预训练学习通用能力，微调学习具体任务。微调不是只训练新 Head 的同义词：可以只更新 Head，也可以更新部分或全部骨干网络参数。

**全参数微调（Full Fine-tuning）**更新全部模型参数，适应能力强但显存和存储成本高。

**参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）**冻结大部分原模型，只训练少量新增或选定参数，以降低成本。

### 16.2 Domain Adaptation

**领域适应（Domain Adaptation）**让通用模型继续在法律、医学、金融、电影评论等领域语料上学习，使表示更符合特定领域的词汇和语言分布。

它可以发生在下游任务微调之前，并被多个领域任务复用。

### 16.3 Knowledge Distillation

**知识蒸馏（Knowledge Distillation）**用较大的教师模型指导较小的学生模型。学生不仅学习真实标签，还学习教师输出分布或中间表示。

蒸馏的目标是减少参数和推理成本，同时尽量保留教师能力。DistilBERT 是典型例子。

### 16.4 Checkpoint 与 Model Card

**检查点（Checkpoint）**是某一训练时刻保存的模型状态，通常包含权重、配置以及恢复训练所需信息。它不是正在运行的模型对象，而是一份可加载的保存状态。

**模型卡片（Model Card）**是描述模型用途、训练数据、评估结果、限制、偏见和使用方法的文档。它的意义是帮助使用者判断模型是否适合当前场景，而不是只看模型名称或下载量。

## 17. 训练为什么能并行，生成为什么通常串行

训练 CLM 或 Seq2Seq Decoder 时，整段正确目标已知。Seq2Seq 可先构造右移后的目标输入；CLM 可在损失内部错开 Logits 与标签。二者再配合因果掩码，就能一次构造所有训练位置并用矩阵并行计算预测。

推理时没有未来正确答案。第 t+1 步输入依赖第 t 步实际生成的 Token，因此必须先生成前一个 Token，再生成后一个。

所以：

- 因果掩码限制的是“能看哪些位置”。
- 并行计算描述的是“硬件能否同时算多个位置”。
- 自回归依赖描述的是“后一步输入是否依赖前一步输出”。

三者相关，但不是同一个概念。

## 18. 生成式模型的推理过程

### 18.1 Inference

**推理（Inference）**是使用训练好的模型对新输入产生预测，不更新模型参数。

### 18.2 Prefill

**预填充阶段（Prefill）**一次处理完整提示词，计算所有输入 Token 的隐藏状态和注意力 Key、Value。

预填充能在多个提示 Token 之间并行，但长提示仍会消耗大量计算和显存。

### 18.3 Decoding

**解码阶段（Decoding）**逐步生成新 Token。每一步根据当前 Logits 选择一个或多个候选 Token，再继续下一步。

模型内部的一次完整循环是：从词表 Logits 选出 Token ID → 把新 ID 追加到 ID 序列 → 将新 ID 通过 Embedding 形成下一步输入表示 → 结合历史 KV Cache 再计算下一组 Logits。循环直到生成 EOS 或满足其他停止条件。Tokenizer 可以同步把 ID 转成文本供用户查看，也可以在生成结束后统一转换；文本转换不是下一步神经网络计算的必要环节，而且单个子词或字节 Token 未必能独立形成可读片段。

这里的“解码”有两个相关含义：Tokenizer 解码是把 Token ID 还原成文本；生成解码是根据概率策略逐步选择 Token。二者不等同于 Transformer 的 Decoder 架构。

### 18.4 KV Cache

**KV 缓存（Key-Value Cache，KV Cache）**保存历史 Token 在各层注意力中的 Key 和 Value。生成新 Token 时，不必重新计算全部历史 K、V，只计算新位置并读取缓存。

KV Cache 大幅减少重复计算，但会随上下文长度、层数、Head 数和隐藏维度增长，占用显存。

### 18.5 Context Window

**上下文窗口（Context Window）**是模型一次可利用的最大 Token 范围，包括提示词、历史对话、检索内容和已生成文本。

上下文窗口不是长期记忆。超出窗口的信息若没有被重新提供，模型无法直接访问。

## 19. Token 选择与生成策略

### 19.1 Greedy Decoding

**贪心解码（Greedy Decoding）**每一步选择概率最高的 Token。它速度快、结果确定，但容易陷入局部最优、重复或单调表达。

### 19.2 Beam Search

**波束搜索（Beam Search）**同时保留若干条累计得分最高的候选序列，每一步扩展并再次筛选。

**波束宽度（Beam Width）**是每一步保留的候选数量。宽度越大，搜索更充分，但计算更贵，也可能偏向安全、常见和较短的表达。

### 19.3 Sampling

**采样（Sampling）**按照概率分布随机选择 Token，而不是始终取最大值。它能增加多样性，但也提高不稳定和错误风险。

### 19.4 Temperature

**温度（Temperature）**在 Softmax 前缩放 Logits。温度低时分布更尖锐，输出更确定；温度高时分布更平坦，低概率 Token 更容易被选中。

温度不增加模型知识，只改变选择分布。

### 19.5 Top-k Sampling

**Top-k 采样**只保留概率最高的 k 个 Token，再从中采样。它固定候选数量，但不同场景中第 k 个 Token 的实际概率可能差异很大。

### 19.6 Top-p Sampling

**Top-p 采样**也称核采样（Nucleus Sampling），选择累计概率达到阈值 p 的最小候选集合，再从中采样。

它的候选数量会随分布变化：模型很确定时集合小，不确定时集合大。

### 19.7 Repetition Penalty

**重复惩罚（Repetition Penalty）**降低已出现 Token 再次被选择的倾向，用于缓解循环重复。

惩罚过强会破坏必须重复的名称、术语和语法结构，因此它只是解码控制，不是从根本上修复模型能力。

### 19.8 Length Penalty 与停止条件

**长度惩罚（Length Penalty）**调整搜索时对长短序列的偏好，常用于 Beam Search。

**停止条件（Stopping Criteria）**规定何时结束生成，例如生成 EOS、达到最大新 Token 数或出现指定终止串。

## 20. 主要 NLP 任务的内部逻辑

### 20.1 Sequence Classification

**序列分类（Sequence Classification）**为整个输入预测一个或多个标签。

模型先产生整段文本的表示，再由分类 Head 输出各类别 Logits。整段表示可以来自 CLS Token、平均池化或其他聚合方式。

**单标签分类（Single-label Classification）**要求互斥类别中只能选择一个，例如正面、负面或中性；**多标签分类（Multi-label Classification）**允许多个标签同时成立，例如一篇新闻同时属于科技、商业和国际类别。两者的输出解释、损失和阈值策略不同。

**池化（Pooling）**是把可变长度的 Token 表示聚合为固定长度向量。常见方式包括取 CLS、平均池化和最大池化。

### 20.2 Sentence Pair Classification

**句子对分类（Sentence Pair Classification）**判断两段文本之间的关系，例如语义是否等价、前提是否蕴含假设。**蕴含（Entailment）**表示在前提为真时，假设能够由前提推出；它不同于两句话字面相同。

它与 Batch 不同：句子对是一个样本内部的两个序列；Batch 是多个独立样本的集合。

### 20.3 Token Classification

**Token 分类（Token Classification）**为每个 Token 输出一个标签。输出长度通常与输入 Token 序列长度对应。

它适合 NER、POS 和 Chunking，但必须处理原始词标签与子词 Token 之间的对齐。

### 20.4 Named Entity Recognition

**命名实体识别（Named Entity Recognition，NER）**识别文本中的实体边界和类别，如人名、组织、地点、日期、时间、金额、产品和其他领域实体。它识别的对象由数据集的实体类型体系决定，不只限于专名。

仅知道某个 Token 属于“人名”还不够，还要知道多个 Token 是否属于同一个实体，因此需要边界标注。

### 20.5 POS Tagging

**词性标注（Part-of-Speech Tagging，POS）**为词标注名词、动词、形容词等语法类别。

### 20.6 Chunking

**句法分块（Chunking）**识别非递归的短语块，例如名词短语和动词短语。它比完整句法树简单，但比单纯词性标注更关注局部结构。

### 20.7 IOB2

**IOB2 标注**中的 IOB 是 Inside、Outside、Beginning 的缩写，它用位置前缀表示实体或短语边界：

- B（Begin）：一个实体的第一个 Token。
- I（Inside）：同一实体内部的后续 Token。
- O（Outside）：不属于任何实体。

即使两个相邻实体类别相同，第二个实体也必须重新从 B 开始，才能区分边界。

IOB2 中的“2”用于区别早期 IOB1 规则：IOB2 要求每个实体无论前面是什么都从 B 开始，因此边界更统一、直观。

### 20.8 Subword Label Alignment

**子词标签对齐（Subword Label Alignment）**处理“一个原始词被切成多个 Token，但只有一个词级标签”的问题。

常见思想是：特殊 Token 和 Padding 不参与损失；第一个子词继承原标签；后续子词根据标注规范转为实体内部标签，或直接忽略。选择哪种策略会影响训练信号和评估方式。

### 20.9 Machine Translation

机器翻译是典型 Seq2Seq 任务。Encoder 表示源语言，Decoder 使用目标语言真实前文训练，并在推理时自回归生成。

**平行语料（Parallel Corpus）**是彼此对应的源语言与目标语言句子集合。对齐错误会让模型学到错误映射。

翻译不仅追求词面对应，还要处理语序、歧义、术语、语气和文化表达。

### 20.10 Text Summarization

**抽取式摘要（Extractive Summarization）**从原文选择重要句子或片段。

**生成式摘要（Abstractive Summarization）**理解原文后重新组织语言，可以产生原文中没有连续出现的新表述。

摘要的核心矛盾是压缩与信息保留。生成结果流畅不代表事实正确，可能出现把原文内容改写错或凭空增加事实的情况。

### 20.11 Extractive Question Answering

**抽取式问答（Extractive QA）**要求答案是上下文中的连续片段。模型为每个 Token 预测成为答案起点和终点的分数，再选择有效跨度。

**答案跨度（Answer Span）**是答案在上下文中的连续起止范围。

训练数据的答案常以原文中的字符起点和终点保存。**字符偏移（Character Offset）**是字符在原文中的位置；预处理需要借助 Offset Mapping 把字符范围映射成 Token 起止索引，才能监督模型的 Token 级预测。

长上下文通常要切成多个有重叠的窗口，再将各窗口候选答案合并。训练时，如果答案不在某个窗口内，该窗口应使用数据集约定的“无答案”位置，而不能伪造一个普通文本跨度。验证时还要保留每个窗口所属的原始样本 ID 与有效 Offset Mapping，才能把 Token 预测还原成原文答案。

**滑动窗口（Sliding Window）**让相邻文本块保留部分重叠，减少答案被边界切断的风险。

选择最终答案时，不能简单地分别取最高起点和最高终点，因为二者组合后可能终点早于起点、跨越特殊 Token 或长度不合理。通常先从高分起点与终点中组合候选，只保留有效跨度，再用起点与终点 Logits 之和等联合分数比较；同一问题的多个窗口也会据此汇总候选。但跨窗口分数未必经过严格校准，这种直接排序是常见启发式方法，不能把不同窗口各自的 Softmax 概率当成天然可比。若任务允许无答案，还要把最佳文本跨度与无答案分数比较。

### 20.12 Generative Question Answering

**生成式问答（Generative QA）**由模型逐 Token 生成答案。它能综合、解释和改写，但答案不一定是原文片段，也更容易产生无依据内容。

### 20.13 Hallucination

**幻觉（Hallucination）**是模型生成了流畅、看似合理但缺乏输入或可靠事实支持的内容。

幻觉不是简单的语法错误，而是内容依据问题。检索、引用、事实核验和任务约束能降低风险，但不能保证完全消失。

## 21. Embedding、语义搜索与 RAG

### 21.1 Contextual Embedding

**上下文化嵌入（Contextual Embedding）**是结合上下文后得到的 Token 向量。同一个词在不同句子中的向量可以不同。

### 21.2 Sentence Embedding

**句子嵌入（Sentence Embedding）**是表示整句话或整段文本的固定长度向量。

它可以通过 CLS Pooling、平均池化或专门训练的句向量模型获得。

**CLS Pooling**取 CLS Token 的最终隐藏状态作为整段表示。

**平均池化（Mean Pooling）**对有效 Token 隐藏状态求平均，通常要排除 Padding。

句子向量通常需要专门训练，不能默认任意模型的 CLS 状态都适合语义相似度。

**对比学习（Contrastive Learning）**让语义相关的文本向量靠近，让无关文本向量远离。

**正样本对（Positive Pair）**是语义应接近的文本组合，如问题与正确答案；**负样本对（Negative Pair）**是语义不相关或不匹配的组合。

**难负样本（Hard Negative）**表面上与查询很相似、实际上并不正确。它比随机无关文本更能训练模型分辨细微语义差异。

### 21.3 Vector Space

**向量空间（Vector Space）**是由多个数值维度组成的表示空间。语义嵌入希望含义相近的文本距离近，含义不同的文本距离远。

单个维度通常没有稳定、明确的人类解释，语义由整体方向和位置共同表达。

### 21.4 Cosine Similarity 与 Dot Product

**余弦相似度（Cosine Similarity）**比较两个向量方向的夹角，弱化长度影响。值越大通常表示方向越接近。

**点积相似度（Dot Product Similarity）**同时受方向和向量长度影响。若向量已归一化，点积与余弦相似度等价。

**归一化（Normalization）**在这里指把向量缩放到单位长度，不要与文本规范化或 LayerNorm 混淆。

**距离度量（Distance Metric）**是判断两个向量接近程度的规则，例如余弦距离、内积或欧氏距离。Embedding 模型的训练目标、是否归一化以及向量索引采用的度量必须相互匹配；否则“模型认为相近”和“索引实际返回相近”可能不是同一件事。若用内积索引实现余弦检索，应先把向量归一化。

### 21.5 Symmetric 与 Asymmetric Search

**对称语义搜索（Symmetric Semantic Search）**的查询与候选文本形式相似，例如句子找相似句子。

**非对称语义搜索（Asymmetric Semantic Search）**的查询和文档形式不同，例如短问题检索长段落。模型需要专门学会把不同形式但语义相关的文本映射到相近空间。

### 21.6 Vector Index 与 FAISS

**向量索引（Vector Index）**是一种用于快速近邻搜索的数据结构。

**最近邻搜索（Nearest Neighbor Search）**是在大量向量中寻找与查询向量最接近的若干向量。

**近似最近邻（Approximate Nearest Neighbor，ANN）**牺牲少量精确性换取大规模搜索速度。

**FAISS（Facebook AI Similarity Search）**是用于高效向量相似度搜索的工具。它不是生成模型，也不理解文本；它只根据已经计算好的向量和距离规则寻找近邻。

### 21.7 RAG 的完整逻辑

RAG 的一般链路是：准备外部知识 → 根据查询检索相关内容 → 组织检索上下文 → 让生成模型依据上下文回答。检索不只限于向量检索，也可以使用关键词检索、结构化查询或混合检索。采用向量检索时，常见链路是：文档切分 → 文档向量化 → 建立向量索引 → 查询向量化 → Top-k 检索。

查询和文档必须由彼此兼容的 Embedding 模型映射到同一个语义空间，否则两类向量的距离没有可靠含义。建立索引时还应随每个文本块保存来源、文档 ID、页码或位置等**元数据（Metadata）**，以便检索后恢复上下文、去重并给出可追溯引用。

**文档切分（Chunking）**把长文档拆成可检索片段。这里的 Chunking 是文档工程中的切块，不是前面的句法分块任务。

**Top-k 检索**返回相似度最高的 k 个候选片段。

RAG 的意义是把模型参数之外的外部知识在推理时提供给模型，使知识可更新、答案可引用，并降低完全依赖参数记忆造成的幻觉。

RAG 不自动保证正确：检索可能漏掉相关文档，向量模型可能排序错误，切块可能破坏语义，生成模型也可能忽略证据。

### 21.8 Transformer 为什么也能处理图像和音频

Transformer 的核心输入是向量序列，而不是只能接收文字。只要能把其他模态转换为序列，注意力就能在这些序列元素之间建立关系。

**模态（Modality）**是信息的表现形式，如文本、图像、音频和视频。

**视觉 Transformer（Vision Transformer，ViT）**把图像切成多个固定大小的 Patch，把每个 Patch 映射为向量，再像处理 Token 一样交给 Encoder。

**图像块（Patch）**是图像中的矩形局部区域，相当于视觉序列的基本单位。

Whisper 一类语音模型先把音频转换为频谱特征序列，再由 Encoder-Decoder 生成文字。

**频谱（Spectrogram）**描述声音能量如何随时间和频率变化，是音频波形的时频表示。

这说明 Transformer 的本质是序列关系建模；文本 Token、图像 Patch 和音频帧只是不同输入单位。

## 22. 评价指标

### 22.1 Accuracy

**准确率（Accuracy）**是预测正确样本数占全部样本的比例。

评价二分类时常用四种计数：**真正例（True Positive，TP）**是正例被正确预测为正；**假正例（False Positive，FP）**是负例被误报为正；**假负例（False Negative，FN）**是正例被漏报为负；**真负例（True Negative，TN）**是负例被正确预测为负。

类别严重不平衡时，Accuracy 可能误导。例如 NER 中大量 Token 都是 O，全部预测 O 也可能得到很高 Token 准确率。

### 22.2 Precision

**精确率（Precision）**衡量模型预测为正的结果中，有多少确实为正，即 TP ÷（TP + FP）。

精确率高表示误报少。

### 22.3 Recall

**召回率（Recall）**衡量所有真实为正的结果中，有多少被模型找出，即 TP ÷（TP + FN）。

召回率高表示漏报少。

### 22.4 F1 Score

**F1 分数**是 Precision 与 Recall 的调和平均，即 2PR ÷（P + R）。**调和平均（Harmonic Mean）**对较小值更敏感，因此只有 Precision 与 Recall 都较高时，F1 才会高。

“正例”单位必须明确。NER 常按完整实体边界与类别计算，而不是只按单个 Token 计算。

多分类指标通常把每个类别轮流视为“正类”、其余类别视为“负类”，再按 Macro、Micro 或 Weighted 方式汇总。因此同名 F1 若平均方式不同，含义和数值都可能不同。

**Macro Average**先分别计算每个类别的指标再等权平均，因此更关注小类别。

**Micro Average**先汇总所有类别的预测计数再计算指标，因此更受大类别影响。

**Weighted Average**按各类别真实样本数量加权平均。三种平均方式回答的问题不同，报告结果时必须说明采用哪一种。

### 22.5 Perplexity

**困惑度（Perplexity）**通常是平均交叉熵损失的指数形式，用来描述语言模型对数据有多“意外”。数值越低，模型对相同评价条件下的文本预测越确定。

**指数函数（Exponential Function）**把对数尺度还原到原始乘法尺度；若交叉熵使用自然对数，困惑度通常写成 e 的“平均交叉熵”次方。它可直观理解为模型在每一步面对的有效候选不确定性，但不是字面上的候选词数量。

不同 Tokenizer、词表、切块和任务定义会改变困惑度，因此不能随意跨设置比较。MLM 的困惑度只基于被遮盖位置，也不同于 CLM 对完整序列的概率建模。

### 22.6 BLEU

**BLEU（Bilingual Evaluation Understudy）**主要用于机器翻译，通过候选译文与参考译文的 N-gram 重合度评价输出，并对过短译文施加惩罚。

BLEU 高不保证语义完全正确；正确但表达不同的译文也可能得分较低。

### 22.7 ROUGE

**ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**常用于摘要，通过生成摘要与参考摘要的词或序列重合衡量信息覆盖。名称中的“Recall-Oriented”强调它最初更关注参考摘要内容被覆盖了多少。

ROUGE-1 看一元组，ROUGE-2 看二元组，ROUGE-L 基于最长公共子序列。

ROUGE 主要衡量词面重合，不直接验证事实一致性。

### 22.8 Exact Match

**完全匹配（Exact Match，EM）**要求规范化后的预测答案与某个参考答案完全一致。它严格、直观，但对同义改写和边界差异不宽容。

这里的**答案规范化（Answer Normalization）**通常按基准规则统一大小写、空白、标点，有时还去除冠词，然后再比较；具体规则由数据集决定。它不同于向量归一化和 LayerNorm。同一问题若有多个参考答案，EM 和问答 F1 通常分别与每个参考答案比较并取最佳分数；问答 F1 常按规范化后的 Token 重合计算 Precision、Recall 与 F1。

### 22.9 Baseline

**基线（Baseline）**是用于比较的简单方法或已有模型。它回答“复杂方法是否真的带来提升”。没有基线，单独一个分数很难判断价值。

## 23. 易混淆概念集中辨析

### 23.1 Token、Word、Character

Character 是书写字符，Word 是语言学上的词，Token 是模型实际处理单位。三者可能一一对应，也可能完全不同。

### 23.2 Tokenizer Model 与 Transformer Model

Tokenizer Model 指分词算法、词表和合并规则；Transformer Model 指通过神经网络参数进行表示和预测的模型。前者不负责理解语义，后者依赖前者提供正确 Token ID。

### 23.3 Token Embedding 与 Hidden State

Token Embedding 是由 Token ID 查表得到的初始向量；Hidden State 是经过一层或多层上下文计算后的内部表示。初始 Token Embedding 通常与上下文无关，Hidden State 与上下文有关。“Embedding”是更宽泛的概念，也可指上下文化 Token 向量或句子向量，不能一概等同于模型输入。

### 23.4 Attention Mask、Causal Mask、MASK Token、Loss Mask

- Attention Mask：标记有效输入与 Padding。
- Causal Mask：阻止 Decoder 查看未来位置。
- MASK Token：MLM 中放进输入序列的特殊 Token。
- Loss Mask：通过忽略标签让某些位置不参与损失。

它们控制的对象分别是输入有效性、注意力方向、输入内容和损失位置。

### 23.5 Self-Attention 与 Cross-Attention

Self-Attention 的 Q、K、V 来自同一序列；Cross-Attention 的 Query 来自 Decoder，Key 和 Value 来自 Encoder 输出。

### 23.6 Encoder 与 Decoder

Encoder 重点是把完整输入转成上下文化表示；Decoder 重点是根据可见前文生成下一 Token。Encoder-Decoder 中的 Decoder 还通过交叉注意力读取源序列。

### 23.7 MLM 与 CLM

MLM 恢复被随机遮盖的 Token，可利用左右上下文；CLM 根据左侧前文预测下一个 Token，不允许读取未来。

### 23.8 Pretraining 与 Fine-tuning

Pretraining 学通用语言能力；Fine-tuning 在已有权重上适配任务或领域。两者都是训练，只是起点、数据规模和目标不同。

### 23.9 Training 与 Inference

Training 有正确答案、计算损失并更新参数；Inference 没有标签监督，不更新参数，只产生预测。

### 23.10 Teacher Forcing 与 Autoregressive Generation

Teacher Forcing 在训练时使用真实目标前文；Autoregressive Generation 在推理时使用模型自己已经生成的前文。

### 23.11 Sequence Classification 与 Token Classification

Sequence Classification 对整段输入输出一个或多个标签；Token Classification 对每个 Token 输出标签。

### 23.12 Extractive 与 Generative

Extractive 方法从原输入中选择片段；Generative 方法可以生成原文中没有连续出现的新表达。生成更灵活，也更需要事实约束。

### 23.13 Keyword Search 与 Semantic Search

Keyword Search 匹配字面词语；Semantic Search 比较向量表达的含义。语义搜索不是永远更好，精确编号、专名和固定术语有时更适合关键词匹配。

### 23.14 Model Parameter 与 Hyperparameter

Parameter 由训练学习；Hyperparameter 由训练前设定或通过验证选择。

### 23.15 Loss 与 Metric

Loss 是用于反向传播的可微优化目标；Metric 是用于评价实际表现的指标。两者可能相关，但不能互相替代。

### 23.16 Context Window 与 Memory

Context Window 是本次推理可见的 Token 范围；Memory 是系统额外保存并在未来重新提供的信息。模型不会因为一次读过内容就永久记住，除非参数被训练更新或外部系统再次提供。

### 23.17 Attention Head 与 Task Head

Attention Head 是多头注意力内部的一条关系计算通道，负责形成一组 Q、K、V 和注意力输出；Task Head 位于骨干网络之后，把隐藏状态映射成分类、Token 标签、答案边界或词表 Logits。前者是 Transformer Block 内部结构，后者是面向任务的输出适配层。

### 23.18 Tokenizer Decoding、Transformer Decoder 与 Generation Decoding

Tokenizer Decoding 把 Token ID 还原成文本；Transformer Decoder 是由因果自注意力、可选交叉注意力和 FFN 等组成的网络架构；Generation Decoding 是依据 Logits 选择下一个 Token 的搜索或采样过程。三者中文都可能叫“解码”，但对象分别是 ID、神经网络表示和概率分布。

## 24. Transformer 信息流的最终统一理解

从最底层看，Transformer 完成的是以下过程：

1. Tokenizer 把语言切成离散 Token，并映射为 Token ID。
2. Token Embedding 把 Token ID 变成连续向量，位置表示加入顺序信息。
3. 注意力让每个位置从其他允许位置读取相关信息。
4. 多头机制让模型在多个表示子空间并行建立关系。
5. FFN 对每个位置的聚合信息做非线性变换。
6. 残差连接和 LayerNorm 改善深层网络的优化稳定性，Dropout 用于降低过拟合风险；它们不能保证训练必然成功。
7. 多层 Block 反复执行信息交换与特征变换，形成上下文化隐藏状态。
8. 任务 Head 把隐藏状态转成类别、实体标签、词表 Logits 或答案边界。
9. 训练阶段先按任务构造标签：MLM 只在被选中的遮盖位置计算损失；CLM 用位置 i 的 Logits 对齐位置 i+1 的 Token；Seq2Seq 使用右移目标作为 Decoder 输入；Padding 和无效位置由损失掩码排除。
10. 模型使用交叉熵等损失衡量预测与标签差距，通过反向传播计算梯度，再由优化器更新参数。
11. Encoder 推理根据任务分成三条常见输出支路：池化后做整段分类、逐 Token 分类、或从每个位置的起止分数中选择答案跨度。
12. 生成推理从 LM Logits 选择 Token ID，追加到序列，重新形成下一步输入并循环，直到满足停止条件。

从更高层看，BERT、GPT、T5、NER、翻译、摘要、问答和 RAG 并不是彼此孤立的知识：

- BERT 是以 Encoder 和双向上下文为核心的预训练模型。
- GPT 是以因果 Decoder 和自回归生成目标为核心的语言模型。
- T5 把任务统一成文本到文本的 Encoder-Decoder 生成问题。
- NER 在 Token 隐藏状态上增加 Token 分类 Head。
- 翻译和摘要使用源序列条件下的 Decoder 生成。
- 抽取式 QA 在输入隐藏状态上寻找答案起止位置。
- Sentence Embedding 把 Token 隐藏状态聚合成可比较的句子或段落向量。
- 语义搜索用向量距离检索文本。
- RAG 把检索结果重新放入生成模型的上下文，组合外部知识与语言生成。

理解这条统一主线后，新的 NLP 模型或任务主要是在四个位置发生变化：**输入怎样组织、哪些位置可见、训练目标是什么、输出怎样解释。**

## 25. 核心术语速查表

| 术语 | 核心定义 | 最关键的辨析 |
|---|---|---|
| NLP | 让计算机处理、理解和生成人类语言的领域 | 不等于真正的人类理解 |
| Corpus | 一组语言材料 | 不等于单个样本 |
| Token | 模型处理的离散文本单位 | 不一定是完整词 |
| Tokenizer | 文本与 Token ID 之间的完整转换系统 | 不负责神经网络语义计算 |
| Vocabulary | Token 与 ID 的映射集合 | 必须与模型权重匹配 |
| OOV | 词表无法直接表示的内容 | 子词方法可显著减少 |
| Embedding | 把对象映射到连续向量空间的表示方法 | Token Embedding、上下文化表示和句向量含义不同 |
| Hidden State | 模型内部的上下文化表示 | 不是最终概率或标签 |
| Parameter | 训练学习的数值 | 不同于人工设置的超参数 |
| Logits | 未归一化原始分数 | 不是概率 |
| Softmax | 把一组 Logits 变成概率分布 | 不改变同组分数排序 |
| Loss | 用于训练优化的误差数值 | 不等于业务评价指标 |
| Gradient | 损失对参数变化的敏感方向 | 用于指导参数更新 |
| Attention | 按相关性聚合其他位置的信息 | 不是模型的全部结构 |
| Query | 当前表示要寻找的信息条件 | 与 Key 匹配 |
| Key | 可被查询匹配的索引表示 | 决定相关性分数 |
| Value | 被关注后实际汇总的信息 | 不直接参与匹配含义 |
| Self-Attention | Q、K、V 来自同一序列 | 可双向，也可带因果掩码 |
| Cross-Attention | Query 与 Key/Value 来自不同序列 | 常连接 Decoder 与 Encoder |
| Multi-Head Attention | 多组注意力并行后融合 | 多个表示子空间，不是简单重复 |
| FFN | 对每个位置独立应用的非线性网络 | 负责位置内部特征变换 |
| Residual | 子层输入直接加到输出 | 帮助信息和梯度传播 |
| LayerNorm | 在隐藏特征维度上归一化 | 不依赖 Batch 统计为主 |
| Dropout | 训练时随机丢弃部分激活 | 推理时关闭 |
| Encoder | 把完整输入变成上下文化表示 | 通常适合理解任务 |
| Decoder | 根据可见前文预测后续 Token | 依靠因果约束生成 |
| Causal Mask | 屏蔽所有未来位置 | 不等于 MASK Token |
| MLM | 恢复被遮盖 Token | 可使用左右上下文 |
| CLM | 用前文预测下一个 Token | 不能查看未来 |
| Seq2Seq | 一个序列映射到另一个序列 | 常用 Encoder-Decoder |
| Teacher Forcing | 训练时使用真实目标前文 | 推理时无法使用 |
| Autoregressive | 把已生成结果作为下一步输入 | 推理通常串行 |
| Attention Head | 多头注意力中的一条关系计算通道 | 位于 Transformer Block 内部 |
| Task Head | 把隐藏状态转成任务输出 | 面向下游任务，通常需要训练 |
| Pretraining | 大规模数据上学习通用能力 | 通常先于具体任务微调 |
| Fine-tuning | 在已有权重上适配新目标 | 可更新部分或全部参数 |
| Domain Adaptation | 让模型适应特定领域分布 | 不等于只训练分类 Head |
| Inference | 用训练好模型处理新输入 | 不更新参数 |
| Context Window | 单次推理可利用的 Token 范围 | 不等于永久记忆 |
| KV Cache | 缓存历史注意力的 Key 与 Value | 加速生成但占显存 |
| NER | 识别实体边界和类别 | 不能只看 Token 类别准确率 |
| IOB2 | 用 B/I/O 表示实体边界 | 相邻同类实体需重新用 B |
| Embedding Search | 用向量距离检索语义相近文本 | 质量依赖嵌入模型和索引 |
| RAG | 检索外部知识后条件生成 | 检索和生成都可能出错 |
| Hallucination | 生成缺乏可靠依据的内容 | 流畅不代表真实 |
| Accuracy | 正确数占总数比例 | 类别不平衡时可能误导 |
| Precision | 预测正例中真正为正的比例 | 关注误报 |
| Recall | 真实正例中被找出的比例 | 关注漏报 |
| F1 | Precision 与 Recall 的调和平均 | 必须明确按什么单位统计 |
| Perplexity | 语言模型平均损失的指数表达 | 只能在相同设置下比较 |
| BLEU | 翻译候选与参考的 N-gram 重合指标 | 不直接等于语义质量 |
| ROUGE | 摘要与参考的重合指标 | 不直接验证事实一致性 |
| Exact Match | 预测与参考完全一致 | 对同义改写不宽容 |
