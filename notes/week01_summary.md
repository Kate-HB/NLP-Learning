# Week 01 周志：Hugging Face 入门与 Tokenizer（Day 01–04）

## 概述

本周是 NLP 系统学习的起点，目标是从零搭建可用的实验环境，并深入理解 Hugging Face `pipeline()` 背后的完整推理流程——从原始文本输入到模型输出标签的每一步发生了什么。前四天完成了环境配置、Pipeline 体验、Transformer 架构入门，以及 Tokenizer 的全面学习。

## 环境搭建与 Pipeline 初体验（Day 01）

第一天的工作重心是让代码跑起来。在 Windows 环境下配置了 Conda 虚拟环境、VS Code 解释器和 Jupyter Notebook 内核，安装了 PyTorch CUDA 12.8 版本并验证 GPU 可用。修复了 Torchaudio WinError 127 和 SymPy 版本冲突两个环境问题，将 Transformers 固定为 4.57.3 以兼容课程示例。登录 Hugging Face Hub 后，通过 `pipeline("sentiment-analysis")` 跑通了第一个 NLP 推理，体验了文本生成、掩码填充、Token 分类和问答等多个任务。初步了解了预训练、微调和迁移学习的概念：预训练从随机权重开始，在大规模文本上通过自监督学习掌握通用语言规律；微调在预训练模型基础上使用小规模任务数据继续训练，使模型适应具体任务。同时建立了一条重要的实践原则——Hugging Face Hub 上的模型不一定属于 Transformers 框架，使用前需要检查 `config.json` 中的 `model_type`。

## Transformer 架构与三类模型（Day 02）

第二天深入理解了 Transformer 的三类使用方式。Encoder-only 模型以 BERT 为代表，自注意力可以同时读取一个 Token 左右两侧的信息，适合文本分类和命名实体识别等理解任务。Decoder-only 模型以 GPT 和 GPT-2 为代表，通过因果掩码限制每个位置只能看到前文，适合文本生成任务。Encoder-Decoder 模型以 T5、BART、Whisper 为代表，Encoder 理解输入，Decoder 根据 Encoder 输出生成目标序列，适合翻译、摘要和语音识别等序列到序列任务。

理解了 BERT 的掩码语言模型：训练时完整句子先存在，程序随机遮盖部分 Token 后让模型根据上下文恢复原词，被遮盖的原词作为监督答案。区分了 BERT 的 `[MASK]` Token 和 Decoder 的 attention mask——前者是输入序列中的真实特殊 Token，后者是一套控制信息可见范围的规则。掌握了隐藏状态、logits 和语言建模头的关系：隐藏状态是每个 Token 的上下文化向量表示，语言建模头把隐藏状态线性变换为词表大小的 logits，logits 经过 softmax 后才变成概率。GPT-2 的训练不是"把 logits 向右移动"，而是用当前位置的 logits 预测下一个 Token，预测结果与真实值计算交叉熵损失。

还学习了 LLM 推理的两阶段过程：预填充阶段一次性处理完整提示词建立上下文，解码阶段自回归逐 Token 生成，KV 缓存用于避免重复计算历史 Key 和 Value。了解了温度、Top-k、Top-p、重复惩罚等采样策略如何控制生成的随机性和多样性。Transformer 的应用也不仅限于文本——Whisper 将音频特征序列化后送入 Encoder-Decoder，ViT 将图像切分成 Patch 后送入 Encoder 做分类，核心思想都是"把信息序列化后用 Transformer 处理"。

## 拆解 Pipeline 与理解模型输出（Day 03）

第三天正式拆开了 `pipeline()` 的黑箱，理解了它内部的三个步骤：Tokenizer 将文本预处理为数字张量，模型计算结果，后处理将模型输出转换为可读标签。建立了一条关键区分——checkpoint 是模型文件的保存地址（包含 config.json、model.safetensors、tokenizer.json 等），而 model 是加载到内存中可以执行前向计算的 Python 对象，两者不能混为一谈。

学习了 `AutoModel` 和 `AutoModelForSequenceClassification` 的本质区别：前者输出的是每个 Token 的 hidden states，形状为 `(batch_size, sequence_length, hidden_size)`，是还没有确定具体任务的中间表示；后者在 hidden states 之上多了一个分类 head，输出的是每个类别的 logits。不同任务需要不同的 head——分类 head 输出类别 logits，MLM head 预测被遮盖的 Token，QA head 预测答案起止位置，LM head 预测下一个 Token。模型输出的 logits 是原始分数，不是概率，需要用 softmax 转换为概率后再用 `id2label` 映射为人类可读的标签。

理解了 `tokenizer("A", "B")` 和 `tokenizer(["A", "B"])` 的本质区别：前者是句子对输入，两个句子被拼接为 `[CLS] A [SEP] B [SEP]`，并用 `token_type_ids` 区分 A 和 B；后者才是多个独立样本组成的 batch。这个区分在后续处理句子对任务（如自然语言推理）时至关重要。还学习了 `return_tensors="pt"` 的作用：不加时返回 Python list，加上后返回 PyTorch tensor 并自动添加 batch 维度。

## Tokenizer 深度解析（Day 04）

第四天全面学习了分词器的原理和使用。分词策略有三种粒度：基于单词的分词直观但词表巨大、未知词问题严重；基于字符的分词词表极小但序列变长、语义信息稀薄；基于子词的分词是工业界的主流方案，常用词保持完整，罕见词拆成有意义的子词片段。BERT 使用的 WordPiece 和 GPT-2 使用的 BPE 都属于子词分词算法，`##` 前缀表示该 Token 是前一个词的续接部分，decode 时会自动合并。

Tokenizer 提供三个原子操作：`tokenize()` 把文本拆成 Token 字符串列表，`convert_tokens_to_ids()` 把 Token 映射为词表中的整数 ID，`decode()` 把 ID 序列还原为可读文本。`tokenizer(text)` 一键调用是对这三个操作的封装，额外添加了 `[CLS]` 和 `[SEP]` 等特殊 Token 并生成 attention mask。

在批处理场景中，多个句子长度不同时需要用 padding 补齐到相同长度，但仅做 padding 会导致 Transformer 的注意力层把填充位置也纳入计算，改变有效句子的 logits。通过实验验证了：不加 attention mask 时 padding 句子的 logits 与单独输入时不同；加上 attention mask（padding 位置标 0）后，logits 恢复为单独输入时的值。这说明 attention mask 的核心作用是告诉注意力层"忽略这些位置"，而不仅仅是形式上的存在。

一个贯穿四天学习的核心约束是：Tokenizer 和 Model 必须来自同一个 checkpoint。Tokenizer 的词汇表、特殊 Token 定义和分词算法都是在模型预训练时确定的，用 BERT 的 tokenizer 切词后喂给 GPT-2，Token ID 的语义完全错乱，模型无法正常工作。因此 `AutoTokenizer.from_pretrained()` 和 `AutoModel.from_pretrained()` 必须使用相同的模型名称。

## 已解决问题

四天中解决的主要技术问题包括：Torchaudio CUDA 版本不一致导致的 WinError 127、SymPy 版本冲突、Transformers 新版本不兼容课程 Pipeline（降级至 4.57.3）、`fill-mask` 输入缺少 `[MASK]` Token、手动构造 tensor 缺少 batch 维度导致模型报错，以及 padding 后 logits 不一致的根因分析。环境方面，建立了安装包后必须重启 Notebook Kernel 的工作习惯。

## 下一步

Week 1 剩余三天的任务：Day 05 深入练习不同长度文本的 batch 构造和 padding 策略对比，Day 06 完整走通 tokenizer + model + softmax 的手动推理流程，Day 07 进行 Week 1 复盘，画出从文本到标签的完整推理流程图。
