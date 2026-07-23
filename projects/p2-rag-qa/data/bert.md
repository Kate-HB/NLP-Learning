# BERT

BERT 是 Encoder-only Transformer。它在预训练阶段通过遮盖语言模型学习双向上下文表示，因此适合文本分类、命名实体识别和抽取式问答等理解任务。

文本分类时，tokenizer 通常在句首加入 `[CLS]`。分类模型读取最后一层的 `[CLS]` hidden state，再经过线性分类头输出每个类别的 logits。logits 经过 softmax 后可解释为类别概率。

微调会同时更新预训练编码器和随机初始化的分类头。训练集太小时容易过拟合，因此需要独立验证集、测试集以及简单 baseline。
