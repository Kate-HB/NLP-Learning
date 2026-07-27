# Day 14：中文情感分类实战——从数据到微调

用 bert-base-chinese 在真实中文对话数据上做 8 分类情感识别，完成完整的微调 pipeline，准确率从随机水平（12.5%）提升到 87.2%。

## 探索过程

### 第一轮：tabularisai/multilingual-sentiment-analysis + TIX007

一开始用 `tabularisai/multilingual-sentiment-analysis`（已做过中文情感微调的多语言模型）和 TIX007 数据集（16k 条 6 分类中文情感文本）。

**问题一**：`num_labels` 不匹配。该模型预训练时 classification head 是 5 分类，数据是 6 分类。

解决：`AutoModelForSequenceClassification.from_pretrained(model, num_labels=6, ignore_mismatched_sizes=True)`。`ignore_mismatched_sizes=True` 跳过形状不匹配的 classification head 权重，用随机初始化替代。BERT 主体权重保留。

**问题二**：1 epoch accuracy 就到 100%。三个原因叠加：

1. 训练集和验证集有 828 条重叠文本，过滤后仍有 99.95%。
2. 过滤后仍然 100%——TIX007 是合成数据，模板化严重，BERT 学的不是语义而是模板标记。
3. 验证：bert-base-chinese 随机分类头未训练评估 16.65%，训练后同样 100%，证明是数据问题而非模型问题。

解决：放弃 TIX007，换真实对话数据集。

### 第二轮：bert-base-chinese + zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset

换用 `zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset`（4,159 条真实中文对话，8 分类）。

**问题三**：数据是单个 CSV，没有分 train/val/test。

解决：`train_test_split(stratify=label, random_state=42)` 分层抽样 70/15/15，保存为 JSON 文件。`stratify` 保证每类在各集合中比例一致。

**问题四**：`from evaluate import load` 报错 `cannot import name 'load' from 'evaluate'`。

根因：项目目录下有个 `evaluate.py` 文件，Python 先搜索当前目录，把 `evaluate` 库盖掉了。

解决：删除或重命名本地 `evaluate.py`。

**问题五**：label 映射搞错。

根因：用 `zip(set(d['label']), set(d['label_name']))` 获取映射，但 set 是无序的，两个集合迭代顺序不一致导致错位。

正确做法：`sorted(set((x['label'], x['label_name']) for x in data))` 保持一一对应。

**问题六**：混淆矩阵中文乱码。

解决：

```python
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

SimHei 是 Windows 自带中文字体。

**问题七**：`trainer.train()` 结束时 `KeyboardInterrupt`。

根因：`TrainingArguments` 设置了 `push_to_hub=True`，训练结束后自动上传模型到 Hugging Face Hub，网络超时或认证问题导致卡住。

解决：去掉 `push_to_hub=True`，或训练完成后再手动 `model.push_to_hub()`。

**问题八**：`FutureWarning: tokenizer is deprecated and will be removed in version 5.0.0 for Trainer.__init__. Use processing_class instead.`

Transformers 4.57 开始推荐用 `processing_class` 替代 `tokenizer` 参数。暂时不改也不影响运行。

## 最终 pipeline

```python
# 1. 加载数据
raw_dataset = load_dataset("json", data_files={
    "train": "data/train_dialogue.json",
    "validation": "data/validation_dialogue.json",
    "test": "data/test_dialogue.json",
})

# 2. Tokenize
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# 3. DataCollator 动态填充
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 模型（bert-base-chinese 从零微调）
id2label = {0: "伤心", 1: "关心", 2: "厌恶", 3: "平静", 4: "惊讶", 5: "开心", 6: "生气", 7: "疑问"}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=8, id2label=id2label, label2id=label2id
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments("emotion-classification", eval_strategy="epoch", ...),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 6. 训练
trainer.train()
```

## 结果

bert-base-chinese, 3 epochs, lr=2e-5：

| 数据集 | accuracy | macro-F1 |
|--------|----------|----------|
| train (2,911) | 0.9763 | 0.9759 |
| val (624) | 0.8830 | 0.8814 |
| test (624) | 0.8718 | 0.8701 |

训练前未微调模型 accuracy 约 12.5%（≈1/8 随机水平），证明微调有效。训练集和验证集之间有约 9% 的 gap，属于正常的轻微过拟合。

## 关键概念

### num_labels 与 ignore_mismatched_sizes

预训练模型的 classification head 输出维度通常是预训练任务决定的（如 5 分类），下游任务类别数可能不同。`num_labels` 指定新分类头输出维度，`ignore_mismatched_sizes=True` 让形状不匹配的层权重随机初始化，预训练 BERT 主体的权重保留。这是所有下游微调的标准做法。

### 合成数据 vs 真实数据

TIX007 是模板合成的，验证集和训练集虽文本不同但模式完全相同。BERT 短短 1 epoch 就能学到这些表面模式拿到 100%，但这不叫情感识别。`zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset` 是真实对话，数据量虽少（4k vs 16k），但训练曲线和评估结果是可信任的。

### 过拟合判断

训练集准确率减去验证集准确率就是过拟合程度。本实验 gap ≈ 9%，正常。TIX007 gap = 0 —— 不是不过拟合，是数据本身测不出 gap。

### Trainer 的 label 处理

Trainer 内部自动从 dataset 中取出 `label` 列作为标签，不会传给 `model.forward()`。所以 tokenize 时只需要删除 `text` 和 `label_name` 等字符串列，保留整数 `label` 列即可。

### 混淆矩阵的 display_labels vs labels

`display_labels` 只改标签显示文字，不改矩阵行列顺序。要换顺序必须同时传 `labels=[...]` 参数。

## 今日知识总结

### 完整微调 pipeline

加载数据 → tokenize → DataCollator 动态填充 → 定义 compute_metrics → 组装 Trainer → evaluate 基线 → train → evaluate 对比。每一步的职责是清晰的：tokenizer 负责分词和编码，collator 负责 batch 内对齐，model 负责 forward 计算，Trainer 负责训练循环编排。

### 数据质量决定一切

今天是切身感受到"数据比模型重要"。TIX007 16k 条不管怎么调参都是 100%，换 4k 条真实数据立刻正常。NLP 项目中最容易被忽视的步骤恰恰是数据检查——验证集是否和训练集有重叠、数据是人工标注还是模板合成、类别分布是否合理。

### 合成数据的 100% 不是成功

合成数据的 100% 准确率不是"训练得好"，而是"数据太简单"。BERT 学的不是情感本身，而是每个情绪类别对应的固定句式模板。换成没见过的真实文本立刻崩——用"傻逼你妈死了"测 `tabularisai` 模型居然判成 love 99.99%。真实的微调需要真实数据。

### 模型未训练前的 baseline 很重要

每次微调前应该用随机分类头的模型做一次评估，确认结果接近随机水平（1/类别数）。如果未训练就很高，说明数据有问题或模型已见过同类数据。本节实验中 12.5% 的 baseline 是健康的，微调后涨到 87% 才有说服力。

### 下一步

Day 15 进入 GPT 与解码策略——Decoder-only 架构、自回归生成、greedy/beam search/top-k/top-p 采样。
