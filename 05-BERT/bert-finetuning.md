# BERT 微调基础

## BERT 做句子对分类

BERT 预训练时就包含了句子对建模能力（下一句预测任务），因此微调做句子对分类（如 MRPC 同义判断）时天然适配。

### 输入格式

```text
[CLS] 句子A [SEP] 句子B [SEP]
```

通过 `token_type_ids` 区分两句：
- `0`：句子 A 的 token
- `1`：句子 B 的 token

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("This is sentence A.", "This is sentence B.")
# token_type_ids: [0,0,0,0,0,0,0, 1,1,1,1,1,1,1]
```

### 模型加载

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

预训练 BERT 的原始 head 被丢弃，换成一个随机初始化的二分类 head（线性层）。警告 "Some weights were not initialized" 是正常的——说明分类层需要训练。

## Trainer API 微调流程

### 1. 加载数据

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
# DatasetDict: train(3668), validation(408), test(1725)
```

### 2. 预处理

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

Tokenize 时不传 `padding=True`，留给 DataCollator 动态填充。

### 3. 动态填充

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### 4. 训练配置

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "output-dir",
    eval_strategy="epoch",     # 每个 epoch 评估一次
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
)
```

### 5. 定义评估指标

```python
import numpy as np
import evaluate

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

### 6. 组装并训练

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 7. 评估

```python
# 训练完直接评估（使用内存中的已训练权重）
result = trainer.predict(tokenized_datasets["validation"])

# 或从保存的 checkpoint 加载
from transformers import AutoModelForSequenceClassification

trained_model = AutoModelForSequenceClassification.from_pretrained("output-dir/checkpoint-1377")
trainer.model = trained_model
result = trainer.predict(tokenized_datasets["validation"])
```

## 常见问题

### 训练后评估指标骤降

**原因**：`trainer.train()` 后重新执行 model 定义 cell（`AutoModelForSequenceClassification.from_pretrained(checkpoint)`），classification head 被重新随机初始化。

**解决**：train 完立刻 evaluate，或从 checkpoint 加载已训练权重。

### `evaluation_strategy` 报错

**原因**：Transformers 4.57 重命名为 `eval_strategy`。

**解决**：使用 `TrainingArguments(..., eval_strategy="epoch")`。

### accelerate 未安装

```bash
pip install "accelerate>=0.26.0"
# 或
D:/Soft/Conda/python.exe -m pip install "accelerate>=0.26.0"
```

安装后重启 Notebook Kernel。

## 模型保存与加载

训练后的模型保存在 `output_dir` 下：

```text
output_dir/
├── config.json
├── model.safetensors
├── tokenizer.json
├── training_args.bin
├── checkpoint-500/
├── checkpoint-1000/
└── checkpoint-1377/   ← 最终模型
```

加载已训练模型进行推理：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("output_dir/checkpoint-1377")
model = AutoModelForSequenceClassification.from_pretrained("output_dir/checkpoint-1377")

inputs = tokenizer("Sentence A", "Sentence B", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # (1, 2)，取 argmax 得预测类别
```
