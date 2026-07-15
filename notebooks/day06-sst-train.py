#加载数据集
from datasets import load_dataset
raw_datasets = load_dataset("glue", "sst2")

#加载分词器
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(example["sentence"],  truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

#动态padding
from transformers import  DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



import evaluate
import numpy as np
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "sst2")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#定义训练参数
from transformers import TrainingArguments
training_args = TrainingArguments("notebooks\\outputs\\day06-sst-train", eval_strategy="epoch",push_to_hub=True)

#定义模型
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

#定义Trainer
from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()