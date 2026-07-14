#上复刻上述预处理。它有点不同，因为它是由单句而不是成对的句子组成的
def preprocess_glue(examples, tokenizer, task_keys=None):
    """适用于任何 GLUE 任务的预处理函数。

    参数
    ----
    examples : dict
        数据集的一个 batch，由 Dataset.map(batched=True) 传入。
    tokenizer : AutoTokenizer
        分词器。
    task_keys : tuple[str] | None
        传给 tokenizer 的参数名，如 ("sentence",) 或 ("sentence1", "sentence2")。
        传 None 时自动从 examples 的 key 推断（过滤掉 label、idx 等非文本列）。

    返回
    ----
    dict
        tokenizer 的输出（input_ids、attention_mask、token_type_ids 等），
        同时保留原始 label 列以便 Trainer 使用。
    """
    if task_keys is None:
        # 自动检测文本列：取所有 value 为字符串的列名，保留常见顺序
        text_keys = [k for k, v in examples.items()
                     if k not in ("label", "idx", "index") and isinstance(v[0], str)]
    else:
        text_keys = list(task_keys)

    if len(text_keys) == 1:
        # 单句任务：CoLA, SST-2
        return tokenizer(examples[text_keys[0]], truncation=True)
    elif len(text_keys) == 2:
        # 句子对任务：MRPC, MNLI, QQP 等
        return tokenizer(
            examples[text_keys[0]], examples[text_keys[1]], truncation=True
        )
    else:
        raise ValueError(f"无法确定文本列，检测到: {text_keys}")

#加载数据集
from datasets import load_dataset
raw_datasets = load_dataset("glue", "sst2")
print(raw_datasets)

#预处理
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

"""
def tokenize_function(example):
    return tokenizer(example["sentence"],  truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
"""
# 用法示例
tokenized_datasets = raw_datasets.map(
    preprocess_glue,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=raw_datasets["train"].column_names,
)
print(tokenized_datasets)

#动态填充
samples = tokenized_datasets["train"][:8]
#samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
print(samples)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})






