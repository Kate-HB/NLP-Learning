# P2 BERT 中文问答微调

用 bert-base-chinese 在 Chinese-SQuAD v2 上微调抽取式问答模型，支持不可回答问题（No-Answer）。

## 1. 问题与数据

- 输入：自然语言问题 + 上下文段落。
- 输出：答案文本或空（不可回答时）。
- 数据：[real-jiakai/chinese-squadv2](https://huggingface.co/datasets/real-jiakai/chinese-squadv2)，训练集 90,027 条，验证集 9,936 条。
- 约 60% 的验证样本不可回答（answers 为空），模型需学会区分作答/不答。

## 2. 方法流程

```text
离线：huggingface-cli download parquet → load_dataset → 按 title 降采样 → tokenize → 训练
在线推理：pipeline("question-answering") + 自定义 null_score 判断
```

- 模型：`bert-base-chinese` + `AutoModelForQuestionAnswering`
- 空答案标签：start=0, end=0（CLS token 位置）
- 评估：`evaluate.load("squad_v2")`，输出 HasAns/NoAns 分类 F1 和 Exact
- 不答判定：`start_logits[0] + end_logits[0] > best_answer_score + threshold`

## 3. 数据获取

```bash
# 手动下载 parquet 到本地（国内网络推荐）
huggingface-cli download real-jiakai/chinese-squadv2 \
  --repo-type dataset \
  --local-dir ./data \
  --include "*.parquet"
```

```python
from datasets import load_dataset
raw_dataset = load_dataset("parquet", data_files={
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
})
```

## 4. 训练

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    "rag-qa-base-bert",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 数据降采样

每个 title 最多取 100 条，防止高频文档主导训练：

```python
import numpy as np

def sample_by_title(dataset, max_per_title=100):
    rng = np.random.default_rng(42)
    title_to_idx = {}
    for i, title in enumerate(dataset["title"]):
        title_to_idx.setdefault(title, []).append(i)
    indices = []
    for idxs in title_to_idx.values():
        indices.extend(idxs if len(idxs) <= max_per_title
                       else rng.choice(idxs, max_per_title, replace=False).tolist())
    return dataset.select(sorted(indices))
```

## 5. 评估

```python
import evaluate
metric = evaluate.load("squad_v2")

metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_dataset["validation"])
```

指标含义：
- `exact` / `f1`：整体精确匹配度和 F1
- `HasAns_exact` / `HasAns_f1`：可回答子集上的分数
- `NoAns_exact`：不可回答子集的准确率（预测空答案的比例）
- `best_exact` / `best_f1`：遍历阈值后的最优分

## 6. 核心问题与解决

### 空答案处理

训练集中 answers 可能为空 `{"text": [], "answer_start": []}`。预处理时检测到空列表直接标 `(0, 0)`，损失函数中 `[CLS]` 位置对应"不答"信号。

### compute_metrics 三个坑

1. **squad_v2 需要 `no_answer_probability`**：预测字典必须包含此字段。
2. **null_score 取错 feature**：多 chunk 样本须在循环内跟踪 `best_null_score`，不能取最后一个 feature 的 CLS 分。
3. **softmax 数值溢出**：`np.exp(large_logit)` 会爆成 `inf`，先减 `scores.max()` 再算。

### Pipeline 默认任务错误

`pipeline(model="bert-base-chinese")` 默认走 `fill-mask`，必须显式指定 `pipeline("question-answering", ...)`。

### 自建"不答"推理

pipeline 永远返回最高分 span，不会主动输出空答案。需手动比较 CLS score 和最佳 span score：

```python
start_logits = outputs.start_logits[0].detach().cpu().numpy()
end_logits = outputs.end_logits[0].detach().cpu().numpy()
null_score = start_logits[0] + end_logits[0]
# 遍历 top spans 取最佳非空得分 best_score
if null_score > best_score:
    return {"answer": "", "no_answer": True}
```

注意：tensor 在 GPU 上需 `.detach().cpu().numpy()`，且 tokenizer 输出的 tensor 需 `.to(device)`。

## 7. 上传到 Hugging Face Hub

```python
# 训练完推送模型和 tokenizer
trainer.push_to_hub(commit_message="Training complete", tags="question-answering")
# 或单独推送
model.push_to_hub("rag-qa-base-bert")
tokenizer.push_to_hub("rag-qa-base-bert")
```

模型卡片通过 Hub 网页编辑，填模型用途、评估结果、数据集和注意事项。

## 8. 模型卡片

- 模型：[Kate-lf/rag-qa-base-bert](https://huggingface.co/Kate-lf/rag-qa-base-bert)
- 基座：bert-base-chinese
- 数据集：real-jiakai/chinese-squadv2（降采样后 43,188 训练 / 6,859 验证）
- 评估结果：见 metrics 输出
