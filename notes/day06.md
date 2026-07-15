# Day 06：手动训练循环

Day 05 用 Trainer 完成了 MRPC 微调。Day 06 拆开 Trainer 黑盒，手写 PyTorch 训练循环，理解每一步在做什么。

## 数据准备

Trainer 自动处理了数据清理，手动训练需要自己来：

```python
# 删除模型不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 列名改为模型期望的 "labels"
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 返回 PyTorch 张量而非列表
tokenized_datasets.set_format("torch")
```

构建 DataLoader，`collate_fn=data_collator` 让 `DataCollatorWithPadding` 处理动态 padding：

```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

验证一个 batch 的形状：`{k: v.shape for k, v in batch.items()}`。

## 模型与优化器

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
```

`num_labels=2`：在预训练 BERT 上加一个二分类头，输出 2 个 logits。SST-2 是二分类（正/负面），MRPC 也是二分类（等价/不等价）。

AdamW 是 BERT 微调的标配优化器，与 Adam 相同但加入了权重衰减正则化。

## 学习率调度器

默认使用线性衰减（linear decay）：从 5e-5 线性衰减到 0。总步数决定衰减斜率：

```
总步数 = epochs × len(train_dataloader)
      = epochs × (数据集大小 ÷ batch_size)
```

```python
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

每走一步，学习率等量递减。epoch 越少、batch size 越大，总步数越少，衰减越快。

**三者关系**（以 MRPC 为例，训练集约 3668 条）：

| 参数变化 | 总步数 | 衰减速度 |
|---------|--------|---------|
| epoch=3, bs=8 | 1377 | 基准 |
| epoch=1, bs=8 | 459 | 3倍快 |
| epoch=3, bs=16 | 689 | 2倍快 |

## 训练循环

```python
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

每一步做的事：

1. `model(**batch)` — 前向传播，计算 logits 和 loss
2. `loss.backward()` — 反向传播，计算梯度
3. `optimizer.step()` — 根据梯度更新权重
4. `lr_scheduler.step()` — 衰减学习率
5. `optimizer.zero_grad()` — 清零梯度，避免累积

**注意**：这段纯 PyTorch 循环不保存任何东西。模型权重、日志全在内存里，进程结束就没了。需要手动 `model.save_pretrained()`。而 Trainer 自动保存到 `output_dir`。

## 评估循环

```python
import evaluate
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

关键点：
- `model.eval()` — 切换评估模式，关闭 dropout
- `torch.no_grad()` — 不计算梯度，节省显存
- `metric.add_batch()` — 逐 batch 累积，最后 `compute()` 汇总

## Accelerate

同一份代码，单卡、多卡、TPU 都能跑。核心区别只有一行：

```python
# PyTorch 原生
loss.backward()
# Accelerate
accelerator.backward(loss)
```

然后用 `accelerator.prepare()` 自动处理设备分发，`accelerate config` + `accelerate launch train.py` 启动分布式训练。

Trainer 内部已集成 Accelerate，不需要自己写。只有需要比 Trainer 更细粒度控制时才手写 Accelerate 循环。

## 分布式训练

把训练拆分到多张 GPU 并行跑。

- **数据并行**（最常见）：每张卡持有完整模型副本，吃不同 batch 子集，梯度平均后同步更新
- **模型并行**：模型太大一张卡放不下，切分到不同卡上

BERT-base（1.1 亿参数）+ MRPC/SST-2 单张 4G 显存就够，不需要分布式。

## 今日知识总结

### 手动训练循环 vs Trainer

Trainer 封装了数据准备、训练循环、评估、保存。手动循环用纯 PyTorch，每一步（forward、backward、optimizer step、scheduler step、zero_grad）都要自己写。手动循环不会自动保存模型和日志。

### 微调的本质

不是"数据集 + 预训练模型 + 参数"拼在一起跑。而是：拿预训练权重当起点（不是随机初始化），用自己的数据继续训练，反向传播更新所有权重，让模型从通用语言理解转向具体任务能力。继承了预训练的语法语义知识，少量数据就能收敛。

### 学习率调度器

默认线性衰减，从 5e-5 到 0，总步数 = epochs × (数据集大小 / batch_size)。epoch 数和 batch size 共同决定总步数，进而决定衰减斜率。习惯上 batch size 翻倍时学习率也翻倍（线性缩放规则）。

### 评估循环要点

`model.eval()` 关 dropout，`torch.no_grad()` 省显存，`metric.add_batch()` 逐 batch 累积，最后 `metric.compute()` 汇总。这和 Trainer 的 `compute_metrics` 回调做的事一样，只是写法不同。

### Accelerate 的价值

让同一份训练代码在单卡、多卡、TPU 上都能跑。`accelerator.prepare()` 自动处理设备，`accelerator.backward()` 替代 `loss.backward()`。Trainer 内部已集成，通常不需要手写。

### 分布式训练场景

BERT-base 级别单卡够用。真正需要分布式的是大模型（单卡放不下）或大数据（单卡训练太慢）。数据并行是最常见方式。

### 下一步

Day 07：Week 1 复盘，重跑 Day 02-06，画出完整推理和训练流程图。
