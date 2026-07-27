# P1 中文文本情感分类

任务：输入一条中文文本，输出情绪标签（伤心/关心/厌恶/平静/惊讶/开心/生气/疑问）和置信度。项目经历了从合成数据到真实对话数据的完整微调流程。

## 1. 问题与数据

- 任务类型：单标签、8 分类情感识别。
- 数据集：[zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset)，4,159 条中文对话，真实标注。
- 数据划分：分层抽样 70/15/15 → train 2,911 / val 624 / test 624。
- 标签映射：

| label | 情绪 |
|-------|------|
| 0 | 伤心 |
| 1 | 关心 |
| 2 | 厌恶 |
| 3 | 平静 |
| 4 | 惊讶 |
| 5 | 开心 |
| 6 | 生气 |
| 7 | 疑问 |

## 2. 方法流程

```text
CSV 原始数据
  → 分层切分 train/val/test
  → BERT tokenizer 编码（truncation=True）
  → DataCollatorWithPadding 动态填充
  → bert-base-chinese + 8 分类 classification head
  → Trainer 微调（3 epochs, lr=2e-5）
  → accuracy / macro-F1 / 混淆矩阵 / 分类报告
```

## 3. 运行方式

Notebook 方式（推荐）：`text-classification.ipynb`，按 cell 顺序执行。

关键依赖：

```powershell
pip install transformers datasets evaluate scikit-learn matplotlib pandas
```

## 4. 指标与结果

bert-base-chinese，3 epochs，learning_rate=2e-5：

| 数据集 | accuracy | macro-F1 |
|--------|----------|----------|
| train (2,911) | 0.9763 | 0.9759 |
| val (624) | 0.8830 | 0.8814 |
| test (624) | 0.8718 | 0.8701 |

训练前 baseline（随机分类头）：accuracy 约 12.5%（1/8 随机水平），证明微调有效。

## 5. 核心代码理解

### tokenizer、collator、model 三者关系

- `tokenizer(text, truncation=True)` → input_ids + attention_mask，`truncation=True` 防止超长文本溢出 512。
- `DataCollatorWithPadding` 每个 batch 动态填充到 batch 内最长，无需全局 max_length。
- `model` 接收 padded batch 做 forward，Trainer 自动区分 `input_ids`（给模型）和 `label`（给 loss）。

### 为什么 `ignore_mismatched_sizes=True`

预训练模型的 classification head 权重形状（如 5 分类）与当前任务（8 分类）不匹配。此参数跳过形状不匹配的层，用随机初始化替代。BERT 主体权重保留预训练值，分类头从零学习——这是所有下游任务的标准做法。

### 模型评估 pipeline

`trainer.evaluate()` → 验证集指标；
`trainer.predict()` → 返回 logits + label_ids，用于混淆矩阵和分类报告。

## 6. 踩坑记录

见 [notes/day14.md](../../notes/day14.md)，包含：

- 合成数据 100% 准确率的排查
- `evaluate` 库与本地文件同名冲突
- label 映射顺序错误（`zip(set())` 无序）
- 训练/验证集数据重叠
- 中文绘图乱码
- `push_to_hub=True` 导致 KeyboardInterrupt
