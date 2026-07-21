# BERT 做命名实体识别（NER）

## Token 分类 vs 序列分类

| | Sequence Classification | Token Classification |
|---|---|---|
| 输出 | 一个标签（整个句子） | 每个 token 一个标签 |
| 典型任务 | 情感分析、文本分类 | NER、POS、Chunking |
| Head | `ForSequenceClassification` | `ForTokenClassification` |
| 标签粒度 | 句子级 | token 级 |

## IOB2 标注

实体标签 = 位置前缀 + 实体类别：

| 前缀 | 含义 |
|---|---|
| B- | 实体开头（Begin） |
| I- | 实体内部（Inside） |
| O | 非实体（Outside） |

CoNLL-2003 共 9 标签：`O` + `B/I-PER` + `B/I-ORG` + `B/I-LOC` + `B/I-MISC`。

相邻同类型实体需要边界区分：`B-PER I-PER O B-PER I-PER` → 两个人；全标 `PER` → 无法区分。

## 标签对齐：子词分词的挑战

BERT tokenizer 可能把单词切成子词（"lamb" → "la" + "##mb"），但原始标签是词级别的。

### 三步解决

1. **预分词输入**：`tokenizer(word_list, is_split_into_words=True)`
2. **获取映射**：`inputs.word_ids()` → `[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]`
3. **对齐规则**：
   - `word_id` 为 `None`（特殊 token）→ 标签 `-100`
   - 同一单词的首个 token → 保留原标签
   - 同一单词的后续 token → 若原标签为 B-XXX，改为 I-XXX
   - `-100` 被交叉熵损失自动忽略

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:  # B-XXX → I-XXX
                label += 1
            new_labels.append(label)
    return new_labels
```

## 数据整理：DataCollatorForTokenClassification

区别于 `DataCollatorWithPadding`（只填充输入），NER 整理器用 `-100` 填充标签，与对齐函数中忽略特殊 token 的值一致。

## 评估：seqeval（实体级）

NER 不能直接用 accuracy——O 标签占绝大多数会虚高。seqeval 按**实体边界 + 类别完全匹配**才算正确：

- 需要字符串标签而非整数 ID
- 返回每个实体类别的 precision/recall/F1 + overall
- 评估前需过滤 `-100` 标签

## 环境注意事项

- **datasets 5.0.0**：`load_dataset("conll2003")` 需加 `revision="refs/convert/parquet"`
- **Transformers 4.57**：`TrainingArguments` 参数名 `evaluation_strategy` → `eval_strategy`
