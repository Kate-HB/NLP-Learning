# 语义搜索：嵌入向量 + FAISS

## 什么是语义搜索

传统关键词搜索只匹配字面，语义搜索理解**意思**。查询"如何离线加载数据集"能匹配到"load a dataset without internet"——即使没有共同词汇。

核心流程：文本 → 嵌入向量 → 相似度比较 → 返回最相似文档。

## 文本嵌入

### CLS Pooling

BERT/MPNet 给每个 token 一个 hidden state。要得到**整句**的向量，取 `[CLS]` token 的最后一层 hidden state：

```python
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]  # (batch, hidden_dim)
```

`[CLS]` 在预训练时被训练为"整个序列的摘要"。

### 嵌入生成最佳实践

- 批处理：`batched=True, batch_size=32`
- 关闭梯度：`torch.no_grad()`
- 模型放 GPU：`model.to(device)`
- 返回 numpy：`.detach().cpu().numpy()`

```python
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    return cls_pooling(model_output).detach().cpu().numpy()

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": list(get_embeddings(x["text"]))},
    batched=True,
    batch_size=32
)
```

## FAISS 相似度搜索

### 为什么用 FAISS

遍历所有向量找最近的 K 个是 O(n) 复杂度。FAISS 建索引后近似搜索 O(log n)，数百万向量也能毫秒级返回。

### 基础用法

```python
# 1. 装 faiss-cpu (Windows 无 faiss-gpu)
embeddings_dataset.add_faiss_index(column="embeddings")

# 2. 编码查询
question_embedding = get_embeddings([question])  # (1, 768)

# 3. 搜索
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
```

### 非对称语义搜索

短查询 vs 长文档 → 嵌入向量不在同一空间分布。应选用针对这种场景训练的模型，如 `sentence-transformers/multi-qa-mpnet-base-dot-v1`。

查询用原始向量，文档用索引加速检索，点积（dot product）作为相似度得分。

## 数据集准备流程

GitHub Issues → 过滤（排除 PR、无评论、短评论） → explode 评论 → 拼接标题+正文+评论为 text → 嵌入 → FAISS 索引 → 查询。

关键：每行应是一个 (title, body, comment, url) 四元组，而非每条 issue 一行带评论列表。
