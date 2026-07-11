# P2 RAG 文档问答

目标：输入问题，从本地文档中检索相关片段，再生成带引用来源的回答。

## 学习目的

- 理解 embedding 和语义检索。
- 理解 chunk、Top-k、上下文拼接。
- 理解 RAG 为什么能降低幻觉。

## 文件规划

```text
p2-rag-qa/
├── README.md
├── ingest.py
├── retrieve.py
├── ask.py
├── data/
└── error_analysis.md
```

## 最低功能

1. 读取本地文档。
2. 切分为 chunk。
3. 生成 embedding。
4. 根据 query 返回 Top-k。
5. 回答时显示引用来源。

## 验收问题

1. embedding 表示什么。
2. chunk 太大或太小有什么问题。
3. Top-k 如何影响回答。
4. 检索失败会造成什么后果。

