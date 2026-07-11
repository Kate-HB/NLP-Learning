# Week 7：Embedding、语义检索与 RAG 项目

目标：完成 P2：RAG 文档问答。重点理解检索如何减少模型胡编。

## 本周资源

| 资源 | 链接 |
|---|---|
| Sentence Transformers | https://www.sbert.net/ |
| Semantic Search | https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html |
| FAISS | https://faiss.ai/ |
| Transformers RAG Docs | https://huggingface.co/docs/transformers/model_doc/rag |
| LangChain RAG Concepts | https://python.langchain.com/docs/concepts/rag/ |

## 每日计划

| Day | 学习内容 | 项目任务 | 产出 |
|---|---|---|---|
| Day 43 | Sentence Embedding | 句子转向量，算相似度 | `08-RAG/embedding_demo.py` |
| Day 44 | 语义相似度 | 对比关键词和语义检索 | `08-RAG/similarity_demo.py` |
| Day 45 | Top-k 检索 | 返回最相关 5 条文本 | `projects/p2-rag-qa/retrieve.py` |
| Day 46 | chunk 与索引 | 文档切分、保存索引 | `projects/p2-rag-qa/ingest.py` |
| Day 47 | RAG 上下文拼接 | 拼接 query + retrieved docs | `projects/p2-rag-qa/ask.py` |
| Day 48 | P2 README | 写 RAG 流程和引用来源 | `projects/p2-rag-qa/README.md` |
| Day 49 | 复盘 | 分析检索失败案例 | `notes/week07_summary.md` |

## P2 最低要求

1. 能导入一批文档。
2. 能切分 chunk。
3. 能生成 embedding。
4. 能 Top-k 检索。
5. 回答必须带引用来源。
6. README 能解释 RAG 流程。

## 必须掌握

1. embedding 是什么。
2. 余弦相似度是什么。
3. chunk size 为什么重要。
4. RAG 为什么需要引用来源。
5. 检索失败会导致什么问题。

