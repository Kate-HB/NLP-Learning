# RAG

RAG 是 Retrieval-Augmented Generation，即检索增强生成。系统先把本地文档切成 chunk 并生成 embedding；收到问题后，将问题也转换为向量，通过余弦相似度找出 Top-k 相关 chunk。

检索到的 chunk 会与问题一起放入提示词，再交给生成模型回答。上下文限制了模型可使用的事实范围，引用来源让使用者可以核验，因此 RAG 能降低但不能彻底消除幻觉。

chunk 太小会丢失上下文，太大则会混入无关内容并浪费上下文窗口。Top-k 太小可能漏掉证据，太大则会引入噪声。实际项目需要用带标准答案的问题集调节 chunk size、overlap、Top-k 和拒答阈值。
