# P3 设计

## 架构

```text
Gradio app.py
├── ClassificationService → P1 TextClassifier → tokenizer + BERT
└── RAGService → P2 VectorIndex + Embedder → 检索 → 生成/拒答
```

`app.py` 不导入 PyTorch、Transformers 或 SentenceTransformers。两个服务首次收到请求时才加载模型，减少启动等待和无谓显存占用。

## 模块契约

分类输入是非空字符串；输出包含 `label`、`confidence`、`probabilities`。RAG 输入是问题和 Top-k；输出包含 `answer`、`citations`、`retrieved`。UI 只依赖这些字段，因此底层模型可以替换而不改页面。

## 错误处理

缺少 P1 模型或 P2 索引时，页面显示对应的准备命令提示。空输入、模型加载失败和推理异常会显示在当前 Tab，不会导致整个 Gradio 进程退出。

## 为什么展示不确定性

分类概率不是“正确概率”，但可作为模型相对犹豫程度的信号。RAG 相似度也不是答案正确率，因此必须同时展示引用文本位置，让使用者回到原文核验。
