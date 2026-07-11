# P3 LLM 应用 Demo

目标：把 P1 分类和 P2 RAG 整合成一个可交互 Demo。

## 学习目的

- 学会把模型能力封装成应用。
- 学会组织项目 README、运行命令和错误分析。
- 学会向别人讲清楚 NLP 项目。

## 文件规划

```text
p3-llm-app/
├── README.md
├── app.py
├── classify.py
├── rag.py
├── design.md
└── demo_notes.md
```

## 最低功能

1. Gradio 页面。
2. 中文文本分类 Tab。
3. RAG 文档问答 Tab。
4. 输出置信度和引用来源。
5. README 能复现。

## 验收问题

1. 分类模块输入输出是什么。
2. RAG 模块输入输出是什么。
3. UI 如何展示模型不确定性。
4. 项目最大限制是什么。

