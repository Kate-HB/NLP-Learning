# Week 8：LLM 应用整合项目

目标：完成 P3：整合 P1 中文分类和 P2 RAG 的 LLM 应用 Demo。

## 本周资源

| 资源 | 链接 |
|---|---|
| Gradio Quickstart | https://www.gradio.app/guides/quickstart |
| Hugging Face Hub | https://huggingface.co/docs/hub/index |
| GitHub README Docs | https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes |
| Transformers Docs | https://huggingface.co/docs/transformers/index |

## 每日计划

| Day | 学习内容 | 项目任务 | 产出 |
|---|---|---|---|
| Day 50 | 应用设计 | 确定输入、输出、页面结构 | `projects/p3-llm-app/design.md` |
| Day 51 | 接入分类模块 | 调用 P1 推理函数 | `projects/p3-llm-app/classify.py` |
| Day 52 | 接入 RAG 模块 | 调用 P2 检索和问答 | `projects/p3-llm-app/rag.py` |
| Day 53 | Gradio UI | 做两个 Tab：分类、问答 | `projects/p3-llm-app/app.py` |
| Day 54 | 结果展示 | 加入置信度、引用来源、错误提示 | `projects/p3-llm-app/demo_notes.md` |
| Day 55 | README 和讲解稿 | 写复现步骤和项目讲解 | `projects/p3-llm-app/README.md` |
| Day 56 | 总复盘 | 整理 8 周知识图谱 | `notes/week08_summary.md` |

## P3 最低要求

1. 一个 Gradio 入口。
2. 分类功能可运行。
3. RAG 问答功能可运行。
4. 输出包含标签、置信度、引用来源。
5. README 写清楚安装、运行、数据、模型、限制。

## 必须掌握

1. 如何把模型能力封装成函数。
2. 如何把多个 NLP 模块组合成应用。
3. 如何解释项目输入输出。
4. 如何写错误分析。
5. 如何向别人讲清楚项目价值。

