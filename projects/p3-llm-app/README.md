# P3 LLM 应用 Demo

将 P1 中文情绪分类和 P2 中文问答模型封装为 Gradio 交互页面，并部署到 HuggingFace Spaces。

## 文件

| 文件 | 用途 |
|------|------|
| `text-classification.py` | 情绪分类 Gradio 应用，加载 `Kate-lf/emotion-classification` |
| `rag-qa.py` | 抽取式问答 Gradio 应用，加载 `Kate-lf/rag-qa-base-bert` |
| `llm-app.ipynb` | 开发 Notebook，包含 Pipeline 调用和 Gradio 实验 |

## 启动

### 分类应用

```bash
cd projects/p3-llm-app
python text-classification.py
```

打开 http://127.0.0.1:7860 ，输入中文句子，返回 8 类情绪标签和置信度。

### 问答应用

```bash
cd projects/p3-llm-app
python rag-qa.py
```

输入上下文和问题，模型从上下文中抽取答案。无答案时返回"无法回答"。

### 生成公网链接

```python
demo.launch(share=True)
```

会生成一个 `*.gradio.live` 临时公网链接，72 小时有效，无需服务器。

## 模型

- 分类：`Kate-lf/emotion-classification`（bert-base-chinese，8 类中文情绪）
- 问答：`Kate-lf/rag-qa-base-bert`（bert-base-chinese，Chinese-SQuAD v2）

## 踩坑记录

### Transformers 版本与 huggingface-hub 冲突

`transformers==4.57.3` 要求 `huggingface-hub<1.0`，新版 Conda 环境默认装 1.x 导致 ImportError。解决方案：升级 `transformers` 到 5.x。

### Transformers 5.x 移除 `question-answering` 任务名

`pipeline("question-answering", model="...")` 报 `KeyError: Unknown task`。修复：去掉 task 参数，让 pipeline 从模型 config 自动推断——`pipeline(model="Kate-lf/rag-qa-base-bert")`。

### HuggingFace Spaces 三种部署方式对比

| 方式 | 原理 | 配额 | 网络要求 |
|------|------|------|------|
| Gradio Space | HF 服务器跑 Python + Gradio | 免费 2 个 CPU | 访问公网即可 |
| Static Space + Inference API | 静态 HTML 调用 HF API | 免费，不限 | 需能访问 `api-inference.huggingface.co` |
| 本地 + share=True | 本地跑模型，Gradio 穿透 | 无限制 | 本地能跑模型 |

Static Space 方案在国内网络下不可用，`api-inference.huggingface.co` 被阻断。推荐用 Gradio Space 或本地 `share=True`。
