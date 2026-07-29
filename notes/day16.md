# Day 16：Gradio 交互页面——将 NLP 模型封装为 Web 应用

用 Gradio 将 P1 中文情绪分类和 P2 中文问答模型封装成交互式 Web 页面，并探索 HuggingFace Spaces 三种部署方式。

## 1. Gradio 最小示例

### `gr.Interface` 三要素

```python
import gradio as gr

def predict(text):
    return f"你输入了：{text}"

gr.Interface(fn=predict, inputs="textbox", outputs="text").launch()
```

`fn` 是处理函数，`inputs` 定义输入控件类型，`outputs` 定义输出显示方式。`launch()` 不带参数时默认启动本地服务器 `http://127.0.0.1:7860`。

### 核心概念

- `fn` 的参数个数必须和 `inputs` 列表长度一致
- `fn` 的返回值类型必须和 `outputs` 匹配
- `inputs` 和 `outputs` 可以是字符串简写（`"textbox"`、`"text"`），也可以是组件实例（`gr.Textbox(label="...", lines=5)`）
- `examples` 参数提供点击即可填充的示例，降低使用门槛

## 2. 封装 Pipeline 为 Gradio 应用

### 分类模型

之前 P1 的 8 类中文情绪分类模型推送到了 `Kate-lf/emotion-classification`。用 `pipeline` 加载后包装一个 `predict` 函数传入 `gr.Interface`：

```python
from transformers import pipeline
import gradio as gr

model = pipeline("text-classification", model="Kate-lf/emotion-classification")

def predict(text):
    result = model(text)[0]
    return f"{result['label']}（置信度: {result['score']:.2%}）"

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title="句子情绪分类",
    description="输入句子，判断属于 伤心 / 关心 / 厌恶 / 平静 / 惊讶 / 开心 / 生气 / 疑问",
    examples=[["曾经最好的朋友，现在连点赞都不会了。"]],
).launch()
```

分类 pipeline 返回 `[{"label": "伤心", "score": 0.96}]`，取 `[0]` 再格式化展示。`{score:.2%}` 把浮点数转为百分比格式（0.96 → "96.00%"）。

### 问答模型

问答模型输入是**上下文 + 问题**两个参数，与分类的单输入不同。`inputs` 传一个列表对应两个参数：

```python
model = pipeline(model="Kate-lf/rag-qa-base-bert")

def predict(context, question):
    result = model(question=question, context=context)
    score = result["score"]
    answer = result["answer"]
    if score < 0.1 or not answer.strip():
        return "上下文不包含答案，无法回答。"
    return f"答案：{answer}\n\n置信度：{score:.2%}"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="上下文", lines=5),
        gr.Textbox(label="问题"),
    ],
    outputs="text",
    title="抽取式问答",
    description="输入上下文和问题，模型从上下文中抽取答案。",
    examples=[
        ["苏轼是北宋著名文学家，号东坡居士。", "苏轼的号是什么？"],
    ],
).launch()
```

QA pipeline 返回 `{"score": 0.87, "answer": "东坡居士", "start": 20, "end": 24}`，取 `answer` 和 `score` 字段。分数过低或无内容时返回"无法回答"。

### 生成公网链接

```python
demo.launch(share=True)
```

Gradio 用 Ngrok 穿透生成 `*.gradio.live` 临时公网链接，72 小时有效。不需要服务器、不需要公网 IP。

## 3. HuggingFace Spaces 部署

### 三种方式对比

| 方式 | 原理 | 配额限制 | 网络要求 |
|------|------|------|------|
| Gradio Space | HF 服务器跑 Python + Gradio | 免费 2 个 CPU | 访问公网即可 |
| Static Space + Inference API | 静态 HTML，JS 调 API | 不限，API 免费额度 | 需访问 `api-inference.huggingface.co` |
| 本地 + share=True | 本地跑模型，Gradio 穿透 | 无 | 本地能跑模型 |

### Gradio Space 完整流程

**创建 Space**：访问 [huggingface.co/new-space](https://huggingface.co/new-space)，SDK 选 **Gradio**，Hardware 选 **CPU (free)**。创建后 Space 是一个独立的 Git 仓库，地址为 `https://huggingface.co/spaces/<用户名>/<Space名>`。

**添加 `requirements.txt`**：

```
transformers
torch
gradio
```

Space 构建时自动 `pip install -r requirements.txt`，不需要指定版本号（用最新稳定版）。

**Space 的 `README.md`** 用 YAML 头声明 SDK 类型：

```yaml
---
title: Emotion Classification
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
---
```

`app_file` 指定入口脚本，默认 `app.py`。`sdk_version` 固定 Gradio 版本避免未来 API 变动。

**推送**：在项目目录初始化为 Git 仓库，关联远端，推送到 `main` 分支：

```bash
cd projects/p3-llm-app
git init
git remote add origin https://huggingface.co/spaces/<用户名>/<Space名>
git add app.py requirements.txt
git commit -m "Initial commit"
git push origin master:main
```

推送后 Space 自动构建。构建日志在页面顶部的 **Builder** 标签查看。首次构建 + 模型下载约 2-3 分钟。

**免费配额**：同一账户最多 2 个 CPU Space 同时运行。报"配额已达上限"时去 HF Settings 暂停不用的 Space。

### Static Space 方案与问题

Static Space 不支持后端代码，只托管 HTML/CSS/JS。推理靠浏览器 JS 发 HTTP POST 到 `api-inference.huggingface.co/models/<model-id>`，HF 后端自动加载模型返回结果。

**前置条件**：模型必须正确设置 `pipeline_tag`。例如分类模型推送时被误标为 `summarization`（`trainer.push_to_hub(tags="summarization")`），导致 Inference API 按摘要任务调用，返回错误结果。在模型页面 Settings → Pipeline tag 修改为 `text-classification`。

**关键问题**：`api-inference.huggingface.co` 在国内直连被阻断（`ERR_CONNECTION_CLOSED`）。Static Space + Inference API 方案在国内基本不可用。推荐 Gradio Space——模型运行在 HF 服务器上，用户只访问网页，不存在 API 阻断问题。

### `gr.Blocks` 与 `gr.Tab`

Gradio 除 `Interface` 外还有 `Blocks`——低成本 API，能做多 Tab 页面、条件逻辑、状态管理等复杂交互。P3 原始设计用 `gr.Blocks` + `gr.Tab` 把分类和问答放同一个页面，通过标签切换。

```python
with gr.Blocks(title="NLP Demo") as demo:
    gr.Markdown("# NLP 学习项目 Demo")
    with gr.Tab("中文文本分类"):
        text = gr.Textbox(label="中文文本")
        classify_btn = gr.Button("开始分类")
        result = gr.Markdown()
        classify_btn.click(classify_handler, inputs=text, outputs=result)
    with gr.Tab("RAG 文档问答"):
        question = gr.Textbox(label="问题")
        ask_btn = gr.Button("检索并回答")
        answer = gr.Markdown()
        ask_btn.click(rag_handler, inputs=question, outputs=answer)
```

`Blocks` 的 `click()` 事件绑定比 `Interface` 的自动映射更灵活，适合多步骤、多组件的应用。

## 4. Transformers 5.x 版本兼容

### huggingface-hub 版本冲突

`transformers==4.57.3` 固定要求 `huggingface-hub>=0.34.0,<1.0`，Conda 环境默认装 1.x 导致导入失败：

```
ImportError: huggingface-hub>=0.34.0,<1.0 is required,
but found huggingface-hub==1.25.1
```

三种解法：
- 升级 transformers：`pip install transformers -U`，4.58+ 解除了 <1.0 限制
- 降级 huggingface-hub：`pip install "huggingface-hub<1.0"`
- 新建隔离环境单独装新版全家桶

这里选升级方案，升到 5.14.1。

### `question-answering` 任务名移除

Transformers 5.x 不再注册 `question-answering` 任务名，`pipeline("question-answering", ...)` 报 `KeyError: Unknown task`。去掉 task 参数让 pipeline 根据模型 config 自动推断：

```python
# 5.x 报错
model = pipeline("question-answering", model="Kate-lf/rag-qa-base-bert")

# 正确：自动推断
model = pipeline(model="Kate-lf/rag-qa-base-bert")
```

模型 config 声明了 `"architectures": ["BertForQuestionAnswering"]`，pipeline 据此匹配正确的任务实现。这套自动推断机制对所有 HuggingFace 模型都适用——只要 `config.json` 里有 `architectures` 字段。

## 今日知识总结

### Gradio 是模型到产品的最后一公里

`gr.Interface(fn, inputs, outputs).launch()` 三行代码把一个 Python 函数变成网页。不需要写 HTML/CSS/JS，不需要理解 HTTP 协议。核心是把模型函数的输入输出签名映射到 Gradio 组件——单输入用字符串简写，多输入用组件列表。

### Pipeline 的输出格式决定 predict 函数的写法

分类返回 `[{"label": ..., "score": ...}]`，取第一个元素格式化展示。问答返回 `{"answer": ..., "score": ...}`，需要额外处理无答案情况。`predict` 函数在模型输出和界面展示之间做格式转换——把浮点数转成百分比、把空答案转成提示语、把多个字段拼成易读文本。

### 部署选型取决于网络和持久化需求

本地 `share=True` 最快但不持久（72 小时）；Gradio Space 正式但需 GPGPU 配额（免费 2 个 CPU）；Static Space 轻量但在国内因 API 阻断不可用。国内推荐用 Gradio Space——模型跑在 HF 服务器上，不经过被阻的 Inference API 域名。

### 版本升级会产生连锁问题

升级 transformers 修复了 huggingface-hub 冲突，但也移除了 `question-answering` 任务名——写死的 task 字符串在新版报错。解决方式是依赖 pipeline 的自动推断而非硬编码任务名，这是 HuggingFace 设计上的容错机制。每次升级后跑关键 notebook 验证，不假设"小版本升级完全兼容"。
