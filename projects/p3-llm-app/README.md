# P3 LLM 应用 Demo

用一个 Gradio 页面整合 P1 中文文本分类和 P2 RAG 文档问答。页面不重新实现模型，只调用两个项目的稳定服务接口。

## 1. 输入输出

- 分类 Tab：输入中文文本；输出标签、置信度、所有类别概率。
- RAG Tab：输入问题和 Top-k；输出回答、引用文件、chunk 编号、相似度。
- 置信度低于 60% 时提示人工复核；RAG 低于检索阈值时明确拒答。
- 数据与模型完全复用 P1 的分类 CSV/checkpoint 和 P2 的本地文档/向量索引，P3 不复制训练数据。

## 2. 前置准备

在仓库根目录执行：

```powershell
python -m venv .venv-projects
.\.venv-projects\Scripts\Activate.ps1
python -m pip install -r projects/requirements.txt

cd projects/p1-text-classification
python prepare_data.py
python train.py

cd ../p2-rag-qa
python ingest.py
```

P3 默认读取：

- P1 模型：`projects/p1-text-classification/outputs/bert-best/`
- P2 索引：`projects/p2-rag-qa/outputs/index/`

## 3. 启动

```powershell
cd projects/p3-llm-app
python app.py
```

打开终端输出的本地地址。运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 4. 文件职责

- `classify.py`：懒加载 P1 模型，暴露 `classify(text)`。
- `rag.py`：懒加载 P2 embedding 与索引，暴露 `answer(question, top_k)`。
- `app.py`：只负责输入校验、事件绑定和结果展示。
- `design.md`：模块边界和数据流。
- `demo_notes.md`：复习讲解稿与演示步骤。

## 5. 结果与限制

单元测试验证模块接口和展示逻辑；真实分类指标来自 P1 的 `metrics.json`，检索结果直接显示每条相似度。应用本身不创造新指标。

集成验证已让 P3 在同一进程中加载 P1 的 BERT checkpoint 和 P2 的真实向量索引：分类返回置信度，RAG 返回 `rag.md#chunk-1` 及 0.6401 的相似度；Gradio `Blocks` 页面可成功构建。

最大限制是上游质量：分类器受小样本限制，RAG 受文档、切块、embedding 和生成器限制。UI 只能把不确定性与来源展示出来，不能修复错误模型或错误知识。

详细设计见 [design.md](design.md)，演示与复习问答见 [demo_notes.md](demo_notes.md)。

## 6. 错误分析与后续改进

- 分类错误：回到 P1 的 `errors.csv`，检查标签混淆和低置信度样本。
- 问答错误：先确认 P2 Top-k 是否包含正确证据，再判断生成器是否误用证据。
- 服务错误：区分模型/索引缺失、依赖未安装和运行时推理异常；页面会直接显示原因。
- 后续先增加真实数据与评测集，再加入模型缓存状态、批量评测页和引用一致性校验。
