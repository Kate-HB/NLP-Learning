# P2 RAG 文档问答

输入问题，先从本地 Markdown/TXT 文档检索相关 chunk，再返回带 `[数字]` 引用的回答。默认离线摘录回答；配置 OpenAI 兼容接口后，可由 LLM 基于检索上下文生成。

## 1. 问题与数据

- 输入：自然语言问题。
- 输出：答案、引用来源、Top-k 文档片段和余弦相似度。
- 示例知识库：`data/bert.md`、`data/rag.md`、`data/transformer.md`。
- 支持格式：UTF-8 编码的 `.md`、`.txt`，会递归读取子目录。

## 2. 方法流程

```text
离线入库：文档 → 重叠切块 → embedding → vectors.npy + metadata.json
在线问答：问题 → embedding → 余弦 Top-k → 阈值判断 → 生成/拒答 → 引用
```

- `ingest.py`：读取文档、切 chunk、生成并保存 embedding。
- `retrieve.py`：加载索引、计算余弦相似度、返回 Top-k。
- `ask.py`：构造有引用编号的上下文，生成回答或低分拒答。

## 3. 安装与运行

```powershell
python -m venv .venv-projects
.\.venv-projects\Scripts\Activate.ps1
python -m pip install -r projects/requirements.txt
cd projects/p2-rag-qa
python ingest.py
python retrieve.py "RAG 为什么能降低幻觉" --top-k 3
python ask.py "chunk 太大会有什么问题"
python -m unittest discover -s tests -v
```

首次执行 `ingest.py` 会下载 `shibing624/text2vec-base-chinese`。索引默认写入 `outputs/index/`。

### 可选：接入生成模型

PowerShell 当前会话设置三个变量后，`ask.py` 自动使用 OpenAI 兼容接口：

```powershell
$env:LLM_BASE_URL="https://你的服务地址/v1"
$env:LLM_API_KEY="你的密钥"
$env:LLM_MODEL="模型名称"
python ask.py "BERT 为什么适合文本分类"
```

密钥只从环境变量读取，不写入仓库。未配置时返回最相关原文并保留引用，整个检索链仍可学习和验证。

## 4. 参数与结果

- `--chunk-size 300`：每块最多 300 个字符。
- `--overlap 50`：相邻块重复 50 个字符，保护边界信息。
- `--top-k 3`：向生成器提供最相关的 3 块。
- `--min-score 0.15`：最高相似度低于阈值时拒答。

入库后 `outputs/index/manifest.json` 记录 chunk 数和向量维度。每次问答返回 `score`，可据此观察检索质量。阈值不应直接照搬到真实数据，应使用评测问题集调参。

本仓库在 2026-07-22 的集成验证生成 4 个 chunks。“RAG 为什么能降低幻觉”的首条来源为 `rag.md`，相似度 0.4875；“chunk 太大会有什么问题”的首条来源为 `rag.md`，相似度 0.6401。分数只用于同一 embedding 模型内排序，不是答案正确率。

## 5. 核心概念

### embedding 表示什么

embedding 是文本的连续数值表示。模型训练目标使语义相近文本在向量空间中方向接近；它不是原文压缩包，单个维度通常没有可直接命名的含义。

### chunk 大小与 overlap

chunk 太小会切断论证和指代；太大会混入多个主题、降低检索精度并占用 LLM 上下文。overlap 能保护边界，但过大将产生重复结果并增大索引。

### Top-k 如何影响回答

Top-k 太小容易漏证据；太大会加入噪声，甚至让冲突片段误导生成。应同时评估“证据是否被召回”和“最终回答是否正确”。

### RAG 为什么只能降低幻觉

检索提供可核验事实和边界，但检索可能失败，生成模型也可能忽略上下文。本项目增加引用与低分拒答，使失败更容易发现，不能保证答案绝对正确。

## 6. 错误分析与改进

详见 [error_analysis.md](error_analysis.md)。优先建立评测集，再考虑 reranker、混合检索、FAISS 或更强生成模型。
