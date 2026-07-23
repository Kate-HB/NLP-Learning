# P1 中文文本分类

输入一条中文新闻标题，输出“科技 / 体育 / 财经”标签、置信度和各类别概率。本项目先训练 TF-IDF baseline，再微调中文 BERT，便于比较传统特征与预训练语义表示。

## 1. 问题与数据

- 任务类型：单标签、多分类。
- 输入：`text` 字符串；输出：`label` 类别。
- 示例数据：`data/raw/news.csv`，3 类、每类 15 条，仅用于跑通流程。
- 实际学习时应替换为规模更大、来源更真实的数据；保持 `text,label` 两列即可。

## 2. 方法流程

```text
原始 CSV
  → 校验、按标签分层切分
  → TF-IDF baseline
  → BERT tokenizer 编码
  → [CLS] 表征进入分类 head
  → logits 经 softmax 变为概率
  → accuracy / macro-F1 / 错误样本
```

文件职责：

- `prepare_data.py`：校验数据，确定性分层切分。
- `baseline.py`：字符 TF-IDF + 逻辑回归。
- `train.py`：使用 `AutoTokenizer`、`AutoModelForSequenceClassification`、`Trainer` 微调。
- `evaluate.py`：计算 accuracy、macro-F1，导出错误样本。
- `predict.py`：封装单句推理。

## 3. 安装与运行

在仓库根目录安装三个项目的共用依赖：

```powershell
python -m venv .venv-projects
.\.venv-projects\Scripts\Activate.ps1
python -m pip install -r projects/requirements.txt
```

进入本项目后按顺序执行：

```powershell
cd projects/p1-text-classification
python prepare_data.py
python baseline.py
python train.py --model-name bert-base-chinese --epochs 3
python evaluate.py
python predict.py "国产芯片公司发布新一代处理器"
python -m unittest discover -s tests -v
```

显存不足时使用 `--batch-size 2`；没有 GPU 也能运行，但 BERT 训练会更慢。

## 4. 指标与结果

- `accuracy`：预测正确数 / 总样本数，直观但容易掩盖类别不平衡。
- `macro-F1`：分别计算每类 F1 后平均，让少数类拥有同等权重。
- baseline 结果：`outputs/baseline/metrics.json`。
- BERT 测试结果：`outputs/evaluation/metrics.json`。
- 错误明细：`outputs/evaluation/errors.csv`。

示例数据很小，指标波动大，不能代表真实泛化能力。脚本会把当前数据和模型的真实结果写入以上文件；下表仅记录一次集成验证，便于复习对照。

本仓库在 2026-07-22 的集成验证结果（`bert-base-chinese`、1 epoch、seed 42）：

| 模型/数据 | accuracy | macro-F1 |
|---|---:|---:|
| TF-IDF baseline / test（3 条） | 0.3333 | 0.1667 |
| BERT / eval（6 条） | 0.6667 | 0.5333 |
| BERT / test（3 条） | 0.6667 | 0.5556 |

测试集仅 3 条，数值只证明流程可运行，不支持模型优劣结论。示例“国产芯片公司发布新一代处理器”被误判为财经，置信度 0.4101；这说明小数据微调后的概率接近，不应把 argmax 当作可靠结论。

## 5. 核心代码理解

### 为什么使用 `[CLS]`

BERT 会在句首加入 `[CLS]`。经过多层双向注意力后，它聚合了整句上下文；分类 head 将该位置的 hidden state 映射为每个类别的分数。它不是天然“句向量”，而是通过微调学会服务当前分类任务。

### logits、label 和概率如何对应

若 `id2label={0:"体育", 1:"科技", 2:"财经"}`，模型输出的 3 个 logits 与编号 0、1、2 一一对应。`argmax` 得到预测编号；softmax 只负责把 logits 转为总和为 1 的相对概率。

### 为什么先做 baseline

baseline 成本低，可快速发现标签错误、数据泄漏或任务本身不可学。只有 BERT 稳定超过 baseline，额外训练成本才有意义。

## 6. 错误分析与改进

详见 [error_analysis.md](error_analysis.md)。优先改进数据规模、类别定义和难例覆盖，再尝试更大模型或更复杂超参数。
