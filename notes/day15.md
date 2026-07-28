# Day 15：BERT 中文问答微调——No-Answer 处理与 HF 数据交互

用 bert-base-chinese 在 Chinese-SQuAD v2 上微调抽取式问答，核心挑战是让模型学会该作答时抽取答案、不该作答时输出空。

## 1. HF 数据手动下载到本地

### 为什么手动下载

国内网络直连 HF Hub 不稳定，`load_dataset("real-jiakai/chinese-squadv2")` 容易超时或下载极慢。用 CLI 下 parquet 到本地再加载最稳。

### 下载与本地加载

```bash
huggingface-cli download real-jiakai/chinese-squadv2 --repo-type dataset --local-dir ./data --include "*.parquet"
```

`--repo-type dataset` 告诉 CLI 这是数据集而非模型仓库。`--include "*.parquet"` 只下载 parquet 文件跳过其他格式。

Python 侧用 `load_dataset("parquet", data_files={...})` 直接从本地加载，完全不走网络：

```python
raw_dataset = load_dataset("parquet", data_files={
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
})
```

每个 parquet 自带 schema，datasets 库自动解析为 Dataset 对象。

### 数据规模与降采样

训练集 90,027 条，验证集 9,936 条，约 60% 验证样本不可回答（answers 为空列表）。高频文档样本过多会主导训练，按 title 分组，每个 title 最多随机取 100 条：

```python
def sample_by_title(dataset, max_per_title=100):
    rng = np.random.default_rng(42)
    title_to_idx = {}
    for i, title in enumerate(dataset["title"]):
        title_to_idx.setdefault(title, []).append(i)
    indices = []
    for idxs in title_to_idx.values():
        indices.extend(idxs if len(idxs) <= max_per_title
                       else rng.choice(idxs, max_per_title, replace=False).tolist())
    return dataset.select(sorted(indices))
```

降采样后训练集 43,188 条，验证集 6,859 条。`random.default_rng(42)` 固定种子保证结果可复现。

## 2. 空答案的处理逻辑

### 训练时：标签标在 CLS 位置

BERT QA 模型预测两个 token 位置：answer_start 和 answer_end。对于不可回答的问题，标签标为 `(0, 0)`，即 `[CLS]` token 的位置。`[CLS]` 的原始功能是聚合整句信息做分类，这里被复用来表示"没有答案"。

预处理时在每个样本开头检查 answer_start 是否为空。如果空列表直接 append `(0, 0)` 并 continue，跳过后续的字符偏移→token 位置计算。跨熵损失最小化 CLS 位置的误差，模型逐渐学到：选 CLS = 不答，选其他位置 = 抽答案。

### 推理时：比较 CLS 得分和最佳 span 得分

HuggingFace 的 `pipeline("question-answering")` 不会主动判断"不答"——它永远返回得分最高的 span，哪怕最高分恰好在 CLS 位置。传入一个完全不相关的问题和上下文做验证：`"太阳在哪"` + `"小明是曹操的爸爸"`，模型输出 `"李逵的妈妈"` 而非空答案，说明 pipeline 没有做 null 比较。

需要手动获取 logits：`start_logits[0][0] + end_logits[0][0]` 作为 null_score，和遍历所有有效 span 找到的最佳非空得分对比。如果 null_score > best_score，返回空字符串并标记 no_answer=True。

这个过程涉及三个设备细节：tokenizer 输出的 tensor 在 CPU 需 `.to(device)` 移到 GPU；GPU tensor 不能直接 `.numpy()`，需先 `.detach().cpu()`；以及排除 CLS 位置（s>0, e>0）避免 null span 和最佳非空比较时把自己也算进去。

## 3. 评估函数 debug：squad_v2 的三个坑

### 指标选择

SQuAD v1 指标不支持不可回答问题，必须用 `evaluate.load("squad_v2")`。v2 返回的字段比 v1 多：`HasAns_exact/f1` 衡量可回答子集，`NoAns_exact` 衡量不可回答子集的准确率，`best_exact/f1` 是遍历 `no_answer_probability` 阈值后的最优分。

### 坑一：预测字典缺 no_answer_probability

squad_v2 的 `metric.compute()` 要求每个预测 dict 中包含 `no_answer_probability` 字段。这是在 add_batch 时校验的，缺了就报 `KeyError`。

squad_v2 内部用这个概率值和可配置的阈值做对比，决定预测是否为空答案。计算方式是取 null_score 和 best_score 做 softmax 归一化，null 那一支的概率就是 `no_answer_probability`。

### 坑二：null_score 取了错误的 feature

长上下文被 tokenizer 切成多个 chunk（feature），每个 chunk 有自己的 CLS。`compute_metrics` 中遍历 example_to_features 收集完所有 span 候选后，null_score 取的是循环结束后残留的那个 `feature_index`，只拿到了最后一个 chunk 的 CLS 分。

修正：在循环内计算每个 feature 的 null_score，用变量跟踪最大值。多 chunk 的样本要以最强的 CLS 信号为准。

### 坑三：softmax 数值溢出

BERT 输出的原始 logit 可能很大（几百甚至上千），`np.exp(700)` 直接溢出为 `inf`。计算 softmax 时先整体减去最大值：`e^(a-m) / (e^(a-m) + e^(b-m))`，其中 m = max(a, b)。数学上等价，计算上安全——最大项变成 `e^0 = 1`，其余 ≤ 1。

## 4. 训练与上传到 HF Hub

### 训练配置

`bert-base-chinese` + `AutoModelForQuestionAnswering`，3 epochs，lr=2e-5，fp16=True。`eval_strategy="no"` 因为验证集没有 start_positions/end_positions 标签，Trainer 无法自动计算 eval loss。

### 模型上传

训练完用 `trainer.push_to_hub(commit_message="...", tags="question-answering")` 一次性推送模型权重、配置和 tokenizer。也可以 `model.push_to_hub()` 和 `tokenizer.push_to_hub()` 分开调。

首次使用需要登录：`huggingface_hub.notebook_login()` 或命令行 `hf auth login`，输入 [hf.co/settings/tokens](https://huggingface.co/settings/tokens) 生成的 Access Token。

### 模型卡片

推送后 HF Hub 自动生成基础 README。通过网页编辑完善卡片内容：模型用途（中文抽取式问答）、基座模型（bert-base-chinese）、数据集、评估结果、注意事项（pipeline 需配合自定义 null 判断）。模型卡片是对外说明模型能力边界的地方，评估结果必须如实写。

## 5. 训练结果

| 指标 | 未训练 baseline | 微调后 |
|------|:---:|:---:|
| 整体 exact | 0.04 | 58.71 |
| 整体 f1 | 0.05 | 58.75 |
| HasAns exact | 0.0 | 34.58 |
| HasAns f1 | 0.03 | 34.67 |
| NoAns exact | 0.07 | 74.42 |
| best_exact | 59.83 | 63.58 |

未训练 baseline 的整体 exact 只有 0.04，但 best_exact 竟有 59.83——因为 60% 样本不可回答，squad_v2 扫到最优阈值 ≈ "全部猜不答"，刚好拿满 NoAns 分。本质是随机模型被阈值优化后看起来还行，实际没有任何问答能力。

微调后 NoAns 从 0.07 涨到 74.42，说明模型真的学会了何时沉默。HasAns 从 0 涨到 ~34.7，虽然绝对值不高，但数据集中很多答案是长句子的片段，精确匹配本身就难。best_f1 的 63.6 更接近模型真实水平。

## 今日知识总结

### No-Answer 的 CLS 信号机制

抽取式问答中让模型"不答"的标准做法是用 `[CLS]` token 的 start+end logits 代表"无答案"分数。训练时不可回答样本标 (0, 0)，交叉熵自然会推动模型在没有答案可抽时给 CLS 高分。这是 SQuAD v2 引入的设计，比额外加一个二分类头更简洁，也复用已有的输出结构。

### HF Hub 数据的完整交互链路

数据下载：`huggingface-cli download` → 本地 parquet → `load_dataset("parquet", data_files={...})`。国内网络不稳定时这一步是必须的，parquet 格式自带 schema 无需额外配置。模型上传：`push_to_hub()` 一个方法搞定权重、配置、tokenizer；模型卡片在网页上补充用途、评估结果和注意事项。

### 评估指标不能只看一个数

squad_v2 的 overall exact/f1 受 NoAns 比例影响很大——一个只会闭嘴的模型也能拿不错的整体分。必须分开看 HasAns 和 NoAns 子集指标，才知道模型是"会抽答案"还是"只会沉默"。best_exact 遍历阈值后取最优，比固定阈值更客观，适合做模型对比。

### 排查 problem 的基本功

今天 debug 了五个问题：数据空答案处理、squad_v2 格式要求、多 chunk 变量作用域、softmax 溢出、pipeline 任务推断错误。都是小问题但每个都会导致静默错误或直接报错。定位问题的方法是：先看报错信息确定出错阶段，再看变量值是取早了还是取错了，最后对照官方示例或文档确认格式要求。
