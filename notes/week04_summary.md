# Week 04 周志：三项目实战——从模型训练到 Web 部署（Day 14–16）

## 概述

本周完成了三个 NLP 项目的完整闭环：P1 中文情绪分类的端到端微调、P2 中文问答模型的 No-Answer 处理、P3 Gradio 交互页面与 HuggingFace Spaces 部署。三条线各有侧重——P1 覆盖数据准备到模型评估的完整 pipeline，P2 聚焦问答任务特有的空答案处理逻辑，P3 把前两个项目的模型封装为可分享的 Web 应用。

本周最核心的收获是**链路完整意识**：一个 NLP 项目不只是 notebook 里的几十行训练代码，还包括数据质量校验、错误分析、模型上传、接口封装、页面部署。任何一个环节断裂，模型就无法从"实验产出"变成"可用产品"。

## P1 中文情绪分类：一条完整的微调流（Day 14）

### 数据质量管理

Day 14 的核心经验不是模型架构或训练技巧，而是**数据质量决定模型上限**。经历了两次数据切换：

- TIX007 合成数据：模板化严重，每种情绪是同几个句式换词，训练 1 epoch 即 100%。根因不是模型过拟合，而是数据本身没有多样性——训练集和验证集高度重叠（828 条交集），模型不需要学语义，背模板就行。
- zzhdbw 真实对话数据：8 分类 4,159 条，标签分布不均衡但语义多样。

在开始训练之前做了三件事：`train_test_split(stratify=label)` 分层抽样保证 70/15/15 各集标签分布一致；`filter()` 剔除训练验证重叠样本；验证未训练 baseline（16.65% 随机水平）确认数据有用。这些是"训练前的必要步骤"，跳过任何一步都会导致不可靠的评估。

### 模型与训练

`bert-base-chinese` + `AutoModelForSequenceClassification(num_labels=8, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)`。`id2label` 是 0→"伤心" 的映射，让模型输出的数字转为可读标签。`ignore_mismatched_sizes=True` 接受 classification head 权重不匹配（新随机初始化），只加载预训练 backbone。

训练配置：3 epochs，lr=2e-5，weight_decay=0.01。分类报告 + 混淆矩阵做评估，`plt.rcParams["font.sans-serif"] = ["SimHei"]` 解决 matplotlib 中文乱码。混淆矩阵的 `labels` 参数控制数字顺序，`display_labels` 控制显示文字，两者容易搞混。

### 训练结果与问题

最终：训练集 97.6%，验证集 88.3%，测试集 87.2%。训练前 baseline 12.5%。几个问题：

- `push_to_hub=True` 时网络不通导致 `trainer.train()` 结束时 KeyboardInterrupt 卡死——模型已训完，只是上传失败。去掉 `push_to_hub` 先训完再手动推送。
- `evaluate` 库与本地 `evaluate.py` 文件同名导致 import 错乱——本地文件覆盖了第三方包。
- label 映射用 `zip(set(), set())` 乱序——集合无序，`list(set())` 每次顺序不同，label 错位导致分类报告对不上标签文字。

### 模型上传

`trainer.push_to_hub(commit_message="...", tags="summarization")` 推送到 `Kate-lf/emotion-classification`。这行代码后来在 P3 部署时暴露了问题——tags 里写 `summarization` 导致 HF Inference API 把模型当成摘要模型，pipeline_tag 错误。

## P2 中文问答：No-Answer 与 HF 数据互操作（Day 15）

### 手动下载到本地

国内网络直连 HF Hub 不稳定，用 `huggingface-cli download --repo-type dataset --local-dir ./data --include "*.parquet"` 下载 Chinese-SQuAD v2 parquet 文件，再用 `load_dataset("parquet", data_files={...})` 本地加载，完全不走网络。parquet 自带 schema，datasets 库自动解析。

训练集 90,027 条降采样到 43,188——按 title 分组每标题最多随机 100 条，用 `np.random.default_rng(42)` 固定种子保证可复现。

### CLS 做"不答"信号

深度学习模型没有"不知道"的天然概念。SQuAD v2 的解决方案：不可回答的问题，answer_start/end 标为 `(0, 0)`，即 `[CLS]` token 位置。训练时交叉熵最小化 CLS 位置的误差，推理时对比 `null_score = start_logits[0][0] + end_logits[0][0]` 和最佳非空 span 的得分。

pipeline API 不会自动做这个比较——它永远返回最高分 span，即使最高分在 CLS 位置。必须手动：获取 logits → 遍历所有 span 找最佳非空得分 → softmax 比较 → 决定答或不答。三个实现细节：tokenizer 输出在 CPU 需 `.to(device)` 移到 GPU；GPU tensor 不能 `.numpy()`，先 `.detach().cpu()`；排除 CLS 位置避免 null 把自己算进比较。

### 评估函数五个 bug

1. squad_v2 要求 `no_answer_probability` 字段，缺了直接 KeyError
2. null_score 取了多 chunk 循环最后一个而非最强 CLS
3. softmax 原始 logit 过大溢出——先减最大值
4. GPU/CPU tensor 混用导致无法 `.numpy()`
5. pipeline 默认不处理 No-Answer——官网示例误导

最终结果：整体 F1 58.75，NoAns 准确率 74.42%，模型学会了何时闭嘴。

## P3 Gradio 交互页面：从模型到网页（Day 16）

### Gradio Interface 最小模式

`gr.Interface(fn, inputs, outputs).launch()` 三要素：处理函数、输入控件、输出展示。`fn` 的参数签名和 `inputs` 数量必须对应。`inputs`/`outputs` 可用字符串简写（`"textbox"`、`"text"`）也可用组件实例（`gr.Textbox(label="...", lines=5)`）。`examples` 参数提供可点击填充的示例，降低使用门槛。

### 封装 Pipeline 为 predict 函数

分类模型返回 `[{"label": "伤心", "score": 0.96}]`，取 `[0]` 后用 `{score:.2%}` 把浮点转百分比。问答模型返回 `{"answer": "东坡居士", "score": 0.87}`，需额外处理空答案（分数 < 0.1 或无内容时返回"无法回答"）。`predict` 函数做两件事：调模型拿到原始输出，把输出格式化为用户可读的文本。

### HuggingFace Spaces 三种部署方式

| 方式 | 原理 | 配额 | 适用 |
|------|------|------|------|
| Gradio Space | HF 服务器跑 Python + Gradio | 免费 2 CPU | 正式部署 |
| Static + Inference API | 静态 HTML 调 `api-inference.huggingface.co` | 不限 | 轻量前端（国内不可用） |
| 本地 + share=True | 本地跑模型，Ngrok 穿透 | 无 | 临时测试 |

Gradio Space 流程：创建 Space（SDK: Gradio, Hardware: CPU free）→ Git push `app.py` + `requirements.txt` → 自动构建 2-3 分钟可用。免费配额只有 2 个同时运行的 CPU Space，需暂停不用的。

Static Space 方案在国内不可用，因为 `api-inference.huggingface.co` 被阻断（`ERR_CONNECTION_CLOSED`）。

### Transformers 5.x 两个兼容问题

升级到 5.14.1 后：`question-answering` 任务名被移除，`pipeline("question-answering", ...)` 报 KeyError。解决：去掉 task 参数，让 pipeline 根据模型 config 的 `architectures` 字段自动推断任务类型。这是 HuggingFace 设计的容错机制——`config.json` 中的 `architectures: ["BertForQuestionAnswering"]` 比 task 字符串更可靠。

P1 模型 `push_to_hub(tags="summarization")` 导致 pipeline_tag 错误，Inference API 按摘要处理而非分类。在模型页面编辑 README 的 YAML 头或 Settings 修改 Pipeline tag。

### Gradio Blocks vs Interface

`gr.Blocks` + `gr.Tab` 能构建多 Tab 页面，适合需要分类+问答双功能的场景。`click()` 事件绑定比 `Interface` 的自动映射更灵活，但复杂度也更高。P3 设计用 Blocks 把两个模型放同一页面通过标签切换——单页面方案比两个独立应用更整洁。

## 已解决问题

- **数据模板化导致虚假高分**（Day 14）：合成数据 100% 准确率是数据问题不是模型问题；切换真实数据后恢复正常。
- **label 映射错乱**（Day 14）：`zip(set(), set())` 无序，改用 `sorted(set(tuple))` 保证一致性。
- **matplotlib 中文乱码**（Day 14）：`plt.rcParams["font.sans-serif"] = ["SimHei"]` 设置中文字体。
- **evaluate 库同名冲突**（Day 14）：本地 `evaluate.py` 覆盖第三方包，重命名本地文件。
- **push_to_hub 网络卡死**（Day 14）：训练完成后网络等待导致 KeyboardInterrupt，先训练再手动推送。
- **squad_v2 评估五连坑**（Day 15）：字段缺失、变量作用域、softmax 溢出、GPU tensor 混用、pipeline 空答案处理。
- **datasets 5.0.0 脚本拒绝**（Day 15）：`revision="refs/convert/parquet"` 绕过。
- **huggingface-hub 版本冲突**（Day 16）：`transformers==4.57.3` 要求 <1.0，升级 transformers 解决。
- **transformers 5.x 移除 question-answering 任务名**（Day 16）：改用 pipeline 自动推断。
- **Static Space Inference API 国内阻断**（Day 16）：`ERR_CONNECTION_CLOSED`，改用 Gradio Space。
- **Gradio Space 硬件配额**（Day 16）：选错硬件或旧 Space 占用导致，确认 CPU (free) 并暂停不用的 Space。
- **模型 pipeline_tag 错误**（Day 16）：`tags="summarization"` 导致分类模型被当摘要模型，编辑模型卡片修正。

## 下一步

三个 NLP 项目（分类、问答、Gradio 部署）已完成。下一步进入 GPT 与生成：Decoder-only 架构、自回归生成、贪婪搜索 vs beam search vs top-k/top-p 采样策略、温度参数、KV 缓存机制。这些是理解 ChatGPT 类模型如何"说话"的基础，也是 Week 07 RAG 生成环节的前置知识。

```text
notes/day17.md
06-GPT/
```
