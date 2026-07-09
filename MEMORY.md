# 项目记忆

> 用于后续继续学习时快速恢复上下文。只记录稳定信息，不保存账号、Token、密码等敏感数据。

## 学习者

- NLP 初学者
- 目标：通过 Hugging Face 和项目实践系统学习 NLP
- 学习方式：概念学习、Notebook 实验、Markdown 笔记、项目产出结合
- 解释要求：先讲直观含义，再给示例，最后说明技术原理
- 回复要求：中文、直接、简洁；不提供无关方案

## 项目

```text
C:\Users\27729\Desktop\NLP-Learning
```

主学习路线为 8 周：

1. 环境、Python、PyTorch、Hugging Face 入门
2. NLP 基础与文本表示
3. Transformer 基础
4. 手写 Mini Transformer
5. BERT 与下游任务
6. GPT 与文本生成
7. LoRA、PEFT、RAG 等现代 LLM 技术
8. 综合项目

详细周计划位于：

```text
resources/week01.md
...
resources/week08.md
```

## 当前环境

项目使用 D 盘 Conda `base` 环境：

```text
Python: D:\Soft\Conda\python.exe
Python version: 3.13.9
PyTorch: 2.11.0+cu128
CUDA Runtime: 12.8
CUDA available: True
Transformers: 4.57.3
Datasets: 5.0.0
```

关键约束：

- VS Code 解释器固定为 `D:\Soft\Conda\python.exe`
- Notebook Kernel 也必须选择该解释器
- Transformers 固定为 `4.57.3`，用于兼容课程 Pipeline 示例
- 安装或切换包版本后必须重启 Notebook Kernel
- 当前 `pip check` 无依赖冲突

环境文档：

```text
00-Environment/env.md
```

## 当前进度

Day 01 已完成：

- Conda、VS Code、PyTorch CUDA 环境验证
- Hugging Face Hub 登录
- Jupyter Notebook 基础使用
- `pipeline()` 工作流程
- 情感分析 Pipeline
- 文本生成、掩码填充、Token 分类和问答任务体验
- 预训练、微调和迁移学习概念
- Transformer 架构入门
- 已学习 Hugging Face LLM Course Chapter 1/4

Day 01 笔记：

```text
notes/day01.md
```

实验 Notebook：

```text
notebooks/day01.ipynb
```

## 已掌握的 Transformer 概念

- Encoder：理解完整输入并生成上下文化表示
- Decoder：根据目标前文生成下一个 Token
- Encoder-only：适合理解任务
- Decoder-only：适合生成任务
- Encoder-Decoder：适合翻译、摘要等条件生成任务
- Masked Self-Attention：只能查看目标序列当前位置及之前允许的信息
- Cross-Attention：解码器可以读取编码器的完整源句表示
- 因果掩码：阻止当前位置看到未来 Token
- 目标右移：防止当前位置直接看到当前正确答案
- Teacher Forcing：训练时后续位置使用真实前文，不使用前一位置的预测结果
- 训练可以并行：完整目标已知，所有位置的输入可提前构造
- 自回归推理通常串行：下一步依赖上一步实际生成的 Token

详细文档：

```text
04-Transformer/transformer-basics.md
```

## 已解决问题

### Torchaudio WinError 127

根因：PyTorch、Torchvision、Torchaudio CUDA 版本不一致。

结果：统一安装为 PyTorch CUDA 12.8 版本后解决。

### SymPy 版本冲突

根因：PyTorch 对 SymPy 有精确版本要求。

结果：依赖已修复，`pip check` 通过。

### Hugging Face 未认证警告

通过以下命令登录：

```powershell
hf auth login
hf auth whoami
```

### Windows 符号链接警告

不是运行错误。开启 Windows 开发者模式可减少 Hugging Face 缓存中的重复文件。

### `No mask_token found`

`fill-mask` 输入必须包含 `tokenizer.mask_token`。若其值为 `None`，模型不支持掩码填充。

### `model_type` 缺失

Hugging Face Hub 中的模型可能属于 GGUF、Flair 等其他格式，不能直接传给 Transformers `pipeline()`。

### `Unknown task question-answering`

Transformers 5.13.0 未注册该 Pipeline，而课程示例仍依赖它。已降级并固定为 `4.57.3`。

## 模型使用原则

使用模型前检查：

1. 模型任务是否匹配当前 Pipeline。
2. 模型框架是否为 Transformers。
3. 仓库是否包含有效的 `config.json`。
4. Tokenizer 是否包含任务需要的特殊 Token。
5. 模型文件大小是否适合本机。
6. 模型页是否提供官方调用示例。

不要因为模型位于 Hugging Face Hub，就默认它一定支持 Transformers `pipeline()`。

## 文档约定

- 每天学习记录放入 `notes/dayXX.md`
- 每天实验放入 `notebooks/dayXX.ipynb`
- 环境说明放入 `00-Environment`
- PyTorch 练习放入 `02-PyTorch`
- NLP 基础放入 `03-NLP-Basic`
- Transformer 原理与实现放入 `04-Transformer`
- 每周计划放入 `resources/weekXX.md`
- 主 README 只保留路线、索引、目标和阶段产出
- 新内容优先写入对应主题文档，避免所有内容堆积在每日笔记

## 下一步

Day 02 学习 Tokenizer：

- Token 与词表
- `AutoTokenizer`
- `tokenize()`、`encode()`、`decode()`
- `input_ids`
- 特殊 Token
- Padding
- Truncation
- Attention Mask

预期产出：

```text
notebooks/day02-tokenizer.ipynb
notes/day02.md
```

## 恢复上下文顺序

后续继续学习时，依次读取：

1. `MEMORY.md`
2. `README.md`
3. 当前周的 `resources/weekXX.md`
4. 最近一天的 `notes/dayXX.md`
5. 对应主题目录中的文档
