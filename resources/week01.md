# Week 1：基础准备

本周目标：完成 NLP 项目学习环境，掌握最基础的 PyTorch 操作，跑通 Hugging Face 第一个模型，并形成第一周学习记录。

---

## 1. 本周学习什么

### 1.1 环境能力

- Conda 环境管理
- Python 解释器路径确认
- pip 依赖安装
- PyTorch CUDA 版本确认
- GPU 是否可用验证
- VS Code 项目目录使用
- Git 基础命令

### 1.2 PyTorch 基础

- Tensor 是什么
- Tensor 的 shape、dtype、device
- CPU Tensor 和 GPU Tensor
- `requires_grad`
- `loss.backward()`
- `nn.Module` 基础概念
- 简单训练循环

### 1.3 Hugging Face 入门

- `pipeline` 是什么
- 情感分析任务是什么
- 模型如何自动下载
- tokenizer、model、prediction 的关系
- 默认模型提示是什么意思

### 1.4 文档和项目管理

- Markdown 基础
- 每日学习笔记
- 项目 README
- 目录结构整理
- 实验结果记录

---

## 2. 本周学习资料

## 2.1 主线资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| PyTorch Tutorials | Tensor、Autograd、训练流程 | https://pytorch.org/tutorials/ |
| Hugging Face LLM Course Chapter 1 | pipeline、NLP 任务概览 | https://huggingface.co/learn/llm-course/chapter1/1 |
| Transformers Quicktour | pipeline、AutoTokenizer、AutoModel | https://huggingface.co/docs/transformers/quicktour |

## 2.2 补充资料

| 资料 | 学习内容 | 地址 |
|---|---|---|
| NLP-LOVE/ML-NLP | 机器学习、深度学习基础 | https://github.com/NLP-LOVE/ML-NLP |
| Dive into Deep Learning | 张量、线性模型、深度学习基础 | https://d2l.ai/ |
| GitHub Markdown Guide | Markdown 写法 | https://docs.github.com/en/get-started/writing-on-github |

## 2.3 本周资料使用顺序

1. 先看 Hugging Face LLM Course Chapter 1。
2. 跑通 `pipeline`。
3. 再看 PyTorch Tutorials 的 Tensor 和 Autograd。
4. 最后补 NLP-LOVE/ML-NLP 里的机器学习基础。

---

## 3. 本周最终产出

完成 Week 1 后，仓库里至少要有：

```text
00-Environment/
└── env.md

02-PyTorch/
├── tensor_demo.py
└── autograd_demo.py

notes/
├── day01.md
├── day02.md
├── day03.md
├── day04.md
└── day05.md

outputs/
└── week01_pipeline_result.md
```

必须完成：

- PyTorch CUDA 验证记录
- Hugging Face pipeline 运行记录
- Tensor 基础代码
- Autograd 基础代码
- 第一周学习总结

---

## 4. 每日计划

## Day 1：环境配置与 Hugging Face pipeline

### 学习目标

- 确认 Conda 环境可用
- 确认 PyTorch CUDA 可用
- 跑通 Hugging Face 情感分析 pipeline

### 学习资料

- Hugging Face LLM Course Chapter 1
- Transformers Quicktour

### 任务

1. 确认 Python 路径。
2. 确认 PyTorch 版本。
3. 确认 CUDA 可用。
4. 运行 sentiment-analysis pipeline。
5. 记录运行结果。

### 命令

```powershell
where python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love NLP'))"
```

### 产出

- `00-Environment/env.md`
- `notes/day01.md`
- `outputs/week01_pipeline_result.md`

### 验收标准

- `torch.cuda.is_available()` 输出 `True`
- pipeline 输出 `POSITIVE`
- 能解释 `pipeline` 做了什么

---

## Day 2：PyTorch Tensor

### 学习目标

- 理解 Tensor
- 掌握 Tensor 创建、索引、shape、dtype、device
- 会把 Tensor 放到 GPU

### 学习资料

- PyTorch Tutorials：Tensors

### 任务

1. 创建标量、向量、矩阵。
2. 查看 shape 和 dtype。
3. 创建随机 Tensor。
4. 做加法、乘法、矩阵乘法。
5. 把 Tensor 移动到 GPU。

### 代码文件

`02-PyTorch/tensor_demo.py`

### 示例代码

```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x)
print(x.shape)
print(x.dtype)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)
```

### 产出

- `02-PyTorch/tensor_demo.py`
- `notes/day02.md`

### 验收标准

- 能说清 Tensor 和 NumPy array 的相似点
- 能说清 CPU 和 GPU device 的区别
- 代码能正常运行

---

## Day 3：Autograd 自动求导

### 学习目标

- 理解自动求导
- 理解 `requires_grad`
- 理解 `backward()`
- 理解 `.grad`

### 学习资料

- PyTorch Tutorials：Autograd

### 任务

1. 创建带梯度的 Tensor。
2. 构造一个简单函数。
3. 调用 `backward()`。
4. 查看梯度。
5. 解释梯度含义。

### 代码文件

`02-PyTorch/autograd_demo.py`

### 示例代码

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()

print("x:", x)
print("y:", y)
print("grad:", x.grad)
```

### 产出

- `02-PyTorch/autograd_demo.py`
- `notes/day03.md`

### 验收标准

- 能解释 `requires_grad=True`
- 能解释 `backward()` 是反向传播
- 能知道 `.grad` 保存梯度

---

## Day 4：nn.Module 与简单模型

### 学习目标

- 理解 `nn.Module`
- 理解模型参数
- 理解 forward
- 写一个简单神经网络

### 学习资料

- PyTorch Tutorials：Neural Networks

### 任务

1. 定义一个继承 `nn.Module` 的模型。
2. 使用 `nn.Linear`。
3. 输入一批数据。
4. 查看模型输出。
5. 查看模型参数。

### 代码文件

`02-PyTorch/simple_model.py`

### 示例代码

```python
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(3, 4)
y = model(x)

print(y)
for name, param in model.named_parameters():
    print(name, param.shape)
```

### 产出

- `02-PyTorch/simple_model.py`
- `notes/day04.md`

### 验收标准

- 能解释 `nn.Module`
- 能解释 `forward`
- 能看懂 `nn.Linear(4, 2)`

---

## Day 5：完整训练流程

### 学习目标

- 理解训练循环
- 理解 loss
- 理解 optimizer
- 理解 `zero_grad()`、`backward()`、`step()`

### 学习资料

- PyTorch Tutorials：Training a Classifier

### 任务

1. 构造一批假数据。
2. 定义简单分类模型。
3. 定义 loss。
4. 定义 optimizer。
5. 跑 10 次训练循环。
6. 观察 loss 变化。

### 代码文件

`02-PyTorch/train_loop_demo.py`

### 示例代码

```python
import torch
from torch import nn

x = torch.randn(20, 4)
y = torch.randint(0, 2, (20,))

model = nn.Linear(4, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
```

### 产出

- `02-PyTorch/train_loop_demo.py`
- `notes/day05.md`
- `notes/week01_summary.md`

### 验收标准

- 能解释训练循环每一步
- 能解释 loss 的作用
- 能解释 optimizer 的作用
- 代码能运行

---

## 5. 本周必须掌握的问题

1. Conda 环境是什么？
2. 怎么确认当前 Python 路径？
3. 怎么确认 PyTorch CUDA 可用？
4. Tensor 是什么？
5. Tensor 的 shape、dtype、device 是什么？
6. `requires_grad` 是什么？
7. `backward()` 做什么？
8. `nn.Module` 是什么？
9. `loss` 是什么？
10. `optimizer.step()` 做什么？
11. Hugging Face `pipeline` 做了什么？
12. 为什么不指定模型时会出现默认模型提示？

---

## 6. 本周学习笔记模板

```md
# Day XX

## 今日目标

## 学习资料

## 学习内容

## 代码实验

## 运行结果

## 遇到的问题

## 解决方法

## 今日总结

## 明日任务
```

---

## 7. Week 1 总结模板

```md
# Week 1 总结

## 本周完成

- [ ] 环境配置
- [ ] PyTorch CUDA 验证
- [ ] Hugging Face pipeline
- [ ] Tensor demo
- [ ] Autograd demo
- [ ] 简单模型 demo
- [ ] 训练循环 demo

## 本周学到的核心概念

## 最重要的代码

## 遇到的问题

## 还没理解的内容

## Week 2 准备
```

---

## 8. 本周 Git 提交建议

每天至少提交一次：

```powershell
git add .
git commit -m "day01 environment and pipeline"
git commit -m "day02 pytorch tensor demo"
git commit -m "day03 autograd demo"
git commit -m "day04 simple model demo"
git commit -m "day05 training loop demo"
```

---

## 9. 本周完成标准

Week 1 结束时，你应该能做到：

- 独立打开 Conda 环境
- 独立验证 GPU
- 独立运行 Hugging Face pipeline
- 独立创建 Tensor
- 独立写一个 `nn.Module`
- 独立写一个最小训练循环
- 写出 5 篇每日笔记
- 写出 1 篇周总结

