# Week 2：PyTorch 最小训练基础

目标：只学微调必须懂的 PyTorch，不追求系统学完。

## 本周资源

| 资源 | 链接 |
|---|---|
| PyTorch Tensors | https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html |
| PyTorch Data Tutorial | https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html |
| PyTorch Build Model | https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html |
| PyTorch Autograd | https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html |
| PyTorch Optimization | https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html |
| Save and Load Model | https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html |

## 每日计划

| Day | 学习内容 | 代码任务 | 笔记重点 |
|---|---|---|---|
| Day 08 | Tensor、shape、dtype、device | 创建 tensor，移动 CPU/GPU | tensor 与 array 的区别 |
| Day 09 | Dataset、DataLoader | 写最小 Dataset | batch 从哪里来 |
| Day 10 | `nn.Module` | 两层线性分类模型 | forward 和参数 |
| Day 11 | autograd | 手算一次 `loss.backward()` | 梯度保存在哪里 |
| Day 12 | loss、optimizer | 跑 `zero_grad/backward/step` | optimizer 为什么更新参数 |
| Day 13 | 完整训练循环 | 假数据二分类训练 | 训练循环顺序 |
| Day 14 | 复盘 | 保存和加载模型 | 训练和推理的区别 |

## 本周产出

```text
02-PyTorch/
├── tensor_demo.py
├── dataloader_demo.py
├── simple_model.py
├── autograd_demo.py
├── train_loop_demo.py
└── save_load_demo.py
```

## 必须掌握

1. `shape`、`dtype`、`device`。
2. `Dataset` 和 `DataLoader` 的关系。
3. `nn.Module` 的作用。
4. `loss.backward()` 做什么。
5. 为什么每轮训练要 `optimizer.zero_grad()`。

