# 27 Epoch、Step 与梯度累积

> 一句话：Epoch 是"把训练集完整过一遍"，Step 是"处理一个 Batch 并更新一次参数"，梯度累积用小显存近似大 Batch。


## 前置知识

- 先读 [16-Batch-Padding-Truncation-与-GPU](16-Batch-Padding-Truncation-与-GPU.md)。
- 先读 [16-Batch-Padding-Truncation-与-GPU](16-batch-padding-truncation-gpu.md)。
- 想理解"为什么要累积梯度"，需要知道显存有限的现实约束。

## 一、直观含义

- **Epoch（训练轮次）**：把整本习题册从头到尾做一遍 = 1 个 Epoch。
- **Step（训练步）**：做一题（处理一个 Batch）并对照答案订正一次 = 1 个 Step。
- **梯度累积（Gradient Accumulation）**：一次想做"大卷子"，但桌子（显存）放不下，就分成几份做，先不订正，攒够几份的"错题汇总"再一次性订正。

## 二、具体例子

- 训练集 4000 条，Batch Size 32 → 1 个 Epoch = 125 个 Step。
- 3 个 Epoch → 375 个 Step。
- 显存只放得下 Batch=8，但想要等效 Batch=32：累积 4 个小 Batch 的梯度再更新一次参数（4×8=32）。

## 三、技术要点（严谨部分）

### 1. 训练轮次（Epoch）

- 模型完整遍历一次训练集。
- 轮次是超参数，由人设定（如 3 epochs）。

### 2. 训练步（Step）

- 通常指处理一个 Batch 并完成一次（或一次累计后）的参数更新。
- 总步数 = Epoch × 每个 Epoch 的 Batch 数。
- 学习率调度的"刻度"就是 Step（见 [26-优化器-Adam-AdamW-与学习率调度](26-优化器-Adam-AdamW-与学习率调度.md)）。
- 学习率调度的"刻度"就是 Step（见 [26-优化器-Adam-AdamW-与学习率调度](26-optimizer-adam-adamw-scheduler.md)）。

### 3. 梯度累积（Gradient Accumulation）

- 先处理多个小 Batch 并**累积梯度**，再**更新一次参数**。
- 效果：用较小显存近似更大的有效 Batch。
- 注意：显存省的是"参数更新次数"，每个小 Batch 的前向反向仍要完整算一遍，只是不立刻更新。

### 4. Batch 大小如何影响训练

- Batch 越大：梯度估计越稳，但显存占用越高，每次更新用的数据越多。
- Batch 与学习率联动：Batch 翻倍，学习率通常也翻倍（见 [26](26-优化器-Adam-AdamW-与学习率调度.md)）。
- Batch 与学习率联动：Batch 翻倍，学习率通常也翻倍（见 [26](26-optimizer-adam-adamw-scheduler.md)）。

## 四、一句话总结

Epoch 数轮次、Step 数更新、梯度累积在显存不够时把小 Batch 攒成大更新——它们共同决定了"训练到底要走多少步、每步吃多少数据"。
