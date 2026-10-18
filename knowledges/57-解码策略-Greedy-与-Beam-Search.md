# 57 解码策略：Greedy 与 Beam Search

> 一句话：Greedy 每一步都选概率最高的 Token（快但容易重复）；Beam Search 同时保留多条候选序列择优扩展（更稳但可能偏保守）。


## 前置知识

- 先读 [46-自回归生成](46-自回归生成.md)。
- 先读 [46-自回归生成](46-autoregressive-generation.md)。
- 了解 Logits/Softmax 后的概率分布，见 [22-Hidden-State-Logits-与-Softmax](22-Hidden-State-Logits-与-Softmax.md)。
- 了解 Logits/Softmax 后的概率分布，见 [22-Hidden-State-Logits-与-Softmax](22-hidden-state-logits-softmax.md)。

## 一、直观含义

- **Greedy（贪心）**：每一步只看当下，闭眼选"现在最可能的字"——目光短浅，不回头。
- **Beam Search（波束搜索）**：每一步同时保留 k 条"最有希望"的候选句子，逐步扩展、定期淘汰，最后挑总分最高的那条——像面试时留几个备选人，一路筛到最后。

## 二、具体例子

- Greedy：`[我, 爱, 自, ...]` 每一步取概率最大。但它可能在第 3 步选了"自"，后来才发现当初该选"一"——一步错无法挽回。
- Beam Search（beam=3）：每一步保留 3 条序列，如：
  - `我爱自`（总分 0.9）
  - `我喜自`（总分 0.85）
  - `我一开`（总分 0.83）
  - 下一步各自扩展、再保留总分最高的 3 条，直到全部结束，再选总体最优。

## 三、技术要点（严谨部分）

### 1. Greedy Decoding（贪心解码）

- 每一步选择**概率最高**的 Token。
- 优点：速度快、结果确定（无随机性）。
- 缺点：容易陷入**局部最优**、重复、单调表达（连续选高分但平庸的词）。

### 2. Beam Search（波束搜索）

- **同时保留若干条累计得分最高的候选序列**，每一步扩展并再次筛选。
- **波束宽度（Beam Width）**：每一步保留的候选数量。
  - 越大 → 搜索更充分，但计算更贵。
  - 也**可能偏向安全、常见和较短的表达**——因为长度惩罚、高分词偏好等因素。
- 常用作**翻译、摘要**等"追求最好答案"任务的选择（这类任务答案相对确定）。
- 对**开放式生成**（对话、写诗）通常不如采样（见 [58-解码策略-Sampling-Temperature-Top-k-Top-p](58-解码策略-Sampling-Temperature-Top-k-Top-p.md)）。
- 对**开放式生成**（对话、写诗）通常不如采样（见 [58-解码策略-Sampling-Temperature-Top-k-Top-p](58-decoding-sampling-temperature-topk-topp.md)）。

### 3. 分数怎么算

- 候选序列的得分通常是各 Token **对数概率之和**（乘性变加性，见 [12-动态规划与-Viterbi-算法](12-动态规划与-Viterbi-算法.md)），常配合**长度惩罚**（见 [58](58-解码策略-Sampling-Temperature-Top-k-Top-p.md)）。
- 候选序列的得分通常是各 Token **对数概率之和**（乘性变加性，见 [12-动态规划与-Viterbi-算法](12-dynamic-programming-and-viterbi.md)），常配合**长度惩罚**（见 [58](58-decoding-sampling-temperature-topk-topp.md)）。

### 4. 选择建议

| 场景 | 常用策略 |
|---|---|
| 翻译、摘要（答案确定） | Beam Search |
| 对话、创作（追求多样） | Sampling 系列 |
| 快速确定输出 | Greedy |

## 四、一句话总结

Greedy 每步选最火的字（快但短视），Beam Search 保留 k 条路择优（稳但偏保守）；翻译摘要用 Beam，开放创作用采样。
