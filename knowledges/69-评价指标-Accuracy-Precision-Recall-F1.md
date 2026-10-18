# 69 评价指标：Accuracy、Precision、Recall、F1

> 一句话：Accuracy 看"总的对错"，Precision 看"报出来的准不准"，Recall 看"该找的找没找全"，F1 是两者的调和平均；类别不平衡时 Accuracy 会骗人。


## 前置知识

- 先读 [23-标签-损失函数-与交叉熵](23-标签-损失函数-与交叉熵.md)（区分 Loss 与 Metric）。
- 先读 [23-标签-损失函数-与交叉熵](23-labels-loss-cross-entropy.md)（区分 Loss 与 Metric）。
- 若 NER"按实体算"的细节不熟，可回看 [60-Token-分类-NER-POS-Chunking-与-IOB2](60-Token-分类-NER-POS-Chunking-与-IOB2.md)。
- 若 NER"按实体算"的细节不熟，可回看 [60-Token-分类-NER-POS-Chunking-与-IOB2](60-token-classification-ner-pos-chunking-iob2.md)。

## 一、直观含义

想象一个"抓逃犯"的场景：

- **Accuracy（准确率）**：我这一天判断对了几次（含"没逃犯"的正确判断）。
- **Precision（精确率）**：我喊"抓人"的时候，抓对的比例——**误报多不多**。
- **Recall（召回率）**：真正的逃犯里，我抓到几个——**漏网多不多**。
- **F1**：把"抓得准"和"抓得全"合成一个分——两个都好 F1 才高。

## 二、具体例子

- NER 里绝大多数 Token 是"O"（不是实体）。模型把所有 Token 都判成"O"→ Accuracy 高达 95%+，但一个实体都没识别出来——**Accuracy 严重误导**。这时要看实体级的 Precision/Recall/F1。

## 三、技术要点（严谨部分）

### 1. Accuracy（准确率）

- 预测正确样本数 ÷ 全部样本数。
- 类别严重不平衡时**可能误导**。

### 2. 二分类的四种计数

- **TP（真正例）**：正例被正确预测为正。
- **FP（假正例）**：负例被误报为正。
- **FN（假负例）**：正例被漏报为负。
- **TN（真负例）**：负例被正确预测为负。

### 3. Precision（精确率）

- 预测为正的结果中，有多少确实为正：**TP ÷ (TP + FP)**。
- 精确率高 = **误报少**。

### 4. Recall（召回率）

- 所有真实为正的结果中，有多少被找出：**TP ÷ (TP + FN)**。
- 召回率高 = **漏报少**。

### 5. F1 Score

- Precision 与 Recall 的**调和平均**：`2PR ÷ (P + R)`。
- **调和平均（Harmonic Mean）**对较小值更敏感——只有两者都较高，F1 才会高。
- 例：P=0.9、R=0.1 → 算术平均 0.5，但 F1≈0.18——因为它惩罚短板。

### 6. ⚠️ "正例"单位必须明确

- NER 常按**完整实体边界与类别**计算，而不是只按单个 Token 算。
- 多个 Token 组成的实体，只有整块都标对才算对（seqeval 这类评估按实体匹配）。

### 7. 多分类怎么汇总

- 把每个类别轮流视为"正类"、其余为"负类"，再按 Macro / Micro / Weighted 汇总——见 [70-Macro-Micro-Weighted-与-Baseline](70-Macro-Micro-Weighted-与-Baseline.md)。
- 把每个类别轮流视为"正类"、其余为"负类"，再按 Macro / Micro / Weighted 汇总——见 [70-Macro-Micro-Weighted-与-Baseline](70-macro-micro-weighted-baseline.md)。

## 四、一句话总结

Accuracy 数总账、Precision 管误报、Recall 管漏报、F1 是它们的调和平均；不平衡时别只看 Accuracy，NER 等任务还要按"完整实体"而不是按 Token 算。
