# Baseline vs RAG 显著性检验报告

## 总体结果

- 总样本数: 208
- Baseline Accuracy: 0.091
- RAG Accuracy: 0.226
- 准确率差异: 0.135 (13.5 个百分点)

## McNemar's Test

配对混淆矩阵:

| | RAG正确 | RAG错误 |
|---|---|---|
| Baseline正确 | 19 | 0 |
| Baseline错误 | 28 | 161 |

- 卡方统计量: 26.0357
- p 值: 0.000000
- 优势比: ∞

**结果显著 (p < 0.05)：RAG 与 Baseline 的准确率差异具有统计学意义**

## 按问题类型的显著性检验

### 事实提取

- 样本数: 50
- Baseline Accuracy: 0.040
- RAG Accuracy: 0.260
- 差异: 0.220
- McNemar p 值: 0.002569
- **显著 (p < 0.05)**

### 判断验证

- 样本数: 50
- Baseline Accuracy: 0.340
- RAG Accuracy: 0.560
- 差异: 0.220
- McNemar p 值: 0.002569
- **显著 (p < 0.05)**

### 推理分析

- 样本数: 50
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.040
- 差异: 0.040
- McNemar p 值: 0.479500
- 不显著 (p >= 0.05)

### 比较计算

- 样本数: 50
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.080
- 差异: 0.080
- McNemar p 值: 0.133614
- 不显著 (p >= 0.05)

