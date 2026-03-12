# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.086
- RAG Accuracy: 0.134
- Accuracy difference: 0.048 (4.8 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 37 | 6 |
| Baseline Wrong | 30 | 427 |

- Chi-square statistic: 14.6944
- p-value: 0.000126
- Odds ratio: 5.000

**Significant (p < 0.05): The accuracy difference between RAG and Baseline is statistically meaningful.**

## Significance by Question Type

### 事实提取

- Samples: 100
- Baseline Accuracy: 0.010
- RAG Accuracy: 0.050
- Difference: 0.040
- McNemar p-value: 0.220671
- Not significant (p >= 0.05)

### 列举枚举

- Samples: 100
- Baseline Accuracy: 0.110
- RAG Accuracy: 0.190
- Difference: 0.080
- McNemar p-value: 0.043308
- **Significant (p < 0.05)**

### 判断验证

- Samples: 100
- Baseline Accuracy: 0.240
- RAG Accuracy: 0.320
- Difference: 0.080
- McNemar p-value: 0.013328
- **Significant (p < 0.05)**

### 推理分析

- Samples: 100
- Baseline Accuracy: 0.070
- RAG Accuracy: 0.080
- Difference: 0.010
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

### 比较计算

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.030
- Difference: 0.030
- McNemar p-value: 0.248213
- Not significant (p >= 0.05)

