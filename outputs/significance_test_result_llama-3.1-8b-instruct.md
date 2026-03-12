# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.066
- RAG Accuracy: 0.040
- Accuracy difference: -0.026 (-2.6 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 6 | 27 |
| Baseline Wrong | 14 | 453 |

- Chi-square statistic: 3.5122
- p-value: 0.060919
- Odds ratio: 0.519

**Not significant (p >= 0.05): Cannot reject H0 (no accuracy difference between systems).**

## Significance by Question Type

### 事实提取

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.000
- Difference: 0.000
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

### 列举枚举

- Samples: 100
- Baseline Accuracy: 0.230
- RAG Accuracy: 0.040
- Difference: -0.190
- McNemar p-value: 0.000086
- **Significant (p < 0.05)**

### 判断验证

- Samples: 100
- Baseline Accuracy: 0.090
- RAG Accuracy: 0.150
- Difference: 0.060
- McNemar p-value: 0.238593
- Not significant (p >= 0.05)

### 推理分析

- Samples: 100
- Baseline Accuracy: 0.010
- RAG Accuracy: 0.010
- Difference: 0.000
- McNemar p-value: 0.479500
- Not significant (p >= 0.05)

### 比较计算

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.000
- Difference: 0.000
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

