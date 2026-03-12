# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.086
- RAG Accuracy: 0.102
- Accuracy difference: 0.016 (1.6 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 36 | 7 |
| Baseline Wrong | 15 | 442 |

- Chi-square statistic: 2.2273
- p-value: 0.135593
- Odds ratio: 2.143

**Not significant (p >= 0.05): Cannot reject H0 (no accuracy difference between systems).**

## Significance by Question Type

### 事实提取

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.020
- Difference: 0.020
- McNemar p-value: 0.479500
- Not significant (p >= 0.05)

### 列举枚举

- Samples: 100
- Baseline Accuracy: 0.090
- RAG Accuracy: 0.110
- Difference: 0.020
- McNemar p-value: 0.751830
- Not significant (p >= 0.05)

### 判断验证

- Samples: 100
- Baseline Accuracy: 0.260
- RAG Accuracy: 0.310
- Difference: 0.050
- McNemar p-value: 0.073638
- Not significant (p >= 0.05)

### 推理分析

- Samples: 100
- Baseline Accuracy: 0.080
- RAG Accuracy: 0.050
- Difference: -0.030
- McNemar p-value: 0.248213
- Not significant (p >= 0.05)

### 比较计算

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.020
- Difference: 0.020
- McNemar p-value: 0.479500
- Not significant (p >= 0.05)

