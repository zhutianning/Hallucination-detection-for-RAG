# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.020
- RAG Accuracy: 0.022
- Accuracy difference: 0.002 (0.2 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 1 | 9 |
| Baseline Wrong | 10 | 480 |

- Chi-square statistic: 0.0000
- p-value: 1.000000
- Odds ratio: 1.111

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
- Baseline Accuracy: 0.050
- RAG Accuracy: 0.020
- Difference: -0.030
- McNemar p-value: 0.449692
- Not significant (p >= 0.05)

### 判断验证

- Samples: 100
- Baseline Accuracy: 0.030
- RAG Accuracy: 0.090
- Difference: 0.060
- McNemar p-value: 0.113846
- Not significant (p >= 0.05)

### 推理分析

- Samples: 100
- Baseline Accuracy: 0.020
- RAG Accuracy: 0.000
- Difference: -0.020
- McNemar p-value: 0.479500
- Not significant (p >= 0.05)

### 比较计算

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.000
- Difference: 0.000
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

