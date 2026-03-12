# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.098
- RAG Accuracy: 0.066
- Accuracy difference: -0.032 (-3.2 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 23 | 26 |
| Baseline Wrong | 10 | 441 |

- Chi-square statistic: 6.2500
- p-value: 0.012419
- Odds ratio: 0.385

**Significant (p < 0.05): The accuracy difference between RAG and Baseline is statistically meaningful.**

## Significance by Question Type

### 事实提取

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.010
- Difference: 0.010
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

### 列举枚举

- Samples: 100
- Baseline Accuracy: 0.170
- RAG Accuracy: 0.050
- Difference: -0.120
- McNemar p-value: 0.005960
- **Significant (p < 0.05)**

### 判断验证

- Samples: 100
- Baseline Accuracy: 0.270
- RAG Accuracy: 0.250
- Difference: -0.020
- McNemar p-value: 0.789268
- Not significant (p >= 0.05)

### 推理分析

- Samples: 100
- Baseline Accuracy: 0.050
- RAG Accuracy: 0.020
- Difference: -0.030
- McNemar p-value: 0.371093
- Not significant (p >= 0.05)

### 比较计算

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.000
- Difference: 0.000
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

