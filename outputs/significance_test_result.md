# Baseline vs RAG Significance Test Report

## Overall Results

- Total samples: 500
- Baseline Accuracy: 0.096
- RAG Accuracy: 0.120
- Accuracy difference: 0.024 (2.4 pp)

## McNemar's Test

Paired contingency table:

| | RAG Correct | RAG Wrong |
|---|---|---|
| Baseline Correct | 34 | 14 |
| Baseline Wrong | 26 | 426 |

- Chi-square statistic: 3.0250
- p-value: 0.081990
- Odds ratio: 1.857

**Not significant (p >= 0.05): Cannot reject H0 (no accuracy difference between systems).**

## Significance by Question Type

### 事实提取  fact_extraction

- Samples: 100
- Baseline Accuracy: 0.020
- RAG Accuracy: 0.070
- Difference: 0.050
- McNemar p-value: 0.182422
- Not significant (p >= 0.05)

### 列举枚举 enumeration

- Samples: 100
- Baseline Accuracy: 0.140
- RAG Accuracy: 0.130
- Difference: -0.010
- McNemar p-value: 1.000000
- Not significant (p >= 0.05)

### 判断验证 judgment_verification

- Samples: 100
- Baseline Accuracy: 0.260
- RAG Accuracy: 0.330
- Difference: 0.070
- McNemar p-value: 0.023342
- **Significant (p < 0.05)**

### 推理分析 reasoning_analysis

- Samples: 100
- Baseline Accuracy: 0.060
- RAG Accuracy: 0.040
- Difference: -0.020
- McNemar p-value: 0.617075
- Not significant (p >= 0.05)

### 比较计算 comparative_calculation

- Samples: 100
- Baseline Accuracy: 0.000
- RAG Accuracy: 0.030
- Difference: 0.030
- McNemar p-value: 0.248213
- Not significant (p >= 0.05)

