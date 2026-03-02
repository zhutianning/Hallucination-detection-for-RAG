# Baseline vs RAG Comparison Experiment (Answer Quality + Hallucination Detection)
- Loaded Gold Standard: 250 items
- Loaded Baseline Answers: 250 items
- Loaded RAG Answers: 250 items

## **System: Baseline (No RAG, LLM Only)**

- Total Samples: 208
- Correct Answers: 19
- Accuracy (Answer Correctness): 0.091

**Answer Accuracy by Question Type:**

- Fact Extraction: 2/50 (accuracy=0.040)
- Comparison & Calculation: 0/50 (accuracy=0.000)
- Judgment & Verification: 17/50 (accuracy=0.340)
- Reasoning & Analysis: 0/50 (accuracy=0.000)
- List Enumeration: 0/8 (accuracy=0.000)

**Hallucination-Related Metrics (positive class = true hallucination/contradiction; not all errors are treated as hallucinations):**

- Hallucination Rate: 0.990
- Precision: 0.908
- Recall: 1.000
- F1: 0.952

## System: RAG

- Total Samples: 208
- Correct Answers: 47
- Accuracy (Answer Correctness): 0.226

**Answer Accuracy by Question Type:**

- Fact Extraction: 13/50 (accuracy=0.260)
- Comparison & Calculation: 4/50 (accuracy=0.080)
- Judgment & Verification: 28/50 (accuracy=0.560)
- Reasoning & Analysis: 2/50 (accuracy=0.040)
- List Enumeration: 0/8 (accuracy=0.000)

**Hallucination-Related Metrics (positive class = true hallucination/contradiction; not all errors are treated as hallucinations):**

- Hallucination Rate: 0.471
- Precision: 0.980
- Recall: 0.667
- F1: 0.793