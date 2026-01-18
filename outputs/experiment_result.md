# Baseline vs RAG 对比实验（答案质量 + 幻觉检测）
- 加载 Gold 标准: 250 条
- 加载 Baseline 答案: 250 条
- 加载 RAG 答案: 250 条

## **系统: Baseline（无RAG，仅LLM）**

- 总样本数: 208
- 答案正确数: 19
- Accuracy（答案正确率）: 0.091

**按问题类型的答案正确率：**

- 事实提取: 2/50 (accuracy=0.040)
- 比较计算: 0/50 (accuracy=0.000)
- 判断验证: 17/50 (accuracy=0.340)
- 推理分析: 0/50 (accuracy=0.000)
- 列举枚举: 0/8 (accuracy=0.000)

**幻觉相关指标（正类=真实幻觉/矛盾，非将全部错误视为幻觉）:**

- Hallucination Rate: 0.990
- Precision: 0.908
- Recall: 1.000
- F1: 0.952

## 系统: RAG

- 总样本数: 208
- 答案正确数: 47
- Accuracy（答案正确率）: 0.226

**按问题类型的答案正确率：**

- 事实提取: 13/50 (accuracy=0.260)
- 比较计算: 4/50 (accuracy=0.080)
- 判断验证: 28/50 (accuracy=0.560)
- 推理分析: 2/50 (accuracy=0.040)
- 列举枚举: 0/8 (accuracy=0.000)

**幻觉相关指标（正类=真实幻觉/矛盾，非将全部错误视为幻觉）:**

- Hallucination Rate: 0.471
- Precision: 0.980
- Recall: 0.667
- F1: 0.793