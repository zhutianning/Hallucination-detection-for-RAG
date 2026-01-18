"""
显著性检验脚本：Baseline vs RAG

使用 McNemar's Test 进行配对样本的显著性检验
适用于：同一批问题，两个系统分别回答（配对数据）

依赖：
- datas/gold_standard.json
- no_rag_top1_pred_test_advanced_250.json
- rag_top1_pred.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict

# 导入 compare_rag_vs_baseline.py 中的函数
import sys
from pathlib import Path

# 确保可以导入同目录下的模块
sys.path.insert(0, str(Path(__file__).parent))

from compare_rag_vs_baseline import (
    load_json_list,
    index_by_question,
    is_answer_correct,
    BASE_DIR,
    GOLD_PATH,
    BASELINE_PATH,
    RAG_PATH,
)


def collect_paired_results(
    gold_dict: Dict[str, Dict],
    baseline_dict: Dict[str, Dict],
    rag_dict: Dict[str, Dict],
) -> Tuple[List[bool], List[bool], Dict[str, List[Tuple[bool, bool]]]]:
    """
    收集配对结果：对每个问题，记录 Baseline 和 RAG 是否正确
    
    返回：
    - baseline_correct: List[bool] - Baseline 是否正确
    - rag_correct: List[bool] - RAG 是否正确
    - type_pairs: Dict[str, List[Tuple[bool, bool]]] - 按问题类型分组的配对结果
    """
    baseline_correct = []
    rag_correct = []
    type_pairs = defaultdict(list)
    
    for q, gold_item in gold_dict.items():
        gold_ans = gold_item.get("answer", "")
        qtype = gold_item.get("type", "未知")
        
        baseline_item = baseline_dict.get(q)
        rag_item = rag_dict.get(q)
        
        if not baseline_item or not rag_item:
            continue
        
        baseline_ans = baseline_item.get("answer", "")
        rag_ans = rag_item.get("answer", "")
        
        baseline_is_correct = is_answer_correct(gold_ans, baseline_ans, qtype)
        rag_is_correct = is_answer_correct(gold_ans, rag_ans, qtype)
        
        baseline_correct.append(baseline_is_correct)
        rag_correct.append(rag_is_correct)
        type_pairs[qtype].append((baseline_is_correct, rag_is_correct))
    
    return baseline_correct, rag_correct, type_pairs


def build_mcnemar_table(baseline_correct: List[bool], rag_correct: List[bool]) -> np.ndarray:
    """
    构建 McNemar 检验的配对混淆矩阵
    
    矩阵结构：
                RAG 正确  RAG 错误
    Baseline 正确   a        b
    Baseline 错误   c        d
    
    其中：
    - a: 两个都正确
    - b: Baseline 正确，RAG 错误
    - c: Baseline 错误，RAG 正确
    - d: 两个都错误
    """
    a = b = c = d = 0
    
    for bl_corr, rag_corr in zip(baseline_correct, rag_correct):
        if bl_corr and rag_corr:
            a += 1
        elif bl_corr and not rag_corr:
            b += 1
        elif not bl_corr and rag_corr:
            c += 1
        else:
            d += 1
    
    return np.array([[a, b], [c, d]])


def mcnemar_test(table: np.ndarray, correction: bool = True) -> Dict:
    """
    执行 McNemar's Test
    
    参数：
    - table: 2x2 配对混淆矩阵
    - correction: 是否使用连续性校正（推荐用于小样本）
    
    返回：
    - statistic: 卡方统计量
    - pvalue: p 值
    - odds_ratio: 优势比 (c/b)，如果 b+c > 0
    """
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]
    
    # McNemar's Test
    if correction:
        # 使用连续性校正
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    else:
        statistic = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0
    
    # p 值（卡方分布，自由度=1）
    pvalue = 1 - stats.chi2.cdf(statistic, df=1) if (b + c) > 0 else 1.0
    
    # 优势比（odds ratio）
    odds_ratio = c / b if b > 0 else float('inf') if c > 0 else 1.0
    
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "odds_ratio": odds_ratio,
        "table": table,
        "discordant_pairs": b + c,  # 不一致的对数
    }


def proportion_test(baseline_correct: List[bool], rag_correct: List[bool]) -> Dict:
    """
    比例差异检验（使用正态近似）
    
    检验 H0: p_baseline = p_rag vs H1: p_baseline != p_rag
    """
    n = len(baseline_correct)
    p_baseline = sum(baseline_correct) / n
    p_rag = sum(rag_correct) / n
    
    # 计算标准误（配对样本）
    diff = [r - b for r, b in zip(rag_correct, baseline_correct)]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # t 检验
    if se_diff > 0:
        t_stat = mean_diff / se_diff
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    else:
        t_stat = 0
        pvalue = 1.0
    
    return {
        "p_baseline": p_baseline,
        "p_rag": p_rag,
        "difference": p_rag - p_baseline,
        "t_statistic": t_stat,
        "pvalue": pvalue,
        "n": n,
    }


def print_significance_report(
    mcnemar_result: Dict,
    proportion_result: Dict,
    type_results: Dict[str, Dict],
):
    """打印显著性检验报告"""
    print("=" * 70)
    print("Baseline vs RAG 显著性检验报告")
    print("=" * 70)
    
    # 总体结果
    print("\n【总体结果】")
    print(f"- 总样本数: {proportion_result['n']}")
    print(f"- Baseline Accuracy: {proportion_result['p_baseline']:.3f}")
    print(f"- RAG Accuracy: {proportion_result['p_rag']:.3f}")
    print(f"- 准确率差异: {proportion_result['difference']:.3f} ({proportion_result['difference']*100:.1f} 个百分点)")
    
    # McNemar's Test
    print("\n【McNemar's Test（配对样本检验）】")
    table = mcnemar_result['table']
    print(f"配对混淆矩阵:")
    print(f"                RAG正确  RAG错误")
    print(f"Baseline正确    {table[0,0]:4d}    {table[0,1]:4d}")
    print(f"Baseline错误    {table[1,0]:4d}    {table[1,1]:4d}")
    print(f"\n不一致的对数: {mcnemar_result['discordant_pairs']}")
    print(f"卡方统计量: {mcnemar_result['statistic']:.4f}")
    print(f"p 值: {mcnemar_result['pvalue']:.6f}")
    
    # 优势比
    if mcnemar_result['odds_ratio'] != float('inf'):
        print(f"优势比 (RAG正确/Baseline正确): {mcnemar_result['odds_ratio']:.3f}")
    else:
        print(f"优势比: ∞ (RAG 在 Baseline 错误时总是正确)")
    
    # 显著性判断
    alpha = 0.05
    if mcnemar_result['pvalue'] < alpha:
        print(f"\n✓ 结果显著 (p < {alpha})：RAG 与 Baseline 的准确率差异具有统计学意义")
        if proportion_result['difference'] > 0:
            print(f"  → RAG 显著优于 Baseline")
        else:
            print(f"  → Baseline 显著优于 RAG")
    else:
        print(f"\n✗ 结果不显著 (p >= {alpha})：无法拒绝 H0（两个系统准确率无差异）")
    
    # 比例差异检验
    print("\n【比例差异 t 检验】")
    print(f"t 统计量: {proportion_result['t_statistic']:.4f}")
    print(f"p 值: {proportion_result['pvalue']:.6f}")
    if proportion_result['pvalue'] < alpha:
        print(f"✓ 结果显著 (p < {alpha})")
    else:
        print(f"✗ 结果不显著 (p >= {alpha})")
    
    # 按问题类型的结果
    if type_results:
        print("\n【按问题类型的显著性检验】")
        for qtype, result in sorted(type_results.items()):
            print(f"\n{qtype}:")
            print(f"  - 样本数: {result['n']}")
            print(f"  - Baseline Accuracy: {result['p_baseline']:.3f}")
            print(f"  - RAG Accuracy: {result['p_rag']:.3f}")
            print(f"  - 差异: {result['difference']:.3f}")
            print(f"  - McNemar p 值: {result['mcnemar_p']:.6f}")
            if result['mcnemar_p'] < alpha:
                print(f"  ✓ 显著 (p < {alpha})")
            else:
                print(f"  ✗ 不显著 (p >= {alpha})")


def main():
    print("=" * 70)
    print("加载数据...")
    print("=" * 70)
    
    gold_data = load_json_list(GOLD_PATH)
    baseline_data = load_json_list(BASELINE_PATH)
    rag_data = load_json_list(RAG_PATH)
    
    gold_dict = index_by_question(gold_data)
    baseline_dict = index_by_question(baseline_data)
    rag_dict = index_by_question(rag_data)
    
    print(f"- 加载 Gold 标准: {len(gold_dict)} 条")
    print(f"- 加载 Baseline 答案: {len(baseline_dict)} 条")
    print(f"- 加载 RAG 答案: {len(rag_dict)} 条")
    
    # 收集配对结果
    print("\n收集配对结果...")
    baseline_correct, rag_correct, type_pairs = collect_paired_results(
        gold_dict, baseline_dict, rag_dict
    )
    
    print(f"- 成功配对: {len(baseline_correct)} 条")
    
    # 总体显著性检验
    table = build_mcnemar_table(baseline_correct, rag_correct)
    mcnemar_result = mcnemar_test(table, correction=True)
    proportion_result = proportion_test(baseline_correct, rag_correct)
    
    # 按问题类型的显著性检验
    type_results = {}
    for qtype, pairs in type_pairs.items():
        if len(pairs) < 10:  # 样本量太小，跳过
            continue
        
        bl_corr = [p[0] for p in pairs]
        rag_corr = [p[1] for p in pairs]
        
        type_table = build_mcnemar_table(bl_corr, rag_corr)
        type_mcnemar = mcnemar_test(type_table, correction=True)
        type_prop = proportion_test(bl_corr, rag_corr)
        
        type_results[qtype] = {
            "n": len(pairs),
            "p_baseline": type_prop["p_baseline"],
            "p_rag": type_prop["p_rag"],
            "difference": type_prop["difference"],
            "mcnemar_p": type_mcnemar["pvalue"],
            "table": type_table,
        }
    
    # 打印报告
    print_significance_report(mcnemar_result, proportion_result, type_results)
    
    # 保存结果到文件
    output_path = BASE_DIR / "outputs" / "significance_test_result.md"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vs RAG 显著性检验报告\n\n")
        f.write("## 总体结果\n\n")
        f.write(f"- 总样本数: {proportion_result['n']}\n")
        f.write(f"- Baseline Accuracy: {proportion_result['p_baseline']:.3f}\n")
        f.write(f"- RAG Accuracy: {proportion_result['p_rag']:.3f}\n")
        f.write(f"- 准确率差异: {proportion_result['difference']:.3f} ({proportion_result['difference']*100:.1f} 个百分点)\n\n")
        
        f.write("## McNemar's Test\n\n")
        f.write("配对混淆矩阵:\n\n")
        f.write("| | RAG正确 | RAG错误 |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Baseline正确 | {table[0,0]} | {table[0,1]} |\n")
        f.write(f"| Baseline错误 | {table[1,0]} | {table[1,1]} |\n\n")
        f.write(f"- 卡方统计量: {mcnemar_result['statistic']:.4f}\n")
        f.write(f"- p 值: {mcnemar_result['pvalue']:.6f}\n")
        if mcnemar_result['odds_ratio'] != float('inf'):
            f.write(f"- 优势比: {mcnemar_result['odds_ratio']:.3f}\n")
        else:
            f.write(f"- 优势比: ∞\n")
        
        alpha = 0.05
        if mcnemar_result['pvalue'] < alpha:
            f.write(f"\n**结果显著 (p < {alpha})：RAG 与 Baseline 的准确率差异具有统计学意义**\n")
        else:
            f.write(f"\n**结果不显著 (p >= {alpha})：无法拒绝 H0（两个系统准确率无差异）**\n")
        
        if type_results:
            f.write("\n## 按问题类型的显著性检验\n\n")
            for qtype, result in sorted(type_results.items()):
                f.write(f"### {qtype}\n\n")
                f.write(f"- 样本数: {result['n']}\n")
                f.write(f"- Baseline Accuracy: {result['p_baseline']:.3f}\n")
                f.write(f"- RAG Accuracy: {result['p_rag']:.3f}\n")
                f.write(f"- 差异: {result['difference']:.3f}\n")
                f.write(f"- McNemar p 值: {result['mcnemar_p']:.6f}\n")
                if result['mcnemar_p'] < alpha:
                    f.write(f"- **显著 (p < {alpha})**\n")
                else:
                    f.write(f"- 不显著 (p >= {alpha})\n")
                f.write("\n")
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

