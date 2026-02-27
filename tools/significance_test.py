"""
Significance testing script: Baseline vs RAG

Uses McNemar's Test for paired-sample significance testing.
Applicable when two systems answer the same set of questions (paired data).

Dependencies:
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

# Import helper functions from compare_rag_vs_baseline.py
import sys
from pathlib import Path

# Ensure imports work from the same directory
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
    Collect paired outcomes for each question: whether Baseline and RAG are correct.

    Returns:
    - baseline_correct: List[bool] - Baseline correctness flags
    - rag_correct: List[bool] - RAG correctness flags
    - type_pairs: Dict[str, List[Tuple[bool, bool]]] - Paired outcomes grouped by question type
    """
    baseline_correct = []
    rag_correct = []
    type_pairs = defaultdict(list)
    
    for q, gold_item in gold_dict.items():
        gold_ans = gold_item.get("answer", "")
        qtype = gold_item.get("type", "unknown")
        
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
    Build the paired 2x2 contingency table for McNemar's test.

    Table layout:
                    RAG correct  RAG wrong
    Baseline correct      a           b
    Baseline wrong        c           d

    Where:
    - a: both correct
    - b: Baseline correct, RAG wrong
    - c: Baseline wrong, RAG correct
    - d: both wrong
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
    Run McNemar's Test.

    Args:
    - table: 2x2 paired contingency table
    - correction: whether to apply continuity correction (recommended for small samples)

    Returns:
    - statistic: chi-square statistic
    - pvalue: p-value
    - odds_ratio: odds ratio (c/b), when b+c > 0
    """
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]
    
    # McNemar's test
    if correction:
        # Apply continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    else:
        statistic = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0
    
    # p-value (chi-square distribution, df=1)
    pvalue = 1 - stats.chi2.cdf(statistic, df=1) if (b + c) > 0 else 1.0
    
    # Odds ratio
    odds_ratio = c / b if b > 0 else float('inf') if c > 0 else 1.0
    
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "odds_ratio": odds_ratio,
        "table": table,
        "discordant_pairs": b + c,  # Number of discordant pairs
    }


def proportion_test(baseline_correct: List[bool], rag_correct: List[bool]) -> Dict:
    """
    Test difference in proportions (normal approximation).

    Test H0: p_baseline = p_rag vs H1: p_baseline != p_rag
    """
    n = len(baseline_correct)
    p_baseline = sum(baseline_correct) / n
    p_rag = sum(rag_correct) / n
    
    # Standard error for paired samples
    diff = [r - b for r, b in zip(rag_correct, baseline_correct)]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # Paired t-test on per-item correctness differences
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
    """Print significance testing report."""
    print("=" * 70)
    print("Baseline vs RAG Significance Test Report")
    print("=" * 70)
    
    # Overall results
    print("\n[Overall Results]")
    print(f"- Total samples: {proportion_result['n']}")
    print(f"- Baseline Accuracy: {proportion_result['p_baseline']:.3f}")
    print(f"- RAG Accuracy: {proportion_result['p_rag']:.3f}")
    print(f"- Accuracy difference: {proportion_result['difference']:.3f} ({proportion_result['difference']*100:.1f} pp)")
    
    # McNemar's test
    print("\n[McNemar's Test (Paired Samples)]")
    table = mcnemar_result['table']
    print("Paired contingency table:")
    print("                   RAG Correct  RAG Wrong")
    print(f"Baseline Correct   {table[0,0]:11d}  {table[0,1]:9d}")
    print(f"Baseline Wrong     {table[1,0]:11d}  {table[1,1]:9d}")
    print(f"\nDiscordant pairs: {mcnemar_result['discordant_pairs']}")
    print(f"Chi-square statistic: {mcnemar_result['statistic']:.4f}")
    print(f"p-value: {mcnemar_result['pvalue']:.6f}")
    
    # Odds ratio
    if mcnemar_result['odds_ratio'] != float('inf'):
        print(f"Odds ratio (RAG-correct / Baseline-correct): {mcnemar_result['odds_ratio']:.3f}")
    else:
        print("Odds ratio: ∞ (RAG is always correct when Baseline is wrong)")
    
    # Significance interpretation
    alpha = 0.05
    if mcnemar_result['pvalue'] < alpha:
        print(f"\n✓ Significant result (p < {alpha}): RAG and Baseline differ statistically in accuracy")
        if proportion_result['difference'] > 0:
            print("  -> RAG is significantly better than Baseline")
        else:
            print("  -> Baseline is significantly better than RAG")
    else:
        print(f"\n✗ Not significant (p >= {alpha}): cannot reject H0 (no accuracy difference)")
    
    # Proportion-difference test
    print("\n[Proportion-Difference t-test]")
    print(f"t-statistic: {proportion_result['t_statistic']:.4f}")
    print(f"p-value: {proportion_result['pvalue']:.6f}")
    if proportion_result['pvalue'] < alpha:
        print(f"✓ Significant (p < {alpha})")
    else:
        print(f"✗ Not significant (p >= {alpha})")
    
    # Results by question type
    if type_results:
        print("\n[Significance by Question Type]")
        for qtype, result in sorted(type_results.items()):
            print(f"\n{qtype}:")
            print(f"  - Samples: {result['n']}")
            print(f"  - Baseline Accuracy: {result['p_baseline']:.3f}")
            print(f"  - RAG Accuracy: {result['p_rag']:.3f}")
            print(f"  - Difference: {result['difference']:.3f}")
            print(f"  - McNemar p-value: {result['mcnemar_p']:.6f}")
            if result['mcnemar_p'] < alpha:
                print(f"  ✓ Significant (p < {alpha})")
            else:
                print(f"  ✗ Not significant (p >= {alpha})")


def main():
    print("=" * 70)
    print("Loading data...")
    print("=" * 70)
    
    gold_data = load_json_list(GOLD_PATH)
    baseline_data = load_json_list(BASELINE_PATH)
    rag_data = load_json_list(RAG_PATH)
    
    gold_dict = index_by_question(gold_data)
    baseline_dict = index_by_question(baseline_data)
    rag_dict = index_by_question(rag_data)
    
    print(f"- Gold standard loaded: {len(gold_dict)} items")
    print(f"- Baseline answers loaded: {len(baseline_dict)} items")
    print(f"- RAG answers loaded: {len(rag_dict)} items")
    
    # Collect paired outcomes
    print("\nCollecting paired outcomes...")
    baseline_correct, rag_correct, type_pairs = collect_paired_results(
        gold_dict, baseline_dict, rag_dict
    )
    
    print(f"- Successfully paired: {len(baseline_correct)} items")
    
    # Overall significance test
    table = build_mcnemar_table(baseline_correct, rag_correct)
    mcnemar_result = mcnemar_test(table, correction=True)
    proportion_result = proportion_test(baseline_correct, rag_correct)
    
    # Significance tests by question type
    type_results = {}
    for qtype, pairs in type_pairs.items():
        if len(pairs) < 10:  # Sample size too small; skip
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
    
    # Print report
    print_significance_report(mcnemar_result, proportion_result, type_results)
    
    # Save results to file
    output_path = BASE_DIR / "outputs" / "significance_test_result.md"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vs RAG Significance Test Report\n\n")
        f.write("## Overall Results\n\n")
        f.write(f"- Total samples: {proportion_result['n']}\n")
        f.write(f"- Baseline Accuracy: {proportion_result['p_baseline']:.3f}\n")
        f.write(f"- RAG Accuracy: {proportion_result['p_rag']:.3f}\n")
        f.write(f"- Accuracy difference: {proportion_result['difference']:.3f} ({proportion_result['difference']*100:.1f} pp)\n\n")
        
        f.write("## McNemar's Test\n\n")
        f.write("Paired contingency table:\n\n")
        f.write("| | RAG Correct | RAG Wrong |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Baseline Correct | {table[0,0]} | {table[0,1]} |\n")
        f.write(f"| Baseline Wrong | {table[1,0]} | {table[1,1]} |\n\n")
        f.write(f"- Chi-square statistic: {mcnemar_result['statistic']:.4f}\n")
        f.write(f"- p-value: {mcnemar_result['pvalue']:.6f}\n")
        if mcnemar_result['odds_ratio'] != float('inf'):
            f.write(f"- Odds ratio: {mcnemar_result['odds_ratio']:.3f}\n")
        else:
            f.write(f"- Odds ratio: ∞\n")
        
        alpha = 0.05
        if mcnemar_result['pvalue'] < alpha:
            f.write(f"\n**Significant (p < {alpha}): The accuracy difference between RAG and Baseline is statistically meaningful.**\n")
        else:
            f.write(f"\n**Not significant (p >= {alpha}): Cannot reject H0 (no accuracy difference between systems).**\n")
        
        if type_results:
            f.write("\n## Significance by Question Type\n\n")
            for qtype, result in sorted(type_results.items()):
                f.write(f"### {qtype}\n\n")
                f.write(f"- Samples: {result['n']}\n")
                f.write(f"- Baseline Accuracy: {result['p_baseline']:.3f}\n")
                f.write(f"- RAG Accuracy: {result['p_rag']:.3f}\n")
                f.write(f"- Difference: {result['difference']:.3f}\n")
                f.write(f"- McNemar p-value: {result['mcnemar_p']:.6f}\n")
                if result['mcnemar_p'] < alpha:
                    f.write(f"- **Significant (p < {alpha})**\n")
                else:
                    f.write(f"- Not significant (p >= {alpha})\n")
                f.write("\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

