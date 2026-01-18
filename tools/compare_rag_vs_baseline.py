"""
对比实验脚本：Baseline(仅LLM) vs RAG + Gold Standard(Oracle)

目标：
- 对同一批问题，比较：
  - Baseline 答案质量 vs RAG 答案质量（相对 Oracle）
  - Baseline 幻觉检测表现 vs RAG 幻觉检测表现

依赖：
- gold 标准文件（稍后你标好后再运行本脚本）：
    datas/gold_standard.json
  期望字段至少包括：question, answer, filename, page, type
- Baseline 输出：
    no_rag_top1_pred_test_advanced_250.json
  结构：[{question, answer, filename, page}, ...]
- RAG 输出：
    rag_top1_pred.json
  结构：[{question, answer, filename, page}, ...]

评价指标：
- Answer 正确性（相对 Oracle）：
    - accuracy: 正确答案条数 / 总条数
    - 同时输出每种问题类型的正确率
- 幻觉检测（基于 RuleBasedDetector + 证据对齐）：
    - 将 “答案错误” 视为 ground truth 的“幻觉” (y_true = 1)
    - 将 规则层 verdict == "无证据" 视为 预测“幻觉” (y_pred = 1)
    - 输出 Hallucination Rate（预测为幻觉的比例）
    - 以及 Accuracy / Precision / Recall / F1（以“幻觉”为正类）

说明：
- 本脚本不会改动原始文件，只读取并输出对比结果。
- 如果 gold 文件暂时不存在，运行会报错；待你标完 Oracle 再运行即可。
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from hallucination_detector import (
    RuleBasedDetector,
    EvidenceRetriever,
    extract_numbers,
    calculate_text_coverage,
)


def has_numeric_conflict(gold_answer: str, pred_answer: str, tol: float = 0.01) -> bool:
    """
    判断答案中的数值是否与 Gold 存在“明显不一致”
    - gold / pred 都有数值时：
        若存在某个 gold 数值，在 pred 中找不到相对误差 <= tol 的对应值，则视为数值冲突
    - 至少一边没有数值时，返回 False（不按数值冲突处理）
    """
    g_nums = extract_numbers(gold_answer or "")
    p_nums = extract_numbers(pred_answer or "")
    if not g_nums or not p_nums:
        return False

    for g in g_nums:
        matched = False
        for p in p_nums:
            if abs(p) < 1e-8:
                if abs(g) < 1e-8:
                    matched = True
                    break
            else:
                rel_err = abs((g - p) / p)
                if rel_err <= tol:
                    matched = True
                    break
        if not matched:
            # 有一个 gold 数值在 pred 中找不到“足够接近”的对应值 → 认为有冲突
            return True
    return False


BASE_DIR = Path(__file__).resolve().parent.parent

# 路径约定：如有需要可自行修改
GOLD_PATH = BASE_DIR / "datas" / "gold_standard.json"
BASELINE_PATH = BASE_DIR / "no_rag_top1_pred_test_advanced_250.json"
RAG_PATH = BASE_DIR / "rag_top1_pred.json"
CHUNK_JSON_PATH = BASE_DIR / "all_pdf_page_chunks_merged.json"


def classify_error_type(rule_res: Dict, is_correct: bool, num_conflict: bool) -> str:
    """
    基于规则层输出，区分“幻觉”与其他错误类型：
    - hallucination: 声称了证据中不存在的内容（如列表中包含不存在的项）或完全无证据
    - contradiction: 与证据直接冲突（数值明显冲突或 verdict 为矛盾）
    - incompleteness: 答案不完整，但已列出的信息真实（列表类检索不完整）
    - reasoning_error: 计算/逻辑错误
    - other_error: 其他情况
    """
    if is_correct:
        return "correct"
    verdict = rule_res.get("verdict", "")
    scores = rule_res.get("scores", {})
    list_error_type = rule_res.get("list_error_type", "none")
    coverage = scores.get("coverage", 0.0)
    calc_score = scores.get("calculation", 1.0)
    completeness_score = scores.get("completeness", 1.0)

    # 幻觉信号：列表中编造、无证据且覆盖率极低
    if list_error_type == "hallucination":
        return "hallucination"
    if verdict == "无证据" and coverage < 0.3:
        return "hallucination"

    # 矛盾信号：数值冲突或明确矛盾
    if verdict == "矛盾" or num_conflict:
        return "contradiction"

    # 不完整：列表缺项但列出的都真实
    if list_error_type == "incompleteness" or completeness_score < 0.8:
        return "incompleteness"

    # 推理/计算错误
    if calc_score < 0.5:
        return "reasoning_error"

    return "other_error"


def load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_by_question(items: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for it in items:
        q = it.get("question", "").strip()
        if not q:
            continue
        out[q] = it
    return out


def is_answer_correct(
    gold_answer: str,
    pred_answer: str,
    question_type: str,
    num_rel_tol: float = 0.02,
    text_cov_threshold: float = 0.6,
) -> bool:
    """
    简单的“是否正确”判定：
    - 若 gold 中含数值：所有 gold 数值在 pred 中找到相近数值（相对误差<=num_rel_tol）
    - 再看文本覆盖度，>= text_cov_threshold 则视为“正确”
    - 否则视为“不正确”
    """
    if not gold_answer or not pred_answer:
        return False

    gold_nums = extract_numbers(gold_answer)
    pred_nums = extract_numbers(pred_answer)

    # 数值对齐：gold 有数值时，要求 pred 能对上主要数值
    if gold_nums:
        if not pred_nums:
            return False
        for g in gold_nums:
            matched = False
            for p in pred_nums:
                if abs(p) < 1e-8:
                    if abs(g) < 1e-8:
                        matched = True
                        break
                else:
                    rel = abs((g - p) / p)
                    if rel <= num_rel_tol:
                        matched = True
                        break
            if not matched:
                return False  # 有一个关键数值对不上，则认为错误

    # 文本覆盖度
    cov = calculate_text_coverage(gold_answer, pred_answer)
    return cov >= text_cov_threshold


def eval_system_against_gold(
    name: str,
    gold_dict: Dict[str, Dict],
    sys_dict: Dict[str, Dict],
    retriever: EvidenceRetriever,
    rule_detector: RuleBasedDetector,
) -> Dict:
    """
    对某个系统（Baseline / RAG）做整体评估：
    - answer 正确性
    - 基于 rule_detector 的幻觉检测效果
    """
    total = 0
    correct = 0

    # 幻觉检测的混淆矩阵（正类=幻觉）
    tp = fp = tn = fn = 0
    pred_hallu_cnt = 0

    # 按问题类型聚合正确率
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for q, gold_item in gold_dict.items():
        gold_ans = gold_item.get("answer", "")
        gold_fn = gold_item.get("filename", "")
        gold_pg = gold_item.get("page", 0)
        qtype = gold_item.get("type", "未知")

        sys_item = sys_dict.get(q)
        if not sys_item:
            continue

        sys_ans = sys_item.get("answer", "")
        sys_fn = sys_item.get("filename", "")
        sys_pg = sys_item.get("page", 0)

        total += 1
        type_stats[qtype]["total"] += 1

        # 1) QA 正确性（相对 Oracle）
        is_corr = is_answer_correct(gold_ans, sys_ans, qtype)
        if is_corr:
            correct += 1
            type_stats[qtype]["correct"] += 1

        # 基于 gold / pred 计算是否存在明显数值不一致
        num_conflict = has_numeric_conflict(gold_ans, sys_ans)

        # 检索证据：优先用系统自己的 filename/page，缺失则退回 gold
        ev_fn = sys_fn or gold_fn
        try:
            ev_pg = int(sys_pg) if str(sys_pg).strip() != "" else int(gold_pg)
        except Exception:
            ev_pg = int(gold_pg) if gold_pg is not None else 0

        evidence = retriever.retrieve(ev_fn, ev_pg, q) if ev_fn else ""
        if not evidence:
            evidence = "未找到证据"

        rule_res = rule_detector.judge(
            question=q,
            answer=sys_ans,
            evidence=evidence,
            answer_filename=sys_fn,
            answer_page=str(sys_pg),
            gold_filename=gold_fn,
            gold_page=str(gold_pg),
            question_type=qtype,
        )
        verdict = rule_res["verdict"]  # 有证据 / 部分证据 / 无证据

        # 2) 真实幻觉标签（ground truth）：仅将“编造/矛盾”视为幻觉
        error_type = classify_error_type(rule_res, is_corr, num_conflict)
        y_true_hallu = 1 if error_type in ("hallucination", "contradiction") else 0

        # 3) 预测幻觉标签：更保守的判定，避免将所有错误等同幻觉
        if rule_res.get("list_error_type") == "hallucination":
            y_pred_hallu = 1
        elif verdict == "无证据" and rule_res["scores"].get("coverage", 0.0) < 0.3:
            y_pred_hallu = 1
        elif verdict == "矛盾":
            y_pred_hallu = 1
        elif verdict == "部分证据" and num_conflict:
            y_pred_hallu = 1
        else:
            y_pred_hallu = 0

        if y_pred_hallu == 1:
            pred_hallu_cnt += 1

        # 更新混淆矩阵
        if y_true_hallu == 1 and y_pred_hallu == 1:
            tp += 1
        elif y_true_hallu == 1 and y_pred_hallu == 0:
            fn += 1
        elif y_true_hallu == 0 and y_pred_hallu == 1:
            fp += 1
        else:
            tn += 1

    acc = correct / total if total else 0.0

    # 幻觉检测指标
    det_acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    hallu_rate = pred_hallu_cnt / total if total else 0.0

    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "type_stats": type_stats,
        "hallu_detection": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": det_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "hallucination_rate": hallu_rate,
        },
    }


def print_report(res: Dict):
    name = res["name"]
    print("\n" + "=" * 70)
    print(f"系统: {name}")
    print("=" * 70)
    print(f"- 总样本数: {res['total']}")
    print(f"- 答案正确数: {res['correct']}")
    print(f"- Accuracy（答案正确率）: {res['accuracy']:.3f}")

    print("\n按问题类型的答案正确率：")
    for qtype, stats in res["type_stats"].items():
        tot = stats["total"]
        corr = stats["correct"]
        acc = corr / tot if tot else 0.0
        print(f"  - {qtype}: {corr}/{tot} (accuracy={acc:.3f})")

    det = res["hallu_detection"]
    print("\n幻觉相关指标（正类=真实幻觉/矛盾，非将全部错误视为幻觉）:")
    print(f"- Hallucination Rate: {det['hallucination_rate']:.3f}")
    print(f"- Precision: {det['precision']:.3f}")
    print(f"- Recall: {det['recall']:.3f}")
    print(f"- F1: {det['f1']:.3f}")


def main():
    print("=" * 70)
    print("Baseline vs RAG 对比实验（答案质量 + 幻觉检测）")
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

    retriever = EvidenceRetriever(str(CHUNK_JSON_PATH))
    rule_detector = RuleBasedDetector()

    baseline_res = eval_system_against_gold(
        "Baseline（无RAG，仅LLM）", gold_dict, baseline_dict, retriever, rule_detector
    )
    rag_res = eval_system_against_gold(
        "RAG（检索增强）", gold_dict, rag_dict, retriever, rule_detector
    )

    print_report(baseline_res)
    print_report(rag_res)


if __name__ == "__main__":
    main()


