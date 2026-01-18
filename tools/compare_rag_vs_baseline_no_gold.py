import json
from pathlib import Path
from collections import defaultdict

from hallucination_detector import RuleBasedDetector, EvidenceRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "datas" / "test_advanced_250.json"
BASELINE_PATH = BASE_DIR / "no_rag_top1_pred_test_advanced_250.json"
RAG_PATH = BASE_DIR / "rag_top1_pred.json"
CHUNK_JSON_PATH = BASE_DIR / "all_pdf_page_chunks_merged.json"


def load_json_list(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_by_question(items):
    return {it["question"].strip(): it for it in items if it.get("question")}


def eval_one_system(name, qa_dict, test_meta_dict, retriever, rule_detector):
    """
    只做“有证据 / 部分证据 / 无证据”分布统计（无 Oracle）。

    证据来源策略：
    1）优先用系统自己预测的 filename/page 抽取证据；
    2）如系统 filename 或 page 为空，再回退到 test_advanced_250.json 中的真实 filename/page；
    这样可以同时评估“检索/定位能力”和“答案与证据的一致性”。
    """
    total = 0
    verdict_counts = defaultdict(int)
    verdict_counts_by_type = defaultdict(lambda: defaultdict(int))

    for q, ans_item in qa_dict.items():
        meta = test_meta_dict.get(q)
        if not meta:
            continue

        total += 1
        qtype = meta.get("type", "未知")
        gold_fn = meta.get("filename", "")
        gold_pg = meta.get("page", 0)

        answer = ans_item.get("answer", "")

        # 系统预测的 filename/page（可能为空或字符串）
        sys_fn = ans_item.get("filename", "") or ""
        sys_pg_raw = ans_item.get("page", "")

        # 1) 优先用系统自己的引用作为证据来源
        used_fn = sys_fn.strip() if isinstance(sys_fn, str) else ""
        used_pg = None
        try:
            if isinstance(sys_pg_raw, str):
                sys_pg_raw = sys_pg_raw.strip()
            if sys_pg_raw not in (None, "", "无", "unknown"):
                used_pg = int(sys_pg_raw)
        except Exception:
            used_pg = None

        evidence = ""
        if used_fn and used_pg is not None:
            evidence = retriever.retrieve(used_fn, used_pg, q)

        # 2) 若系统引用无效或取证失败，则回退到题目给的真实 filename/page
        if not evidence:
            used_fn = gold_fn
            try:
                used_pg = int(gold_pg)
            except Exception:
                used_pg = 0
            if used_fn:
                evidence = retriever.retrieve(used_fn, used_pg, q)

        if not evidence:
            evidence = "未找到证据"

        # 规则层判定：看“系统答案 vs 所使用的证据”是否一致
        rule_res = rule_detector.judge(
            question=q,
            answer=answer,
            evidence=evidence,
            answer_filename=used_fn,
            answer_page=str(used_pg if used_pg is not None else ""),
            gold_filename=gold_fn,
            gold_page=str(gold_pg),
            question_type=qtype,
        )
        v = rule_res["verdict"]  # 有证据 / 部分证据 / 无证据
        verdict_counts[v] += 1
        verdict_counts_by_type[qtype][v] += 1

    print("\n" + "=" * 60)
    print(f"系统: {name}")
    print("=" * 60)
    print(f"- 总样本数: {total}")
    for v, c in verdict_counts.items():
        print(f"  {v}: {c}  ({c / total:.2%})")

    print("\n按问题类型分布：")
    for qtype in ["事实提取", "列举枚举", "比较计算", "判断验证", "推理分析"]:
        if qtype not in verdict_counts_by_type:
            continue
        print(f"  {qtype}:")
        vs = verdict_counts_by_type[qtype]
        t = sum(vs.values())
        for v, c in vs.items():
            print(f"    {v}: {c}  ({c / t:.2%})")
    print("=" * 60 + "\n")


def main():
    test_items = load_json_list(TEST_PATH)
    baseline_items = load_json_list(BASELINE_PATH)
    rag_items = load_json_list(RAG_PATH)

    test_meta_dict = index_by_question(test_items)
    baseline_dict = index_by_question(baseline_items)
    rag_dict = index_by_question(rag_items)

    retriever = EvidenceRetriever(str(CHUNK_JSON_PATH))
    rule_detector = RuleBasedDetector()

    # 调严一些规则阈值，使“有证据”判定更严格，数值偏差更容易暴露
    # 数值容差：从 1% 收紧到 0.5%
    rule_detector.num_tolerance = 0.005
    # 文本覆盖率阈值：略提高
    rule_detector.coverage_threshold_high = 0.85
    rule_detector.coverage_threshold_low = 0.5

    eval_one_system("Baseline（无 RAG，仅 LLM）", baseline_dict, test_meta_dict, retriever, rule_detector)
    eval_one_system("RAG（检索增强）", rag_dict, test_meta_dict, retriever, rule_detector)


if __name__ == "__main__":
    main()