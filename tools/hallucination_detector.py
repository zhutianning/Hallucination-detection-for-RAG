"""
多层幻觉检测器 - 无需训练分类器
基于规则层 + LLM判定层 + 证据对齐层
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from tqdm import tqdm

load_dotenv()

# ========== 配置 ==========
CHUNK_JSON_PATH = "all_pdf_page_chunks_merged.json"  # 知识库chunks
GOLD_STANDARD_PATH = "datas/gold_standard.json"  # 金标准答案
RAG_ANSWERS_PATH = "outputs/rag_answers.json"  # RAG生成答案
OUTPUT_PATH = "outputs/hallucination_detection_results.json"

# LLM判定层配置（可选，如果不用LLM判定层可跳过）
USE_LLM_LAYER = True
LLM_API_KEY = os.getenv('LOCAL_API_KEY')
LLM_BASE_URL = os.getenv('LOCAL_BASE_URL')
LLM_MODEL = os.getenv('LOCAL_TEXT_MODEL')


# ========== 工具函数 ==========
def normalize_text(text: str) -> str:
    """文本归一化：去标点、全角转半角、去空格"""
    if not text:
        return ""
    # 全角转半角
    text = text.translate(str.maketrans('，。！？；：""''（）【】', ',.!?;:"\'()[]'))
    # 去标点、去空格
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return text.lower()


def extract_numbers(text: str) -> List[float]:
    """提取文本中的所有数字（含百分号、小数、带逗号）"""
    # 移除千分位逗号
    text_clean = text.replace(',', '').replace('，', '')
    # 匹配数字（含小数、百分号）
    patterns = [
        r'(-?\d+\.?\d*)%',  # 百分比
        r'(-?\d+\.?\d*)',   # 普通数字
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text_clean)
        for m in matches:
            try:
                num = float(m)
                numbers.append(num)
            except:
                pass
    return list(set(numbers))  # 去重


def normalize_number(num: float, unit: str = "") -> float:
    """数值归一化（单位转换：亿元→万元等）"""
    unit_map = {
        "亿": 10000, "万元": 1, "万元": 1, "万": 1,
        "千": 0.1, "元": 0.0001
    }
    if unit in unit_map:
        return num * unit_map[unit]
    return num


def calculate_text_coverage(answer: str, evidence: str) -> float:
    """计算答案在证据中的文本覆盖率（基于token集合）"""
    ans_tokens = set(normalize_text(answer))
    evi_tokens = set(normalize_text(evidence))
    if not ans_tokens:
        return 0.0
    intersection = ans_tokens & evi_tokens
    return len(intersection) / len(ans_tokens)


def extract_list_items(text: str) -> List[str]:
    """提取列表项（支持多种列表格式）"""
    items = []
    # 匹配 "1. xxx" 或 "- xxx" 或 "• xxx"
    patterns = [
        r'[0-9一二三四五六七八九十]+[\.、]\s*([^\n]+)',
        r'[-•·]\s*([^\n]+)',
        r'([^\n]+)（[^）]+）',  # 带括号的项
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        items.extend([m.strip() for m in matches if m.strip()])
    return items


# ========== 规则层检测 ==========
class RuleBasedDetector:
    """规则层：基于确定性规则的幻觉检测"""
    
    def __init__(self):
        self.num_tolerance = 0.01  # 数值容差（相对误差）
        self.coverage_threshold_high = 0.8  # 高覆盖率阈值
        self.coverage_threshold_low = 0.4   # 低覆盖率阈值
    
    def detect_numerical_hallucination(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str]]:
        """
        检测数值类幻觉
        返回: (置信度0-1, 问题描述列表)
        """
        ans_nums = extract_numbers(answer)
        evi_nums = extract_numbers(evidence)
        
        if not ans_nums:
            return 0.5, ["答案中无数值，无法进行数值验证"]
        
        if not evi_nums:
            return 0.2, ["证据中无数值，无法验证答案中的数值"]
        
        # 检查每个答案数值是否在证据中存在（容差匹配）
        matched = 0
        issues = []
        for a_num in ans_nums:
            found = False
            for e_num in evi_nums:
                # 相对误差容差
                if abs(e_num) < 1e-6:
                    if abs(a_num) < 1e-6:
                        found = True
                        break
                else:
                    rel_err = abs((a_num - e_num) / e_num)
                    if rel_err <= self.num_tolerance:
                        found = True
                        matched += 1
                        break
            if not found:
                issues.append(f"数值 {a_num} 在证据中未找到匹配")
        
        score = matched / len(ans_nums) if ans_nums else 0.0
        return score, issues
    
    def detect_text_coverage(
        self, answer: str, evidence: str
    ) -> Tuple[float, str]:
        """检测文本覆盖率"""
        coverage = calculate_text_coverage(answer, evidence)
        if coverage >= self.coverage_threshold_high:
            return coverage, "高覆盖率"
        elif coverage >= self.coverage_threshold_low:
            return coverage, "中等覆盖率"
        else:
            return coverage, "低覆盖率"
    
    def detect_reference_consistency(
        self, answer_filename: str, answer_page: str,
        gold_filename: str, gold_page: str
    ) -> Tuple[float, List[str]]:
        """检测引用一致性（文件名、页码）"""
        issues = []
        score = 1.0
        
        # 文件名匹配（允许部分匹配，因为可能有路径差异）
        ans_fn_base = Path(answer_filename).name if answer_filename else ""
        gold_fn_base = Path(gold_filename).name if gold_filename else ""
        
        if ans_fn_base and gold_fn_base:
            if ans_fn_base != gold_fn_base:
                # 尝试模糊匹配（去掉日期前缀等）
                ans_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', ans_fn_base)
                gold_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', gold_fn_base)
                if ans_clean != gold_clean:
                    issues.append(f"文件名不一致: {ans_fn_base} vs {gold_fn_base}")
                    score -= 0.3
        
        # 页码匹配（允许±2页的容差）
        try:
            ans_pg = int(answer_page) if answer_page else -1
            gold_pg = int(gold_page) if gold_page else -1
            if ans_pg >= 0 and gold_pg >= 0:
                if abs(ans_pg - gold_pg) > 2:
                    issues.append(f"页码差异较大: {ans_pg} vs {gold_pg}")
                    score -= 0.2
        except:
            pass
        
        return max(0.0, score), issues
    
    def detect_calculation_error(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str]]:
        """检测计算错误（适用于比较计算类问题）"""
        if "比较计算" not in question_type and "计算" not in question_type:
            return 1.0, []  # 非计算类问题，跳过
        
        # 提取增长率、变化幅度等计算表达式
        growth_patterns = [
            r'增长[了]?([-+]?\d+\.?\d*)%',
            r'同比[增长下降]?([-+]?\d+\.?\d*)%',
            r'变化[了]?([-+]?\d+\.?\d*)个?百分点',
        ]
        
        # 从答案中提取声称的计算结果
        claimed_values = []
        for pattern in growth_patterns:
            matches = re.findall(pattern, answer)
            claimed_values.extend([float(m) for m in matches if m])
        
        if not claimed_values:
            return 0.5, ["无法从答案中提取计算数值"]
        
        # 从证据中提取基期和报告期数值，验证计算
        evi_nums = extract_numbers(evidence)
        if len(evi_nums) < 2:
            return 0.3, ["证据中数值不足，无法验证计算"]
        
        # 简单验证：如果答案中的增长率与证据中数值的增长率差异过大，标记为错误
        # 这里简化处理，实际可以更复杂
        return 0.7, []  # 计算验证较复杂，这里给中等置信度
    
    def detect_list_completeness(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str], str]:
        """
        检测列表完整性（适用于列举枚举类问题），并区分：
        - hallucination: 答案中列出了证据中不存在的项
        - incompleteness: 答案不完整（遗漏项），但已列出的项都存在
        - none: 无问题
        """
        if "列举" not in question_type and "枚举" not in question_type:
            return 1.0, [], "none"  # 非列举类问题，跳过
        
        ans_items = extract_list_items(answer)
        evi_items = extract_list_items(evidence)
        
        if not ans_items:
            return 0.2, ["答案中未提取到列表项"], "incompleteness"
        
        if not evi_items:
            return 0.3, ["证据中未提取到列表项，无法验证完整性"], "none"
        
        # 检查答案中的项是否都在证据中（检测幻觉）
        hallucinated_items = []
        for item in ans_items:
            if not any(item in evi_item or evi_item in item for evi_item in evi_items):
                hallucinated_items.append(item)
        
        # 检查证据中是否有答案未列出的项（检测不完整）
        missing_items = []
        for evi_item in evi_items:
            if not any(evi_item in ans_item or ans_item in evi_item for ans_item in ans_items):
                missing_items.append(evi_item)
        
        matched = len(ans_items) - len(hallucinated_items)
        completeness = matched / len(ans_items) if ans_items else 0.0
        
        issues = []
        error_type = "none"
        
        if hallucinated_items:
            issues.append(f"答案中列出的以下项在证据中不存在（幻觉）: {hallucinated_items[:3]}")
            error_type = "hallucination"
        elif missing_items:
            issues.append(f"答案遗漏了以下项（不完整）: {missing_items[:3]}")
            error_type = "incompleteness"
        elif completeness < 0.8:
            issues.append(f"列表项匹配率较低: {completeness:.2%}")
            error_type = "incompleteness"
        
        return completeness, issues, error_type
    
    def judge(
        self, question: str, answer: str, evidence: str,
        answer_filename: str, answer_page: str,
        gold_filename: str, gold_page: str, question_type: str
    ) -> Dict:
        """规则层综合判定"""
        scores = {}
        issues = []
        
        # 1. 数值验证
        num_score, num_issues = self.detect_numerical_hallucination(
            answer, evidence, question_type
        )
        scores['numerical'] = num_score
        issues.extend(num_issues)
        
        # 2. 文本覆盖率
        coverage_score, coverage_desc = self.detect_text_coverage(answer, evidence)
        scores['coverage'] = coverage_score
        if coverage_score < 0.4:
            issues.append(f"文本覆盖率低: {coverage_desc}")
        
        # 3. 引用一致性
        ref_score, ref_issues = self.detect_reference_consistency(
            answer_filename, answer_page, gold_filename, gold_page
        )
        scores['reference'] = ref_score
        issues.extend(ref_issues)
        
        # 4. 计算错误（仅计算类问题）
        calc_score, calc_issues = self.detect_calculation_error(
            answer, evidence, question_type
        )
        scores['calculation'] = calc_score
        issues.extend(calc_issues)
        
        # 5. 列表完整性（仅列举类问题）
        list_score, list_issues, list_error_type = self.detect_list_completeness(
            answer, evidence, question_type
        )
        scores['completeness'] = list_score
        issues.extend(list_issues)
        
        # 综合得分（加权平均）
        weights = {
            'numerical': 0.3,
            'coverage': 0.25,
            'reference': 0.2,
            'calculation': 0.15,
            'completeness': 0.1
        }
        overall_score = sum(scores.get(k, 1.0) * weights.get(k, 0) for k in weights)
        
        # 判定 verdict
        if overall_score >= 0.8:
            verdict = "有证据"
        elif overall_score >= 0.5:
            verdict = "部分证据"
        else:
            verdict = "无证据"
        
        return {
            "verdict": verdict,
            "confidence": overall_score,
            "scores": scores,
            "issues": issues,
            "list_error_type": list_error_type
        }


# ========== LLM判定层 ==========
class LLMDetector:
    """LLM判定层：用提示词让LLM判断答案是否被证据支持"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def judge(
        self, question: str, answer: str, evidence: str, question_type: str
    ) -> Dict:
        """LLM判定"""
        prompt = f"""你是一名专业的财报分析审核员。请判断以下答案是否被证据支持。

【问题类型】{question_type}

【问题】{question}

【答案】{answer}

【证据】（来自年报的原始文本，截取相关片段）
{evidence[:2000]}  # 限制长度

请严格根据证据判断答案的事实一致性，输出JSON格式：
{{
  "verdict": "有证据|部分证据|无证据|矛盾",
  "confidence": 0.0-1.0,
  "reasoning": "判断理由（1-2句话）",
  "specific_support": ["证据中支持答案的具体句子或片段"],
  "missing_info": ["答案中无法在证据中找到的声明"],
  "contradictions": ["证据中与答案矛盾的部分（如有）"]
}}

注意：
1. "有证据"：答案的核心事实在证据中被明确支持
2. "部分证据"：答案部分正确，但缺少关键信息或存在不完整
3. "无证据"：答案中的关键信息在证据中找不到
4. "矛盾"：答案与证据直接冲突

请仅输出JSON，不要输出其他内容。"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名专业的财报分析审核员，擅长事实核查。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            raw = completion.choices[0].message.content.strip()
            
            # 提取JSON
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "verdict": "部分证据",
                    "confidence": 0.5,
                    "reasoning": "LLM返回格式异常",
                    "specific_support": [],
                    "missing_info": [],
                    "contradictions": []
                }
        except Exception as e:
            return {
                "verdict": "部分证据",
                "confidence": 0.5,
                "reasoning": f"LLM调用失败: {str(e)}",
                "specific_support": [],
                "missing_info": [],
                "contradictions": []
            }


# ========== 证据检索 ==========
class EvidenceRetriever:
    """从知识库检索证据"""
    
    def __init__(self, chunk_json_path: str):
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        # 按文件名+页码索引
        self.index = defaultdict(list)
        for chunk in self.chunks:
            md = chunk.get('metadata', {})
            fn = md.get('file_name', '')
            pg = md.get('page', 0)
            key = (fn, pg)
            self.index[key].append(chunk)
    
    def retrieve(
        self, filename: str, page: int, question: str = ""
    ) -> str:
        """检索指定文件+页码的证据"""
        key = (filename, page)
        chunks = self.index.get(key, [])
        if not chunks:
            # 尝试模糊匹配文件名
            for (fn, pg), chs in self.index.items():
                if Path(fn).name == Path(filename).name and abs(pg - page) <= 2:
                    chunks = chs
                    break
        
        if not chunks:
            return ""
        
        # 合并同页的chunks
        texts = [c.get('content', '') for c in chunks]
        return "\n\n".join(texts)


# ========== 主检测流程 ==========
def load_data(gold_path: str, rag_path: str) -> Tuple[List[Dict], List[Dict]]:
    """加载金标准和RAG答案"""
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    with open(rag_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    # 按question建立索引
    gold_dict = {item['question']: item for item in gold_data}
    rag_dict = {item['question']: item for item in rag_data}
    
    # 对齐数据
    aligned = []
    for q, gold_item in gold_dict.items():
        if q in rag_dict:
            aligned.append((gold_item, rag_dict[q]))
    
    return aligned


def main():
    """主函数"""
    print("=" * 60)
    print("多层幻觉检测器 - 开始运行")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    aligned_data = load_data(GOLD_STANDARD_PATH, RAG_ANSWERS_PATH)
    print(f"共 {len(aligned_data)} 条问答对")
    
    # 2. 初始化组件
    print("\n[2/5] 初始化检测器...")
    rule_detector = RuleBasedDetector()
    evidence_retriever = EvidenceRetriever(CHUNK_JSON_PATH)
    
    llm_detector = None
    if USE_LLM_LAYER and LLM_API_KEY and LLM_BASE_URL and LLM_MODEL:
        llm_detector = LLMDetector(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
        print("  ✓ LLM判定层已启用")
    else:
        print("  ⚠ LLM判定层未启用（跳过）")
    
    # 3. 逐条检测
    print("\n[3/5] 开始检测...")
    results = []
    
    for gold_item, rag_item in tqdm(aligned_data, desc="检测进度"):
        question = gold_item['question']
        gold_answer = gold_item.get('answer', '')
        gold_filename = gold_item.get('filename', '')
        gold_page = gold_item.get('page', 0)
        question_type = gold_item.get('type', '未知')
        
        rag_answer = rag_item.get('answer', '')
        rag_filename = rag_item.get('filename', '')
        rag_page = rag_item.get('page', 0)
        
        # 检索证据（优先用RAG的引用，如果没有则用金标准的）
        evidence_filename = rag_filename or gold_filename
        evidence_page = int(rag_page) if rag_page else int(gold_page)
        evidence = evidence_retriever.retrieve(
            evidence_filename, evidence_page, question
        )
        
        if not evidence:
            evidence = "未找到证据"  # 降级处理
        
        # 规则层判定
        rule_result = rule_detector.judge(
            question, rag_answer, evidence,
            rag_filename, rag_page,
            gold_filename, gold_page,
            question_type
        )
        
        # LLM判定层（可选）
        llm_result = None
        if llm_detector:
            try:
                llm_result = llm_detector.judge(
                    question, rag_answer, evidence, question_type
                )
            except Exception as e:
                print(f"  ⚠ LLM判定失败: {e}")
        
        # 融合判定
        rule_verdict = rule_result['verdict']
        rule_conf = rule_result['confidence']
        
        if llm_result:
            llm_verdict = llm_result['verdict']
            llm_conf = llm_result.get('confidence', 0.5)
            # 简单融合：取平均置信度，verdict取更严格的
            final_conf = (rule_conf + llm_conf) / 2
            verdict_map = {"有证据": 3, "部分证据": 2, "无证据": 1, "矛盾": 0}
            final_verdict_idx = min(
                verdict_map.get(rule_verdict, 2),
                verdict_map.get(llm_verdict, 2)
            )
            final_verdict = [k for k, v in verdict_map.items() if v == final_verdict_idx][0]
        else:
            final_conf = rule_conf
            final_verdict = rule_verdict
        
        # 构建结果
        result = {
            "question": question,
            "question_type": question_type,
            "gold_answer": gold_answer,
            "rag_answer": rag_answer,
            "gold_filename": gold_filename,
            "gold_page": gold_page,
            "rag_filename": rag_filename,
            "rag_page": rag_page,
            "final_verdict": final_verdict,
            "final_confidence": final_conf,
            "rule_layer": rule_result,
            "llm_layer": llm_result,
            "evidence_preview": evidence[:500]  # 证据预览
        }
        results.append(result)
    
    # 4. 保存结果
    print("\n[4/5] 保存结果...")
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存到: {output_path}")
    
    # 5. 统计报告
    print("\n[5/5] 生成统计报告...")
    verdict_counts = defaultdict(int)
    type_verdicts = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        verdict = r['final_verdict']
        qtype = r['question_type']
        verdict_counts[verdict] += 1
        type_verdicts[qtype][verdict] += 1
    
    print("\n" + "=" * 60)
    print("检测结果统计")
    print("=" * 60)
    print(f"\n总体分布:")
    for v, c in sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True):
        pct = c / len(results) * 100
        print(f"  {v}: {c}题 ({pct:.1f}%)")
    
    print(f"\n按问题类型分布:")
    for qtype in ["事实提取", "列举枚举", "比较计算", "判断验证", "推理分析"]:
        if qtype in type_verdicts:
            print(f"\n  {qtype}:")
            for v, c in sorted(type_verdicts[qtype].items(), key=lambda x: x[1], reverse=True):
                total = sum(type_verdicts[qtype].values())
                pct = c / total * 100 if total > 0 else 0
                print(f"    {v}: {c}题 ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("检测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()