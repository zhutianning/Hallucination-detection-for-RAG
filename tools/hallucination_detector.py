"""
Multi-layer hallucination detector (no classifier training required)
Based on rule layer + LLM judging layer + evidence alignment layer
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

# ========== Configuration ==========
CHUNK_JSON_PATH = Path(__file__).resolve().parent.parent / "all_pdf_page_chunks_merged_2.json"  # Knowledge-base chunks
GOLD_STANDARD_PATH = Path(__file__).resolve().parent.parent / "datas" / "gold_standard_500.json" # Gold standard answers
RAG_ANSWERS_PATH = Path(__file__).resolve().parent.parent / "rag_top1_pred_llama-3.1-8b-instruct.json"  # RAG-generated answers
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "hallucination_detection_results_llama-3.1-8b-instruct.json"

# LLM judging layer config (optional)
USE_LLM_LAYER = False
LLM_API_KEY = os.getenv('LOCAL_API_KEY')
LLM_BASE_URL = os.getenv('LOCAL_BASE_URL')
LLM_MODEL = os.getenv('LOCAL_TEXT_MODEL')


# ========== Utility Functions ==========
def normalize_text(text: str) -> str:
    """Normalize text: remove punctuation, convert full-width chars, strip spaces."""
    if not text:
        return ""
    # Convert full-width punctuation to half-width
    text = text.translate(str.maketrans('，。！？；：""''（）【】', ',.!?;:"\'()[]'))
    # Remove punctuation and spaces
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return text.lower()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text (percentages, decimals, comma-separated)."""
    # Remove thousand separators
    text_clean = text.replace(',', '').replace('，', '')
    # Match numeric patterns (including decimals and percentages)
    patterns = [
        r'(-?\d+\.?\d*)%',  # Percentage
        r'(-?\d+\.?\d*)',   # Plain number
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
    return list(set(numbers))  # Deduplicate


def normalize_number(num: float, unit: str = "") -> float:
    """Normalize numeric value with unit conversion (e.g., 100M -> 10K unit base)."""
    unit_map = {
        "亿": 10000, "万元": 1, "万元": 1, "万": 1,
        "千": 0.1, "元": 0.0001
    }
    if unit in unit_map:
        return num * unit_map[unit]
    return num


def calculate_text_coverage(answer: str, evidence: str) -> float:
    """Compute text coverage of answer against evidence (token-set based)."""
    ans_tokens = set(normalize_text(answer))
    evi_tokens = set(normalize_text(evidence))
    if not ans_tokens:
        return 0.0
    intersection = ans_tokens & evi_tokens
    return len(intersection) / len(ans_tokens)


def extract_list_items(text: str) -> List[str]:
    """Extract list items (supports multiple list formats)."""
    items = []
    # Match "1. xxx", "- xxx", or "• xxx"
    patterns = [
        r'[0-9一二三四五六七八九十]+[\.、]\s*([^\n]+)',
        r'[-•·]\s*([^\n]+)',
        r'([^\n]+)（[^）]+）',  # Item with parentheses
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        items.extend([m.strip() for m in matches if m.strip()])
    return items


def parse_page_to_int(page_value, default: int = 0) -> int:
    """
    Parse page field into an integer page number.
    Supports values like 5, "5", "5,9", "5-9", "第5页".
    Returns the first detected integer; falls back to default.
    """
    if page_value is None:
        return default

    if isinstance(page_value, (int, float)):
        return int(page_value)

    page_str = str(page_value).strip()
    if not page_str:
        return default

    matches = re.findall(r"\d+", page_str)
    if matches:
        return int(matches[0])
    return default


# ========== Rule-Based Detection ==========
class RuleBasedDetector:
    """Rule layer: hallucination detection based on deterministic rules."""
    
    def __init__(self):
        self.num_tolerance = 0.01  # Numeric tolerance (relative error)
        self.coverage_threshold_high = 0.8  # High coverage threshold
        self.coverage_threshold_low = 0.4   # Low coverage threshold
    
    def detect_numerical_hallucination(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str]]:
        """
        Detect numeric hallucinations.
        Returns: (confidence score 0-1, issue description list)
        """
        ans_nums = extract_numbers(answer)
        evi_nums = extract_numbers(evidence)
        
        if not ans_nums:
            return 0.5, ["No numeric value found in answer; numeric validation is not possible"]
        
        if not evi_nums:
            return 0.2, ["No numeric value found in evidence; cannot validate answer numbers"]
        
        # Check whether each answer-side number appears in evidence (tolerance match)
        matched = 0
        issues = []
        for a_num in ans_nums:
            found = False
            for e_num in evi_nums:
                # Relative error tolerance
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
                issues.append(f"Numeric value {a_num} has no match in evidence")
        
        score = matched / len(ans_nums) if ans_nums else 0.0
        return score, issues
    
    def detect_text_coverage(
        self, answer: str, evidence: str
    ) -> Tuple[float, str]:
        """Detect text coverage."""
        coverage = calculate_text_coverage(answer, evidence)
        if coverage >= self.coverage_threshold_high:
            return coverage, "High coverage"
        elif coverage >= self.coverage_threshold_low:
            return coverage, "Medium coverage"
        else:
            return coverage, "Low coverage"
    
    def detect_reference_consistency(
        self, answer_filename: str, answer_page: str,
        gold_filename: str, gold_page: str
    ) -> Tuple[float, List[str]]:
        """Detect citation consistency (filename and page)."""
        issues = []
        score = 1.0
        
        # Filename matching (partial match allowed because paths may differ)
        ans_fn_base = Path(answer_filename).name if answer_filename else ""
        gold_fn_base = Path(gold_filename).name if gold_filename else ""
        
        if ans_fn_base and gold_fn_base:
            if ans_fn_base != gold_fn_base:
                # Try fuzzy matching (remove date prefixes, etc.)
                ans_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', ans_fn_base)
                gold_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', gold_fn_base)
                if ans_clean != gold_clean:
                    issues.append(f"Filename mismatch: {ans_fn_base} vs {gold_fn_base}")
                    score -= 0.3
        
        # Page matching (allow ±2-page tolerance)
        try:
            ans_pg = int(answer_page) if answer_page else -1
            gold_pg = int(gold_page) if gold_page else -1
            if ans_pg >= 0 and gold_pg >= 0:
                if abs(ans_pg - gold_pg) > 2:
                    issues.append(f"Large page gap: {ans_pg} vs {gold_pg}")
                    score -= 0.2
        except:
            pass
        
        return max(0.0, score), issues
    
    def detect_calculation_error(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str]]:
        """Detect calculation errors (for computation/comparison questions)."""
        if "比较计算" not in question_type and "计算" not in question_type:
            return 1.0, []  # Non-calculation question, skip
        
        # Extract growth/change expressions
        growth_patterns = [
            r'增长[了]?([-+]?\d+\.?\d*)%',
            r'同比[增长下降]?([-+]?\d+\.?\d*)%',
            r'变化[了]?([-+]?\d+\.?\d*)个?百分点',
        ]
        
        # Extract claimed calculation results from answer
        claimed_values = []
        for pattern in growth_patterns:
            matches = re.findall(pattern, answer)
            claimed_values.extend([float(m) for m in matches if m])
        
        if not claimed_values:
            return 0.5, ["Cannot extract calculation values from answer"]
        
        # Extract baseline/reporting values from evidence for calculation checks
        evi_nums = extract_numbers(evidence)
        if len(evi_nums) < 2:
            return 0.3, ["Insufficient numeric values in evidence; cannot verify calculation"]
        
        # Simple heuristic: if growth rate differs too much from evidence-inferred values, mark as error
        # Simplified for now; can be expanded with stronger formula-level checks
        return 0.7, []  # Calculation validation is complex; use medium confidence for now
    
    def detect_list_completeness(
        self, answer: str, evidence: str, question_type: str
    ) -> Tuple[float, List[str], str]:
        """
        Detect list completeness (for enumerative/listing questions), and distinguish:
        - hallucination: answer contains items not present in evidence
        - incompleteness: answer misses items but listed items are supported
        - none: no issue
        """
        if "列举" not in question_type and "枚举" not in question_type:
            return 1.0, [], "none"  # Non-enumeration question, skip
        
        ans_items = extract_list_items(answer)
        evi_items = extract_list_items(evidence)
        
        if not ans_items:
            return 0.2, ["No list item extracted from answer"], "incompleteness"
        
        if not evi_items:
            return 0.3, ["No list item extracted from evidence; completeness cannot be verified"], "none"
        
        # Check whether answer items are all present in evidence (hallucination check)
        hallucinated_items = []
        for item in ans_items:
            if not any(item in evi_item or evi_item in item for evi_item in evi_items):
                hallucinated_items.append(item)
        
        # Check whether evidence has items omitted by answer (incompleteness check)
        missing_items = []
        for evi_item in evi_items:
            if not any(evi_item in ans_item or ans_item in evi_item for ans_item in ans_items):
                missing_items.append(evi_item)
        
        matched = len(ans_items) - len(hallucinated_items)
        completeness = matched / len(ans_items) if ans_items else 0.0
        
        issues = []
        error_type = "none"
        
        if hallucinated_items:
            issues.append(f"The following answer items are not found in evidence (hallucination): {hallucinated_items[:3]}")
            error_type = "hallucination"
        elif missing_items:
            issues.append(f"The following items are missing from answer (incompleteness): {missing_items[:3]}")
            error_type = "incompleteness"
        elif completeness < 0.8:
            issues.append(f"Low list-item match rate: {completeness:.2%}")
            error_type = "incompleteness"
        
        return completeness, issues, error_type
    
    def judge(
        self, question: str, answer: str, evidence: str,
        answer_filename: str, answer_page: str,
        gold_filename: str, gold_page: str, question_type: str
    ) -> Dict:
        """Rule-layer final judgment."""
        scores = {}
        issues = []
        
        # 1. Numeric validation
        num_score, num_issues = self.detect_numerical_hallucination(
            answer, evidence, question_type
        )
        scores['numerical'] = num_score
        issues.extend(num_issues)
        
        # 2. Text coverage
        coverage_score, coverage_desc = self.detect_text_coverage(answer, evidence)
        scores['coverage'] = coverage_score
        if coverage_score < 0.4:
            issues.append(f"Low text coverage: {coverage_desc}")
        
        # 3. Citation consistency
        ref_score, ref_issues = self.detect_reference_consistency(
            answer_filename, answer_page, gold_filename, gold_page
        )
        scores['reference'] = ref_score
        issues.extend(ref_issues)
        
        # 4. Calculation error (calculation-type questions only)
        calc_score, calc_issues = self.detect_calculation_error(
            answer, evidence, question_type
        )
        scores['calculation'] = calc_score
        issues.extend(calc_issues)
        
        # 5. List completeness (enumeration-type questions only)
        list_score, list_issues, list_error_type = self.detect_list_completeness(
            answer, evidence, question_type
        )
        scores['completeness'] = list_score
        issues.extend(list_issues)
        
        # Overall score (weighted average)
        weights = {
            'numerical': 0.3,
            'coverage': 0.25,
            'reference': 0.2,
            'calculation': 0.15,
            'completeness': 0.1
        }
        overall_score = sum(scores.get(k, 1.0) * weights.get(k, 0) for k in weights)
        
        # Final verdict
        if overall_score >= 0.8:
            verdict = "supported"
        elif overall_score >= 0.5:
            verdict = "partially_supported"
        else:
            verdict = "unsupported"
        
        return {
            "verdict": verdict,
            "confidence": overall_score,
            "scores": scores,
            "issues": issues,
            "list_error_type": list_error_type
        }


# ========== LLM Judging Layer ==========
class LLMDetector:
    """LLM layer: use prompting to judge whether the answer is evidence-supported."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def judge(
        self, question: str, answer: str, evidence: str, question_type: str
    ) -> Dict:
        """LLM judgment."""
        prompt = f"""你是一名专业的财报分析审核员。请判断以下答案是否被证据支持。

【问题类型】{question_type}

【问题】{question}

【答案】{answer}

【证据】（来自年报的原始文本，截取相关片段）
{evidence[:2000]}  # Length limit

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
            
            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "verdict": "partially_supported",
                    "confidence": 0.5,
                    "reasoning": "Invalid JSON format returned by LLM",
                    "specific_support": [],
                    "missing_info": [],
                    "contradictions": []
                }
        except Exception as e:
            return {
                "verdict": "partially_supported",
                "confidence": 0.5,
                "reasoning": f"LLM call failed: {str(e)}",
                "specific_support": [],
                "missing_info": [],
                "contradictions": []
            }


# ========== Evidence Retrieval ==========
class EvidenceRetriever:
    """Retrieve evidence from the chunk knowledge base."""
    
    def __init__(self, chunk_json_path: str):
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        # Build index by filename + page
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
        """Retrieve evidence for specific filename + page."""
        key = (filename, page)
        chunks = self.index.get(key, [])
        if not chunks:
            # Try fuzzy filename matching
            for (fn, pg), chs in self.index.items():
                if Path(fn).name == Path(filename).name and abs(pg - page) <= 2:
                    chunks = chs
                    break
        
        if not chunks:
            return ""
        
        # Merge chunks from the same page
        texts = [c.get('content', '') for c in chunks]
        return "\n\n".join(texts)


# ========== Main Detection Flow ==========
def load_data(gold_path: str, rag_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load gold standard and RAG answers."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    with open(rag_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    # Build index by question
    gold_dict = {item['question']: item for item in gold_data}
    rag_dict = {item['question']: item for item in rag_data}
    
    # Align data pairs by question
    aligned = []
    for q, gold_item in gold_dict.items():
        if q in rag_dict:
            aligned.append((gold_item, rag_dict[q]))
    
    return aligned


def main():
    """Main entrypoint."""
    print("=" * 60)
    print("Multi-layer Hallucination Detector - Start")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    aligned_data = load_data(GOLD_STANDARD_PATH, RAG_ANSWERS_PATH)
    print(f"Loaded {len(aligned_data)} aligned QA pairs")
    
    # 2. Initialize components
    print("\n[2/5] Initializing detectors...")
    rule_detector = RuleBasedDetector()
    evidence_retriever = EvidenceRetriever(CHUNK_JSON_PATH)
    
    llm_detector = None
    if USE_LLM_LAYER and LLM_API_KEY and LLM_BASE_URL and LLM_MODEL:
        llm_detector = LLMDetector(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
        print("  ✓ LLM judging layer enabled")
    else:
        print("  ⚠ LLM judging layer disabled (skipped)")
    
    # 3. Per-item detection
    print("\n[3/5] Running detection...")
    results = []
    
    for gold_item, rag_item in tqdm(aligned_data, desc="Detection progress"):
        question = gold_item['question']
        gold_answer = gold_item.get('answer', '')
        gold_filename = gold_item.get('filename', '')
        gold_page = gold_item.get('page', 0)
        question_type = gold_item.get('type', 'unknown')
        
        rag_answer = rag_item.get('answer', '')
        rag_filename = rag_item.get('filename', '')
        rag_page = rag_item.get('page', 0)
        
        # Retrieve evidence (prefer RAG citation, fallback to gold citation)
        evidence_filename = rag_filename or gold_filename
        evidence_page = parse_page_to_int(rag_page, default=parse_page_to_int(gold_page, default=0))
        evidence = evidence_retriever.retrieve(
            evidence_filename, evidence_page, question
        )
        
        if not evidence:
            evidence = "No evidence found"  # Fallback handling
        
        # Rule-layer judgment
        rule_result = rule_detector.judge(
            question, rag_answer, evidence,
            rag_filename, rag_page,
            gold_filename, gold_page,
            question_type
        )
        
        # LLM judging layer (optional)
        llm_result = None
        if llm_detector:
            try:
                llm_result = llm_detector.judge(
                    question, rag_answer, evidence, question_type
                )
            except Exception as e:
                print(f"  ⚠ LLM judgment failed: {e}")
        
        # Fuse judgments
        rule_verdict = rule_result['verdict']
        rule_conf = rule_result['confidence']
        
        if llm_result:
            llm_verdict = llm_result['verdict']
            llm_conf = llm_result.get('confidence', 0.5)
            # Simple fusion: average confidence; use stricter verdict
            final_conf = (rule_conf + llm_conf) / 2
            verdict_map = {"supported": 3, "partially_supported": 2, "unsupported": 1, "contradiction": 0}
            final_verdict_idx = min(
                verdict_map.get(rule_verdict, 2),
                verdict_map.get(llm_verdict, 2)
            )
            final_verdict = [k for k, v in verdict_map.items() if v == final_verdict_idx][0]
        else:
            final_conf = rule_conf
            final_verdict = rule_verdict
        
        # Build result object
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
            "evidence_preview": evidence[:500]  # Evidence preview
        }
        results.append(result)
    
    # 4. Save results
    print("\n[4/5] Saving results...")
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved to: {output_path}")
    
    # 5. Statistics report
    print("\n[5/5] Generating report...")
    verdict_counts = defaultdict(int)
    type_verdicts = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        verdict = r['final_verdict']
        qtype = r['question_type']
        verdict_counts[verdict] += 1
        type_verdicts[qtype][verdict] += 1
    
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    print(f"\nOverall distribution:")
    for v, c in sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True):
        pct = c / len(results) * 100
        print(f"  {v}: {c} items ({pct:.1f}%)")
    
    print(f"\nDistribution by question type:")
    qtype_display_map = {
        "事实提取": "fact_extraction",
        "列举枚举": "enumeration",
        "比较计算": "comparative_calculation",
        "判断验证": "judgment_verification",
        "推理分析": "reasoning_analysis",
    }
    for qtype in ["事实提取", "列举枚举", "比较计算", "判断验证", "推理分析"]:
        if qtype in type_verdicts:
            print(f"\n  {qtype_display_map.get(qtype, qtype)}:")
            for v, c in sorted(type_verdicts[qtype].items(), key=lambda x: x[1], reverse=True):
                total = sum(type_verdicts[qtype].values())
                pct = c / total * 100 if total > 0 else 0
                print(f"    {v}: {c} items ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Detection completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()