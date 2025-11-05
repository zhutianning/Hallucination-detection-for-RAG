import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
MERGED = (BASE_DIR / "all_pdf_page_chunks_merged.json").resolve()
PDF_DIR = (BASE_DIR / "datas/年报").resolve()
OUT = (BASE_DIR / "datas/test_advanced_250.json").resolve()


def list_target_reports(pdf_dir: Path) -> List[str]:
    files = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        name = p.name
        # 过滤审计报告与摘要
        if "审计报告" in name or "摘要" in name:
            continue
        # 仅保留年度报告（包含“年度报告”或“年年度报告”或“年度报告.pdf”）
        if "年度报告" in name:
            files.append(name)
    return files


def load_merged_chunks(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_pages_by_file(chunks: List[Dict]) -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = defaultdict(dict)
    for c in chunks:
        md = c.get("metadata", {})
        fn = md.get("file_name", "").strip()
        pg = md.get("page", 0)
        t = c.get("content", "").strip()
        if not fn or not isinstance(pg, int) or not t:
            continue
        # 以最后写入为准（同页多块合并后已去重切块，这里简单覆盖）
        if pg not in out[fn]:
            out[fn][pg] = t
        else:
            # 同页累加，便于关键词命中
            out[fn][pg] = (out[fn][pg] + "\n" + t).strip()
    return out


def extract_company_and_year(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split("-")
    # 规则：优先选择包含中文、且不含“年度报告/审计报告/摘要”的片段，
    # 同时排除形如“600030.SH”或纯数字、日期段
    def is_candidate(s: str) -> bool:
        if not re.search(r"[\u4e00-\u9fff]", s):
            return False
        if any(x in s for x in ["年度报告", "审计报告", "摘要"]):
            return False
        if re.match(r"^\d{6}\.[A-Z]{2,}$", s):
            return False
        if re.match(r"^\d{4}$", s):
            return False
        if re.match(r"^\d{2}$", s):
            return False
        return True

    candidates = [p for p in parts if is_candidate(p)]
    if candidates:
        # 选择中文字符数最多的片段作为公司名
        company = max(candidates, key=lambda x: len(re.findall(r"[\u4e00-\u9fff]", x)))
    elif len(parts) >= 5:
        company = parts[4]
    else:
        company = stem

    # 报告年度：匹配“(\d{4})年?年度报告”
    m = re.search(r"(\d{4})年?年度报告", filename)
    year = m.group(1) if m else ""
    return company, year


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "管理层讨论": ["管理层讨论", "讨论与分析", "经营情况", "经营回顾"],
    "风险因素": ["风险", "不确定", "风险提示"],
    "分部业务": ["分部", "分行业", "分产品", "分地区"],
    "财务概览": ["主要财务指标", "财务指标", "关键指标", "利润表", "资产负债表", "现金流量表"],
    "现金流": ["现金流", "经营活动现金流", "投资活动现金流", "筹资活动现金流"],
    "股利分红": ["分红", "股利", "利润分配"],
    "资本开支": ["资本开支", "固定资产投资", "资本性支出", "在建工程"],
    "研发投入": ["研发", "研究与开发", "技术投入"],
    "审计意见": ["审计意见", "保留意见", "无保留意见", "关键审计事项"],
    "治理结构": ["公司治理", "董事会", "监事会", "高级管理人员"],
    "客户供应": ["前五大客户", "前五大供应商", "客户集中", "供应商集中"],
    "地区结构": ["地区", "境内", "境外", "海外"],
    "费用变动": ["费用", "销售费用", "管理费用", "财务费用"],
    "毛利率": ["毛利率", "净利率", "利润率"],
    "存货应收": ["存货", "存货跌价", "应收账款", "周转"],
    "负债结构": ["负债", "债务", "到期", "利息"],
    "非经常损益": ["非经常性损益", "公允价值变动", "资产处置"],
    "ESG": ["ESG", "社会责任", "环境", "可持续"],
    "展望指引": ["展望", "指引", "计划", "目标"],
}


def find_pages_by_keywords(pages: Dict[int, str], keywords: List[str], max_hits: int = 5) -> List[int]:
    hits = []
    for pg, text in pages.items():
        t = text[:10000]  # 限制长度以提速
        for k in keywords:
            if k in t:
                hits.append(pg)
                break
        if len(hits) >= max_hits:
            break
    return sorted(set(hits))


def build_question_pool(company: str, year: str) -> List[Dict[str, str]]:
    """
    为每个公司生成5类×10题=50题的问题池，带类型标签
    返回: List[Dict] 每项包含 'question' 和 'type' 字段
    """
    y = year or "报告期内"
    c = company
    
    questions = []
    
    # ========== 类型1：事实提取类（10题）- 直接查询单个具体数值、名称、日期 ==========
    fact_questions = [
        f"{c}{y}年度的营业收入总额是多少亿元？",
        f"{c}{y}年度归属于母公司股东的净利润是多少亿元？",
        f"{c}{y}年度末总资产规模达到多少亿元？",
        f"{c}{y}年度的基本每股收益（EPS）是多少元？",
        f"{c}{y}年度的加权平均净资产收益率（ROE）是多少？",
        f"{c}{y}年度末的资产负债率是多少？",
        f"{c}{y}年度的研发费用总额是多少亿元？",
        f"{c}{y}年度经营活动产生的现金流量净额是多少亿元？",
        f"{c}{y}年度的毛利率是多少？",
        f"{c}{y}年度支付的现金股利总额是多少亿元？",
    ]
    questions.extend([{"question": q, "type": "事实提取"} for q in fact_questions])
    
    # ========== 类型2：列举枚举类（10题）- 列出多个项目/明细 ==========
    list_questions = [
        f"请列举{c}{y}年度前五大客户的名称及其销售收入占比。",
        f"请列举{c}{y}年度前五大供应商的名称及其采购金额占比。",
        f"{c}{y}年度主要业务板块有哪些？各板块收入占比分别是多少？",
        f"请列举{c}{y}年度主要子公司的名称、持股比例及其对合并报表净利润的贡献。",
        f"{c}{y}年度前十大股东分别是谁？持股比例各是多少？",
        f"请列举{c}{y}年度董事会成员的姓名及其担任的职务。",
        f"{c}{y}年度主要在建工程项目有哪些？各项目的投资进度与预算分别如何？",
        f"请列举{c}{y}年度发生的重大关联交易，包括交易对手、交易内容和金额。",
        f"{c}{y}年度获得的政府补助项目有哪些？各项补助金额是多少？",
        f"请列举{c}{y}年度持有的前五大金融资产或投资项目及其账面价值。",
    ]
    questions.extend([{"question": q, "type": "列举枚举"} for q in list_questions])
    
    # ========== 类型3：比较计算类（10题）- 同比/环比/比率/趋势计算 ==========
    compare_questions = [
        f"计算{c}{y}年度营业收入同比增长率和增长金额。",
        f"计算{c}{y}年度净利润同比增长率和增长金额。",
        f"计算{c}{y}年度与上一年度毛利率的变化幅度（百分点）。",
        f"计算{c}{y}年度与上一年度ROE的变化幅度（百分点）。",
        f"计算{c}{y}年度销售费用率与上一年度的变化。",
        f"计算{c}{y}年度应收账款周转天数与上一年度的差异。",
        f"比较{c}{y}年度Q1至Q4各季度营业收入，确定最高季度及其占比。",
        f"计算{c}{y}年度资产负债率与上一年度的变化幅度（百分点）。",
        f"计算{c}{y}年度研发费用率与上一年度的变化。",
        f"计算{c}{y}年度经营现金流/净利润比值与上一年度的对比。",
    ]
    questions.extend([{"question": q, "type": "比较计算"} for q in compare_questions])
    
    # ========== 类型4：判断验证类（10题）- 是否披露/是否存在 ==========
    judge_questions = [
        f"{c}{y}年度是否实施现金分红？如实施，分红金额和分红率各是多少？",
        f"{c}{y}年度是否存在商誉减值？如存在，减值金额是多少？",
        f"{c}{y}年度是否完成重大并购或资产重组？如完成，交易标的和对价各是什么？",
        f"{c}{y}年度是否实施股权激励计划？如实施，激励对象数量和授予股数各是多少？",
        f"{c}{y}年度审计意见类型是什么？是否存在关键审计事项或强调事项段？",
        f"{c}{y}年度是否存在对外担保？如存在，担保总额和主要被担保方是什么？",
        f"{c}{y}年度是否发生重大诉讼或仲裁？如发生，涉案金额和进展情况如何？",
        f"{c}{y}年度是否发生控股股东或实际控制人变更？如变更，新旧控制人分别是谁？",
        f"{c}{y}年度是否存在募集资金？如存在，使用进度和是否变更募投项目？",
        f"{c}{y}年度是否发生会计政策变更？如变更，变更内容和影响金额各是多少？",
    ]
    questions.extend([{"question": q, "type": "判断验证"} for q in judge_questions])
    
    # ========== 类型5：推理分析类（10题）- 归因/评估/解释 ==========
    analyze_questions = [
        f"归因分析：{c}{y}年度营业收入增长的主要驱动因素（价格vs数量）。",
        f"原因解释：{c}{y}年度毛利率变化是成本驱动还是结构优化？",
        f"差异分析：{c}{y}年度净利润与经营现金流净额差异的会计调整项。",
        f"风险评估：{c}{y}年度应收账款增长是否存在信用风险或激进确认？",
        f"效率分析：{c}{y}年度期间费用率变化与规模效应、战略投入的关系。",
        f"杠杆评估：{c}{y}年度资产负债率变化是扩张需求还是现金流压力？",
        f"库存分析：{c}{y}年度存货增长是正常备货还是存在滞销风险？",
        f"ROE拆解：{c}{y}年度ROE变化的杜邦三因素贡献度分析。",
        f"战略评估：{c}{y}年度研发投入强度与技术壁垒构建的匹配性。",
        f"质量分析：{c}{y}年度非经常性损益占比及对盈利可持续性的影响。",
    ]
    questions.extend([{"question": q, "type": "推理分析"} for q in analyze_questions])
    
    return questions


def assign_pages_for_questions(pages: Dict[int, str], questions: List[str]) -> List[int]:
    # 为每条问题分配一个尽量相关的页码（启发式）：按关键词类别映射
    category_to_pages: Dict[str, List[int]] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        category_to_pages[cat] = find_pages_by_keywords(pages, kws, max_hits=10)

    def pick_page_for(q: str) -> int:
        # 依据类别关键词命中优先选择对应页
        for cat, pgs in category_to_pages.items():
            for kw in CATEGORY_KEYWORDS[cat]:
                if kw in q:
                    if pgs:
                        return pgs[0]
        # 回退：选择中位页
        if pages:
            return sorted(pages.keys())[len(pages)//2]
        return 1

    return [pick_page_for(q) for q in questions]


def main():
    pdfs = list_target_reports(PDF_DIR)
    chunks = load_merged_chunks(MERGED)
    file_to_pages = group_pages_by_file(chunks)

    n_reports = len(pdfs)
    if n_reports == 0:
        raise RuntimeError("未找到可用的年度报告PDF")

    TOTAL = 250
    TARGET_PER_TYPE = 50  # 每种类型严格50题
    
    results: List[Dict] = []
    seen_questions = set()
    type_counts = Counter()  # 实时统计各类型数量
    
    type_order = ["事实提取", "列举枚举", "比较计算", "判断验证", "推理分析"]

    # 第一轮：均衡分配，严格控制每类不超过50题
    for idx, fn in enumerate(pdfs):
        company, year = extract_company_and_year(fn)
        pool_items = build_question_pool(company, year)
        
        # 按类型分组
        type_groups = defaultdict(list)
        for item in pool_items:
            type_groups[item['type']].append(item)
        
        pages = file_to_pages.get(fn, {})
        
        # 每个文件尽量每类取2题（11题/5类≈2.2）
        for qtype in type_order:
            if type_counts[qtype] >= TARGET_PER_TYPE:
                continue  # 该类型已达标，跳过
            
            group = type_groups.get(qtype, [])
            target_for_this_type = min(2, TARGET_PER_TYPE - type_counts[qtype])
            
            added = 0
            for item in group:
                q = item['question']
                if q in seen_questions:
                    continue
                if added >= target_for_this_type:
                    break
                
                sel_pages = assign_pages_for_questions(pages, [q])
                results.append({
                    "filename": fn,
                    "page": int(sel_pages[0]),
                    "question": q,
                    "type": qtype,
                })
                seen_questions.add(q)
                type_counts[qtype] += 1
                added += 1

    # 第二轮：补齐未达50题的类型
    if len(results) < TOTAL:
        for qtype in type_order:
            if type_counts[qtype] >= TARGET_PER_TYPE:
                continue
            need = TARGET_PER_TYPE - type_counts[qtype]
            
            for fn in pdfs:
                if need <= 0:
                    break
                company, year = extract_company_and_year(fn)
                pool_items = build_question_pool(company, year)
                
                for item in pool_items:
                    if item['type'] != qtype:
                        continue
                    q = item['question']
                    if q in seen_questions:
                        continue
                    
                    pg = assign_pages_for_questions(file_to_pages.get(fn, {}), [q])[0]
                    results.append({
                        "filename": fn,
                        "page": int(pg),
                        "question": q,
                        "type": qtype,
                    })
                    seen_questions.add(q)
                    type_counts[qtype] += 1
                    need -= 1
                    if need <= 0:
                        break

    # 截断至250（理论上应该正好250）
    results = results[:TOTAL]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(results)} questions to {OUT}")


if __name__ == "__main__":
    main()


