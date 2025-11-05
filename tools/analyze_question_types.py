import json
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "datas/test_advanced_250.json"

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['question'] for item in data]

# 定义类型识别规则
def classify_question(q: str) -> str:
    # 事实提取类：直接询问"是多少"、"达到多少"
    if re.search(r'(是多少|达到多少|总额是|规模达|占比是)', q):
        if not re.search(r'(请列举|列出|有哪些|分别)', q):
            return "事实提取类"
    
    # 列举枚举类：包含"请列举"、"有哪些"、"分别"
    if re.search(r'(请列举|列出|有哪些|分别是谁|包括哪些)', q):
        return "列举枚举类"
    
    # 比较计算类：包含"同比"、"环比"、"对比"、"变化"、"增长率"
    if re.search(r'(同比|环比|对比|与上一年度相比|各季度|比重多少)', q):
        return "比较计算类"
    
    # 判断验证类：以"是否"开头或包含"是否"
    if re.search(r'(^是否|年度是否|是否存在|是否披露|是否发生|是否实施|是否完成)', q):
        return "判断验证类"
    
    # 推理分析类：包含"分析"、"解释"、"评估"
    if re.search(r'(分析|解释|评估)', q):
        return "推理分析类"
    
    return "未分类"

# 统计类型分布
type_counts = Counter([classify_question(q) for q in questions])
# 使用JSON中保存的类型字段（如果存在）
if data and 'type' in data[0]:
    type_counts = Counter([item['type'] for item in data])
    use_embedded_type = True
else:
    # 定义类型识别规则（备用）
    def classify_question(q: str) -> str:
        if re.search(r'(是多少|达到多少|总额是|规模达|占比是)', q):
            if not re.search(r'(请列举|列出|有哪些|分别)', q):
                return "事实提取类"
        if re.search(r'(请列举|列出|有哪些|分别是谁|包括哪些)', q):
            return "列举枚举类"
        if re.search(r'(同比|环比|对比|与上一年度相比|各季度|比重多少)', q):
            return "比较计算类"
        if re.search(r'(^是否|年度是否|是否存在|是否披露|是否发生|是否实施|是否完成)', q):
            return "判断验证类"
        if re.search(r'(分析|解释|评估)', q):
            return "推理分析类"
        return "未分类"
    type_counts = Counter([classify_question(q) for q in questions])
    use_embedded_type = False



print("=" * 60)
print("问题类型分布统计")
print("=" * 60)
print(f"\n总问题数: {len(questions)}")
print(f"唯一问题数: {len(set(questions))}")
print(f"重复问题数: {len(questions) - len(set(questions))}\n")

print("各类型问题数量：")
for qtype, count in sorted(type_counts.items()):
    pct = count / len(questions) * 100
    print(f"  {qtype}: {count}题 ({pct:.1f}%)")

# 抽样展示每种类型的示例
print("\n" + "=" * 60)
print("每种类型示例问题")
print("=" * 60)

########################################################
samples_per_type = {}
for item in data:
    if use_embedded_type:
        qtype = item.get('type', '未分类')
        q = item['question']
    else:
        q = item['question']
        qtype = classify_question(q)
    
    if qtype not in samples_per_type:
        samples_per_type[qtype] = []
    if len(samples_per_type[qtype]) < 3:  # 每类最多3个示例
        samples_per_type[qtype].append(q)

for qtype in sorted(samples_per_type.keys()):
    print(f"\n【{qtype}】")
    for i, q in enumerate(samples_per_type[qtype], 1):
        print(f"  {i}. {q}")

# 验证每个文件的问题类型分布
print("\n" + "=" * 60)
print("每个文件的问题类型分布检查（前5个文件）")
print("=" * 60)

file_type_dist = {}
for item in data:
    fn = item['filename']
    if use_embedded_type:
        qtype = item.get('type', '未分类')
    else:
        qtype = classify_question(item['question'])
    if fn not in file_type_dist:
        file_type_dist[fn] = Counter()
    file_type_dist[fn][qtype] += 1

for i, (fn, type_counter) in enumerate(sorted(file_type_dist.items())[:5]):
    print(f"\n文件: {Path(fn).stem[:50]}...")
    for qtype, count in sorted(type_counter.items()):
        print(f"  {qtype}: {count}题")