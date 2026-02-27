import json
from pathlib import Path
from collections import defaultdict

# read gold standard and baseline
test_file = Path(__file__).resolve().parent.parent /"datas" /"test_advanced_500.json"
gold_file = Path(__file__).resolve().parent.parent /"datas" /"gold_standard_500.json"

with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(gold_file, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

print(f"test_advanced_500.json has {len(test_data)} questions")
print(f"gold_standard_500.json has {len(gold_data)} questions")
print()

#create questions index 
test_questions = {item["question"].strip(): item for item in test_data}
gold_questions = {item["question"].strip(): item for item in gold_data}

print(f"test_advanced_500.json only has {len(test_questions)} unique questions")
print(f"gold_standard_500.json only has {len(gold_questions)} unique questions")
print()

# find questions in test_data but not in gold_data
missing_in_gold =[]
for q, item in test_questions.items():
    if q not in gold_questions:
        missing_in_gold.append(item)

# find questions in gold_data but not in test_data
missing_in_test =[]
for q, item in gold_questions.items():
    if q not in test_questions:
        missing_in_test.append(item)
    
print(f"gold_standard_500.json has {len(missing_in_gold)} questions not in test_advanced_500.json")
print(f"test_advanced_500.json has {len(missing_in_test)} questions not in gold_standard_500.json")
print()

# Output mismatched questions in details
# 详细输出不匹配的问题
if missing_in_gold:
    print("=" * 80)
    print("In test_advanced_500.json but not in gold_standard_500.json:")
    print("=" * 80)
    for i, item in enumerate(missing_in_gold, 1):
        print(f"\n{i}. 文件名: {item['filename']}")
        print(f"   页码: {item['page']}")
        print(f"   类型: {item['type']}")
        print(f"   问题: {item['question']}")
        
        # 尝试找到相似的问题（可能只是文本略有不同）
        similar = []
        for gq, gitem in gold_questions.items():
            # 检查是否是同一类问题（基于文件名和类型）
            if (gitem['filename'] == item['filename'] and 
                gitem['type'] == item['type'] and
                gitem['page'] == item['page']):
                similar.append((gq, gitem))
        
        if similar:
            print(f"   ⚠️  发现相似问题（同文件、同类型、同页码）:")
            for sq, sitem in similar[:3]:  # 只显示前3个
                print(f"      - {sq[:80]}...")
    
if missing_in_test:
    print("\n" + "=" * 80)
    print("In gold_standard_500.json but not in test_advanced_500.json:")
    print("=" * 80)
    for i, item in enumerate(missing_in_test, 1):
        print(f"\n{i}. 文件名: {item['filename']}")
        print(f"   页码: {item['page']}")
        print(f"   类型: {item['type']}")
        print(f"   问题: {item['question']}")

# 按类型统计不匹配
print("\n" + "=" * 80)
print("按类型统计不匹配问题:")
print("=" * 80)
type_stats = defaultdict(int)
for item in missing_in_gold:
    type_stats[item['type']] += 1

for item_type, count in sorted(type_stats.items()):
    print(f"{item_type}: {count} 条")

if missing_in_gold:
    output_file = Path(__file__).resolve().parent.parent /"outputs" /"mismatched_questions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(missing_in_gold, f, ensure_ascii=False, indent=2)
    print(f"Saved mismatched questions to {output_file}")