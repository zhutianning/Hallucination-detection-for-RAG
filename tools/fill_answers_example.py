"""
示例脚本：使用RAG系统为 test_advanced_250.json 填充答案

使用方法：
1. 确保已有 all_pdf_page_chunks_merged.json
2. 配置 .env 文件中的 API 密钥
3. 运行: python tools/fill_answers_example.py

注意：这是一个示例脚本，实际使用时请根据您的RAG系统调整
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import sys

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """
    使用RAG系统为问题集生成答案的示例流程
    """
    BASE_DIR = Path(__file__).parent.parent
    INPUT_FILE = BASE_DIR / "datas/test_advanced_250.json"
    OUTPUT_FILE = BASE_DIR / "outputs/test_advanced_250_with_answers.json"
    
    # 加载问题集
    print("加载问题集...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"共加载 {len(test_data)} 个问题")
    
    # 统计各类型问题数量
    from collections import Counter
    type_counts = Counter([item['type'] for item in test_data])
    print("\n问题类型分布:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}题")
    
    # ====== 方式1: 使用 rag_from_page_chunks.py 中的 SimpleRAG ======
    try:
        from rag_from_page_chunks import SimpleRAG
        
        print("\n初始化RAG系统...")
        # 根据您的配置调整参数
        chunk_json_path = str(BASE_DIR / "all_pdf_page_chunks_merged.json")
        rag = SimpleRAG(
            chunk_json_path=chunk_json_path,
            use_rerank=True,  # 使用重排序
            recall_top_m_vec=50,
            recall_top_m_bm25=50
        )
        rag.setup()
        
        print("\n开始生成答案（可能需要较长时间）...")
        for item in tqdm(test_data, desc="生成答案"):
            try:
                result = rag.generate_answer(item['question'], top_k=5)
                item['answer'] = result.get('answer', '')
                item['rag_source_filename'] = result.get('filename', '')
                item['rag_source_page'] = result.get('page', '')
                # 可选：保存检索到的chunks用于后续分析
                # item['retrieval_chunks'] = result.get('retrieval_chunks', [])
            except Exception as e:
                print(f"\n生成答案失败: {item['question'][:50]}... 错误: {e}")
                item['answer'] = ""
                item['error'] = str(e)
        
        # 保存结果
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 已保存带答案的问题集到: {OUTPUT_FILE}")
        
        # 统计生成情况
        answered = sum(1 for item in test_data if item.get('answer'))
        print(f"\n生成统计:")
        print(f"  成功生成答案: {answered}/{len(test_data)}题")
        print(f"  失败/空答案: {len(test_data)-answered}题")
        
    except ImportError:
        print("\n[ERROR] 未找到 rag_from_page_chunks.py 或其依赖")
        print("请确保已安装所需依赖并正确配置 .env 文件")
        print("\n备选方案：手动实现RAG逻辑或使用其他检索系统")
        return
    
    # ====== 方式2: 如果没有SimpleRAG，可以手动实现简单检索逻辑 ======
    # （此处省略，可根据需要自行实现）

if __name__ == "__main__":
    print("=" * 60)
    print("test_advanced_250.json 答案填充示例脚本")
    print("=" * 60)
    main()

