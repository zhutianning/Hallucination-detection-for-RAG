import os
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


"""
使用与 RAG 相同的 LLM，在「不检索、不提供上下文」的前提下，
直接对 `datas/test_advanced_250.json` 逐题生成答案，作为 baseline。

输出格式与 RAG 结果保持一致：[{question, answer, filename, page}]
方便后续与 `outputs/rag_top1_pred_10.json` 或其它 RAG 结果做对比。
"""


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "datas" / "test_advanced_250.json"
OUT_PATH = BASE_DIR / "outputs" / "no_rag_top1_pred_test_advanced_250.json"

# 如果只想跑一小部分做 sanity check，可以把这里改成 10、20 等
TEST_SAMPLE_NUM = None  # None 表示全部 250 题


def get_llm_client() -> OpenAI:
    """复用项目中同一套 LOCAL_* 配置，保证与 RAG 使用同一模型。"""
    load_dotenv()
    api_key = os.getenv("LOCAL_API_KEY")
    base_url = os.getenv("LOCAL_BASE_URL")
    model = os.getenv("LOCAL_TEXT_MODEL")
    if not api_key or not base_url or not model:
        raise ValueError("请在 .env 中配置 LOCAL_API_KEY / LOCAL_BASE_URL / LOCAL_TEXT_MODEL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    client._baseline_model = model  # 仅作记录，方便调试
    return client


def call_llm_for_question(client: OpenAI, model: str, question: str) -> dict:
    """
    使用同一 LLM，在“不提供检索内容”的情况下直接回答问题。
    仅约束输出为 JSON：{answer, filename, page}
    """
    system_msg = (
        "你是一名专业的金融分析助手。现在不提供任何具体年报原文或检索内容，"
        "但你应当基于自己对财务报表结构、上市公司披露惯例以及一般行业水平的理解，"
        "尽量给出一个**具体、完整且自洽**的答案，而不是拒绝回答。"
        "即使不能保证完全准确，也不要回答“无法根据已有信息给出精确数值”或类似拒绝语，"
        "可以用“约”“大约”“可能在……区间”来表达不确定性，但仍需给出具体数值或结论。"
        "你的目标是模拟一个在没有原文情况下，倾向于“想象/推测”出的答案，用于幻觉检测的对比基线。"
    )

    user_msg = (
        "请严格按照如下 JSON 格式输出，不要输出多余文字：\n"
        '{"answer": "你的简洁回答（尽量具体，不要直接说无法回答）", '
        '"filename": "你推断的来源文件名（如无法确定可留空）", '
        '"page": "你推断的来源页码（如无法确定可留空）"}\n'
        f"问题：{question}\n"
        "注意：务必只输出一个 JSON 对象，不能包含注释或额外说明。"
    )

    import time

    max_retries = 3
    last_err = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
                max_tokens=512,
            )
            raw = completion.choices[0].message.content.strip()
            break  # 成功就退出循环
        except Exception as e:
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"[no-rag] 调用 LLM 失败，第 {attempt+1}/{max_retries} 次重试，等待 {wait}s，错误: {e}")
            time.sleep(wait)
    else:
        # 多次重试仍失败，返回一个占位答案，避免整个脚本崩掉
        print(f"[no-rag] 多次重试仍失败，跳过该问题: {question[:40]}..., 最后错误: {last_err}")
        return {
            "answer": "【基线生成失败：模型接口多次报错，无法给出答案】",
            "filename": "",
            "page": "",
        }

    # 尝试直接解析 JSON；失败则降级为把原文塞到 answer 字段
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "") or ""
            filename = parsed.get("filename", "") or ""
            page = parsed.get("page", "") or ""
        else:
            answer, filename, page = raw, "", ""
    except Exception:
        answer, filename, page = raw, "", ""

    return {
        "answer": answer,
        "filename": filename,
        "page": page,
    }


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"找不到测试集文件: {TEST_PATH}")

    with TEST_PATH.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    all_indices = list(range(total))
    if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0 and total > TEST_SAMPLE_NUM:
        selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))
    else:
        selected_indices = all_indices

    print(f"加载测试集 {TEST_PATH.name}，共 {total} 题，本次生成 {len(selected_indices)} 题的 baseline（无 RAG）答案。")

    load_dotenv()
    api_key = os.getenv("LOCAL_API_KEY")
    base_url = os.getenv("LOCAL_BASE_URL")
    model = os.getenv("LOCAL_TEXT_MODEL")
    client = get_llm_client()

    results = []
    for i, idx in enumerate(tqdm(selected_indices, desc="生成 baseline 答案")):
        item = test_data[idx]
        q = item["question"]
        res = call_llm_for_question(client, model, q)
        results.append({
            "question": q,
            "answer": res["answer"],
            # baseline 不使用检索，因此 filename/page 可以保留 LLM 的猜测或为空
            "filename": res["filename"],
            "page": res["page"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已将 baseline（无 RAG）答案保存至: {OUT_PATH}")
    print("你可以与 RAG 版本结果（如 outputs/rag_top1_pred_10.json 或其它文件）按 question 对齐比较。")


if __name__ == "__main__":
    main()


