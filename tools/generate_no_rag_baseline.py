import os
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


"""
使用与 RAG 相同的 LLM，在「不检索、不提供上下文」的前提下，
直接对 `datas/test_advanced_500.json` 逐题生成答案，作为 baseline。

输出格式与 RAG 结果保持一致：[{question, answer, filename, page}]
方便后续与 `outputs/rag_top1_pred_10.json` 或其它 RAG 结果做对比。
"""


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "datas" / "test_advanced_500.json"
OUT_PATH = BASE_DIR / "outputs" / "no_rag_top1_pred_test_advanced_500.json"

# if you want to run a subset of the questions, you can change the number here
TEST_SAMPLE_NUM = None  # None means all 500 questions


def get_llm_client() -> OpenAI:
    # use the same model as the rag model
    load_dotenv()
    api_key = os.getenv("LOCAL_API_KEY")
    base_url = os.getenv("LOCAL_BASE_URL")
    model = os.getenv("LOCAL_TEXT_MODEL")
    if not api_key or not base_url or not model:
        raise ValueError("please set the LOCAL_API_KEY / LOCAL_BASE_URL / LOCAL_TEXT_MODEL in the .env file")
    client = OpenAI(api_key=api_key, base_url=base_url)
    client._baseline_model = model  # only for debugging
    return client


def call_llm_for_question(client: OpenAI, model: str, question: str) -> dict:
    """
    Use the same LLM to answer directly without retrieval context.
    Constrain output to JSON only: {answer, filename, page}
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
            break  # Exit loop on success
        except Exception as e:
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"[no-rag] call llm failed, attempt {attempt+1}/{max_retries}, wait {wait}s, error: {e}")
            time.sleep(wait)
    else:
        # Return a fallback answer after repeated failures.
        print(f"[no-rag] multiple retries failed, skip this question: {question[:40]}..., last error: {last_err}")
        return {
            "answer": "baseline generation failed, model interface multiple errors, cannot give an answer",
            "filename": "",
            "page": "",
        }

    # Try parsing JSON; if it fails, fallback to raw text.
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
        raise FileNotFoundError(f"cannot find the test set file: {TEST_PATH}")

    with TEST_PATH.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    all_indices = list(range(total))
    if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0 and total > TEST_SAMPLE_NUM:
        selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))
    else:
        selected_indices = all_indices

    print(f"loading test set {TEST_PATH.name}, total {total} questions, generate {len(selected_indices)} questions' baseline (no RAG) answer.")
    load_dotenv()
    api_key = os.getenv("LOCAL_API_KEY")
    base_url = os.getenv("LOCAL_BASE_URL")
    model = os.getenv("LOCAL_TEXT_MODEL")
    print(f"Using API: {base_url}, Model: {model}")
    client = get_llm_client()

    results = []
    for i, idx in enumerate(tqdm(selected_indices, desc="Generating baseline answers")):
        item = test_data[idx]
        q = item["question"]
        res = call_llm_for_question(client, model, q)
        results.append({
            "question": q,
            "answer": res["answer"],
            # Baseline does not use retrieval; filename/page can be guessed or empty.
            "filename": res["filename"],
            "page": res["page"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"All baseline (no RAG) answers have been saved to: {OUT_PATH}")
    print("You can compare the results with the RAG version (e.g. outputs/rag_top1_pred_10.json or other files) by question.")


if __name__ == "__main__":
    main()


