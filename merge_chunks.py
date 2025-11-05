# merge_chunks.py
import json, os, re, hashlib, statistics as stats
from collections import defaultdict, Counter
from pathlib import Path

# 可调参数
TARGET_LEN = 800
OVERLAP = 120
MIN_CHUNK_LEN = 60          # 太短的段先并入
MAX_CHUNK_LEN = 1400        # 太长的段先拆
HEADER_FOOTER_FREQ = 0.3    # 同文件跨页高频行阈值（>=30%页）
MIN_KEEP_LINE_LEN = 3       # 过短行视为噪声
OFFSETS_TO_TRY = [-2, -1, 0, 1, 2]  # 估计页码偏移候选

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def norm_filename(fn: str) -> str:
    if not fn: return ""
    fn = fn.strip()
    # 统一全角符号等
    trans = str.maketrans({"（": "(", "）": ")", "，": ",", "：": ":", "“": "\"", "”": "\""})
    fn = fn.translate(trans)
    # 去掉重复空格，大小写统一
    fn = re.sub(r"\s+", " ", fn).strip()
    return fn

def norm_text(t: str) -> str:
    if not t: return ""
    t = t.replace("\u200b","").replace("\ufeff","")
    t = t.replace("\r","")
    # 合并多空白
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def hash_text(t: str) -> str:
    return hashlib.md5(norm_text(t).lower().encode("utf-8")).hexdigest()

def index_by_file_page(arr):
    by_fp = defaultdict(list)
    for x in arr:
        md = x.get("metadata", {})
        fn = norm_filename(md.get("file_name",""))
        pg = md.get("page", "")
        # 页码转 int，异常则跳过
        try:
            pg = int(pg)
        except:
            continue
        y = dict(x)
        y["metadata"] = {"file_name": fn, "page": pg}
        y["content"] = norm_text(y.get("content",""))
        if y["content"]:
            by_fp[(fn, pg)].append(y)
    return by_fp

def pages_set(by_fp):
    return set(by_fp.keys())

def guess_page_offset(base_pages, cand_pages):
    # 在 [-2..2] 中找使重叠最多的偏移；对不同文件可分别估计
    # 这里按“每个文件独立估计”更稳
    files = set(fn for fn,_ in base_pages)
    per_file_offset = {}
    for fn in files:
        base_pg = set(p for f,p in base_pages if f==fn)
        cand_pg = set(p for f,p in cand_pages if f==fn)
        if not cand_pg:
            continue
        best = 0
        best_off = 0
        for off in OFFSETS_TO_TRY:
            inter = len(base_pg.intersection({p+off for p in cand_pg}))
            if inter > best:
                best = inter
                best_off = off
        per_file_offset[fn] = best_off
    return per_file_offset

def apply_offset(by_fp, per_file_offset):
    out = defaultdict(list)
    for (fn, pg), items in by_fp.items():
        off = per_file_offset.get(fn, 0)
        out[(fn, pg+off)].extend(items)
    return out

def detect_header_footer_lines(by_fp):
    # 统计每文件跨页高频行：出现页占比 >= HEADER_FOOTER_FREQ，且行长合适
    hf_map = {}  # {filename: set(lines)}
    files = set(fn for fn,_ in by_fp.keys())
    for fn in files:
        per_page_lines = []
        pages = sorted(p for f,p in by_fp.keys() if f==fn)
        if not pages:
            continue
        page_set = set(pages)
        # 行出现在哪些页
        line_pages = defaultdict(set)
        for p in pages:
            lines = []
            for item in by_fp[(fn,p)]:
                for line in item["content"].splitlines():
                    s = line.strip()
                    if len(s) >= MIN_KEEP_LINE_LEN:
                        lines.append(s)
            for s in set(lines):
                line_pages[s].add(p)
        total_pages = len(page_set)
        hf = set()
        for s, pset in line_pages.items():
            if len(pset) / total_pages >= HEADER_FOOTER_FREQ and len(s) <= 120:
                hf.add(s)
        hf_map[fn] = hf
    return hf_map

def remove_header_footer(text, hf_lines):
    if not text or not hf_lines:
        return text
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s in hf_lines:
            continue
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def merge_page_texts(items):
    # 同页内简单拼接（保持顺序），用空行隔开
    parts = []
    for it in items:
        t = norm_text(it.get("content",""))
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()

def sentence_split(txt):
    # 简易句切：句末标点或换行
    segs = re.split(r'(?<=[。！？!?;；\.])\s*|\n+', txt)
    return [s.strip() for s in segs if s and s.strip()]

def rechunk_text(txt, target=TARGET_LEN, overlap=OVERLAP):
    if not txt: return []
    sents = sentence_split(txt)
    out = []
    buf = ""
    for s in sents:
        if len(buf) + len(s) + 1 <= target:
            buf = (buf + (" " if buf else "")) + s
        else:
            if buf:
                out.append(buf.strip())
            # 构造重叠
            tail = buf[-overlap:] if buf and len(buf)>overlap else (buf or "")
            buf = (tail + " " + s).strip() if tail else s
    if buf:
        out.append(buf.strip())
    # 对过长段再细拆
    final=[]
    for ch in out:
        if len(ch) <= MAX_CHUNK_LEN:
            final.append(ch)
        else:
            # 粗暴分割
            for i in range(0, len(ch), target):
                piece = ch[i:i+target+overlap]
                if len(piece) >= MIN_CHUNK_LEN:
                    final.append(piece.strip())
    # 合并过短段
    merged=[]
    cur=""
    for ch in final:
        if len(ch) < MIN_CHUNK_LEN:
            cur = (cur + " " + ch).strip()
        else:
            if cur:
                merged.append(cur)
                cur=""
            merged.append(ch)
    if cur:
        merged.append(cur)
    return merged

def dedup_chunks(chunks):
    seen = set()
    out = []
    for c in chunks:
        h = hash_text(c["content"])
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out

def main(pymupdf_path, mineru_path, out_path):
    a = load_json(pymupdf_path)
    b = load_json(mineru_path)

    A = index_by_file_page(a)  # 基准
    B_raw = index_by_file_page(b)

    # 估计 mineru 页码偏移并应用
    per_file_off = guess_page_offset(set(A.keys()), set(B_raw.keys()))
    B = apply_offset(B_raw, per_file_off)

    # 统计并去 header/footer
    hf_map = detect_header_footer_lines({**A, **B})
    for (fn, pg), items in list(A.items()):
        A[(fn,pg)] = [
            {"content": remove_header_footer(it["content"], hf_map.get(fn,set())),
             "metadata": it["metadata"]}
            for it in items if it["content"]
        ]
    for (fn, pg), items in list(B.items()):
        B[(fn,pg)] = [
            {"content": remove_header_footer(it["content"], hf_map.get(fn,set())),
             "metadata": it["metadata"]}
            for it in items if it["content"]
        ]

    # 合并两路 → 页级重切块
    all_keys = sorted(set(A.keys()) | set(B.keys()))
    merged_chunks = []
    for (fn, pg) in all_keys:
        items = []
        if (fn,pg) in A: items += A[(fn,pg)]
        if (fn,pg) in B: items += B[(fn,pg)]
        if not items: continue
        page_txt = merge_page_texts(items)
        page_txt = norm_text(page_txt)
        if not page_txt: continue
        pieces = rechunk_text(page_txt, TARGET_LEN, OVERLAP)
        for t in pieces:
            if not t: continue
            merged_chunks.append({
                "content": t,
                "metadata": {"file_name": fn, "page": int(pg)}
            })

    # 全局去重
    merged_chunks = dedup_chunks(merged_chunks)

    # 输出
    Path(out_path).write_text(json.dumps(merged_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 简要统计
    lens = [len(c["content"]) for c in merged_chunks]
    print(f"合并后 chunks: {len(merged_chunks)} | 平均长度: {sum(lens)/max(1,len(lens)):.1f} | 中位: {stats.median(lens) if lens else 0}")

if __name__ == "__main__":
    # 示例：
    # python merge_chunks.py
    pymupdf_path = "all_pdf_page_chunks.json"
    mineru_path = "all_pdf_page_chunks_mineru.json"
    out_path = "all_pdf_page_chunks_merged.json"
    main(pymupdf_path, mineru_path, out_path)