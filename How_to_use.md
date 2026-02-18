Below is a explaination that is  **full end-to-end workflow** (from PDFs → corpus → question set → Baseline/RAG answers → hallucination evaluation → comparison → significance test), plus a **simple description of what each key file does**. 

---

### Project User Manual 

This repository implements a **multimodal financial-report RAG pipeline** (text + tables + images) and a **rule-based hallucination evaluation** suite. You can follow this manual to reproduce all major artifacts and experiments on their local machine.

---

## 1) What You Can Reproduce

- **Knowledge corpus construction** from annual report PDFs (page-level chunks with metadata).
- **Balanced question set generation** (250/450/500… questions across 5 categories).
- **Baseline experiment** (same LLM, *no retrieval*).
- **RAG experiment** (retrieval + generation with evidence).
- **Hallucination evaluation** using deterministic rule-based metrics.
- **Comparison + significance testing** (McNemar’s test).

---

## 2) Environment Setup

### 2.1 Python
- Recommended: **Python 3.10+**
- Create and activate a virtual environment (optional but recommended)

### 2.2 Install dependencies
```bash
pip install -r requirements.txt
```

**Optional (only if you use MinerU for structured extraction):**
- MinerU is commented out in `requirements.txt` because it can be heavy. Install it only if needed.

### 2.3 Configure `.env` (LLM + embedding via OpenAI-compatible API)
Create a `.env` file in the project root with:

```ini
LOCAL_API_KEY=your_key_or_dummy
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_TEXT_MODEL=qwen2.5:7b
LOCAL_EMBEDDING_MODEL=nomic-embed-text
```

Notes:
- This project uses the **OpenAI Python SDK** but can work with **any OpenAI-compatible server** (e.g., Ollama, vLLM, etc.).
- If you do not run a local server, point `LOCAL_BASE_URL` to a provider endpoint you have access to.

---

## 3) Folder & File Layout (High-Level)

### 3.1 Data folders
- `datas/年报/`: put your annual report PDFs here (recommended naming contains “年度报告”)
- `datas/*.json`: question sets, gold standards, predictions

### 3.2 Core pipeline outputs
- `all_pdf_page_chunks.json`: page-level text chunks from PyMuPDF
- `all_pdf_page_chunks_mineru.json`: page-level structured markdown chunks from MinerU
- `all_pdf_page_chunks_merged.json`: merged + cleaned + re-chunked corpus used for retrieval

---

## 4) Step-by-Step Workflow (Recommended Order)

### Step 0 — Put PDFs into the dataset folder
Copy your annual report PDFs into:
- `datas/年报/`

> The question generator filters filenames. If your PDFs do not contain “年度报告” in their name, they may be skipped.

---
## !!! Either Run Baseline.ipynb (Same pre-process Step 0-3) or below steps
### Step 1 — Extract page text (PyMuPDF) 
**File:** `fitz_pipeline_all.py`  
**Purpose:** Extracts **page-level plain text** with `(filename, page)` metadata.

Run:
```bash
python fitz_pipeline_all.py
```

Expected output:
- `all_pdf_page_chunks.json`

---

### Step 2 — Extract multimodal structure (MinerU) *(optional but recommended)*
**File:** `mineru_pipeline_all.py`  
**Purpose:** Parses PDF layout and extracts **structured elements**:
- hierarchical text (titles/paragraphs)
- tables
- images (with optional captioning)

Run:
```bash
python mineru_pipeline_all.py
```

Expected outputs:
- `data_base_json_content/...` (intermediate MinerU JSON)
- `data_base_json_page_content/...` (page-level markdown JSON)
- `all_pdf_page_chunks_mineru.json`

**Important:**  
You do **NOT** need to download Qwen2.5 weights to use MinerU parsing.  
Qwen2.5-VL is only used if you enable image captioning (optional enrichment).

---

### Step 3 — Merge, align pages, clean headers/footers, and re-chunk
**File:** `merge_chunks.py`  
**Purpose:**
- aligns page index offsets between PyMuPDF and MinerU
- removes repeated headers/footers
- re-chunks into semantically usable blocks
- deduplicates globally

Run:
```bash
python merge_chunks.py
```

Expected output:
- `all_pdf_page_chunks_merged.json`

---

### Step 4 — Generate a balanced question set (blank answers)
**File:** `tools/generate_advanced_questions.py`  
**Purpose:** Creates a **balanced benchmark** with `type` tags:
- Fact Extraction
- List Enumeration
- Comparison/Calculation
- Judgment/Verification
- Reasoning/Analysis

Run (example):
```bash
python tools/generate_advanced_questions.py
```

Expected output (depends on your script config):
- `datas/test_advanced_250.json` (or `datas/test_advanced_450*.json`, etc.)

Each item contains:
```json
{"filename": "...pdf", "page": 123, "question": "...", "answer": "", "type": "..."}
```

---

### Step 5 — Create Gold Standard template (Oracle skeleton)
If you want a gold standard file with **empty answers** for human annotation:

- Copy the generated question set to:
  - `datas/gold_standard.json`

Then humans fill:
- `answer`
(and optionally verify `filename/page`).

---

## 5) Run Experiments (Baseline vs RAG)

### Step 6 — Baseline: same LLM, no retrieval
**File:** `tools/generate_no_rag_baseline.py`  
**Purpose:** Directly calls the LLM to answer questions **without any evidence context**.

Run:
```bash
python tools/generate_no_rag_baseline.py
```

Expected output:
- `outputs/no_rag_top1_pred_*.json`

---

### Step 7 — RAG: retrieval + generation
**File:** `rag_from_page_chunks_original.py`

**Purpose:**
- build embeddings for all chunks
- retrieve with Vector + BM25
- generate final JSON answers with `filename/page`

Before running, make sure the script points to:
- `all_pdf_page_chunks_merged.json`

Then run:
```bash
python rag_from_page_chunks_original.py
```

Expected output (example names may differ by config):
- `rag_top1_pred.json` or / `outputs/rag_top1_pred_*.json`

---

## 6) Hallucination Evaluation + Comparison

### Step 8 — Rule-based hallucination detection (optional batch analysis)
**File:** `tools/hallucination_detector.py`  
**Purpose:** Uses deterministic checks (numbers, coverage, references, list completeness) to output:
- verdict: Evidenced / Partially Evidenced / No Evidence
- confidence + diagnostic signals

Run:
```bash
python tools/hallucination_detector.py
```

---

### Step 9 — Compare Baseline vs RAG (requires gold standard answers)
**File:** `tools/compare_rag_vs_baseline.py`  
**Purpose:** Computes:
- Accuracy (vs gold answer)
- Hallucination Rate + Precision/Recall/F1 (hallucination as positive class)

Run:
```bash
python tools/compare_rag_vs_baseline.py
```

Expected output:
- printed metrics (and your own markdown logs, if enabled)

---

### Step 10 — Significance testing (McNemar)
**File:** `tools/significance_test.py`  
**Purpose:** Tests whether the accuracy difference between Baseline and RAG is statistically significant for paired samples.

Run:
```bash
python tools/significance_test.py
```

Expected output:
- `outputs/significance_test_result.md`

---

## 7) “What Does Each File Do?” (Quick Reference)

### Corpus construction
- `fitz_pipeline_all.py`: PyMuPDF page text extraction → `all_pdf_page_chunks.json`
- `mineru_pipeline_all.py`: MinerU structured parsing (text/tables/images) → `all_pdf_page_chunks_mineru.json`
- `merge_chunks.py`: align + clean + re-chunk + dedup → `all_pdf_page_chunks_merged.json`

### Question generation / processing
- `tools/generate_advanced_questions.py`: generate balanced questions with `type`
- `datas/process_questions.py`: stratified sampling (e.g., make 250 by type with `random.seed`)
- `datas/process_advanced_questions.ipynb`: ad-hoc validation / formatting / cleaning utilities

### Experiments
- `tools/generate_no_rag_baseline.py`: baseline answers without retrieval
- `rag_from_page_chunks_original.py`: RAG retrieval + generation
- `get_text_embedding.py`: calls embedding API (OpenAI-compatible)

### Evaluation
- `tools/hallucination_detector.py`: rule-based evidence/consistency checks
- `tools/compare_rag_vs_baseline.py`: compare metrics vs gold standard
- `tools/significance_test.py`: McNemar test + report

---

## 8) Common Pitfalls (Read This First)

- **PowerShell vs Bash syntax**: PowerShell does not support `python - <<'PY' ...`.
- **PDF naming filter**: question generator may skip PDFs unless the filename contains “年度报告”.
- **MinerU optional**: if you cannot install MinerU, you can still run a text-only pipeline via PyMuPDF + merge (with MinerU path skipped), but table/image quality will drop.
- **OpenAI-compatible endpoint**: you must have a working `LOCAL_BASE_URL` that supports:
  - chat completions (`LOCAL_TEXT_MODEL`)
  - embeddings (`LOCAL_EMBEDDING_MODEL`)
- **Gold alignment**: comparison experiments only count questions present in both predictions and gold.

---

## 9) Minimal “Quickstart” (If You Only Want One Working Run)

1) Put PDFs into `datas/年报/`  
2) Run:
```bash
python fitz_pipeline_all.py
python merge_chunks.py
python tools/generate_advanced_questions.py
python tools/generate_no_rag_baseline.py
```

Then optionally run RAG and evaluation once you configure models properly.

