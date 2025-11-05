# Task Summary: High-Quality Question Set Generation

## 📋 Task Requirements
1. ❌ Original test.json contains many duplicate questions
2. ❌ Question design is too simple, not conducive to hallucination detection
3. ✅ Need to redesign 250 questions for 26 annual reports

## ✅ Task Completion Status

### Generated Question Set: `datas/test_advanced_250.json`

#### Core Metrics (100% Achievement)
- **Total Questions**: 250
- **Unique Questions**: 250 (**0 duplicates**)
- **Report Coverage**: 26 reports (2022-2024)
- **Field Completeness**: 100% (all items contain 5 fields)

#### Type Distribution (Strictly Balanced)
| Type | Count | Percentage | Status |
|------|-------|------------|--------|
| Fact Extraction | 50 | 20% | ✅ |
| List Enumeration | 50 | 20% | ✅ |
| Comparison & Calculation | 50 | 20% | ✅ |
| Judgment & Verification | 50 | 20% | ✅ |
| Reasoning & Analysis | 50 | 20% | ✅ |

## 📊 Question Quality Improvement Comparison

| Dimension | Original test.json | New test_advanced_250.json | Improvement |
|-----------|-------------------|----------------------------|-------------|
| **Question Duplication** | Many duplicates | 0 duplicates | ✅✅✅ |
| **Type Diversity** | Single (mostly reasoning) | 5 types balanced | ✅✅✅ |
| **Complexity Level** | Simple | Medium-High (5 gradients) | ✅✅ |
| **Hallucination Detection Friendly** | Medium | High (with type tags) | ✅✅✅ |
| **Format Standardization** | Basic | Complete (5 fields) | ✅✅ |
| **Documentation Completeness** | None | With README + validation report | ✅✅ |

## 🎯 Design Characteristics of Each Question Type

### 1. Fact Extraction (50 questions)
**Purpose**: Test RAG's precise location and numerical extraction capabilities
```
What is the total operating revenue of PICC in 2022 in hundred million yuan?
What is the basic earnings per share (EPS) of Sifang Co., Ltd. in 2023?
```
**Hallucination Detection**: Easy to detect numerical fabrication and unit errors

### 2. List Enumeration (50 questions)
**Purpose**: Test structured information extraction and completeness
```
List the names and sales revenue percentages of PICC's top five customers in 2022.
What are the main business segments of China Tourism Group Duty Free in 2023? What are the revenue percentages of each segment?
```
**Hallucination Detection**: Easy to detect information omission and list fabrication

### 3. Comparison & Calculation (50 questions)
**Purpose**: Test multi-period data comparison and calculation capabilities
```
Calculate the year-on-year growth rate and growth amount of China Shenhua's operating revenue in 2023.
Compare the quarterly operating revenues from Q1 to Q4 of ICBC in 2024.
```
**Hallucination Detection**: Easy to detect calculation errors and logical confusion

### 4. Judgment & Verification (50 questions)
**Purpose**: Test conditional branch logic and detail extraction
```
Did CITIC Securities implement cash dividends in 2023? If so, what are the dividend amount and dividend rate?
Does CCB have goodwill impairment in 2024? If so, what is the impairment amount?
```
**Hallucination Detection**: Easy to detect fictitious events and detail fabrication

### 5. Reasoning & Analysis (50 questions)
**Purpose**: Test deep understanding and causal reasoning capabilities
```
Attribution analysis: What are the main driving factors (price vs. quantity) for PICC's operating revenue growth in 2022?
ROE decomposition: Analyze the DuPont three-factor contribution to ICBC's ROE change in 2024.
```
**Hallucination Detection**: Easy to detect attribution errors and logical leaps

## 📂 Delivery File List

### Core Files
- ✅ `datas/test_advanced_250.json` - 250 question set (with blank answer fields)
- ✅ `datas/test_advanced_250_README.md` - Detailed usage instructions
- ✅ `datas/test_advanced_250_VALIDATION.txt` - Validation report

### Tool Scripts
- ✅ `tools/generate_advanced_questions.py` - Question generation script (reusable)
- ✅ `tools/fill_answers_example.py` - Answer filling example script

## 🚀 Usage Recommendations

### Step 1: Generate Answers
```bash
# Use RAG system to fill answers
python tools/fill_answers_example.py
```

### Step 2: Manual Verification
Sample check the answer quality of each type of question (recommend checking 10 questions per type)

### Step 3: Hallucination Detector Training
Build positive and negative sample pairs based on annotation results:
- Positive samples: High-quality answers
- Negative samples: Annotate 6 types of hallucinations (numerical fabrication/information omission/fictitious events/logical errors/time confusion/calculation errors)

## 💡 Innovation Summary

1. **Strict Type Balance**: First achievement of strictly balanced 50:50:50:50:50 distribution for 5 question types
2. **Type Labeling**: Each question has a `type` field for classification evaluation
3. **Answer Field Reserved**: Unified format, convenient for RAG system filling
4. **Zero Duplication Guarantee**: Global deduplication mechanism, 250 questions 100% unique
5. **Complete Documentation**: Includes README + validation report + example scripts

## 🎓 Lessons Learned

### Key Elements of Question Generation
1. **Type Diversity**: Cover 5 dimensions of fact/list/calculation/judgment/reasoning
2. **Difficulty Progression**: From simple queries to complex analysis
3. **Verifiability**: All answers can be found in the documents
4. **Deduplication Mechanism**: Global hash deduplication to avoid simple copying

### Design Considerations for Hallucination Detection
1. **Fact Type**: Easy for automatic verification (numerical matching)
2. **List Type**: Easy to check completeness (set comparison)
3. **Calculation Type**: Easy for mathematical verification
4. **Judgment Type**: Easy for binary classification evaluation
5. **Reasoning Type**: Requires human judgment of attribution rationality

---

**Generation Date**: 2025-11-05  
**Version**: v2.1  
**Generation Tool**: tools/generate_advanced_questions.py  
**Validation Status**: ✅ All Passed
