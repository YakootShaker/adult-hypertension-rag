# Day 2 Evaluation Report: Retrieval Optimization
**AI Clinical Decision Support Hackathon · Adult Hypertension RAG**

---

## 1. Executive Summary & Definition of Done

This report presents the empirical optimization and benchmarking of the retrieval pipeline for the **WHO Guideline for Pharmacological Treatment of Hypertension in Adults (2021)**.

### Day 2 Checklist Completion:
- [x] **Ground Truth Test Set**: 10 realistic clinical questions mapped to target guideline sections and page numbers.
- [x] **Top-$k$ Value Justified**: Systematic evaluation of $k \in [1, 2, 3, 4, 5, 8, 10]$ proving $k=4$–$5$ reaches 100% Hit Rate while minimizing context noise.
- [x] **Chunk Size & Overlap Ablation**: Compared 3 configurations (300/40, 600/80, 900/150 tokens) with logged precision and token metrics.
- [x] **Head-to-Head Embedding Benchmark**: Evaluated `gemini-embedding-001` (3072 dims) vs `all-MiniLM-L6-v2` (384 dims) across Hit Rate, Precision@k, MRR, and Latency.
- [x] **Explainability & Verification**: Full explainability view with statement-level inline citations (`[1]`, `[2]`), hover preview popovers, and interactive split-screen PDF viewer with automatic text highlighting.

---

## 2. Ground Truth Test Set (`evaluation/test_set.json`)

10 representative clinical queries testing all major recommendation chapters:

| ID | Query | Target Section | Target Page(s) |
|---|---|---|---|
| **Q01** | What is the recommended BP threshold for initiating pharmacological treatment? | Section 1 (Threshold) | Page 19 |
| **Q02** | Which laboratory tests are recommended before or during treatment initiation? | Section 2 (Lab Testing) | Pages 20–21 |
| **Q03** | Is CVD risk assessment recommended at or after initiation of treatment? | Section 3 (CVD Risk) | Page 22 |
| **Q04** | Which drug classes are recommended as first-line pharmacological agents? | Section 4 (First-Line Drugs) | Pages 23–24 |
| **Q05** | When is combination therapy recommended as initial treatment vs monotherapy? | Section 5 (Combination Rx) | Pages 25–26 |
| **Q06** | What are the recommended target blood pressure levels for adults on treatment? | Section 6 (Target BP) | Page 28 |
| **Q07** | How frequently should patients be reassessed after initiating or changing medications? | Section 7 (Assessment Freq) | Pages 29–30 |
| **Q08** | Can nonphysician healthcare professionals initiate and adjust hypertension treatment? | Section 8 (Task Sharing) | Pages 31–32 |
| **Q09** | What special considerations apply in patients with diabetes, CKD, or CAD? | Section 11 (Special Settings) | Pages 33–34 |
| **Q10** | What implementation tools and clinical treatment protocols are provided? | Section 13 (Implementation) | Pages 39–40 |

---

## 3. Embedding Model Benchmark (Head-to-Head)

We evaluated two models on the exact same 10-query test set against our section-aware index:

| Metric | Google `gemini-embedding-001` | Local `all-MiniLM-L6-v2` | Winner |
|---|:---:|:---:|:---:|
| **Embedding Dimensions** | 3072 | 384 | — |
| **Hit Rate @ 1** (Top-1 contains target) | **90.00%** | 70.00% | 🏆 Gemini (+20%) |
| **Hit Rate @ 3** (Top-3 contains target) | **100.00%** | 70.00% | 🏆 Gemini (+30%) |
| **Hit Rate @ 5** (Top-5 contains target) | **100.00%** | 80.00% | 🏆 Gemini (+20%) |
| **Retrieval Precision @ 3** | **76.67%** | 30.00% | 🏆 Gemini (+46.7%) |
| **Retrieval Precision @ 5** | **54.00%** | 20.00% | 🏆 Gemini (+34.0%) |
| **Mean Reciprocal Rank (MRR)** | **0.9500** | 0.7250 | 🏆 Gemini (+0.225) |
| **Average Query Latency** | 348.3 ms (API) | **10.2 ms** (CPU) | ⚡ MiniLM (Faster) |
| **Cost** | Free tier (1500 RPM) | $0.00 (Local) | Tie |

### Decision & Justification:
We selected **`gemini-embedding-001`** as our primary embedding model. 
1. **Clinical Accuracy**: Clinical decision support demands high recall and precision. Gemini achieved **0.95 MRR** and **100% Hit Rate@3**, finding the exact guideline section on the very first try for 9 out of 10 queries.
2. **Context Window**: Supports 2048 tokens per chunk without truncation (MiniLM truncates at 256 tokens).
3. **Latency**: At 348 ms, latency is well within the acceptable clinical interaction budget (< 1000 ms).

---

## 4. Chunk Size & Overlap Ablation Experiment

We tested 3 distinct parameter configurations by re-chunking the entire WHO guideline:

| Configuration | Target Tokens | Overlap Tokens | Total Chunks | Avg Tokens/Chunk | Hit Rate @ 5 | Precision @ 5 | Notes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Small Chunks** | 300 | 40 | 76 | 292 | 100.00% | 74.00% | High precision per chunk, but fragments clinical recommendations across multiple cards. |
| **Balanced Chunks** *(Current)* | **600** | **80** | **36** | **597** | **100.00%** | **54.00%** | **Optimal**: Preserves complete clinical recommendations and tables in a single coherent context. |
| **Large Chunks** | 900 | 150 | 25 | 861 | 100.00% | 40.00% | Dilutes specific recommendations; wastes prompt context window on unrelated surrounding text. |

### Justification for Balanced Configuration (600 / 80 tokens):
- WHO guideline recommendations are structured as 1–2 page self-contained clinical chapters with rationale and remarks.
- 600 tokens captures the **entire recommendation + remarks + drug tables** in one cohesive chunk without fragmentation, preventing multi-hop assembly errors during generation.

---

## 5. Top-$k$ Parameter Tuning

Evaluated across candidate values of $k$:

| $k$ Value | Hit Rate (Recall) | Precision@$k$ | Avg Context Tokens | Evaluation Assessment |
|:---:|:---:|:---:|:---:|---|
| **$k = 1$** | 90.00% | 90.00% | 607 | ⚠️ Too narrow: misses multi-part recommendations (e.g. drug choices + contraindications). |
| **$k = 2$** | 100.00% | 85.00% | 1,229 | Good for simple queries, but limits comparative drug table context. |
| **$k = 3$** | 100.00% | 76.67% | 1,812 | Strong balance for standard prompts. |
| **$k = 4$** | **100.00%** | **62.50%** | **2,412** | ⭐ **Recommended Default**: 100% recall with tight context boundary. |
| **$k = 5$** | **100.00%** | **54.00%** | **3,075** | ⭐ **Robust Option**: Captures cross-chapter comorbidities & special settings. |
| **$k = 8$** | 100.00% | 35.00% | 4,974 | ⚠️ Context dilution: introduces irrelevant background text; increases LLM latency. |
| **$k = 10$** | 100.00% | 28.00% | 6,270 | ⚠️ Wasted context window; risk of hallucinated contradictions. |

### Justification:
We chose **$k = 4$–$5$** as the sweet spot, providing **100% recall** with an average context load of **~2,400–3,000 tokens**, fitting comfortably within LLM attention heads.

---

## 6. Explainability & Trust UX

To satisfy Module 4 (Explainability):
- **Grounded Inline Citations**: Statements in LLM answers include clickable badges `[1]`, `[2]`.
- **Hover Preview Popovers**: Real-time quote cards showing source section, page number, and text excerpt.
- **Side-by-Side PDF Viewer**: Opens `WHO_guideline_01.pdf` directly to the referenced page with **animated glowing text highlighting** and auto-scroll.

---

*Report generated automatically from empirical evaluation test runs in `d:\coding\adult-hypertension-rag\evaluation\`.*
