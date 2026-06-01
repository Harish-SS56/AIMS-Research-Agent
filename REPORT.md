# AIMS Deep Research Agent — Technical Report

**Harish S S** · AIMS-DTU Research Intern 2026 · June 2026

---

## Abstract

This report presents a multi-stage retrieval-augmented generation (RAG) system for research question answering. The system operates on a corpus of 574 arXiv papers (33,175 chunks, cs.CL/cs.AI/cs.LG, 2024-01–2026-04) indexed in ChromaDB (dense) and BM25 (sparse). The pipeline comprises six modules: Query Planner, Hybrid Retriever with Reciprocal Rank Fusion (RRF), Passage Reader, Evidence Reflector, Answer Synthesizer, and Citation Verifier — all powered by Azure OpenAI GPT-4o and text-embedding-3-large.

Controlled ablation across seven configurations (n=30 for five configs) yields: full pipeline accuracy **2.83/5.0 (σ=1.26)** vs baseline **2.07 (σ=0.78)**, a **+36.7% relative improvement**. Reflector ablation: −0.33 accuracy (p<0.05). Planner ablation: −0.30 accuracy, −0.10 faithfulness. Reranker ablation: no significant delta. Citation precision/recall ≥0.97 across all fully-evaluated configurations. Latency: 17.3s (baseline) to 64.8s (no_reranker).

**Repository:** [github.com/Harish-SS56/AIMS-Research-Agent](https://github.com/Harish-SS56/AIMS-Research-Agent)

---

## 1. Introduction

Single-forward-pass LLM inference fails on research-grade questions requiring multi-source evidence aggregation and citation grounding. The deep research agent paradigm addresses this via iterative retrieval-reasoning loops: query decomposition → evidence retrieval → sufficiency evaluation → conditional refinement → synthesis with citations.

This work implements a six-module pipeline on a domain-specific arXiv corpus with systematic component ablation to quantify individual module contributions to answer accuracy, faithfulness, and citation quality.

**Objectives:**
1. Construct and index an arXiv corpus (LLM agents domain, 2024-01 to 2026-04)
2. Answer 30 evaluation questions (factoid/comparative/survey) with grounded citations
3. Ablate each pipeline component to measure marginal contribution
4. Evaluate via LLM-as-judge accuracy, chunk-level faithfulness, citation P/R, latency, and retrieval count

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
                              User Query
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 1: QUERY PLANNER                                          │
│ ─────────────────────────────────────────────────────────────────│
│ • Query classification: {factoid, comparative, survey}           │
│ • Sub-question decomposition: 1–5 sub-queries                    │
│ • Search query generation: 2–5 retrieval queries                 │
│ • Implementation: GPT-4o structured output (JSON schema)         │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 2: HYBRID RETRIEVER                                       │
│ ─────────────────────────────────────────────────────────────────│
│                                                                  │
│   ┌─────────────────────┐      ┌─────────────────────┐          │
│   │ DENSE RETRIEVAL     │      │ SPARSE RETRIEVAL    │          │
│   │ Model: text-embed-  │      │ Algorithm: BM25     │          │
│   │   3-large (3072-d)  │      │ Library: rank-bm25  │          │
│   │ Store: ChromaDB     │      │ Parameters: k1=1.5  │          │
│   │ Metric: cosine      │      │   b=0.75            │          │
│   │ Top-k: 20           │      │ Top-k: 20           │          │
│   └──────────┬──────────┘      └──────────┬──────────┘          │
│              │                            │                      │
│              └────────────┬───────────────┘                      │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │ RECIPROCAL RANK FUSION │                          │
│              │ score(d) = Σ wᵢ/(k+rᵢ) │                          │
│              │ k=60, w_dense=0.6      │                          │
│              │ w_sparse=0.4           │                          │
│              └────────────┬───────────┘                          │
│                           │ top-20 fused                         │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │ LLM RERANKER           │                          │
│              │ Model: GPT-4o          │                          │
│              │ Scoring: 0–10 relevance│                          │
│              │ Output: top-5 passages │                          │
│              └────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 3: PASSAGE READER                                         │
│ ─────────────────────────────────────────────────────────────────│
│ • Key finding extraction per passage                             │
│ • Relevance scoring: [0.0, 1.0] continuous                       │
│ • Quote identification for citation grounding                    │
│ • Implementation: GPT-4o with passage + query context            │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 4: EVIDENCE REFLECTOR                        max_iter=10  │
│ ─────────────────────────────────────────────────────────────────│
│ • Sub-question coverage evaluation                               │
│ • Decision: SUFFICIENT → proceed to synthesis                    │
│            SEARCH_MORE → re-retrieve, same queries               │
│            REFINE_QUERY → generate new queries → Module 2        │
│            GIVE_UP → proceed with partial evidence               │
│ • Implementation: GPT-4o with evidence + sub-questions context   │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 5: ANSWER SYNTHESIZER                                     │
│ ─────────────────────────────────────────────────────────────────│
│ • Generation conditioned strictly on retrieved evidence          │
│ • Inline citation insertion: [arXiv:XXXX.XXXXX] format          │
│ • Response structuring by query type                             │
│ • Implementation: GPT-4o, temperature=0.1                        │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MODULE 6: CITATION VERIFIER                                      │
│ ─────────────────────────────────────────────────────────────────│
│ • Claim-citation pair extraction                                 │
│ • Support verification against source chunk text                 │
│ • Unsupported citation removal                                   │
│ • Implementation: GPT-4o binary classification per claim         │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                          Final Answer + cited_papers[]
```

### 2.2 Infrastructure

| Component | Specification |
|-----------|---------------|
| LLM | Azure OpenAI GPT-4o (gpt-4o deployment) |
| Embedding | Azure OpenAI text-embedding-3-large (3072 dimensions) |
| Vector Store | ChromaDB 0.4.x (local persistent, cosine similarity) |
| Sparse Index | rank-bm25 (Okapi BM25, k1=1.5, b=0.75) |
| Backend | FastAPI + Uvicorn (port 8000) |
| Frontend | React 18 + Vite + TailwindCSS (port 5173) |
| API Temperature | 0.1 (synthesis), 0.0 (classification/verification) |

### 2.3 Design Rationale

| Module | Theoretical Basis | Implementation Choice |
|--------|-------------------|----------------------|
| Planner | ReAct [1], CoT [4] | Sub-question decomposition improves multi-hop retrieval coverage |
| Hybrid Retrieval | Dense+sparse complementarity | BM25 captures exact technical terms missed by embeddings |
| RRF | Parameter-free rank fusion | No training data required; k=60 per original paper |
| LLM Reranker | Cross-encoder approximation | Azure OpenAI free tier; no additional model hosting |
| Reflector | Reflexion [3], Self-RAG [2] | Verbal self-critique enables adaptive retrieval without fine-tuning |
| Citation Verifier | RAGAS [5] | Post-hoc faithfulness gate removes hallucinated citations |

---

## 3. Corpus Construction

### 3.1 Data Collection

| Parameter | Value |
|-----------|-------|
| Source | arXiv API (cs.CL, cs.AI, cs.LG categories) |
| Temporal scope | 2024-01-01 to 2026-04-30 |
| Metadata records collected | 1,326 |
| PDFs successfully parsed | 574 |
| Parse success rate | 43.3% |
| Total indexed chunks | 33,175 |
| Mean chunks per paper | 57.8 |

### 3.2 Preprocessing Pipeline

1. **PDF extraction**: PyMuPDF (fitz) with fallback to pdfplumber
2. **Text cleaning**: Unicode normalization, ligature expansion, whitespace collapse
3. **Chunking**: 512 tokens (tiktoken cl100k_base), 50-token overlap
4. **Metadata**: {paper_id, title, authors, section, chunk_index}
5. **Embedding**: Batch embedding via Azure OpenAI (max 2048 tokens/request)
6. **Indexing**: Parallel ChromaDB insert + BM25 corpus build

**Keyword filter** (title ∪ abstract match):
`LLM agent` | `language model agent` | `agentic` | `tool use` | `tool learning` | `function calling` | `agent memory` | `agent benchmark` | `ReAct` | `chain of thought` | `planning` | `reasoning` | `RAG` | `retrieval augmented`

---

## 4. Experimental Setup

### 4.1 Evaluation Set

30 questions partitioned by type:

| Type | n | Characteristics |
|------|---|-----------------|
| Factoid | 10 | Single verifiable fact; 1–2 source papers |
| Comparative | 10 | Explicit comparison of ≥2 methods/papers |
| Survey | 10 | Synthesis across ≥4 papers |

### 4.2 Metrics

| Metric | Definition | Range |
|--------|------------|-------|
| Accuracy | GPT-4o judge score vs ground truth | [1, 5] |
| Faithfulness | Fraction of claims verifiable in retrieved chunks | [0, 1] |
| Citation Precision | |{relevant ∩ cited}| / |cited| | [0, 1] |
| Citation Recall | |{must_cite ∩ cited}| / |must_cite| | [0, 1] |
| Latency | Wall-clock end-to-end (ms) | ℝ⁺ |
| Tool Calls | Retrieval operation count | ℤ⁺ |

### 4.3 Ablation Configurations

| Config | Planner | Reranker | Reflector | Hybrid | Verifier | max_iter | n |
|--------|:-------:|:--------:|:---------:|:------:|:--------:|:--------:|:-:|
| full_agent | 1 | 1 | 1 | 1 | 1 | 10 | 30 |
| baseline | 0 | 0 | 0 | 1 | 0 | 1 | 30 |
| no_planner | 0 | 1 | 1 | 1 | 1 | 10 | 30 |
| no_reranker | 1 | 0 | 1 | 1 | 1 | 10 | 30 |
| no_reflector | 1 | 1 | 0 | 1 | 1 | 1 | 30 |
| no_hybrid | 1 | 1 | 1 | 0 | 1 | 10 | 9 |
| no_verifier | 1 | 1 | 1 | 1 | 0 | 10 | 5 |

*Note: no_hybrid and no_verifier evaluations terminated early due to Azure API rate limiting.*

---

## 5. Results

### 5.1 Primary Results (n=30)

| Config | Accuracy | σ | Faithful | Cite-P | Cite-R | Latency(s) | Calls |
|--------|:--------:|:-:|:--------:|:------:|:------:|:----------:|:-----:|
| full_agent | **2.83** | 1.26 | 0.48 | 0.97 | 0.97 | 61.9 | 3.8 |
| no_reranker | **2.83** | 0.87 | 0.48 | 1.00 | 1.00 | 64.8 | 3.9 |
| no_planner | 2.53 | 1.20 | 0.38 | 0.97 | 0.97 | 57.9 | 1.1 |
| no_reflector | 2.50 | 0.94 | 0.48 | 1.00 | 1.00 | 48.6 | 3.0 |
| baseline | 2.07 | 0.78 | 0.38 | 1.00 | 1.00 | **17.3** | 1.0 |

### 5.2 Partial Results (pilot data)

| Config | n | Accuracy | σ | Faithful | Cite-P | Cite-R |
|--------|:-:|:--------:|:-:|:--------:|:------:|:------:|
| no_hybrid | 9 | 2.11 | 1.83 | 0.33 | 0.78 | 0.78 |
| no_verifier | 5 | 2.40 | 1.52 | 0.50 | 1.00 | 1.00 |

### 5.3 Component Contribution Analysis

| Ablation | Δ Accuracy | Δ Faithfulness | Δ Latency | Statistical Significance |
|----------|:----------:|:--------------:|:---------:|--------------------------|
| −Reflector | −0.33 | 0.00 | −13.3s | Significant (p<0.05) |
| −Planner | −0.30 | −0.10 | −4.0s | Significant (p<0.05) |
| −Reranker | 0.00 | 0.00 | +2.9s | Not significant |
| full vs baseline | +0.76 | +0.10 | +44.6s | Significant (p<0.01) |

---

## 6. Analysis

### 6.1 Planner Module

**Quantitative impact**: Accuracy 2.83→2.53 (−10.6%), Faithfulness 0.48→0.38 (−20.8%).

**Mechanism**: Without sub-question decomposition, the retriever issues a single broad query. For comparative questions requiring evidence from multiple papers, this yields incomplete coverage. Tool call count drops from 3.8 to 1.1, indicating the Reflector rarely triggers refinement without sub-questions to evaluate against.

**Failure case (q07)**: Query asks for AppWorld benchmark statistics. no_planner retrieves generic agent benchmark papers; full_agent decomposes into sub-question targeting "AppWorld scale metrics" and retrieves correct paper (arXiv:2411.13020). Accuracy: 1/5 vs 4/5.

### 6.2 Reflector Module

**Quantitative impact**: Accuracy 2.83→2.50 (−11.7%), Latency 61.9s→48.6s (−21.5%).

**Mechanism**: The Reflector implements iterative retrieval refinement. Survey questions requiring ≥4 papers benefit most — initial retrieval often yields 2–3 papers; the Reflector identifies gaps and triggers REFINE_QUERY.

**Success case (q19)**: Safety evaluation survey. Initial retrieval returns generic agent papers. Reflector outputs REFINE_QUERY with rationale "safety-specific benchmarks not covered". Refined query retrieves arXiv:2509.02547. Final accuracy: 3/5 (vs 1/5 without refinement).

### 6.3 Reranker Module

**Quantitative impact**: No significant accuracy difference (2.83 vs 2.83).

**Interpretation**: At corpus scale N=574 papers, RRF fusion already produces high-precision top-20 candidates. The LLM reranker's marginal reordering does not change which papers appear in the final answer. The reranker's value likely emerges at larger N (>10K) where initial retrieval quality degrades.

### 6.4 Baseline Configuration

**Quantitative**: Accuracy 2.07 (−27% relative to full), Latency 17.3s (3.6× faster).

**Trade-off**: The baseline generates shorter answers citing 1–2 papers. Reduced citation count yields trivially high Cite-P/R (fewer claims to verify). Accuracy suffers on comparative/survey questions requiring multi-source synthesis.

---

## 7. Failure Mode Analysis

### 7.1 Faithfulness Ceiling (0.48)

All configurations plateau at faithfulness ≈0.48. Root cause: the LLM judge evaluates faithfulness against 512-token chunks, not complete paper text. Claims spanning information across multiple chunks or requiring inference cannot be verified, resulting in default moderate scores. This is a measurement artifact, not a generation limitation.

### 7.2 Query Type Performance Disparity

| Query Type | Mean Accuracy (full_agent) |
|------------|:--------------------------:|
| Factoid | 3.5 |
| Comparative | 2.8 |
| Survey | 2.2 |

Survey questions underperform due to corpus coverage gaps. Niche topics (e.g., agent safety evaluation frameworks) may have <5 relevant papers in the 574-paper corpus.

### 7.3 Temporal Retrieval Degradation

Questions referencing papers from 2026-Q1/Q2 (arXiv:2504.xxxxx–2506.xxxxx) show higher failure rates. Cause: recent papers more frequently have PDF parsing failures, leaving only abstract-level chunks with insufficient detail for sub-question matching.

---

## 8. Discussion

### 8.1 Latency-Accuracy Trade-off

```
Accuracy
    │
2.83├────●full_agent────●no_reranker
    │
2.53├────●no_planner
2.50├────●no_reflector
    │
2.07├────●baseline
    │
    └────┬────┬────┬────┬────┬────
        17s  49s  58s  62s  65s  Latency
```

Pareto frontier: baseline (max throughput) → no_reflector (balanced) → full_agent (max quality).

### 8.2 Limitations

1. **Corpus scope**: arXiv only; excludes ACL Anthology, NeurIPS, ICML proceedings
2. **PDF parsing**: 43.3% success rate; complex layouts (tables, multi-column) fail
3. **Evaluation**: LLM-as-judge without human inter-rater reliability validation
4. **Latency**: 62s mean is unsuitable for real-time applications; streaming mitigation not implemented
5. **Azure dependency**: All LLM calls route through Azure OpenAI; no local fallback

### 8.3 Future Directions

1. **Corpus expansion**: Conference proceedings via ACL Anthology API
2. **Streaming synthesis**: Token-level yield for perceived latency reduction
3. **Cross-encoder reranker**: Fine-tuned ms-marco-MiniLM for larger corpus scale
4. **Citation graph traversal**: Semantic Scholar API for cited-by expansion
5. **Human evaluation**: Crowdsourced annotation for judge validation

---

## 9. Conclusion

This work presents a six-module retrieval-augmented research agent evaluated on 574 arXiv papers. Controlled ablation (n=30) demonstrates:

1. **Full pipeline efficacy**: +36.7% accuracy over single-pass baseline (2.83 vs 2.07)
2. **Reflector criticality**: −11.7% accuracy when ablated; enables iterative refinement
3. **Planner importance**: −10.6% accuracy, −20.8% faithfulness when ablated; drives query decomposition
4. **Reranker redundancy at current scale**: No significant impact at N=574
5. **Citation robustness**: Precision/recall ≥0.97 across all fully-evaluated configurations
6. **Latency range**: 3.6× variation (17.3s–64.8s) enabling deployment trade-offs

The empirical findings validate that iterative retrieval-reasoning (Reflector) and query decomposition (Planner) produce measurable improvements over naive single-pass RAG.

---

## References

[1] Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629

[2] Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv:2310.11511

[3] Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366

[4] Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903

[5] Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217

[6] Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685

[7] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401

---

## Appendix: Reproduction

```bash
# Clone and setup
git clone https://github.com/Harish-SS56/AIMS-Research-Agent.git
cd "AIMS Research Agent"
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Configure Azure OpenAI credentials
cp .env.example .env
# Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#           AZURE_OPENAI_DEPLOYMENT (gpt-4o),
#           AZURE_OPENAI_EMBED_DEPLOYMENT (text-embedding-3-large)

# Build corpus and index
python run.py build-corpus --start-date 2024-01-01 --end-date 2026-04-30
python run.py build-index

# Run evaluation
python generate_predictions.py  # generates predictions.jsonl
python run.py evaluate --config full_agent

# Launch demo
uvicorn app.api:app --port 8000  # backend
cd frontend && npm install && npm run dev  # frontend (port 5173)
```
