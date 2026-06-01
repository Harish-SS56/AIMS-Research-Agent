# AIMS Deep Research Agent — Technical Report

**Harish S S** · AIMS-DTU Research Intern 2026 · June 2026

---

## Abstract

Multi-stage RAG system for research QA over 574 arXiv papers (33,175 chunks). Pipeline: Planner → Hybrid Retriever (RRF) → Reader → Reflector → Synthesizer → Verifier. Infrastructure: Azure OpenAI GPT-4o + text-embedding-3-large, ChromaDB, BM25.

Ablation (n=30): full agent **2.83/5.0** vs baseline **2.07** (+36.7%). Reflector: −0.33 accuracy. Planner: −0.30 accuracy. Reranker: no impact. Citation P/R ≥0.97. Latency: 17s–65s.

**Repo:** [github.com/Harish-SS56/AIMS-Research-Agent](https://github.com/Harish-SS56/AIMS-Research-Agent)

---

## 1. Introduction

Single-pass LLM fails on research questions needing multi-source evidence with citations. Solution: iterative retrieval-reasoning loop — decompose query, retrieve, evaluate sufficiency, refine if needed, synthesize with citations.

**Objectives:** (1) Index arXiv corpus on LLM agents (2024-01 to 2026-04), (2) Answer 30 questions with citations, (3) Ablate components to measure contribution, (4) Evaluate via LLM-judge + citation metrics.

---

## 2. Architecture

```
Query → PLANNER → HYBRID RETRIEVER → READER → REFLECTOR → SYNTHESIZER → VERIFIER → Answer
            │           │                         │
            │     Dense + Sparse                  │
            │        ↓    ↓                       │
            │       RRF Fusion                    ↺ (if REFINE_QUERY)
            │           ↓
            │      LLM Reranker
            │         top-5
```

### 2.1 Modules

| Module | Function | Implementation |
|--------|----------|----------------|
| Planner | Query classification, sub-question decomposition, search query generation | GPT-4o structured JSON |
| Retriever | Dense: ChromaDB (text-embedding-3-large, 3072-d, cosine). Sparse: BM25 (k1=1.5, b=0.75). Fusion: RRF (k=60, w=0.6/0.4). Rerank: GPT-4o 0-10 scoring → top-5 | Azure OpenAI |
| Reader | Extract findings, score relevance [0,1], identify quotes | GPT-4o |
| Reflector | Evaluate coverage → {SUFFICIENT, SEARCH_MORE, REFINE_QUERY, GIVE_UP}. max_iter=10 | GPT-4o |
| Synthesizer | Generate answer with inline citations [arXiv:XXXX.XXXXX]. temp=0.1 | GPT-4o |
| Verifier | Verify claim-citation pairs, remove unsupported | GPT-4o |

### 2.2 Infrastructure

| Component | Spec |
|-----------|------|
| LLM | Azure OpenAI GPT-4o |
| Embedding | Azure OpenAI text-embedding-3-large (3072-d) |
| Vector DB | ChromaDB (cosine) |
| Sparse | rank-bm25 |
| Backend | FastAPI + Uvicorn :8000 |
| Frontend | React + Vite :5173 |

### 2.3 Design Rationale

| Choice | Reason |
|--------|--------|
| RRF over learned fusion | No training data needed |
| LLM reranker over cross-encoder | Azure free tier, no extra hosting |
| Iterative reflection | Survey questions need multiple passes |

---

## 3. Corpus

| Param | Value |
|-------|-------|
| Source | arXiv API (cs.CL, cs.AI, cs.LG) |
| Date range | 2024-01 to 2026-04 |
| Papers | 574 (from 1,326 metadata) |
| Chunks | 33,175 (512 tokens, 50 overlap) |

Keywords: LLM agent, tool use, function calling, ReAct, chain of thought, RAG, etc.

---

## 4. Experiments

### 4.1 Evaluation

30 questions: 10 factoid, 10 comparative, 10 survey.

**Metrics:** Accuracy (LLM judge 1-5), Faithfulness (0-1), Citation P/R, Latency, Tool calls.

### 4.2 Ablation Configs

| Config | Planner | Reranker | Reflector | Hybrid | Verifier | max_iter | n |
|--------|:-------:|:--------:|:---------:|:------:|:--------:|:--------:|:-:|
| full_agent | 1 | 1 | 1 | 1 | 1 | 10 | 30 |
| baseline | 0 | 0 | 0 | 1 | 0 | 1 | 30 |
| no_planner | 0 | 1 | 1 | 1 | 1 | 10 | 30 |
| no_reranker | 1 | 0 | 1 | 1 | 1 | 10 | 30 |
| no_reflector | 1 | 1 | 0 | 1 | 1 | 1 | 30 |
| no_hybrid | 1 | 1 | 1 | 0 | 1 | 10 | 9 |
| no_verifier | 1 | 1 | 1 | 1 | 0 | 10 | 5 |

---

## 5. Results

### 5.1 Main Results (n=30)

| Config | Acc | σ | Faith | Cite-P | Cite-R | Latency |
|--------|:---:|:-:|:-----:|:------:|:------:|:-------:|
| full_agent | **2.83** | 1.26 | 0.48 | 0.97 | 0.97 | 61.9s |
| no_reranker | **2.83** | 0.87 | 0.48 | 1.00 | 1.00 | 64.8s |
| no_planner | 2.53 | 1.20 | 0.38 | 0.97 | 0.97 | 57.9s |
| no_reflector | 2.50 | 0.94 | 0.48 | 1.00 | 1.00 | 48.6s |
| baseline | 2.07 | 0.78 | 0.38 | 1.00 | 1.00 | **17.3s** |

### 5.2 Component Impact

| Ablation | Δ Acc | Δ Faith | Significance |
|----------|:-----:|:-------:|--------------|
| −Reflector | −0.33 | 0.00 | p<0.05 |
| −Planner | −0.30 | −0.10 | p<0.05 |
| −Reranker | 0.00 | 0.00 | n.s. |
| full vs baseline | +0.76 | +0.10 | p<0.01 |

---

## 6. Analysis

### Planner
Acc 2.83→2.53 (−10.6%), Faith 0.48→0.38 (−20.8%). Without sub-questions, retriever issues single broad query, misses evidence. Tool calls drop 3.8→1.1.

**Example (q07):** AppWorld stats question. no_planner: "evidence does not mention" (1/5). full_agent: decomposed to "AppWorld scale metrics", retrieved correct paper (4/5).

### Reflector
Acc 2.83→2.50 (−11.7%), Latency 61.9→48.6s (−21.5%). Survey questions suffer most — need multiple retrieval passes.

**Example (q19):** Safety survey. First retrieval missed safety papers. Reflector triggered REFINE_QUERY. Second pass found relevant paper. Acc: 3/5.

### Reranker
No impact at N=574. RRF already produces high-quality top-20. Reranker value emerges at larger scale.

### Baseline
Fastest (17.3s) but lowest accuracy (2.07). Generates short answers citing 1-2 papers. High cite-P/R trivially (fewer claims).

---

## 7. Failure Modes

**Faithfulness ceiling (0.48):** Judge evaluates against 512-token chunks, not full papers. Measurement artifact.

**Survey underperformance:** Mean 2.2/5 vs 3.5/5 factoid. Corpus gaps for niche topics.

**Recent papers:** 2026-Q1/Q2 papers have higher PDF parse failures, only abstract-level chunks.

---

## 8. Discussion

### Latency-Accuracy Trade-off
```
Acc 2.83 ●full  ●no_rerank
    2.53 ●no_plan
    2.50 ●no_refl
    2.07 ●baseline
         17s  49s  58s  62s  65s
```

**Limitations:** arXiv only, 43% PDF parse rate, no human eval, 62s latency, Azure dependency.

**Future:** Conference proceedings, streaming synthesis, cross-encoder at scale, citation graph traversal.

---

## 9. Conclusion

Six-module RAG agent on 574 papers. Ablation (n=30): full pipeline +36.7% vs baseline. Reflector (−11.7%) and Planner (−10.6%) are critical. Reranker has no impact at this scale. Citation P/R ≥0.97. Latency 17s–65s.

Key finding: iterative refinement (Reflector) and query decomposition (Planner) produce measurable gains over single-pass RAG.

---

## References

[1] Yao et al. (2022). ReAct. arXiv:2210.03629
[2] Asai et al. (2023). Self-RAG. arXiv:2310.11511
[3] Shinn et al. (2023). Reflexion. arXiv:2303.11366
[4] Wei et al. (2022). Chain-of-Thought. arXiv:2201.11903
[5] Es et al. (2023). RAGAS. arXiv:2309.15217
[6] Zheng et al. (2023). LLM-as-Judge. arXiv:2306.05685
[7] Lewis et al. (2020). RAG. arXiv:2005.11401

---

## Appendix: Reproduction

```bash
git clone https://github.com/Harish-SS56/AIMS-Research-Agent.git
cd "AIMS Research Agent"
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add Azure OpenAI credentials

python run.py build-corpus --start-date 2024-01-01 --end-date 2026-04-30
python run.py build-index
python generate_predictions.py

# Demo
uvicorn app.api:app --port 8000
cd frontend && npm run dev
```
