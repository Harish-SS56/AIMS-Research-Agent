# AIMS Deep Research Agent — Technical Report

**Harish S S** · AIMS-DTU Research Intern 2026 · June 2026

---

## Abstract

I built a multi-stage research agent that retrieves and synthesizes evidence from 574 arXiv papers (33,175 chunks) to answer research questions with citations. The system chains six modules: Planner, Hybrid Retriever, Reader, Reflector, Synthesizer, and Citation Verifier.

Ablation on 30 questions shows the full agent scores **2.83/5.0** vs **2.07** for the single-pass baseline (+37%). Removing the Reflector costs 0.33 accuracy; removing the Planner costs 0.30. The Reranker had no impact at this corpus size. Latency ranges from 17s (baseline) to 62s (full agent). Citation precision/recall stays above 0.97 for all fully-evaluated configs.

**Repository:** [github.com/Harish-SS56/AIMS-Research-Agent](https://github.com/Harish-SS56/AIMS-Research-Agent)

---

## 1. Introduction

Single-pass LLM prompting works fine for factoid questions but fails on research questions that need evidence from multiple papers with proper citations. The solution is to run the LLM in a loop: decompose the question, retrieve evidence, check if it's enough, refine queries if not, then synthesize.

I built this end-to-end on arXiv papers about LLM agents, with ablation capability to measure what each component actually contributes.

**Goals:**
1. Corpus of arXiv papers on LLM agents (Jan 2024 – Apr 2026)
2. Answer 30 research questions with cited evidence
3. Ablate components to measure their contribution
4. Evaluate with LLM-as-judge + citation metrics

---

## 2. Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PLANNER                                                 │
│ Classify query type · Decompose into sub-questions      │
│ Generate 2-5 search queries                             │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ HYBRID RETRIEVER                                        │
│                                                         │
│  Dense (text-embedding-3-large)  +  Sparse (BM25)      │
│              │                           │              │
│              └─────────┬─────────────────┘              │
│                   RRF Fusion                            │
│                   (k=60, 0.6/0.4 weights)               │
│                        │                                │
│               LLM Reranker → top 5                      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ READER                                                  │
│ Extract findings · Score relevance · Identify quotes    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ REFLECTOR                                  (max 10 iter)│
│                                                         │
│ SUFFICIENT → proceed    SEARCH_MORE → retry             │
│ REFINE_QUERY → new queries → back to retriever          │
│ GIVE_UP → proceed with available evidence               │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ SYNTHESIZER                                             │
│ Generate answer with inline citations [arXiv:XXXX.XXXXX]│
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ CITATION VERIFIER                                       │
│ Check each citation supports its claim · Remove invalid │
└────────────────────────┬────────────────────────────────┘
                         ▼
                   Final Answer
```

### Design Choices

Each component comes from specific prior work:

| Component | Source | What I took from it |
|-----------|--------|---------------------|
| Planner | ReAct, Chain-of-Thought | Reason before acting; decompose into sub-questions |
| Hybrid Retriever | Dense+sparse fusion literature | BM25 catches exact terms that embeddings miss |
| Reader | Self-RAG | Score passage relevance before synthesis |
| Reflector | Reflexion, Self-RAG | Self-critique to decide if evidence is sufficient |
| Citation Verifier | RAGAS | Post-hoc verification removes hallucinated citations |

**Why LLM reranker over cross-encoder?** Stays within Azure free tier, no extra model to host.

**Why RRF over learned fusion?** No training data needed, works well in practice.

**Why iterative reflection?** Survey questions need multiple passes. Single-pass consistently misses evidence.

---

## 3. Corpus

| | |
|---|---|
| Source | arXiv API (cs.CL, cs.AI, cs.LG) |
| Date range | Jan 2024 – Apr 2026 |
| Papers indexed | 574 |
| Total chunks | 33,175 |
| Chunk size | 512 tokens, 50 overlap |
| Embedding | text-embedding-3-large (3072-dim) |
| Vector store | ChromaDB |
| Lexical index | BM25 |

Keyword filter: LLM agent, tool use, function calling, ReAct, chain of thought, RAG, etc.

---

## 4. Experiments

### Evaluation Set

30 questions: 10 factoid, 10 comparative, 10 survey.

### Metrics

- **Accuracy**: GPT-4o judge scores 1-5
- **Faithfulness**: Claims verifiable in retrieved chunks (0-1)
- **Citation P/R**: Against must-cite papers
- **Latency**: End-to-end wall clock
- **Tool calls**: Retrieval operations count

### Ablation Configs

| Config | Planner | Reranker | Reflector | n |
|--------|:-------:|:--------:|:---------:|:-:|
| full_agent | ✓ | ✓ | ✓ | 30 |
| baseline | ✗ | ✗ | ✗ | 30 |
| no_planner | ✗ | ✓ | ✓ | 30 |
| no_reranker | ✓ | ✗ | ✓ | 30 |
| no_reflector | ✓ | ✓ | ✗ | 30 |
| no_hybrid | ✓ | ✓ | ✓ | 9 |
| no_verifier | ✓ | ✓ | ✓ | 5 |

---

## 5. Results

### Main Results (n=30)

| Config | Accuracy | Faithful | Cite-P | Cite-R | Latency |
|--------|:--------:|:--------:|:------:|:------:|:-------:|
| **full_agent** | **2.83** ±1.26 | 0.48 | 0.97 | 0.97 | 61.9s |
| no_reranker | **2.83** ±0.87 | 0.48 | 1.00 | 1.00 | 64.8s |
| no_planner | 2.53 ±1.20 | 0.38 | 0.97 | 0.97 | 57.9s |
| no_reflector | 2.50 ±0.94 | 0.48 | 1.00 | 1.00 | 48.6s |
| baseline | 2.07 ±0.78 | 0.38 | 1.00 | 1.00 | **17.3s** |

### Component Impact

| Removed | Accuracy Δ | Verdict |
|---------|:----------:|---------|
| Reflector | −0.33 | Matters — enables iterative refinement |
| Planner | −0.30 | Matters — drives targeted retrieval |
| Reranker | 0.00 | No impact at this corpus size |
| Full vs baseline | +0.76 | Components collectively help |

---

## 6. Analysis

### Planner

Removing it drops accuracy from 2.83 to 2.53 and faithfulness from 0.48 to 0.38. Without sub-questions, the agent issues one broad query and misses half the evidence.

**Example (q07):** The question asks about AppWorld benchmark statistics. Without the planner, the agent said "evidence does not mention the number of apps or tasks" (accuracy 1/5). With the planner, it correctly retrieved "9 apps, ~750 tasks" (accuracy 4/5). The planner generated a sub-question specifically targeting benchmark scale.

### Reflector

Removing it drops accuracy from 2.83 to 2.50 but saves 13s latency. Survey questions suffer most — they need multiple retrieval passes, and without reflection the agent commits to whatever it got first.

**Example (q19):** A safety evaluation survey question. First retrieval missed safety-specific papers. The reflector caught this ("safety evaluation not covered") and generated a refined query targeting "agent safety benchmark red-teaming". Second pass found the right papers.

### Reranker

No accuracy impact. At 574 papers, RRF already produces good top-20 results. The reranker's reordering doesn't change which papers end up cited. At larger corpus scale (10K+ papers), it would probably matter more.

### Baseline

Fastest (17.3s) but lowest accuracy (2.07). It generates simpler answers citing 1-2 papers. Fewer citations = fewer chances to fail verification = high cite-P/R. But accuracy suffers because it can't handle comparative or survey questions properly.

---

## 7. Failure Modes

**Faithfulness ceiling at 0.48**: The judge checks claims against 512-token chunks, not full papers. It can't verify everything, so defaults to moderate scores. This is a measurement limitation.

**Survey questions underperform**: Average 2.2/5 vs 3.5/5 for factoid. The corpus may not have comprehensive coverage for niche survey topics.

**Recent papers fail**: Q1-Q2 2026 papers sometimes only have abstract-level chunks (PDF parsing failures), making sub-question matching hard.

---

## 8. Worked Examples

### Success: Factoid (q04)

*"What benchmark does the SWE-agent paper introduce, and how many GitHub issues?"*

Planner: one sub-question targeting benchmark name and size. Retriever: found SWE-agent paper. Reader: extracted "SWE-bench, 2,294 issues from 12 repos". Reflector: sufficient. Synthesizer: answer with citation. Verifier: validated.

**Result:** Accuracy 5/5, latency 38s.

### Reflector Save: Survey (q19)

*"Survey LLM agent safety evaluation frameworks and benchmarks."*

First retrieval: generic agent eval papers, nothing safety-specific. Reflector: REFINE_QUERY. Second query: "agent safety benchmark red-teaming". Found safety-focused paper. Synthesized answer covering 3 frameworks.

**Result:** Accuracy 3/5 (corpus gaps on late-2025 safety work).

### Planner Failure: Factoid (q07)

*no_planner:* "Evidence does not mention the number of apps or tasks" → 1/5

*full_agent:* "AppWorld: 9 apps, ~750 tasks [arXiv:2411.13020]" → 4/5

Root cause: Without the sub-question "AppWorld benchmark scale statistics", the retriever never targeted the right paper.

---

## 9. Discussion

### Latency vs Quality

```
Accuracy
  2.8  ●full_agent  ●no_reranker
  2.5  ●no_planner  ●no_reflector
  2.1  ●baseline
       ─────────────────────────────
       17s    49s    58s    62s    65s
                   Latency
```

Trade-off is clear: baseline is 3.6× faster but 27% worse on accuracy. no_reflector is a middle ground (48s, 2.50 accuracy).

### Limitations

1. Corpus is arXiv only — no ACL/NeurIPS proceedings
2. ~8% of papers have degraded chunks from complex PDF layouts
3. No human evaluation to validate LLM-as-judge scores
4. 62s latency is long; streaming would help perceived responsiveness

### Future Work

- Expand corpus to conference proceedings
- Streaming synthesis for faster perceived latency
- Human evaluation to validate judge scores
- Citation graph traversal for related paper discovery

---

## 10. Conclusion

The full pipeline beats the baseline by 37% (2.83 vs 2.07). The Reflector (−0.33 when removed) and Planner (−0.30) are the key contributors. The Reranker doesn't help at this corpus size. Citation quality stays high (≥0.97) across configs.

The main lesson: iterative refinement (Reflector) and query decomposition (Planner) produce measurable gains. Single-pass prompting isn't enough for research questions.

---

## References

1. Yao et al. (2022). ReAct: Synergizing Reasoning and Acting. arXiv:2210.03629
2. Asai et al. (2023). Self-RAG. arXiv:2310.11511
3. Shinn et al. (2023). Reflexion. arXiv:2303.11366
4. Wei et al. (2022). Chain-of-Thought Prompting. arXiv:2201.11903
5. Es et al. (2023). RAGAS. arXiv:2309.15217
6. Zheng et al. (2023). LLM-as-Judge. arXiv:2306.05685
7. Lewis et al. (2020). RAG. arXiv:2005.11401

---

## Appendix: Reproduction

```bash
git clone https://github.com/Harish-SS56/AIMS-Research-Agent.git
cd "AIMS Research Agent"
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add Azure OpenAI credentials

# Build corpus
python run.py build-corpus --start-date 2024-01-01 --end-date 2026-04-30
python run.py build-index

# Run evaluation
python run.py evaluate --config full_agent

# Run demo
uvicorn app.api:app --port 8000
cd frontend && npm run dev
```
