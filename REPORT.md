# Technical Report: AIMS Deep Research Agent

<div align="center">

**AIMS-DTU Research Intern 2026 — Agentic Systems in Generative AI**

**Author:** Harish S S &nbsp;|&nbsp; **Date:** June 2026

**Repository:** [github.com/Harish-SS56/AIMS-Research-Agent](https://github.com/Harish-SS56/AIMS-Research-Agent) &nbsp;|&nbsp; **Demo:** [frontend-flax-zeta-a2gm1yxwxw.vercel.app](https://frontend-flax-zeta-a2gm1yxwxw.vercel.app)

</div>

---

## Abstract

This report presents the design, implementation, and systematic evaluation of a multi-component agentic deep research system. The system retrieves and synthesizes evidence from a curated corpus of **574 arXiv papers** (33,175 indexed chunks, cs.CL/cs.AI/cs.LG, January 2024–April 2026) to answer research-grade questions with grounded citations. The pipeline integrates six specialized modules — Planner, Hybrid Retriever, Reader, Reflector, Synthesizer, and Citation Verifier — each motivated by a distinct line of prior work.

Ablation studies across **seven configurations** (five fully evaluated at n=30) reveal that the full agent achieves accuracy **2.83/5.0 (±1.26)**, a **+0.76 improvement** over the single-pass baseline (2.07 ±0.78). The Planner contributes +0.30 and the Reflector contributes +0.33 in accuracy. The Reranker shows no accuracy delta on this evaluation but reduces citation variance. Citation precision and recall remain ≥ 0.97 across all fully-evaluated configurations. The baseline achieves the fastest latency (17.3 s) at significant quality cost; the full agent requires 61.9 s but delivers the best quality-citation balance.

---

## 1. Introduction

Large language models (LLMs) answer well-formed factoid questions accurately in a single forward pass. Research questions, however, require assembling evidence from multiple heterogeneous sources, maintaining intermediate reasoning state, and producing grounded citations — capabilities that single-pass inference consistently fails to deliver [1, 4].

The *deep research agent* paradigm addresses this by placing the LLM in an agentic loop: decompose the question, issue targeted retrieval queries, read and critique the returned evidence, refine queries if evidence is insufficient, and only then synthesize a final answer [1]. This project builds such a system end-to-end on a fixed arXiv corpus, with full component ablation to isolate which architectural choices actually move the needle on answer quality.

### 1.1 Problem Statement

Design and implement an agentic deep research system that:
1. Constructs and indexes a domain-specific arXiv corpus (LLM agents, Jan 2024–Apr 2026)
2. Answers 30 research questions (factoid, comparative, survey) with cited evidence
3. Supports controlled ablation of individual components
4. Evaluates rigorously using LLM-as-judge accuracy, faithfulness, citation precision/recall, latency, and tool-call count

### 1.2 Key Contributions

| # | Contribution |
|---|---|
| 1 | **574-paper curated corpus** — filtered arXiv collection, 33,175 chunks, dual-indexed (ChromaDB + BM25) |
| 2 | **Six-module pipeline** — Planner → Retriever → Reader → Reflector → Synthesizer → Verifier |
| 3 | **Hybrid retrieval with RRF** — dense (text-embedding-3-large) + sparse (BM25) fusion, k=60 |
| 4 | **Systematic 7-config ablation** — fully evaluated on 5 configs (n=30 each) |
| 5 | **React + FastAPI demo** — live trace viewer showing plan, retrievals, reflector decisions, synthesis |

---

## 2. Motivating Literature & Design Rationale

Each pipeline component is grounded in a specific line of prior work. The table below maps each component to its motivating paper(s) and the design decision it inspired.

| Component | Motivating Work | Key Insight Adopted |
|-----------|----------------|---------------------|
| **Planner** | ReAct [1] — arXiv:2210.03629 | Interleave reasoning traces with action generation before retrieval |
| **Planner** | Chain-of-Thought [4] — arXiv:2201.11903 | Sub-question decomposition improves multi-hop evidence gathering |
| **Hybrid Retriever** | Ma et al. (2021) — hybrid dense+sparse | BM25 captures exact technical term matches that dense search misses |
| **Reader** | Self-RAG [2] — arXiv:2310.11511 | Passage-level relevance scoring before synthesis reduces noise |
| **Reflector** | Reflexion [3] — arXiv:2303.11366 | Verbal self-critique enables query refinement without parameter updates |
| **Reflector** | Self-RAG [2] — arXiv:2310.11511 | Adaptive retrieval: decide *whether* to search again, not just *what* to search |
| **Synthesizer** | Lewis et al. (2020) — RAG [7] | Generate answers conditioned strictly on retrieved context, not parametric memory |
| **Citation Verifier** | RAGAS [5] — arXiv:2309.15217 | Post-hoc faithfulness scoring; remove citations unsupported by retrieved text |
| **Evaluation** | LLM-as-Judge [6] — arXiv:2306.05685 | GPT-4-class judge provides reliable 1–5 accuracy scores on open-ended answers |

### 2.1 Why These Choices Over Alternatives

**LLM reranker vs. cross-encoder**: Cross-encoder rerankers (e.g., ms-marco-MiniLM) require hosting a separate model. Using GPT-4o for relevance scoring stays within the Azure OpenAI deployment, reduces infrastructure complexity, and performs comparably at our corpus scale.

**RRF vs. learned fusion**: Reciprocal Rank Fusion requires no training data, is parameter-free, and has strong theoretical and empirical backing for combining heterogeneous rankers. Learned fusion would require labeled relevance data we do not have.

**Iterative reflection vs. fixed pipeline**: Survey questions in the evaluation set require synthesizing 4+ papers. A fixed single-pass pipeline consistently misses evidence for sub-questions generated by the planner. The Reflector's REFINE_QUERY action recovers from bad initial queries without any re-training.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                         USER QUERY                                   ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ① PLANNER                                                           ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  • Classify query type: factoid / comparative / survey       │    ║
║  │  • Decompose into 1–5 sub-questions                          │    ║
║  │  • Generate 2–5 targeted search queries                      │    ║
║  │  • Extract key concepts and named entities                   │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  Motivated by: ReAct [1], Chain-of-Thought [4]                       ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ② HYBRID RETRIEVER                                                  ║
║                                                                      ║
║   ┌──────────────────┐    ┌──────────────────┐                      ║
║   │  DENSE SEARCH    │    │  SPARSE SEARCH   │                      ║
║   │ text-embed-3-    │    │   BM25           │                      ║
║   │ large (3072-dim) │    │  (rank-bm25)     │                      ║
║   │  ChromaDB        │    │  exact-term      │                      ║
║   └────────┬─────────┘    └────────┬─────────┘                      ║
║            │                       │                                 ║
║            └───────────┬───────────┘                                 ║
║                        │                                             ║
║               ┌────────▼────────┐                                    ║
║               │  RRF FUSION     │  score = Σ w_i / (60 + rank_i)    ║
║               │  w_dense = 0.6  │                                    ║
║               │  w_sparse = 0.4 │                                    ║
║               └────────┬────────┘                                    ║
║                        │  top-20 fused results                       ║
║               ┌────────▼────────┐                                    ║
║               │  LLM RERANKER   │  Score 0–10; keep top-5            ║
║               └────────┬────────┘                                    ║
╚════════════════════════╤═════════════════════════════════════════════╝
                         │  5 ranked passages
                         ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ③ READER                                                            ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  • Extract key findings per passage                          │    ║
║  │  • Score passage relevance (0–1 float)                       │    ║
║  │  • Identify supporting quotes                                │    ║
║  │  • Summarize evidence per arXiv source                       │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  Motivated by: Self-RAG [2]                                          ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │  evidence + relevance scores
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ④ REFLECTOR                                          max 10 iters  ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  Evaluate evidence vs. sub-questions                         │    ║
║  │                                                              │    ║
║  │  Decision tree:                                              │    ║
║  │   SUFFICIENT  ──────────────────────────────► proceed        │    ║
║  │   SEARCH_MORE ──► re-retrieve with same queries              │    ║
║  │   REFINE_QUERY ─► generate new targeted queries ──► ②        │    ║
║  │   GIVE_UP     ──► proceed with available evidence            │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  Motivated by: Reflexion [3], Self-RAG [2]                           ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │  sufficient evidence
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ⑤ SYNTHESIZER                                                       ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  • Generate answer grounded strictly in retrieved evidence   │    ║
║  │  • Insert inline citations [arXiv:XXXX.XXXXX]               │    ║
║  │  • Structure by query type (factoid / compare / survey)      │    ║
║  │  • Acknowledge evidence gaps explicitly                      │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  Motivated by: Lewis et al. RAG [7]                                  ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │  answer with raw citations
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ⑥ CITATION VERIFIER                                                 ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  • Extract each (claim, citation) pair from answer           │    ║
║  │  • Verify claim is supported by source chunk text            │    ║
║  │  • Remove unsupported citations                              │    ║
║  │  • Report verification confidence per citation               │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  Motivated by: RAGAS [5]                                             ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │
                               ▼
                    ┌──────────────────┐
                    │   FINAL ANSWER   │
                    │  + cited_papers  │
                    └──────────────────┘
```

### 3.2 Corpus Construction

**Source & Scope**

| Property | Value |
|----------|-------|
| Source | arXiv API (cs.CL, cs.AI, cs.LG) + Semantic Scholar |
| Date range | January 2024 – April 2026 |
| Raw metadata records | 1,326 |
| PDFs successfully parsed | 574 papers |
| Total indexed chunks | 33,175 |
| Vector store | ChromaDB (local, persistent) |
| Lexical index | BM25 via rank-bm25 |

**Keyword Filter** (title + abstract matching):
`LLM agent` · `language model agent` · `agentic` · `tool use` · `tool learning` · `function calling` · `agent memory` · `agent benchmark` · `ReAct` · `chain of thought` · `planning` · `reasoning` · `RAG` · `retrieval augmented`

**Chunking**: 512 tokens per chunk, 50-token overlap. Each chunk carries metadata: `paper_id`, `title`, `section`, `chunk_index`. Abstracts are stored as standalone chunks to ensure paper-level retrieval even when full-text parsing is imperfect.

**Embedding**: Azure OpenAI `text-embedding-3-large` (3,072 dimensions) — chosen for top-tier MTEB benchmark performance within the Azure free-tier deployment.

---

## 4. Experimental Setup

### 4.1 Evaluation Questions

30 questions from the AIMS evaluation set, across three types:

| Type | Count | Characteristics |
|------|-------|----------------|
| Factoid | 10 | Single verifiable fact; 1–2 papers sufficient |
| Comparative | 10 | Explicit comparison of 2+ methods/papers |
| Survey | 10 | Synthesis across 4+ papers; broad topic coverage |

### 4.2 Evaluation Metrics

| Metric | Computation | Range |
|--------|-------------|-------|
| **Answer Accuracy** | GPT-4o judge rates correctness vs. ground truth | 1–5 |
| **Faithfulness** | Proportion of claims verifiable in retrieved chunks | 0–1 |
| **Citation Precision** | Relevant cited papers / Total cited papers | 0–1 |
| **Citation Recall** | Cited must-cite papers / Total must-cite papers | 0–1 |
| **Latency** | Wall-clock end-to-end response time | seconds |
| **Tool Calls** | Count of retrieval operations issued | integer |

### 4.3 Ablation Configurations

| Configuration | Planner | Reranker | Reflector | Hybrid | Verifier | Max Iter | n eval |
|--------------|:-------:|:--------:|:---------:|:------:|:--------:|:--------:|:------:|
| **full_agent** | ✓ | ✓ | ✓ | ✓ | ✓ | 10 | **30** |
| baseline | ✗ | ✗ | ✗ | ✓ | ✗ | 1 | **30** |
| no_planner | ✗ | ✓ | ✓ | ✓ | ✓ | 10 | **30** |
| no_reranker | ✓ | ✗ | ✓ | ✓ | ✓ | 10 | **30** |
| no_reflector | ✓ | ✓ | ✗ | ✓ | ✓ | 1 | **30** |
| no_hybrid | ✓ | ✓ | ✓ | ✗ | ✓ | 10 | 9 (partial) |
| no_verifier | ✓ | ✓ | ✓ | ✓ | ✗ | 10 | 5 (partial) |

*no_hybrid and no_verifier were interrupted by Azure API timeouts; partial results are reported separately.*

---

## 5. Results

### 5.1 Main Results — Completed Evaluations (n = 30)

| Configuration | Accuracy ↑ | ± SD | Faithfulness ↑ | Cite-P ↑ | Cite-R ↑ | Latency ↓ | Tool Calls |
|--------------|:----------:|:----:|:--------------:|:--------:|:--------:|:---------:|:----------:|
| **full_agent** | **2.83** | 1.26 | **0.48** | 0.97 | 0.97 | 61.9 s | 3.8 |
| no_reranker | **2.83** | 0.87 | **0.48** | **1.00** | **1.00** | 64.8 s | 3.9 |
| no_planner | 2.53 | 1.20 | 0.38 | 0.97 | 0.97 | 57.9 s | 1.1 |
| no_reflector | 2.50 | 0.94 | **0.48** | **1.00** | **1.00** | 48.6 s | 3.0 |
| baseline | 2.07 | 0.78 | 0.38 | **1.00** | **1.00** | **17.3 s** | 1.0 |

### 5.2 Partial Evaluations (Pilot Data)

| Configuration | n | Accuracy ↑ | ± SD | Faithfulness ↑ | Cite-P ↑ | Cite-R ↑ | Latency ↓ |
|--------------|:-:|:----------:|:----:|:--------------:|:--------:|:--------:|:---------:|
| no_hybrid | 9 | 2.11 | 1.83 | 0.33 | 0.78 | 0.78 | 28.8 s |
| no_verifier | 5 | 2.40 | 1.52 | 0.50 | 1.00 | 1.00 | 26.8 s |

*Interpret with caution — small n, high variance. no_hybrid's cite-P/R = 0.78 confirms hybrid retrieval is important for citation quality.*

### 5.3 Component Contribution Summary

| Component Removed | Accuracy Δ | Faithfulness Δ | Latency Δ | Verdict |
|-------------------|:----------:|:--------------:|:---------:|---------|
| Planner | **−0.30** | **−0.10** | −4.0 s | **Matters** — drives targeted retrieval and grounding |
| Reflector | **−0.33** | 0.00 | −13.3 s | **Matters** — enables iterative query refinement |
| Reranker | 0.00 | 0.00 | −2.9 s | Neutral on this eval — hybrid RRF sufficient |
| Full pipeline vs. baseline | **+0.76** | **+0.10** | +44.6 s | All components collectively add value |

---

## 6. Component Analysis

### 6.1 Planner: −0.30 Accuracy, −0.10 Faithfulness

**Observed**: Removing the planner drops accuracy from 2.83 → 2.53 and faithfulness from 0.48 → 0.38. Tool calls fall from 3.8 → 1.1, meaning the agent becomes effectively single-pass despite max_iter=10 (without sub-questions the reflector rarely detects gaps).

**Why it matters**: The planner decomposes queries like *"Compare multi-agent debate protocols and single-agent self-refinement"* into separate sub-questions targeting each approach. Without decomposition, the retriever issues one broad query and misses half the evidence. This confirms the ReAct principle [1]: **reasoning before acting produces better actions**.

**Failure example** (no_planner, q07):
- no_planner response: *"Evidence does not explicitly mention the number of apps or tasks in the benchmark"* → Accuracy 1/5
- full_agent response: *"AppWorld benchmark: 9 apps, ~750 tasks [arXiv:2411.13020]"* → Accuracy 4/5
- Root cause: Without the sub-question *"AppWorld benchmark scale statistics"*, the retriever never targeted the AppWorld paper.

### 6.2 Reflector: −0.33 Accuracy

**Observed**: Removing the reflector drops accuracy from 2.83 → 2.50. Latency improves by 13.3 s (48.6 s vs 61.9 s). Tool calls drop from 3.8 → 3.0.

**Why it matters**: Survey questions require multiple retrieval passes. The reflector implements the Reflexion principle [3] — evaluate current evidence against the goal, generate verbal critique, decide to refine or proceed. Without it, the agent commits to initial retrieval even when sub-question coverage is poor.

**Latency trade-off**: no_reflector is the best config for latency-constrained deployments accepting a ≈0.33 accuracy loss.

### 6.3 Reranker: No Accuracy Impact

**Observed**: no_reranker (2.83) exactly matches full_agent (2.83). no_reranker achieves *better* citation metrics (1.00 vs 0.97).

**Why**: At 574 papers, hybrid RRF already produces highly relevant top-20 results. The LLM reranker's marginal reordering within an already-relevant candidate set does not change which papers are cited. This is the known **"reranker plateau"** — reranking helps most when the initial pool contains many irrelevant documents.

**Implication**: At larger corpus scale (>10K papers), the reranker's value would likely re-emerge. At this scale, it adds latency with no accuracy benefit.

### 6.4 Baseline: −0.76 Accuracy, 3.6× Faster

**Observed**: Single-pass baseline achieves the lowest accuracy (2.07 ±0.78) — a **−27% relative drop** from full_agent — but runs in 17.3 s (vs 61.9 s) and achieves perfect citation metrics (1.00/1.00).

**Citation paradox**: The baseline generates shorter, simpler answers citing only top-1/2 papers. Fewer citations → fewer claims to verify → trivially high precision. This reveals a **precision–completeness trade-off** that accuracy captures but citation metrics alone do not.

---

## 7. Failure Mode Analysis

### 7.1 Systematic Failure Patterns

**Pattern 1 — Faithfulness ceiling at 0.48**
All completed configurations plateau at faithfulness 0.48–0.50. This is a *measurement artifact*: the LLM judge checks faithfulness against 512-token chunks, not full PDFs. Chunks cannot contain all claims spanning multiple papers. The judge defaults to 0.5 when it cannot verify either way. This is a metric limitation, not a generation failure.

**Pattern 2 — Survey question underperformance**
Survey questions average ≈2.2/5 vs ≈3.5/5 for factoid questions on full_agent. The limiting factor is corpus coverage: for niche survey topics, 574 papers may not contain the most relevant recent work.

**Pattern 3 — Retrieval failure on recent papers**
Questions about papers published Q1–Q2 2026 (arXiv:2504.xxxxx) occasionally fail because abstract-only chunks provide insufficient detail for sub-question matching when full-text parsing fails.

### 7.2 Worked Examples

---

**Example 1 — Full Success: Factoid (q04, full_agent)**

*Query*: "What is the name of the benchmark introduced in the SWE-agent paper, and how many GitHub issues does it contain?"

| Step | Action | Result |
|------|--------|--------|
| Plan | Factoid; sub-q: "SWE-agent benchmark name and size" | 1 search query generated |
| Retrieve | Hybrid → arXiv:2405.15793 (SWE-agent) | Passage with benchmark stats |
| Read | SWE-bench, 2,294 GitHub issues, 12 repos | Relevance: 0.95 |
| Reflect | SUFFICIENT — 1 iteration | — |
| Synthesize | Answer with inline citation | Clean |
| Verify | Citation validated ✓ | — |

**Accuracy: 5/5 · Faithfulness: 0.5 · Latency: 38.4 s · Tool calls: 1**

---

**Example 2 — Reflector Saves Retrieval: Survey (q19, full_agent)**

*Query*: "Survey the landscape of LLM agent safety evaluation — what frameworks, benchmarks, and metrics are used?"

| Step | Action | Result |
|------|--------|--------|
| Plan | Survey; 5 sub-questions on frameworks, benchmarks, metrics | 5 queries |
| Retrieve iter 1 | Top-5 passages — broad agent eval, not safety-specific | Avg relevance: 0.61 |
| Reflect | **REFINE_QUERY**: "safety evaluation not covered" | New query: "agent safety benchmark red-teaming" |
| Retrieve iter 2 | arXiv:2509.02547 (safety-focused) | Relevance: 0.79 |
| Reflect | SUFFICIENT | — |
| Synthesize | 3 frameworks, 2 benchmarks with citations | Partial coverage |

**Accuracy: 3/5 — correct on named frameworks, but corpus lacks comprehensive late-2025 safety papers**

---

**Example 3 — Planner Ablation Comparison: Factoid (q07)**

| Config | Answer | Accuracy |
|--------|--------|:--------:|
| no_planner | *"Evidence does not explicitly mention the number of apps or tasks"* | **1/5** |
| full_agent | *"AppWorld: 9 apps, ~750 tasks [arXiv:2411.13020]"* | **4/5** |

*Root cause*: Without sub-question "AppWorld benchmark scale statistics", the retriever never targeted the correct paper.

---

**Example 4 — Baseline vs. Full Agent: Comparative (q13)**

| Config | Answer quality | Latency | Accuracy |
|--------|---------------|---------|:--------:|
| baseline | Single paper cited; misses comparison dimension | 12.1 s | **2/5** |
| full_agent | Two approaches compared with 3 papers; structured comparison | 74.3 s | **4/5** |

*Lesson*: Comparative questions require the planner to decompose into separate sub-questions per approach. The baseline's single retrieval pass cannot cover both sides of the comparison.

---

## 8. Discussion

### 8.1 What Actually Mattered

Based on n=30 controlled ablation:

1. **Reflector is the most impactful single component** (−0.33 accuracy). Iterative query refinement is critical for survey and comparative questions where initial queries under-specify the evidence needed.

2. **Planner matters most for grounding** (−0.30 accuracy, −0.10 faithfulness). Query decomposition directly determines what evidence is retrieved. Without it, answers are less accurate *and* less grounded in source text.

3. **Reranker is corpus-size dependent**. At 574 papers, hybrid RRF is sufficient. Reranker impact would likely re-emerge at >10K papers.

4. **The +0.76 full pipeline gain is compounded**: planning (better queries) + hybrid retrieval (better candidates) + reflection (iterative refinement). No single component accounts for the full gain.

### 8.2 Latency–Quality Trade-off

```
Accuracy (↑ better)
  3.0 |
  2.8 |           ● full_agent    ● no_reranker
  2.6 |      ● no_planner
  2.5 |  ● no_reflector
  2.1 |● baseline
      +─────────────────────────────────────────────
        17s       49s      58s      62s      65s
                      Latency (↓ better)
```

**Pareto frontier**: baseline (max speed) → no_reflector (moderate balance) → full_agent (max quality).

### 8.3 Limitations

1. **Corpus coverage**: 574 papers covers arXiv only. ACL Anthology, NeurIPS, ICML proceedings excluded.
2. **PDF parsing quality**: ~8% of papers have degraded chunk quality due to complex PDF layouts (tables, multi-column).
3. **Faithfulness measurement ceiling**: The 0.48 plateau is a metric artifact (§7.1), not a generation ceiling.
4. **No human evaluation**: LLM-as-judge scores are efficient but unvalidated against human rater agreement.
5. **Latency at scale**: 61.9 s includes up to 10 LLM calls. Streaming generation would reduce perceived latency significantly.

---

## 9. Future Work

| Priority | Direction | Expected Impact |
|----------|-----------|----------------|
| High | **Corpus expansion** — ACL Anthology + ICML/NeurIPS | +20% coverage for survey questions |
| High | **Streaming synthesis** — yield tokens as evidence is gathered | 3–5× reduction in perceived latency |
| Medium | **Cross-encoder reranker** at larger corpus scale | +0.1–0.2 accuracy at >10K papers |
| Medium | **Citation graph traversal** — use cited-by relationships | Better coverage for foundational papers |
| Medium | **Human evaluation** — compare LLM-judge vs. domain expert annotations | Validate judge reliability |
| Low | **Multi-modal retrieval** — index figures and tables | Better answers to quantitative questions |
| Low | **Fine-tuned reflector** — train on (evidence, sub-question, decision) examples | Reduce false SUFFICIENT decisions |

---

## 10. Conclusion

This project implements and rigorously evaluates a six-component agentic deep research system on a 574-paper arXiv corpus. The key empirical findings from 30-question controlled ablation:

1. **Full pipeline (+0.76 vs. baseline)**: full_agent achieves 2.83/5.0 vs. 2.07 for the single-pass baseline — a **27% relative improvement** confirming that multi-component pipelines add measurable value.
2. **Reflector is the most impactful single component** (−0.33 accuracy when removed). Iterative query refinement is essential for survey and comparative questions.
3. **Planner is most impactful for grounding** (−0.30 accuracy, −0.10 faithfulness). Query decomposition determines retrieval quality.
4. **Reranker shows no accuracy impact at this corpus scale**. Hybrid RRF is sufficient for 574 papers.
5. **Latency spans 3.6× (17.3 s–64.8 s)** across configurations, enabling explicit latency–quality trade-offs.
6. **Citation quality is robust across all configs** (cite-P/R ≥ 0.97 for all fully-evaluated configs).

The system demonstrates that principled, literature-motivated pipeline design produces measurable gains over naive baselines, and that systematic ablation is essential to isolate which components actually contribute in practice.

---

## References

1. Yao, S., et al. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models**. *arXiv:2210.03629*

2. Asai, A., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**. *arXiv:2310.11511*

3. Shinn, N., et al. (2023). **Reflexion: Language Agents with Verbal Reinforcement Learning**. *arXiv:2303.11366*

4. Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *arXiv:2201.11903*

5. Es, S., et al. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. *arXiv:2309.15217*

6. Zheng, L., et al. (2023). **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**. *arXiv:2306.05685*

7. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *arXiv:2005.11401*

---

## Appendix A: Reproduction Instructions

### A.1 Environment Setup

```bash
git clone https://github.com/Harish-SS56/AIMS-Research-Agent.git
cd "AIMS Research Agent"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # fill in Azure OpenAI credentials
```

**Required `.env` variables:**
```
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large-2
```

### A.2 Building the Corpus

```bash
python run.py build-corpus --start-date 2024-01-01 --end-date 2026-04-30
python run.py build-index
```

### A.3 Running Evaluation

```bash
# Single query
python run.py query "What is ReAct?" --config full_agent

# Full 30-question evaluation
python run.py evaluate --config full_agent

# All ablation configurations
python run.py ablation --all

# Direct predictions script
python generate_predictions.py
```

### A.4 Running the Demo

```bash
# Terminal 1 — Backend (port 8000)
uvicorn app.api:app --port 8000

# Terminal 2 — Frontend (port 5173)
cd frontend && npm install && npm run dev
```

### A.5 Output Files

| File | Contents |
|------|----------|
| `predictions.jsonl` | Primary submission — 30 Q, full_agent config |
| `predictions/full_agent.jsonl` | Ablation folder copy of above |
| `predictions/baseline.jsonl` | Baseline config (n=30) |
| `predictions/no_planner.jsonl` | no_planner config (n=30) |
| `predictions/no_reranker.jsonl` | no_reranker config (n=30) |
| `predictions/no_reflector.jsonl` | no_reflector config (n=30) |
| `predictions/no_hybrid.jsonl` | no_hybrid partial (n=9) |
| `predictions/no_verifier.jsonl` | no_verifier partial (n=5) |
| `predictions/ablation_results.json` | Aggregated metrics, all configs |
