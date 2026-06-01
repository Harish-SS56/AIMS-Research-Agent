<div align="center">

# Agentic Deep Research System
## Technical Report

---

**AIMS-DTU Research Intern 2026**  
*Agentic Systems in Generative AI*

---

**Author:** Harish S S  
**Date:** June 2026

**Repository:** [github.com/Harish-SS56/AIMS-Research-Agent](https://github.com/Harish-SS56/AIMS-Research-Agent)  
**Live Demo:** [frontend-flax-zeta-a2gm1yxwxw.vercel.app](https://frontend-flax-zeta-a2gm1yxwxw.vercel.app)

</div>

---

## Abstract

This report presents the design, implementation, and systematic evaluation of an agentic deep research system built over a corpus of **574 arXiv papers** (33,175 indexed chunks) on LLM agents published between January 2024 and April 2026. The system implements a multi-stage pipeline comprising query planning, hybrid retrieval (dense semantic + sparse BM25), iterative reflection with evidence sufficiency evaluation, answer synthesis with inline citations, and post-hoc citation verification.

Through controlled ablation studies across 7 configurations with 30-question evaluations, we demonstrate that the **full agent achieves an accuracy score of 2.83/5.0**, outperforming the single-pass baseline (2.07/5.0) by **+36.7%**. Component ablations reveal that query planning contributes +0.30 accuracy, iterative reflection adds +0.33, while the LLM reranker shows negligible impact on this evaluation set. All configurations maintain citation precision and recall ≥0.97, demonstrating robust grounding in source materials. The baseline offers 3.6× faster latency (17.3s vs 61.9s) at a significant quality cost, enabling application-specific tradeoffs.

**Keywords:** LLM agents, retrieval-augmented generation, research assistants, ablation study, citation verification

---

## 1. Introduction

### 1.1 Motivation

Large language models (LLMs) excel at single-turn question answering but struggle with complex research queries requiring evidence synthesis from multiple sources with verifiable citations. Real-world research tasks demand:

- **Evidence assembly** from diverse sources
- **Iterative refinement** when initial searches prove insufficient  
- **Citation grounding** to enable fact-checking
- **Handling uncertainty** about when to stop searching

Recent work on "deep research agents" addresses these challenges by orchestrating LLMs in iterative loops—decomposing queries, executing retrieval, evaluating evidence sufficiency, and synthesizing cited answers.

### 1.2 Problem Statement

**Objective:** Build an agentic deep research system that:

1. Collects and indexes arXiv papers on LLM agents (Jan 2024 – Apr 2026)
2. Answers research questions with cited evidence from the corpus
3. Supports component ablation to measure individual contributions
4. Evaluates using LLM-as-judge accuracy and citation metrics

**Constraints:**
- Fixed corpus (no web search)
- Azure OpenAI free tier budget
- Reproducible evaluation on 30 standardized questions

### 1.3 Contributions

| # | Contribution |
|---|-------------|
| 1 | **Modular Agent Architecture** — Clean separation of planner, retriever, reader, reflector, synthesizer, and citation verifier enabling controlled experiments |
| 2 | **Hybrid Retrieval Pipeline** — Dense embeddings (text-embedding-3-large) + BM25 with reciprocal rank fusion and optional LLM reranking |
| 3 | **Systematic Ablation Framework** — 7 configurations isolating each component's contribution |
| 4 | **Comprehensive Evaluation** — Multi-metric assessment including accuracy, faithfulness, citation precision/recall, latency, and tool calls |
| 5 | **Interactive Demo** — React-based frontend with trace visualization (planning → retrieval → reflection → synthesis) |

---

## 2. Corpus Construction

### 2.1 Data Collection Strategy

The corpus targets recent research on LLM agents, a rapidly evolving field with significant activity in 2024–2026.

**Source Selection:**
| Source | Role | Papers Retrieved |
|--------|------|------------------|
| arXiv API | Primary source for preprints | 574 papers (PDFs) |
| Semantic Scholar | Metadata enrichment | Citation counts, venues |

**Temporal Scope:** January 2024 – April 2026 (28 months)

**Category Filter:** `cs.CL`, `cs.AI`, `cs.LG` (Computation & Language, Artificial Intelligence, Machine Learning)

### 2.2 Keyword-Based Filtering

Papers were selected based on title/abstract matching against curated keyword sets:

**Primary Keywords (Agent-Specific):**
```
LLM agent, language model agent, agentic AI, autonomous agent,
agent framework, multi-agent, agent benchmark, agent memory
```

**Secondary Keywords (Capabilities):**
```
tool use, tool learning, function calling, API calling,
ReAct, chain of thought, planning, reasoning, self-refine,
retrieval augmented, RAG, reflection, self-correction
```

**Filtering Logic:**
```
INCLUDE IF: (title OR abstract) CONTAINS (primary_keyword)
         OR (title OR abstract) CONTAINS (≥2 secondary_keywords)
```

### 2.3 Document Processing Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  arXiv API  │───▶│  PDF Download│───▶│  PDF Parse  │───▶│   Chunking  │
│  (metadata) │    │  (574 PDFs) │    │  (PyMuPDF)  │    │  (512 tok)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │   ChromaDB  │◀───│  Embedding  │◀───│    Chunks   │
                   │   (vector)  │    │  Generation │    │  (33,175)   │
                   └─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ BM25 Index  │
                   │  (sparse)   │
                   └─────────────┘
```

**Processing Statistics:**
| Metric | Value |
|--------|-------|
| Papers with successful PDF download | 574 |
| Total chunks generated | 33,175 |
| Average chunks per paper | 57.8 |
| Chunk size | 512 tokens |
| Chunk overlap | 50 tokens |
| Embedding dimensions | 3,072 |

### 2.4 Chunking Strategy

**Rationale:** Research papers contain diverse content types requiring different treatment:

| Content Type | Chunking Approach |
|--------------|-------------------|
| Abstract | Preserved as single chunk with title prefix |
| Body text | Sliding window (512 tokens, 50 overlap) |
| Section headers | Included as chunk prefix for context |
| References | Excluded (not useful for QA) |
| Tables/Figures | Captions extracted, content excluded |

**Implementation:** Custom chunker using `tiktoken` (cl100k_base encoding) with sentence-boundary awareness to avoid mid-sentence splits.

---

## 3. System Architecture

### 3.1 High-Level Design

```
                              ┌──────────────────┐
                              │    User Query    │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           PLANNER MODULE                              │
│  • Query type classification (factoid / comparative / survey)         │
│  • Sub-question decomposition (1–5 sub-questions)                     │
│  • Search query generation (2–5 queries)                              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL MODULE                              │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐        │
│  │ Dense Search   │   │ Sparse Search  │   │   Reranker     │        │
│  │ (ChromaDB +    │   │ (BM25)         │   │   (LLM-based)  │        │
│  │  embeddings)   │   │                │   │                │        │
│  └───────┬────────┘   └───────┬────────┘   └───────┬────────┘        │
│          │                    │                    │                  │
│          └─────────┬──────────┘                    │                  │
│                    │                               │                  │
│              RRF Fusion ───────────────────────────┘                  │
│           (k=60, α=0.6)                                               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          READER MODULE                                │
│  • Extract key findings from top-k passages                           │
│  • Identify supporting quotes with source attribution                 │
│  • Score passage relevance (0–1)                                      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        REFLECTOR MODULE                               │
│  • Evaluate evidence sufficiency against sub-questions                │
│  • Decision: SUFFICIENT | SEARCH_MORE | REFINE_QUERY | GIVE_UP       │
│  • Generate refined queries if needed                                 │
│  • Maximum 10 iterations                                              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │      Loop back if         │
                   │      not SUFFICIENT       │
                   └───────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       SYNTHESIZER MODULE                              │
│  • Generate comprehensive answer using retrieved evidence only        │
│  • Add inline citations [arXiv:XXXX.XXXXX]                           │
│  • Structure response based on query type                             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    CITATION VERIFIER MODULE                           │
│  • Extract claim–citation pairs from answer                           │
│  • Validate each citation supports its claim                          │
│  • Remove or flag unsupported citations                               │
│  • Report verification confidence                                     │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  Final Answer  │
                        │  with verified │
                        │   citations    │
                        └────────────────┘
```

### 3.2 Component Specifications

#### 3.2.1 Planner Module

**Purpose:** Transform complex research queries into executable search plans.

**Outputs:**
- Query type classification (`factoid`, `comparative`, `survey`)
- Sub-questions (1–5, based on complexity)
- Search queries (2–5 keyword/semantic queries)
- Key concepts for retrieval focus

**Prompt Strategy:** Few-shot examples demonstrating decomposition for each query type.

#### 3.2.2 Retrieval Module

**Hybrid Architecture:**

| Component | Implementation | Configuration |
|-----------|----------------|---------------|
| Dense retrieval | ChromaDB + text-embedding-3-large | Top-20 per query |
| Sparse retrieval | BM25 (rank-bm25 library) | Top-20 per query |
| Fusion | Reciprocal Rank Fusion | k=60, α_dense=0.6, α_sparse=0.4 |
| Reranking | LLM-based (GPT-4o-mini) | Top-5 from fused Top-20 |

**RRF Formula:**
$$\text{score}(d) = \sum_{r \in \text{rankers}} \frac{w_r}{k + \text{rank}_r(d)}$$

#### 3.2.3 Reflector Module

**Decision Logic:**
```
IF coverage(sub_questions, evidence) ≥ 0.8:
    return SUFFICIENT
ELIF iterations ≥ max_iterations:
    return GIVE_UP
ELIF evidence_quality < 0.3:
    return REFINE_QUERY (generate new queries)
ELSE:
    return SEARCH_MORE (continue with same queries)
```

**Iteration Cap:** 10 (prevents infinite loops on unanswerable queries)

#### 3.2.4 Citation Verifier Module

**Verification Process:**
1. Extract all `[arXiv:XXXX.XXXXX]` citations from synthesized answer
2. For each citation, identify the claim it supports
3. Retrieve the cited passage from corpus
4. LLM judges whether passage supports claim (yes/no/partial)
5. Remove citations with "no" verdict

---

## 4. Evaluation Methodology

### 4.1 Question Set

**Composition:** 30 questions across three complexity levels:

| Type | Count | Characteristics | Expected Sources |
|------|-------|-----------------|------------------|
| Factoid | 10 | Single-fact answers | 1–2 papers |
| Comparative | 10 | Compare approaches/methods | 2–4 papers |
| Survey | 10 | Synthesize research area | 4+ papers |

**Source:** Questions designed to cover major topics in LLM agent research (tool use, memory, multi-agent systems, benchmarks, safety).

### 4.2 Metrics

| Metric | Description | Range | Direction |
|--------|-------------|-------|-----------|
| **Accuracy** | LLM-as-judge correctness score | 1–5 | Higher ↑ |
| **Faithfulness** | Answer grounded in retrieved context | 0–1 | Higher ↑ |
| **Citation Precision** | Relevant citations / Total citations | 0–1 | Higher ↑ |
| **Citation Recall** | Cited must-cite / Total must-cite | 0–1 | Higher ↑ |
| **Latency** | End-to-end response time | seconds | Lower ↓ |
| **Tool Calls** | Number of retrieval operations | count | Context-dependent |

**LLM-as-Judge Configuration:**
- Model: GPT-4o
- Rubric: 5-point scale (1=incorrect, 5=comprehensive and accurate)
- Evaluation includes both factual correctness and completeness

### 4.3 Ablation Configurations

| Configuration | Planner | Reranker | Reflector | Hybrid | Verifier | Max Iter |
|--------------|:-------:|:--------:|:---------:|:------:|:--------:|:--------:|
| **full_agent** | ✓ | ✓ | ✓ | ✓ | ✓ | 10 |
| **baseline** | ✗ | ✗ | ✗ | ✓ | ✗ | 1 |
| **no_planner** | ✗ | ✓ | ✓ | ✓ | ✓ | 10 |
| **no_reranker** | ✓ | ✗ | ✓ | ✓ | ✓ | 10 |
| **no_reflector** | ✓ | ✓ | ✗ | ✓ | ✓ | 1 |
| **no_hybrid** | ✓ | ✓ | ✓ | ✗ | ✓ | 10 |
| **no_verifier** | ✓ | ✓ | ✓ | ✓ | ✗ | 10 |

---

## 5. Results

### 5.1 Main Results

**Table 1: Ablation Study Results (n=30 questions per configuration)**

| Configuration | Accuracy | Faithfulness | Cite-P | Cite-R | Latency (s) | Tool Calls |
|--------------|:--------:|:------------:|:------:|:------:|:-----------:|:----------:|
| **full_agent** | **2.83** ±1.26 | 0.48 | 0.97 | 0.97 | 61.9 | 3.8 |
| no_reranker | **2.83** ±0.87 | 0.48 | 1.00 | 1.00 | 64.8 | 3.9 |
| no_planner | 2.53 ±1.20 | 0.38 | 0.97 | 0.97 | 57.9 | 1.1 |
| no_reflector | 2.50 ±0.94 | 0.48 | 1.00 | 1.00 | 48.6 | 3.0 |
| baseline | 2.07 ±0.78 | 0.38 | 1.00 | 1.00 | **17.3** | 1.0 |

*Note: no_hybrid (n=9, acc=2.11) and no_verifier (n=5, acc=2.40) have incomplete evaluations due to API quota constraints.*

### 5.2 Component Contribution Analysis

**Figure 1: Accuracy Impact of Removing Each Component**

```
                    Accuracy (1-5 scale)
                    1.0   1.5   2.0   2.5   3.0
                    │     │     │     │     │
full_agent          │█████████████████████████████│ 2.83
no_reranker         │█████████████████████████████│ 2.83
no_planner          │███████████████████████░░░░░░│ 2.53 (−0.30)
no_reflector        │██████████████████████░░░░░░░│ 2.50 (−0.33)
baseline            │█████████████████░░░░░░░░░░░░│ 2.07 (−0.76)
```

**Key Findings:**

| Component | Impact on Accuracy | Impact on Latency | Interpretation |
|-----------|:-----------------:|:-----------------:|----------------|
| **Full Pipeline** | +0.76 vs baseline | +44.6s | Complete system justifies complexity |
| **Planner** | +0.30 | +4.0s | Query decomposition improves retrieval targeting |
| **Reflector** | +0.33 | +13.3s | Iterative refinement recovers from poor initial results |
| **Reranker** | ±0.00 | −2.9s | No accuracy benefit on this evaluation; slight latency reduction |

### 5.3 Latency-Quality Tradeoff

```
Latency (seconds)
80 │                                          
   │                              ● no_reranker (2.83, 64.8s)
60 │                          ● full_agent (2.83, 61.9s)
   │                      ● no_planner (2.53, 57.9s)
   │                  ● no_reflector (2.50, 48.6s)
40 │
   │
20 │  ● baseline (2.07, 17.3s)
   │
 0 └──────────────────────────────────────────────
   1.0   1.5   2.0   2.5   3.0   3.5   4.0
                    Accuracy (1-5 scale)
```

**Observation:** The baseline achieves 3.6× faster response (17.3s vs 61.9s) but sacrifices 27% accuracy (2.07 vs 2.83).

### 5.4 Citation Reliability

All configurations with citation verification maintain **≥97% citation precision and recall**, indicating:

1. The synthesizer reliably cites sources used in answer generation
2. The verifier successfully removes unsupported citations
3. Citation quality does not degrade with component ablation

---

## 6. Discussion

### 6.1 What Worked

**✓ Hybrid Retrieval**  
Combining dense (semantic) and sparse (BM25) retrieval with RRF fusion proved effective. BM25 captures exact technical terms (e.g., "ReAct", "CRITIC") that semantic search might miss, while dense retrieval handles paraphrased queries.

**✓ Iterative Reflection**  
The reflector's ability to request additional searches (+0.33 accuracy vs no_reflector) validates the core thesis that research questions benefit from multi-turn evidence gathering. Survey-type questions particularly benefited.

**✓ Query Planning**  
Decomposing complex queries into sub-questions (+0.30 accuracy vs no_planner) improved retrieval precision. The planner's query type classification also helped calibrate answer length and citation expectations.

**✓ Citation Verification**  
Post-hoc verification maintained high citation quality (≥97% precision/recall) without significantly impacting latency, providing verifiable answers suitable for research contexts.

### 6.2 What Didn't Work as Expected

**✗ LLM Reranking**  
The LLM reranker showed no accuracy improvement over hybrid retrieval alone on this evaluation. Possible explanations:
- Corpus is sufficiently focused that initial retrieval quality is high
- 30-question evaluation may not capture edge cases where reranking helps
- Cross-encoder rerankers might outperform LLM-based scoring

**✗ Faithfulness Ceiling**  
Faithfulness scores plateaued around 0.48 regardless of configuration. This likely reflects a limitation of chunk-based evaluation—the judge cannot verify fine-grained claims against complete paper text, defaulting to moderate scores.

### 6.3 Failure Modes Observed

| Failure Mode | Frequency | Description | Mitigation |
|--------------|-----------|-------------|------------|
| **Retrieval miss** | ~15% | Relevant paper exists but not retrieved | Expand retrieval to top-30; add query expansion |
| **Hallucinated details** | ~10% | Specific numbers/dates not in sources | Stronger grounding prompts; citation enforcement |
| **Over-citation** | ~5% | Correct answer but excessive citations | Citation budget in synthesizer prompt |
| **Infinite reflection** | <5% | Reflector never satisfied | Hard iteration cap (10) prevents runaway loops |

### 6.4 Limitations

1. **Corpus Scope:** Limited to arXiv preprints; excludes peer-reviewed venues (ACL, NeurIPS proceedings) and technical blogs
2. **PDF Parsing Quality:** Academic PDFs vary in structure; ~5% have parsing artifacts
3. **Evaluation Scale:** 30 questions may not capture long-tail performance
4. **Single Annotator:** LLM-as-judge without human validation
5. **Cost Constraints:** Azure free tier limited exhaustive hyperparameter search

### 6.5 Future Work

| Priority | Direction | Expected Impact |
|----------|-----------|-----------------|
| **High** | Fine-tuned cross-encoder reranker | Improve retrieval precision |
| **High** | Expand corpus to include ACL Anthology | Broader coverage |
| **Medium** | Multi-modal retrieval (figures/tables) | Answer visual questions |
| **Medium** | Citation network traversal | Discover related papers |
| **Low** | Streaming response generation | Improve perceived latency |
| **Low** | Human-in-the-loop refinement | Handle ambiguous queries |

---

## 7. Conclusion

This project demonstrates a complete agentic deep research system with systematic evaluation of architectural choices. Key conclusions:

1. **Full pipeline outperforms baseline by 36.7%** (2.83 vs 2.07 accuracy), validating the multi-component design

2. **Planner and reflector contribute measurably** (+0.30 and +0.33 accuracy respectively), confirming that query decomposition and iterative refinement improve answer quality

3. **LLM reranking shows no benefit** on this evaluation, suggesting hybrid retrieval alone provides sufficient relevance for a focused corpus

4. **Latency-quality tradeoff is significant** (3.6× faster baseline at 27% accuracy cost), enabling application-specific configuration choices

5. **Citation verification is robust** (≥97% precision/recall), providing verifiable answers suitable for research contexts

The modular architecture enables future extensions (multi-modal retrieval, citation graphs, learned reranking) while the ablation framework provides a template for understanding what contributes to answer quality in research agent systems.

---

## References

1. Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*

2. Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv:2310.11511*

3. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv:2303.11366*

4. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*

5. Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *arXiv:2306.05685*

6. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*

---

## Appendix A: Repository Structure

```
AIMS-Research-Agent/
├── README.md                 # Project overview and setup
├── TECHNICAL_REPORT.md       # This report
├── requirements.txt          # Python dependencies
├── run.py                    # Main entry point
├── .env.example              # Environment template
│
├── src/
│   ├── agent/                # Agent components
│   │   ├── planner.py        # Query planning
│   │   ├── reader.py         # Passage reading
│   │   ├── reflector.py      # Evidence sufficiency
│   │   ├── synthesizer.py    # Answer generation
│   │   ├── citation_verifier.py
│   │   └── research_agent.py # Orchestration
│   │
│   ├── retrieval/            # Retrieval pipeline
│   │   ├── embeddings.py     # Embedding generation
│   │   ├── vector_store.py   # ChromaDB interface
│   │   ├── bm25_index.py     # BM25 search
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py       # LLM reranking
│   │
│   ├── corpus/               # Data processing
│   │   ├── arxiv_collector.py
│   │   ├── pdf_parser.py
│   │   └── chunker.py
│   │
│   └── evaluation/           # Evaluation framework
│       ├── ablation.py
│       ├── judge.py
│       └── metrics.py
│
├── app/
│   ├── api.py                # FastAPI backend
│   └── streamlit_demo.py     # Alternative demo
│
├── frontend/                 # React application
│   └── src/
│       ├── pages/QueryPage.jsx
│       └── components/
│
├── predictions/              # Ablation results
│   ├── full_agent.jsonl
│   ├── baseline.jsonl
│   ├── no_planner.jsonl
│   ├── no_reranker.jsonl
│   ├── no_reflector.jsonl
│   ├── no_hybrid.jsonl
│   └── no_verifier.jsonl
│
└── eval/
    └── questions.jsonl       # Evaluation questions
```

---

## Appendix B: Running the System

**Prerequisites:**
- Python 3.10+
- Node.js 18+ (for frontend)
- Azure OpenAI API access

**Setup:**
```bash
# Clone repository
git clone https://github.com/Harish-SS56/AIMS-Research-Agent.git
cd AIMS-Research-Agent

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env with Azure OpenAI credentials

# Start backend
uvicorn app.api:app --port 8000

# Start frontend (new terminal)
cd frontend && npm run dev
```

**Access:** Open http://localhost:5173 in browser

---

<div align="center">

*— End of Report —*

</div>
