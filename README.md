# Agentic Deep Research System

A research project for AIMS-DTU Research Intern 2026 that builds an agentic deep-research system over a corpus of recent LLM-agent papers from arXiv.

## Overview

This system implements an agentic research assistant that:
- **Plans**: Decomposes complex research questions into sub-questions
- **Retrieves**: Uses hybrid retrieval (semantic + lexical) over a curated arXiv corpus
- **Reflects**: Decides whether gathered evidence is sufficient or needs more searching
- **Synthesizes**: Writes comprehensive answers with inline citations
- **Verifies**: Validates that citations actually support their claims

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLANNER MODULE                              │
│  - Decomposes query into sub-questions                          │
│  - Identifies key concepts and search terms                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL MODULE                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Semantic Search │  │  BM25 Search    │  │   Reranker      │  │
│  │ (ChromaDB +     │  │  (Lexical)      │  │ (Cross-Encoder) │  │
│  │  Embeddings)    │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     READER MODULE                                │
│  - Extracts relevant information from retrieved passages        │
│  - Summarizes key findings per sub-question                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REFLECTOR MODULE                              │
│  - Evaluates if evidence is sufficient                          │
│  - Decides: answer now OR search more OR refine query           │
│  - Caps iterations to prevent infinite loops                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │ Sufficient?         │
                   │ NO ──► Back to      │
                   │        Planner      │
                   │ YES ──► Continue    │
                   └─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SYNTHESIZER MODULE                             │
│  - Generates answer using only retrieved evidence               │
│  - Adds inline citations [arXiv:XXXX.XXXXX]                    │
│  - Structures response appropriately                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                CITATION VERIFIER MODULE                          │
│  - Validates each citation supports its claim                   │
│  - Removes unsupported citations                                │
│  - Flags potential hallucinations                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Final Answer                               │
│  - Comprehensive response with verified citations               │
│  - Execution trace available for inspection                     │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
AIMS Research Agent/
├── src/
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── arxiv_collector.py      # arXiv API interface & paper collection
│   │   ├── pdf_parser.py           # PDF text extraction
│   │   └── chunker.py              # Text chunking strategies
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── vector_store.py         # ChromaDB interface
│   │   ├── bm25_index.py           # Lexical search
│   │   ├── hybrid_retriever.py     # Combined retrieval
│   │   └── reranker.py             # Cross-encoder reranking
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── planner.py              # Query decomposition
│   │   ├── retriever.py            # Agent retrieval actions
│   │   ├── reader.py               # Passage reading & extraction
│   │   ├── reflector.py            # Evidence sufficiency evaluation
│   │   ├── synthesizer.py          # Answer generation with citations
│   │   ├── citation_verifier.py    # Citation validation
│   │   └── research_agent.py       # Main agent orchestration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── judge.py                # LLM-as-judge implementation
│   │   └── ablation.py             # Ablation study runner
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── llm.py                  # Azure OpenAI interface
│       └── logging_utils.py        # Logging utilities
├── eval/
│   ├── questions.jsonl             # Evaluation questions
│   └── SUBMISSION_FORMAT.md        # Submission format specification
├── predictions/
│   ├── full_agent.jsonl
│   ├── baseline.jsonl
│   ├── no_planner.jsonl
│   ├── no_reranker.jsonl
│   ├── no_reflector.jsonl
│   ├── no_hybrid.jsonl
│   └── no_verifier.jsonl
├── data/
│   ├── papers/                     # Downloaded PDFs
│   ├── processed/                  # Processed paper data
│   └── index/                      # Vector store files
├── app/
│   └── streamlit_demo.py           # Web demo
├── notebooks/
│   └── analysis.ipynb              # Analysis notebook
├── tests/
│   └── test_agent.py               # Unit tests
├── requirements.txt
├── .env
├── README.md
├── run.py                          # Main entry point
└── REPORT.md                       # Technical report
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd "AIMS Research Agent"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large-2
```

### 3. Build the Corpus

```bash
python run.py build-corpus --start-date 2024-01-01 --end-date 2026-04-30 --max-papers 700
```

### 4. Build the Index

```bash
python run.py build-index
```

### 5. Run the Agent

```bash
# Interactive mode
python run.py query "What are the main approaches to tool use in LLM agents?"

# Batch evaluation
python run.py evaluate --config full_agent
```

### 6. Run Ablation Studies

```bash
python run.py ablation --all
```

### 7. Launch Demo (Optional)

```bash
streamlit run app/streamlit_demo.py
```

## Evaluation

The system is evaluated on 30 questions across three types:
- **Factoid** (10 questions): Single-fact answers from 1 paper
- **Comparative** (10 questions): Compare 2+ papers
- **Survey** (10 questions): Synthesize 4+ papers

### Metrics

| Metric | Description |
|--------|-------------|
| Answer Accuracy | LLM-as-judge score (1-5) for correctness |
| Faithfulness | Whether answer is grounded in retrieved context |
| Citation Precision | Fraction of cited papers that are relevant |
| Citation Recall | Fraction of must-cite papers that are cited |
| Latency | End-to-end response time |
| Tool Calls | Number of retrieval operations |

## Ablation Configurations

| Configuration | Description |
|--------------|-------------|
| `full_agent` | Complete system with all components |
| `baseline` | Single-shot retrieval + one LLM call |
| `no_planner` | Skip query decomposition |
| `no_reranker` | Remove cross-encoder reranking |
| `no_reflector` | No iterative reflection loop |
| `no_hybrid` | Semantic-only retrieval (no BM25) |
| `no_verifier` | Skip citation verification |

## Live Demo

🌐 **Frontend Demo**: [https://frontend-flax-zeta-a2gm1yxwxw.vercel.app](https://frontend-flax-zeta-a2gm1yxwxw.vercel.app)

> ⚠️ **Note**: The live demo shows the frontend UI only. The backend API requires Azure OpenAI credentials and runs on a private network. To use the full research agent functionality:
> 1. Clone the repository
> 2. Set up your own Azure OpenAI credentials in `.env`
> 3. Run the backend locally: `uvicorn app.api:app --port 8000`
> 4. The frontend will connect to your local backend

### Running the Full Demo Locally

```bash
# Terminal 1: Start backend
cd "AIMS Research Agent"
venv\Scripts\activate  # Windows
uvicorn app.api:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Key Design Decisions

1. **Embedding Model**: Azure OpenAI `text-embedding-3-large` for high-quality semantic search
2. **Hybrid Retrieval**: Combine BM25 (lexical) with dense embeddings to capture both exact matches and semantic similarity
3. **Chunking Strategy**: 512-token chunks with 50-token overlap to balance context and precision
4. **Reflection Cap**: Maximum 10 iterations to prevent runaway loops
5. **Citation Format**: Inline `[arXiv:XXXX.XXXXX]` for clear attribution

## License

MIT License - See LICENSE file for details.

## Author

Harish S S - AIMS-DTU Research Intern 2026 Candidate
