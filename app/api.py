"""
FastAPI backend for the AIMS Research Agent frontend.
Run with: uvicorn app.api:app --reload --port 8000
"""
import json
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import config as app_config
from src.agent import ResearchAgent
from src.evaluation import ABLATION_CONFIGS

app = FastAPI(title="AIMS Research Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:5174", "http://localhost:5175",
        "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
DATA_DIR = Path(__file__).parent.parent / "data"


# ── Models ──────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    config_name: str = "full_agent"


class ResearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    confidence: float
    iterations: int
    tool_calls: int
    latency_seconds: float
    config_name: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AIMS Research Agent"}


@app.get("/api/configs")
def get_configs():
    """Return all ablation configurations."""
    return {
        name: {
            "name": name,
            "use_planner": cfg.get("use_planner", True),
            "use_reranker": cfg.get("use_reranker", True),
            "use_reflector": cfg.get("use_reflector", True),
            "use_hybrid": cfg.get("use_hybrid", True),
            "use_verifier": cfg.get("use_verifier", True),
            "max_iterations": cfg.get("max_iterations", 10),
        }
        for name, cfg in ABLATION_CONFIGS.items()
    }


@app.post("/api/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    """Run a research query with the specified agent configuration."""
    if req.config_name not in ABLATION_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown config: {req.config_name}")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.time()
    try:
        agent = ResearchAgent.create_from_ablation(req.config_name)
        result = agent.research(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(
        query=req.query,
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        latency_seconds=result.latency_seconds,
        config_name=req.config_name,
    )


@app.get("/api/ablation")
def get_ablation_results():
    """Return aggregated ablation study results."""
    results_file = PREDICTIONS_DIR / "ablation_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Ablation results not found. Run the ablation study first.")

    raw = json.loads(results_file.read_text(encoding='utf-8'))

    # Build per-question data for each config
    configs_detail = {}
    config_order = ["full_agent", "baseline", "no_planner", "no_reranker", "no_reflector", "no_hybrid", "no_verifier"]
    for cfg_name in config_order:
        pred_file = PREDICTIONS_DIR / f"{cfg_name}.jsonl"
        if pred_file.exists():
            questions = []
            for line in pred_file.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    d = json.loads(line)
                    m = d.get("metrics", {})
                    questions.append({
                        "id": d.get("id"),
                        "query": d.get("query", ""),
                        "accuracy": m.get("answer_accuracy"),
                        "faithfulness": m.get("faithfulness"),
                        "latency": round(m.get("latency_seconds", 0), 1),
                        "tool_calls": m.get("tool_calls"),
                    })
            configs_detail[cfg_name] = questions

    # Build summary table
    summary = []
    for cfg_name in config_order:
        pred_file = PREDICTIONS_DIR / f"{cfg_name}.jsonl"
        if pred_file.exists():
            lines = [json.loads(l) for l in pred_file.read_text(encoding='utf-8').splitlines() if l.strip()]
            metrics_list = [d["metrics"] for d in lines if d.get("metrics")]
            if metrics_list:
                n = len(metrics_list)
                summary.append({
                    "config": cfg_name,
                    "accuracy": round(sum(m["answer_accuracy"] for m in metrics_list) / n, 2),
                    "faithfulness": round(sum(m["faithfulness"] for m in metrics_list) / n, 2),
                    "citation_precision": round(sum(m["citation_precision"] for m in metrics_list) / n, 2),
                    "citation_recall": round(sum(m["citation_recall"] for m in metrics_list) / n, 2),
                    "latency": round(sum(m["latency_seconds"] for m in metrics_list) / n, 1),
                    "tool_calls": round(sum(m["tool_calls"] for m in metrics_list) / n, 1),
                    "num_questions": n,
                })

    return {"summary": summary, "per_question": configs_detail}


@app.get("/api/papers")
def get_papers():
    """Return only papers that are actually indexed in the retrieval system."""
    import re as _re
    chunks_file = DATA_DIR / "processed" / "chunks.json"
    indexed_ids: set = set()
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text(encoding='utf-8'))
        indexed_ids = set(
            c["arxiv_id"] for c in chunks
            if _re.match(r"^\d{4}\.\d{4,5}", str(c.get("arxiv_id") or ""))
        )
    papers_file = DATA_DIR / "processed" / "papers_metadata.json"
    if not papers_file.exists():
        raise HTTPException(status_code=404, detail="Papers metadata not found.")
    all_papers = json.loads(papers_file.read_text(encoding='utf-8'))
    # Filter to only papers present in the retrieval index
    papers = [p for p in all_papers if str(p.get("arxiv_id") or "") in indexed_ids]
    # Return lightweight list
    return [
        {
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "authors": p.get("authors", [])[:3],
            "published": p.get("published", "")[:10],
            "categories": p.get("categories", []),
            "abstract": p.get("abstract", "")[:300] + "...",
        }
        for p in papers
    ]


@app.get("/api/stats")
def get_stats():
    """Return corpus and index statistics."""
    import re as _re
    chunks_file = DATA_DIR / "processed" / "chunks.json"
    num_chunks = 0
    num_papers = 0
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text(encoding='utf-8'))
        num_chunks = len(chunks)
        # Count only papers actually indexed in the retrieval system (arXiv IDs in chunks)
        num_papers = len(set(
            c["arxiv_id"] for c in chunks
            if _re.match(r"^\d{4}\.\d{4,5}", str(c.get("arxiv_id") or ""))
        ))
    ablation_done = (PREDICTIONS_DIR / "ablation_results.json").exists()
    configs_done = sum(
        1 for c in ["full_agent", "baseline", "no_planner", "no_reranker", "no_reflector", "no_hybrid", "no_verifier"]
        if (PREDICTIONS_DIR / f"{c}.jsonl").exists()
    )
    return {
        "num_papers": num_papers,
        "num_chunks": num_chunks,
        "ablation_complete": ablation_done,
        "configs_evaluated": configs_done,
        "total_configs": 7,
    }
