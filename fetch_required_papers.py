#!/usr/bin/env python3
"""
Fetch specific papers required to answer the official AIMS questions.
These papers are referenced by name in questions.jsonl.
"""
import arxiv
import json
import time
from pathlib import Path

# Papers explicitly mentioned in the questions
REQUIRED_PAPERS = [
    # Factoid questions
    ("Mem0", "Mem0 memory LLM agents"),
    ("tau-bench", "tau-bench τ-bench reliability benchmark agents"),
    ("OSWorld", "OSWorld benchmark computer use tasks"),
    ("SWE-agent", "SWE-agent ACI agent computer interface"),
    ("Agent Interoperability", "Agent Interoperability Protocols MCP A2A ACP ANP"),
    ("Agentic RAG", "Agentic RAG survey retrieval augmented generation"),
    ("AppWorld", "AppWorld benchmark apps tasks agent"),
    ("UI-TARS", "UI-TARS GUI agent computer use"),
    ("OpenHands", "OpenHands event-driven agent platform"),
    ("OS-MAP", "OS-MAP taxonomy computer using agent"),
    
    # Comparative questions
    ("A-MEM", "A-MEM agentic memory LLM"),
    ("UI-TARS-2", "UI-TARS-2 GUI agent"),
    ("Multi-Turn Multi-Agent", "Multi-Turn Multi-Agent Orchestration"),
    ("Evolving Orchestration", "Multi-Agent Collaboration Evolving Orchestration"),
    ("LLM Agents Debate", "Can LLM Agents Really Debate"),
    ("Multi-Agent Collaboration Mechanisms", "Multi-Agent Collaboration Mechanisms survey"),
    ("Deep Research Agents", "Deep Research Agents survey autonomous"),
    ("Deep Research Survey", "Deep Research Survey Autonomous Research Agents"),
    ("Open Reproducible Deep Research", "Open Reproducible Deep Research"),
    ("Web Search Agentic Deep Research", "Web Search Agentic Deep Research"),
    ("SWE-EVO", "SWE-EVO code agent evolution"),
    
    # Additional papers for survey questions (memory, reflection, tool use)
    ("MemGPT", "MemGPT memory management LLM"),
    ("Cognitive Memory", "Cognitive memory architecture agent"),
    ("Long-term Memory LLM", "Long-term memory LLM agents"),
    ("Self-Refine", "Self-Refine LLM iterative refinement"),
    ("CRITIC", "CRITIC self-correction LLM"),
    ("Toolformer", "Toolformer language models tools"),
    ("ToolLLM", "ToolLLM tool learning large language models"),
    ("API-Bank", "API-Bank benchmark tool use"),
    ("WebArena", "WebArena benchmark web agents"),
    ("Mind2Web", "Mind2Web web agent benchmark"),
    ("VisualWebArena", "VisualWebArena multimodal web agent"),
    ("WorkArena", "WorkArena benchmark enterprise"),
    ("AgentBench", "AgentBench evaluating LLMs agents"),
    ("GAIA", "GAIA benchmark general AI assistants"),
    
    # Deep research / RAG papers
    ("STORM", "STORM writing articles knowledge curation"),
    ("Co-STORM", "Co-STORM collaborative knowledge curation"),
    ("RARE", "RARE retrieval augmented reasoning"),
    ("Adaptive RAG", "Adaptive RAG retrieval augmented generation"),
    ("Corrective RAG", "Corrective RAG CRAG"),
    ("Self-RAG", "Self-RAG learning retrieve generate critique"),
    ("RAG Survey 2024", "RAG retrieval augmented generation survey 2024"),
    
    # Code agents
    ("CodeAct", "CodeAct code action agent"),
    ("Aider", "Aider AI pair programming"),
    ("Devin", "Devin AI software engineer"),
    ("AutoCodeRover", "AutoCodeRover automated program repair"),
    
    # GUI agents
    ("CogAgent", "CogAgent visual language model GUI"),
    ("SeeClick", "SeeClick GUI grounding"),
    ("ScreenAgent", "ScreenAgent computer control"),
    ("OmniParser", "OmniParser screen parsing"),
]

# Known arXiv IDs for some papers (to ensure we get the right ones)
KNOWN_ARXIV_IDS = {
    "2406.12528": "Mem0",  # Mem0
    "2406.12045": "tau-bench",  # τ-bench
    "2404.07972": "OSWorld",  # OSWorld
    "2405.15793": "SWE-agent",  # SWE-agent
    "2411.13020": "AppWorld",  # AppWorld
    "2501.02395": "UI-TARS",  # UI-TARS
    "2407.16741": "OpenHands",  # OpenHands (OpenDevin)
    "2504.00906": "OS-MAP",  # OS-MAP (if it exists)
    "2409.07985": "A-MEM",  # A-MEM
    "2503.05900": "UI-TARS-2",  # UI-TARS-2
    "2406.07155": "SWE-EVO",  # SWE-EVO (if it exists)
    "2310.06117": "AgentBench",
    "2401.13178": "GAIA",
    "2307.13854": "WebArena",
    "2306.06070": "Mind2Web",
    "2401.13649": "VisualWebArena",
    "2403.07323": "WorkArena",
    "2310.08560": "MemGPT",
    "2401.10020": "STORM",
    "2408.15232": "Co-STORM",
    "2310.11511": "Self-RAG",
    "2312.10997": "CRAG",  # Corrective RAG
    "2403.14403": "HippoRAG",
    "2402.14848": "CRAG-Benchmark",
    "2405.04434": "OpenDevin",  # OpenHands original
    "2401.14196": "CodeAct",
    "2312.00849": "CogAgent",
    "2401.10935": "SeeClick",
}


def search_arxiv(query: str, max_results: int = 5) -> list:
    """Search arXiv for papers matching query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in client.results(search):
        results.append({
            "arxiv_id": r.entry_id.split("/")[-1].split("v")[0],
            "title": r.title,
            "abstract": r.summary,
            "authors": [a.name for a in r.authors],
            "published": r.published.strftime("%Y-%m-%d"),
            "updated": r.updated.strftime("%Y-%m-%d") if r.updated else None,
            "categories": r.categories,
            "pdf_url": r.pdf_url,
        })
    return results


def fetch_by_id(arxiv_id: str) -> dict | None:
    """Fetch a specific paper by arXiv ID."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    for r in client.results(search):
        return {
            "arxiv_id": r.entry_id.split("/")[-1].split("v")[0],
            "title": r.title,
            "abstract": r.summary,
            "authors": [a.name for a in r.authors],
            "published": r.published.strftime("%Y-%m-%d"),
            "updated": r.updated.strftime("%Y-%m-%d") if r.updated else None,
            "categories": r.categories,
            "pdf_url": r.pdf_url,
        }
    return None


def main():
    data_dir = Path("data/processed")
    papers_file = data_dir / "papers_metadata.json"
    
    # Load existing papers
    existing = []
    existing_ids = set()
    if papers_file.exists():
        existing = json.loads(papers_file.read_text())
        existing_ids = {p["arxiv_id"] for p in existing}
        print(f"Loaded {len(existing)} existing papers")
    
    new_papers = []
    
    # First, fetch all known arXiv IDs
    print("\n=== Fetching papers by known arXiv ID ===")
    for arxiv_id, name in KNOWN_ARXIV_IDS.items():
        if arxiv_id in existing_ids:
            print(f"  [skip] {name} ({arxiv_id}) - already in corpus")
            continue
        print(f"  [fetch] {name} ({arxiv_id})...", end=" ")
        try:
            paper = fetch_by_id(arxiv_id)
            if paper:
                new_papers.append(paper)
                existing_ids.add(arxiv_id)
                print(f"OK - {paper['title'][:50]}...")
            else:
                print("NOT FOUND")
            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Then search for papers by query
    print("\n=== Searching for papers by query ===")
    for name, query in REQUIRED_PAPERS:
        print(f"\n  Searching: {name}")
        try:
            results = search_arxiv(query, max_results=3)
            for paper in results:
                if paper["arxiv_id"] not in existing_ids:
                    new_papers.append(paper)
                    existing_ids.add(paper["arxiv_id"])
                    print(f"    [+] {paper['arxiv_id']}: {paper['title'][:60]}...")
            time.sleep(1)
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Merge and save
    all_papers = existing + new_papers
    print(f"\n=== Summary ===")
    print(f"  Existing: {len(existing)}")
    print(f"  New: {len(new_papers)}")
    print(f"  Total: {len(all_papers)}")
    
    # Save updated papers
    papers_file.write_text(json.dumps(all_papers, indent=2))
    print(f"\nSaved to {papers_file}")
    
    # Also update chunks
    print("\nCreating chunks...")
    chunks = []
    for paper in all_papers:
        # Create one chunk per paper (abstract)
        chunks.append({
            "chunk_id": f"{paper['arxiv_id']}_abstract",
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "text": paper["abstract"],
            "section": "abstract",
        })
    
    chunks_file = data_dir / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, indent=2))
    print(f"Saved {len(chunks)} chunks to {chunks_file}")


if __name__ == "__main__":
    main()
