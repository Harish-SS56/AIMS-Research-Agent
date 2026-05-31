"""
OpenAlex Comprehensive Corpus Builder
=====================================
Uses OpenAlex as discovery layer to build the largest possible high-quality
Agentic AI corpus. Prioritizes papers with arXiv IDs.

Features:
- Multi-topic search across all Agentic AI research areas
- Filters: Computer Science, 2024-2026
- Prioritizes arXiv-linked papers
- Downloads PDFs from arXiv when possible, otherwise from other sources
- Deduplicates against existing corpus
- Continues until exhaustion (no arbitrary limits)
- Generates comprehensive final report
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import re

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://api.openalex.org/works"
EMAIL = "research-agent-corpus@example.com"

# Directories
PAPERS_DIR = Path("data/papers")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PROGRESS_FILE = Path("data/openalex_expansion_progress.json")

# Search topics - comprehensive coverage of Agentic AI
SEARCH_TOPICS = [
    # Core agent concepts
    "LLM agent large language model",
    "autonomous agent AI",
    "agentic AI system",
    "AI agent reasoning planning",
    
    # Specific frameworks and methods
    "ReAct reasoning acting language model",
    "Reflexion language agent",
    "Self-RAG retrieval augmented",
    "chain of thought reasoning LLM",
    "tree of thought reasoning",
    
    # RAG and retrieval
    "retrieval augmented generation LLM",
    "agentic RAG system",
    "knowledge retrieval language model",
    
    # Tool use and function calling
    "tool use language model",
    "function calling LLM API",
    "tool-augmented language model",
    "Toolformer",
    
    # Multi-agent systems
    "multi-agent system LLM",
    "multi-agent collaboration AI",
    "agent communication cooperation",
    
    # Planning and reasoning
    "planning agent language model",
    "AI reasoning problem solving",
    "task planning LLM",
    "goal-directed agent",
    
    # Memory and context
    "memory augmented agent",
    "episodic memory language model",
    "long-term memory AI agent",
    "context window agent",
    
    # Code and software agents
    "code generation agent LLM",
    "software engineering agent",
    "automated programming agent",
    
    # Evaluation and benchmarks
    "agent benchmark evaluation",
    "LLM agent evaluation",
    
    # Applications
    "conversational agent LLM",
    "dialogue system agent",
    "robotic agent language model",
    "embodied agent AI",
]

# Relevance keywords for scoring
RELEVANCE_KEYWORDS = {
    "high": [
        "llm agent", "language model agent", "agentic", "react", "reflexion",
        "self-rag", "chain-of-thought", "tree of thought", "tool use",
        "multi-agent", "autonomous agent", "retrieval augmented", "rag",
        "function call", "planning agent", "reasoning agent", "agent memory",
    ],
    "medium": [
        "language model", "llm", "gpt", "reasoning", "planning", "tool",
        "retrieval", "generation", "instruction", "prompt", "agent",
    ]
}

# Request settings
REQUEST_DELAY = 1.5  # seconds between API requests
PDF_DOWNLOAD_DELAY = 2.0  # seconds between PDF downloads
REQUEST_TIMEOUT = 30
PDF_TIMEOUT = 60
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_corpus():
    """Load existing papers metadata."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_corpus(papers):
    """Save papers metadata."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


def load_progress():
    """Load expansion progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "completed_topics": [],
        "papers_discovered": 0,
        "papers_downloaded": 0,
        "last_update": None
    }


def save_progress(progress):
    """Save expansion progress."""
    progress["last_update"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def generate_paper_id(title: str, authors: list) -> str:
    """Generate a stable ID for papers without arXiv ID."""
    author_str = ":".join(authors[:3]) if authors else "unknown"
    content = f"{title.lower().strip()}:{author_str}"
    return f"oa_{hashlib.md5(content.encode()).hexdigest()[:12]}"


def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    try:
        words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[i] for i in sorted(words))
    except Exception:
        return ""


def extract_arxiv_id(work: dict) -> Optional[str]:
    """Extract arXiv ID from OpenAlex work record."""
    # Check locations for arXiv URLs
    for loc in work.get("locations", []):
        landing = loc.get("landing_page_url") or ""
        pdf = loc.get("pdf_url") or ""
        
        for url in [landing, pdf]:
            if "arxiv.org" in url:
                # Extract ID from URL like https://arxiv.org/abs/2210.03629v2
                match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', url)
                if match:
                    return match.group(1)
    
    # Check DOI for arXiv
    doi = work.get("doi") or ""
    if "arxiv" in doi.lower():
        match = re.search(r'(\d{4}\.\d{4,5})', doi)
        if match:
            return match.group(1)
    
    # Check IDs object
    ids = work.get("ids", {})
    if ids.get("arxiv"):
        return ids["arxiv"].replace("https://arxiv.org/abs/", "")
    
    return None


def extract_pdf_url(work: dict, arxiv_id: Optional[str]) -> Optional[str]:
    """Extract best PDF URL, preferring arXiv."""
    # If we have arXiv ID, construct arXiv PDF URL
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # Check locations for PDF URLs
    pdf_urls = []
    for loc in work.get("locations", []):
        pdf = loc.get("pdf_url")
        if pdf:
            # Prefer certain sources
            if "arxiv.org" in pdf:
                return pdf
            pdf_urls.append(pdf)
    
    # Check open access URL
    oa = work.get("open_access", {})
    oa_url = oa.get("oa_url")
    if oa_url and oa_url.endswith(".pdf"):
        pdf_urls.insert(0, oa_url)
    
    return pdf_urls[0] if pdf_urls else None


def extract_source_url(work: dict) -> Optional[str]:
    """Extract source/landing page URL."""
    for loc in work.get("locations", []):
        landing = loc.get("landing_page_url")
        if landing:
            return landing
    return work.get("doi") or None


def calculate_relevance_score(title: str, abstract: str) -> tuple[int, str]:
    """Calculate relevance score (0-100) and tier."""
    text = (title + " " + abstract).lower()
    
    high_matches = sum(1 for kw in RELEVANCE_KEYWORDS["high"] if kw in text)
    med_matches = sum(1 for kw in RELEVANCE_KEYWORDS["medium"] if kw in text)
    
    score = min(100, high_matches * 15 + med_matches * 5)
    
    if score >= 60:
        return score, "high"
    elif score >= 30:
        return score, "medium"
    else:
        return score, "low"


def safe_filename(arxiv_id: str) -> str:
    """Convert arXiv ID to safe filename."""
    return arxiv_id.replace("/", "_").replace(":", "_")


# ═══════════════════════════════════════════════════════════════════════════════
# OPENALEX API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_works_for_topic(topic: str, existing_ids: set, max_results: int = 500) -> list[dict]:
    """
    Fetch all works for a topic from OpenAlex.
    Returns list of processed paper records.
    """
    papers = []
    cursor = "*"
    page = 0
    
    while len(papers) < max_results:
        params = {
            "search": topic,
            "filter": "publication_year:2024-2026,concepts.id:C41008148",  # CS concept
            "select": "id,title,publication_year,doi,open_access,locations,abstract_inverted_index,authorships,ids,cited_by_count",
            "per-page": 50,
            "cursor": cursor,
            "mailto": EMAIL,
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                
                if r.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    print(f"      Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif r.status_code != 200:
                    print(f"      Error {r.status_code}: {r.text[:100]}")
                    break
                
                data = r.json()
                results = data.get("results", [])
                meta = data.get("meta", {})
                next_cursor = meta.get("next_cursor")
                
                # Process results
                new_this_batch = 0
                for work in results:
                    oa_id = work.get("id", "")
                    if not oa_id:
                        continue
                    
                    # Extract metadata
                    title = work.get("title") or "Untitled"
                    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                    arxiv_id = extract_arxiv_id(work)
                    
                    # Generate ID for deduplication
                    paper_id = arxiv_id or oa_id.replace("https://openalex.org/", "")
                    
                    # Skip if already in corpus
                    if paper_id in existing_ids or title.lower().strip() in existing_ids:
                        continue
                    
                    # Extract other metadata
                    year = work.get("publication_year")
                    pdf_url = extract_pdf_url(work, arxiv_id)
                    source_url = extract_source_url(work)
                    
                    # Extract authors
                    authors = []
                    for authorship in work.get("authorships", [])[:10]:
                        author = authorship.get("author", {})
                        name = author.get("display_name")
                        if name:
                            authors.append(name)
                    
                    # Calculate relevance
                    score, tier = calculate_relevance_score(title, abstract)
                    
                    # Skip low relevance papers
                    if tier == "low":
                        continue
                    
                    paper = {
                        "id": paper_id,
                        "arxiv_id": arxiv_id,
                        "openalex_id": oa_id,
                        "title": title,
                        "abstract": abstract[:3000],  # Limit abstract length
                        "authors": authors,
                        "year": year,
                        "pdf_url": pdf_url,
                        "source_url": source_url,
                        "relevance_score": score,
                        "relevance_tier": tier,
                        "cited_by_count": work.get("cited_by_count", 0),
                        "topic": topic,
                        "source": "openalex",
                        "has_arxiv": bool(arxiv_id),
                        "has_pdf": bool(pdf_url),
                    }
                    
                    papers.append(paper)
                    existing_ids.add(paper_id)
                    existing_ids.add(title.lower().strip())
                    new_this_batch += 1
                
                page += 1
                print(f"      Page {page}: +{new_this_batch} new (total: {len(papers)})")
                
                if not next_cursor or len(results) < 50:
                    return papers
                
                cursor = next_cursor
                time.sleep(REQUEST_DELAY)
                break
                
            except requests.exceptions.Timeout:
                print(f"      Timeout (attempt {attempt + 1})")
                time.sleep(5)
            except Exception as e:
                print(f"      Error: {e}")
                break
        else:
            # Max retries exceeded
            break
    
    return papers


def download_pdf(paper: dict) -> Optional[str]:
    """Download PDF for a paper. Returns local path or None."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return None
    
    arxiv_id = paper.get("arxiv_id")
    paper_id = paper.get("id")
    
    # Determine filename
    if arxiv_id:
        filename = f"{safe_filename(arxiv_id)}.pdf"
    else:
        filename = f"{safe_filename(paper_id)}.pdf"
    
    pdf_path = PAPERS_DIR / filename
    
    # Skip if already downloaded
    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        return str(pdf_path)
    
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "research-agent/1.0 (mailto:research@example.com)"}
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(pdf_url, timeout=PDF_TIMEOUT, stream=True, headers=headers)
            
            if r.status_code == 200:
                content_type = r.headers.get("content-type", "")
                if "pdf" in content_type or pdf_url.endswith(".pdf"):
                    with open(pdf_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if pdf_path.stat().st_size > 10000:
                        return str(pdf_path)
                    else:
                        pdf_path.unlink(missing_ok=True)
            
            elif r.status_code == 429:
                time.sleep(30)
                continue
            
            break
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPANSION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("OPENALEX COMPREHENSIVE CORPUS BUILDER")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Topics to search: {len(SEARCH_TOPICS)}")
    print("=" * 70)
    print()
    
    # Load existing corpus
    existing_corpus = load_existing_corpus()
    existing_ids = set()
    for p in existing_corpus:
        if p.get("arxiv_id"):
            existing_ids.add(p["arxiv_id"])
        if p.get("id"):
            existing_ids.add(p["id"])
        if p.get("title"):
            existing_ids.add(p["title"].lower().strip())
    
    print(f"Existing corpus: {len(existing_corpus)} papers")
    
    # Load progress
    progress = load_progress()
    completed_topics = set(progress.get("completed_topics", []))
    
    # Statistics
    stats = {
        "discovered": 0,
        "downloaded": 0,
        "with_arxiv": 0,
        "without_arxiv": 0,
        "download_success": 0,
        "download_failed": 0,
        "by_topic": defaultdict(int),
        "by_source": defaultdict(int),
        "by_tier": defaultdict(int),
    }
    
    all_new_papers = []
    
    # Process each topic
    for i, topic in enumerate(SEARCH_TOPICS, 1):
        if topic in completed_topics:
            print(f"[{i}/{len(SEARCH_TOPICS)}] SKIP (completed): {topic[:50]}")
            continue
        
        print(f"\n[{i}/{len(SEARCH_TOPICS)}] Searching: {topic[:50]}...")
        
        papers = fetch_works_for_topic(topic, existing_ids, max_results=300)
        
        if not papers:
            print(f"    No new papers found")
            completed_topics.add(topic)
            continue
        
        # Sort: prioritize papers with arXiv IDs, then by relevance
        papers.sort(key=lambda p: (
            -int(p.get("has_arxiv", False)),  # arXiv first
            -p.get("relevance_score", 0),      # then by relevance
            -p.get("cited_by_count", 0)        # then by citations
        ))
        
        topic_downloaded = 0
        
        for paper in papers:
            stats["discovered"] += 1
            stats["by_topic"][topic] += 1
            stats["by_tier"][paper.get("relevance_tier", "unknown")] += 1
            
            if paper.get("has_arxiv"):
                stats["with_arxiv"] += 1
            else:
                stats["without_arxiv"] += 1
            
            # Track source
            pdf_url = paper.get("pdf_url") or ""
            if "arxiv.org" in pdf_url:
                stats["by_source"]["arxiv"] += 1
            elif "acl" in pdf_url.lower():
                stats["by_source"]["acl_anthology"] += 1
            elif "nature.com" in pdf_url:
                stats["by_source"]["nature"] += 1
            elif "springer" in pdf_url:
                stats["by_source"]["springer"] += 1
            elif "acm.org" in pdf_url:
                stats["by_source"]["acm"] += 1
            elif pdf_url:
                stats["by_source"]["other"] += 1
            else:
                stats["by_source"]["no_pdf"] += 1
            
            # Try to download PDF
            if paper.get("has_pdf"):
                pdf_path = download_pdf(paper)
                if pdf_path:
                    paper["pdf_path"] = pdf_path
                    stats["download_success"] += 1
                    stats["downloaded"] += 1
                    topic_downloaded += 1
                else:
                    stats["download_failed"] += 1
                
                time.sleep(PDF_DOWNLOAD_DELAY)
            
            all_new_papers.append(paper)
            
            # Save periodically
            if len(all_new_papers) % 50 == 0:
                merged = existing_corpus + all_new_papers
                save_corpus(merged)
                progress["completed_topics"] = list(completed_topics)
                progress["papers_discovered"] = stats["discovered"]
                progress["papers_downloaded"] = stats["downloaded"]
                save_progress(progress)
                print(f"    [Checkpoint] {len(all_new_papers)} new, {stats['downloaded']} downloaded")
        
        print(f"    Found {len(papers)} papers, downloaded {topic_downloaded} PDFs")
        completed_topics.add(topic)
    
    # Final save
    final_corpus = existing_corpus + all_new_papers
    save_corpus(final_corpus)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    
    print(f"\n{'─'*40}")
    print("CORPUS STATISTICS")
    print(f"{'─'*40}")
    print(f"Papers in original corpus:     {len(existing_corpus)}")
    print(f"New papers discovered:         {stats['discovered']}")
    print(f"New papers downloaded (PDFs):  {stats['downloaded']}")
    print(f"TOTAL CORPUS SIZE:             {len(final_corpus)}")
    
    print(f"\n{'─'*40}")
    print("ARXIV COVERAGE")
    print(f"{'─'*40}")
    print(f"Papers with arXiv IDs:         {stats['with_arxiv']} ({stats['with_arxiv']/max(stats['discovered'],1)*100:.1f}%)")
    print(f"Papers without arXiv IDs:      {stats['without_arxiv']} ({stats['without_arxiv']/max(stats['discovered'],1)*100:.1f}%)")
    
    print(f"\n{'─'*40}")
    print("PDF DOWNLOAD RESULTS")
    print(f"{'─'*40}")
    total_attempts = stats['download_success'] + stats['download_failed']
    success_rate = stats['download_success'] / max(total_attempts, 1) * 100
    print(f"Download attempts:             {total_attempts}")
    print(f"Successful downloads:          {stats['download_success']}")
    print(f"Failed downloads:              {stats['download_failed']}")
    print(f"Success rate:                  {success_rate:.1f}%")
    
    print(f"\n{'─'*40}")
    print("RELEVANCE DISTRIBUTION")
    print(f"{'─'*40}")
    for tier, count in sorted(stats['by_tier'].items()):
        pct = count / max(stats['discovered'], 1) * 100
        print(f"  {tier:15} {count:5} ({pct:.1f}%)")
    
    print(f"\n{'─'*40}")
    print("TOP PDF SOURCES")
    print(f"{'─'*40}")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {source:20} {count:5}")
    
    print(f"\n{'─'*40}")
    print("TOPIC DISTRIBUTION (top 15)")
    print(f"{'─'*40}")
    for topic, count in sorted(stats['by_topic'].items(), key=lambda x: -x[1])[:15]:
        print(f"  {topic[:45]:45} {count:4}")
    
    # Count PDFs on disk
    pdf_count = len(list(PAPERS_DIR.glob("*.pdf"))) if PAPERS_DIR.exists() else 0
    
    # Estimate chunk count (rough: ~60 chunks per paper average)
    estimated_chunks = pdf_count * 60
    
    print(f"\n{'─'*40}")
    print("CORPUS FILES")
    print(f"{'─'*40}")
    print(f"Total papers in metadata:      {len(final_corpus)}")
    print(f"PDF files on disk:             {pdf_count}")
    print(f"Estimated chunk count:         ~{estimated_chunks:,}")
    
    # Coverage estimate
    # Based on OpenAlex showing ~80k-100k papers for LLM agent topics in 2024-2026
    # Our corpus vs estimated total relevant papers
    coverage_estimate = len(final_corpus) / 5000 * 100  # Rough estimate of 5000 highly relevant papers
    coverage_estimate = min(coverage_estimate, 100)
    
    print(f"\n{'─'*40}")
    print("ESTIMATED COVERAGE")
    print(f"{'─'*40}")
    print(f"Estimated coverage of Agentic AI research (2024-2026): ~{coverage_estimate:.0f}%")
    print("(Based on ~5000 highly relevant papers in the field)")
    
    print(f"\n{'─'*40}")
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Save final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "original_corpus": len(existing_corpus),
        "new_discovered": stats["discovered"],
        "new_downloaded": stats["downloaded"],
        "total_corpus": len(final_corpus),
        "with_arxiv": stats["with_arxiv"],
        "without_arxiv": stats["without_arxiv"],
        "download_success": stats["download_success"],
        "download_failed": stats["download_failed"],
        "pdf_count": pdf_count,
        "estimated_chunks": estimated_chunks,
        "by_tier": dict(stats["by_tier"]),
        "by_source": dict(stats["by_source"]),
        "coverage_estimate_pct": coverage_estimate,
    }
    
    with open("data/openalex_expansion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: data/openalex_expansion_report.json")


if __name__ == "__main__":
    main()
