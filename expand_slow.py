"""
Ultra-slow arXiv scraper - respects strict rate limits.
- 1 paper per request
- 25 second delay between requests
- Saves progress after every paper
- Target: 450 papers (~3 hours runtime)
"""
import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Configuration
TARGET_PAPERS = 500
DELAY_BETWEEN_REQUESTS = 30  # seconds between requests
REQUEST_TIMEOUT = 45
RATE_LIMIT_COOLDOWN = 120  # wait after rate limit before retry

PAPERS_DIR = Path("data/papers")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PROGRESS_FILE = Path("data/slow_expansion_progress.json")

ARXIV_API = "http://export.arxiv.org/api/query"
QUERY = "cat:cs.CL"  # Simple category query - arXiv rate-limits complex searches

# Keywords for local relevance filtering - focused on LLM agents research
RELEVANCE_KEYWORDS = [
    # Core agent concepts
    "agent", "agentic", "multi-agent", "autonomous",
    # Specific frameworks
    "react", "reflexion", "self-rag", "toolformer",
    "chain-of-thought", "cot", "tree of thought",
    # RAG and retrieval
    "retrieval", "rag", "retrieval-augmented",
    # Tool use
    "tool use", "tool-use", "function call", "api call",
    # Memory
    "agent memory", "episodic memory", "working memory",
    # LLM fundamentals
    "llm", "large language model", "language model", "gpt",
    "reasoning", "planning", "prompt", "instruction",
    "fine-tun", "rlhf", "in-context learning"
]

def load_existing_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_metadata(papers):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"offset": 0, "new_papers": 0, "skipped": 0, "errors": 0}

def save_progress(offset, new_papers, skipped, errors):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "offset": offset,
            "new_papers": new_papers,
            "skipped": skipped,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

def fetch_one_paper(start):
    """Fetch exactly one paper from arXiv."""
    url = f"{ARXIV_API}?search_query={QUERY}&start={start}&max_results=1&sortBy=submittedDate&sortOrder=descending"
    
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "research-agent/1.0"})
        
        if r.status_code == 429:
            return None, "rate_limited"
        elif r.status_code != 200:
            return None, f"http_{r.status_code}"
        
        # Parse XML
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        entries = root.findall("atom:entry", ns)
        if not entries:
            return None, "no_entries"
        
        return entries[0], ns
        
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:50]

def parse_entry(entry, ns):
    """Parse an arXiv entry into metadata format."""
    def get_text(path, default=""):
        el = entry.find(path, ns)
        return el.text.strip() if el is not None and el.text else default
    
    arxiv_id = get_text("atom:id").replace("http://arxiv.org/abs/", "")
    
    authors = []
    for author in entry.findall("atom:author", ns):
        name_el = author.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    
    pdf_link = None
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf":
            pdf_link = link.get("href")
            break
    
    return {
        "id": arxiv_id,
        "title": get_text("atom:title").replace("\n", " "),
        "abstract": get_text("atom:summary").replace("\n", " "),
        "authors": authors,
        "published": get_text("atom:published"),
        "updated": get_text("atom:updated"),
        "pdf_url": pdf_link,
        "source": "arxiv"
    }

def is_relevant(meta):
    """Check if paper is relevant to LLM/agent research."""
    text = (meta.get("title", "") + " " + meta.get("abstract", "")).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)

def download_pdf(arxiv_id, pdf_url):
    """Download PDF. Returns path or None."""
    if not pdf_url:
        return None
    
    safe_id = arxiv_id.replace("/", "_").replace(":", "_")
    pdf_path = PAPERS_DIR / f"{safe_id}.pdf"
    
    if pdf_path.exists():
        return str(pdf_path)
    
    try:
        r = requests.get(pdf_url, timeout=60, stream=True, 
                        headers={"User-Agent": "research-agent/1.0"})
        if r.status_code == 200:
            PAPERS_DIR.mkdir(parents=True, exist_ok=True)
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return str(pdf_path)
    except:
        pass
    
    return None

def main():
    print("=" * 60)
    print("ULTRA-SLOW arXiv EXPANSION")
    print(f"Target: {TARGET_PAPERS} papers")
    print(f"Delay: {DELAY_BETWEEN_REQUESTS}s between requests")
    print(f"Rate limit cooldown: {RATE_LIMIT_COOLDOWN}s")
    print(f"Estimated time: ~{(TARGET_PAPERS - 91) * DELAY_BETWEEN_REQUESTS // 3600}+ hours")
    print("=" * 60)
    print()
    
    # Load existing
    existing = load_existing_metadata()
    existing_ids = {p.get("id") for p in existing if p.get("id")}
    print(f"Existing papers: {len(existing)}")
    
    # Load progress
    progress = load_progress()
    offset = progress.get("offset", 0)
    session_new = progress.get("new_papers", 0)
    session_skipped = progress.get("skipped", 0)
    session_errors = progress.get("errors", 0)
    
    print(f"Resuming from offset: {offset}")
    print(f"Session stats: +{session_new} new, {session_skipped} skipped, {session_errors} errors")
    print()
    
    consecutive_errors = 0
    max_consecutive_errors = 8  # More tolerance for errors
    
    while len(existing) < TARGET_PAPERS:
        # Status line
        elapsed_est = (offset - progress.get("offset", 0)) * DELAY_BETWEEN_REQUESTS
        print(f"[{len(existing)}/{TARGET_PAPERS}] Offset {offset} | +{session_new} new | ", end="", flush=True)
        
        entry, ns_or_error = fetch_one_paper(offset)
        
        if entry is None:
            error = ns_or_error
            session_errors += 1
            consecutive_errors += 1
            print(f"ERROR: {error}")
            
            if error == "rate_limited":
                print(f"  Rate limited! Waiting {RATE_LIMIT_COOLDOWN}s...")
                time.sleep(RATE_LIMIT_COOLDOWN)
                # Do NOT increment offset — retry the same offset after cooldown
                save_progress(offset, session_new, session_skipped, session_errors)
                continue
            elif consecutive_errors >= max_consecutive_errors:
                print(f"\nToo many consecutive errors ({consecutive_errors}). Stopping.")
                break
            
            offset += 1
            save_progress(offset, session_new, session_skipped, session_errors)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue
        
        consecutive_errors = 0
        ns = ns_or_error
        meta = parse_entry(entry, ns)
        
        # Skip if already have
        if meta["id"] in existing_ids:
            print(f"SKIP (duplicate): {meta['id']}")
            offset += 1
            save_progress(offset, session_new, session_skipped, session_errors)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue
        
        # Skip if no abstract
        if not meta["abstract"]:
            print(f"SKIP (no abstract): {meta['id']}")
            session_skipped += 1
            offset += 1
            save_progress(offset, session_new, session_skipped, session_errors)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue
        
        # Check relevance
        if not is_relevant(meta):
            print(f"SKIP (not relevant): {meta['title'][:40]}...")
            session_skipped += 1
            offset += 1
            save_progress(offset, session_new, session_skipped, session_errors)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue
        
        # Download PDF
        pdf_path = download_pdf(meta["id"], meta.get("pdf_url"))
        meta["pdf_path"] = pdf_path
        
        # Add to corpus
        existing.append(meta)
        existing_ids.add(meta["id"])
        session_new += 1
        
        status = "PDF" if pdf_path else "no-PDF"
        print(f"ADD [{status}]: {meta['title'][:45]}...")
        
        # Save after every paper
        save_metadata(existing)
        offset += 1
        save_progress(offset, session_new, session_skipped, session_errors)
        
        if len(existing) >= TARGET_PAPERS:
            break
        
        # Wait before next request
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print()
    print("=" * 60)
    print("EXPANSION COMPLETE" if len(existing) >= TARGET_PAPERS else "EXPANSION PAUSED")
    print(f"Total papers: {len(existing)}")
    print(f"Session: +{session_new} new, {session_skipped} skipped, {session_errors} errors")
    pdfs = sum(1 for p in existing if p.get("pdf_path"))
    print(f"Papers with PDFs: {pdfs}")
    print("=" * 60)
    print()
    print("To resume later, just run this script again.")

if __name__ == "__main__":
    main()
