"""
Conservative arXiv corpus expansion.
- Small batches (10 papers)
- 5-second delays between requests
- Progress saved after every batch
- Resumes from existing 91-paper corpus
- Target: 400-500 papers
"""
import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Configuration
TARGET_PAPERS = 450
BATCH_SIZE = 25  # Larger batches since we filter locally
DELAY_BETWEEN_BATCHES = 12  # Longer delay - arXiv is sensitive
MAX_RETRIES = 2
RETRY_DELAY = 45

PAPERS_DIR = Path("data/papers")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PROGRESS_FILE = Path("data/arxiv_expansion_progress.json")

ARXIV_API = "http://export.arxiv.org/api/query"

# Simple category query - arXiv rate-limits complex abstract searches
# We filter for LLM/agent relevance locally instead
QUERY = "cat:cs.CL"

# Keywords to filter for LLM/agent relevance locally
RELEVANCE_KEYWORDS = [
    "llm", "large language model", "language model", "gpt", "chatgpt",
    "agent", "reasoning", "chain-of-thought", "cot", "prompt",
    "instruction", "fine-tun", "rlhf", "retrieval", "rag",
    "transformer", "attention", "generation", "dialogue", "chat"
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
    return {"offset": 0, "new_papers": 0, "failed_batches": 0}

def save_progress(offset, new_papers, failed_batches=0):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "offset": offset, 
            "new_papers": new_papers,
            "failed_batches": failed_batches,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

def fetch_batch(query, start, max_results):
    """Fetch a batch from arXiv. Returns (entries, total) or (None, 0) on failure."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    url = f"{ARXIV_API}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Fetching... (attempt {attempt+1}/{MAX_RETRIES})")
            r = requests.get(url, timeout=30, headers={"User-Agent": "research-agent/1.0"})
            
            if r.status_code == 429:
                print(f"  Rate limited (429). Waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            elif r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                time.sleep(RETRY_DELAY)
                continue
            
            # Parse XML
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            total_str = root.find(".//opensearch:totalResults", 
                                  {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"})
            total = int(total_str.text) if total_str is not None else 0
            
            entries = root.findall("atom:entry", ns)
            return entries, total, ns
            
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(RETRY_DELAY)
    
    return None, 0, None

def parse_entry(entry, ns):
    """Parse an arXiv entry into our metadata format."""
    def get_text(elem, path, default=""):
        el = elem.find(path, ns)
        return el.text.strip() if el is not None and el.text else default
    
    arxiv_id = get_text(entry, "atom:id").replace("http://arxiv.org/abs/", "")
    
    authors = []
    for author in entry.findall("atom:author", ns):
        name = get_text(author, "atom:name")
        if name:
            authors.append(name)
    
    # Get PDF link
    pdf_link = None
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf":
            pdf_link = link.get("href")
            break
    
    return {
        "id": arxiv_id,
        "title": get_text(entry, "atom:title").replace("\n", " "),
        "abstract": get_text(entry, "atom:summary").replace("\n", " "),
        "authors": authors,
        "published": get_text(entry, "atom:published"),
        "updated": get_text(entry, "atom:updated"),
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
    except Exception as e:
        print(f"    PDF download failed: {e}")
    
    return None

def main():
    print("=" * 60)
    print("CONSERVATIVE arXiv CORPUS EXPANSION")
    print(f"Target: {TARGET_PAPERS} papers")
    print(f"Batch size: {BATCH_SIZE}, Delay: {DELAY_BETWEEN_BATCHES}s")
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
    failed_batches = progress.get("failed_batches", 0)
    
    print(f"Resuming from offset: {offset}")
    print(f"Papers added this session: {session_new}")
    print()
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while len(existing) < TARGET_PAPERS:
        print(f"--- Batch at offset {offset} ---")
        print(f"Total: {len(existing)} papers | Session: +{session_new} | Target: {TARGET_PAPERS}")
        
        entries, total, ns = fetch_batch(QUERY, offset, BATCH_SIZE)
        
        if entries is None:
            consecutive_failures += 1
            failed_batches += 1
            print(f"  FAILED (consecutive: {consecutive_failures}/{max_consecutive_failures})")
            
            if consecutive_failures >= max_consecutive_failures:
                print("\nToo many consecutive failures. Stopping.")
                print("Run again later to resume from saved progress.")
                break
            
            offset += BATCH_SIZE
            save_progress(offset, session_new, failed_batches)
            time.sleep(DELAY_BETWEEN_BATCHES * 2)
            continue
        
        consecutive_failures = 0
        print(f"  Got {len(entries)} entries (total available: {total})")
        
        batch_new = 0
        batch_skipped = 0
        for entry in entries:
            meta = parse_entry(entry, ns)
            
            if meta["id"] in existing_ids:
                continue
            
            if not meta["abstract"]:
                continue
            
            # Filter for LLM/agent relevance locally
            if not is_relevant(meta):
                batch_skipped += 1
                continue
            
            # Download PDF
            pdf_path = download_pdf(meta["id"], meta.get("pdf_url"))
            meta["pdf_path"] = pdf_path
            
            existing.append(meta)
            existing_ids.add(meta["id"])
            batch_new += 1
            session_new += 1
            
            status = "[PDF]" if pdf_path else "[no PDF]"
            print(f"    + {meta['id']}: {meta['title'][:45]}... {status}")
        
        print(f"  Batch: +{batch_new} new, {batch_skipped} skipped (not relevant)")
        
        # Save after every batch
        save_metadata(existing)
        offset += BATCH_SIZE
        save_progress(offset, session_new, failed_batches)
        
        if len(existing) >= TARGET_PAPERS:
            print("\nTarget reached!")
            break
        
        if offset >= total:
            print(f"\nReached end of results ({total} total)")
            break
        
        print(f"  Waiting {DELAY_BETWEEN_BATCHES}s...")
        time.sleep(DELAY_BETWEEN_BATCHES)
    
    print()
    print("=" * 60)
    print("EXPANSION COMPLETE")
    print(f"Total papers: {len(existing)}")
    print(f"New this session: {session_new}")
    print(f"Failed batches: {failed_batches}")
    pdfs = sum(1 for p in existing if p.get("pdf_path"))
    print(f"Papers with PDFs: {pdfs}")
    print("=" * 60)

if __name__ == "__main__":
    main()
