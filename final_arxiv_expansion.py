"""
Final arXiv corpus expansion pass.
Target: 50-100 new HIGH-QUALITY papers from May 2025 - April 2026.
Strict deduplication, exponential backoff, and progress saving.
"""
import os
import sys
import json
import time
import random
import hashlib
import requests
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_NEW_PAPERS = 75  # Stop between 50-100
MIN_NEW_PAPERS = 50
MAX_NEW_PAPERS = 100

DATE_START = "2025-05-01"
DATE_END = "2026-04-30"

SEARCH_QUERIES = [
    # Primary agent topics
    '"LLM agent" OR "language model agent"',
    '"agentic AI" OR "agentic system"',
    '"multi-agent" AND "language model"',
    '"tool use" AND "language model"',
    '"function calling" AND LLM',
    '"agent memory" AND LLM',
    '"planning agent" OR "agent planning"',
    '"computer use" AND agent',
    '"Self-RAG" OR "self-reflective RAG"',
    '"autonomous agent" AND LLM',
    '"agent benchmark" OR "agent evaluation"',
    # Secondary topics
    '"ReAct" AND agent',
    '"Reflexion" AND agent',
    '"code agent" OR "coding agent"',
    '"web agent" OR "browser agent"',
    '"GUI agent" OR "UI agent"',
]

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CATEGORIES = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"

PDF_DIR = Path("data/papers")
PROGRESS_FILE = Path("data/final_expansion_progress.json")
METADATA_FILE = Path("data/processed/papers_metadata.json")
TEXTS_FILE = Path("data/processed/papers_texts.json")
CHUNKS_FILE = Path("data/processed/chunks.json")

# Rate limiting
INITIAL_BACKOFF = 3.0
MAX_BACKOFF = 300.0  # 5 minutes - severe limit
BACKOFF_MULTIPLIER = 2.0
MAX_CONSECUTIVE_429 = 5
MAX_RETRY_TIME_MINUTES = 30

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EXISTING CORPUS
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_ids():
    """Load all existing arXiv IDs from chunks and downloaded PDFs."""
    existing = set()
    
    # From chunks.json
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, encoding='utf-8') as f:
            chunks = json.load(f)
        for c in chunks:
            aid = c.get('arxiv_id', '')
            if aid:
                existing.add(aid)
    
    # From downloaded PDFs
    for pdf in PDF_DIR.glob('*.pdf'):
        if pdf.stem[0].isdigit():  # arXiv format
            existing.add(pdf.stem)
    
    # From metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding='utf-8') as f:
            meta = json.load(f)
        for m in meta:
            aid = str(m.get('arxiv_id', ''))
            if aid:
                existing.add(aid)
    
    return existing

def load_progress():
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {
        'new_papers': [],
        'failed_downloads': [],
        'skipped_duplicates': 0,
        'total_api_calls': 0,
        'total_wait_time': 0,
        'queries_completed': [],
        'status': 'in_progress',
        'started': datetime.now().isoformat(),
    }

def save_progress(progress):
    """Save progress after each successful paper."""
    progress['last_updated'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# ARXIV API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arxiv_response(xml_text):
    """Parse arXiv API XML response."""
    papers = []
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    
    for entry in root.findall('atom:entry', ns):
        try:
            # Get arXiv ID from id URL
            id_url = entry.find('atom:id', ns).text
            arxiv_id = id_url.split('/abs/')[-1]
            # Remove version suffix (e.g., v1, v2)
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.rsplit('v', 1)[0]
            
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text[:10]
            
            # Categories
            cats = [c.get('term') for c in entry.findall('atom:category', ns)]
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # PDF link
            pdf_url = None
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            
            papers.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'published': published,
                'categories': cats,
                'authors': authors,
                'pdf_url': pdf_url or f'https://arxiv.org/pdf/{arxiv_id}.pdf',
            })
        except Exception as e:
            continue
    
    return papers

def search_arxiv(query, start=0, max_results=100, existing_ids=set(), progress=None):
    """
    Search arXiv API with exponential backoff.
    Returns (papers, severe_limit_reached).
    """
    full_query = f'({query}) AND ({CATEGORIES})'
    params = {
        'search_query': full_query,
        'start': start,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    }
    
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    
    backoff = INITIAL_BACKOFF
    consecutive_429 = 0
    
    while True:
        if progress:
            progress['total_api_calls'] += 1
        
        try:
            print(f"    API call: start={start}, max={max_results}")
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                papers = parse_arxiv_response(response.text)
                # Filter to date range and deduplicate
                filtered = []
                for p in papers:
                    if p['arxiv_id'] in existing_ids:
                        if progress:
                            progress['skipped_duplicates'] += 1
                        continue
                    # Check date range (YYYY-MM-DD format)
                    if DATE_START <= p['published'] <= DATE_END:
                        filtered.append(p)
                return filtered, False
            
            elif response.status_code == 429:
                consecutive_429 += 1
                print(f"    ⚠️  429 Rate Limited (#{consecutive_429})")
                
                if consecutive_429 >= MAX_CONSECUTIVE_429:
                    print(f"    ❌ Severe rate limiting: {MAX_CONSECUTIVE_429} consecutive 429s")
                    return [], True
                
                if backoff >= MAX_BACKOFF:
                    print(f"    ❌ Severe rate limiting: backoff reached {MAX_BACKOFF}s")
                    return [], True
                
                wait_time = backoff + random.uniform(0, backoff * 0.5)
                if progress:
                    progress['total_wait_time'] += wait_time
                    if progress['total_wait_time'] / 60 > MAX_RETRY_TIME_MINUTES:
                        print(f"    ❌ Severe rate limiting: total wait time > {MAX_RETRY_TIME_MINUTES}min")
                        return [], True
                
                print(f"    Waiting {wait_time:.1f}s (backoff={backoff:.1f}s)")
                time.sleep(wait_time)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            
            else:
                print(f"    ⚠️  HTTP {response.status_code}")
                time.sleep(5)
                return [], False
                
        except requests.exceptions.Timeout:
            print("    ⚠️  Timeout, retrying...")
            time.sleep(10)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            time.sleep(5)
            return [], False

def download_pdf(arxiv_id, pdf_url, max_retries=3):
    """Download PDF with retries."""
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return pdf_path
            elif response.status_code == 429:
                wait = (attempt + 1) * 30
                print(f"      PDF 429, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      PDF HTTP {response.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"      PDF error: {e}")
            time.sleep(5)
    
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def collect_papers():
    """Main collection loop."""
    print("=" * 70)
    print("FINAL ARXIV CORPUS EXPANSION")
    print(f"Target: {MIN_NEW_PAPERS}-{MAX_NEW_PAPERS} new papers")
    print(f"Date range: {DATE_START} to {DATE_END}")
    print("=" * 70)
    
    existing_ids = load_existing_ids()
    print(f"\nExisting corpus: {len(existing_ids)} papers")
    
    progress = load_progress()
    if progress['status'] == 'completed':
        print("\n✅ Collection already completed in previous run.")
        return progress
    
    # Track new papers added this session
    session_new = set(p['arxiv_id'] for p in progress['new_papers'])
    existing_ids.update(session_new)
    
    print(f"Already collected this session: {len(session_new)}")
    
    severe_limit = False
    
    for query in SEARCH_QUERIES:
        if severe_limit:
            break
        if query in progress['queries_completed']:
            continue
        if len(progress['new_papers']) >= MAX_NEW_PAPERS:
            break
        
        print(f"\n📚 Query: {query[:60]}...")
        
        offset = 0
        batch_size = 50
        query_papers = 0
        
        while not severe_limit and len(progress['new_papers']) < MAX_NEW_PAPERS:
            papers, severe_limit = search_arxiv(
                query, start=offset, max_results=batch_size,
                existing_ids=existing_ids, progress=progress
            )
            
            if severe_limit:
                print("  ❌ Severe rate limiting detected, stopping collection")
                break
            
            if not papers:
                break
            
            for paper in papers:
                if len(progress['new_papers']) >= MAX_NEW_PAPERS:
                    break
                if paper['arxiv_id'] in existing_ids:
                    continue
                
                print(f"  ✓ [{paper['arxiv_id']}] {paper['title'][:50]}...")
                
                # Download PDF
                pdf_path = download_pdf(paper['arxiv_id'], paper['pdf_url'])
                if pdf_path:
                    paper['pdf_path'] = str(pdf_path)
                    progress['new_papers'].append(paper)
                    existing_ids.add(paper['arxiv_id'])
                    query_papers += 1
                    
                    # Save progress after each successful paper
                    save_progress(progress)
                    print(f"    → Saved ({len(progress['new_papers'])}/{TARGET_NEW_PAPERS})")
                else:
                    progress['failed_downloads'].append(paper['arxiv_id'])
                
                # Polite delay
                time.sleep(1.5)
            
            offset += batch_size
            time.sleep(3)  # Polite API delay
            
            if len(papers) < batch_size:
                break
        
        if not severe_limit:
            progress['queries_completed'].append(query)
            save_progress(progress)
        
        print(f"  → Found {query_papers} new papers from this query")
    
    # Mark completion
    if len(progress['new_papers']) >= MIN_NEW_PAPERS or severe_limit:
        progress['status'] = 'completed' if not severe_limit else 'stopped_rate_limit'
    
    progress['finished'] = datetime.now().isoformat()
    save_progress(progress)
    
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"New papers collected: {len(progress['new_papers'])}")
    print(f"Failed downloads: {len(progress['failed_downloads'])}")
    print(f"Duplicates skipped: {progress['skipped_duplicates']}")
    print(f"Total API calls: {progress['total_api_calls']}")
    print(f"Total wait time: {progress['total_wait_time']:.1f}s")
    print(f"Status: {progress['status']}")
    
    return progress

if __name__ == "__main__":
    collect_papers()
