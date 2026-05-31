"""
Targeted arXiv expansion script.
Adds 109-259 new arXiv papers to reach 450-600 total.
"""
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path
from datetime import datetime

# Configuration
BASE_URL = 'http://export.arxiv.org/api/query'
PDF_BASE = 'https://arxiv.org/pdf'
PAPERS_DIR = Path('data/papers')
PROGRESS_FILE = Path('data/expansion_progress.json')
MIN_TARGET = 450
MAX_TARGET = 600
REQUEST_DELAY = 25  # seconds between API requests
PDF_DELAY = 25      # seconds between PDF downloads

# Search queries for targeted topics
QUERIES = [
    # Core LLM Agent topics
    ('all:"LLM agent" OR all:"language model agent"', 'cs.CL'),
    ('all:"LLM agent" OR all:"language model agent"', 'cs.AI'),
    ('all:"agentic RAG" OR all:"retrieval augmented generation agent"', 'cs.CL'),
    ('all:ReAct AND all:agent', 'cs.CL'),
    ('all:Reflexion AND all:agent', 'cs.CL'),
    ('all:"Self-RAG" OR all:"self-reflective RAG"', 'cs.CL'),
    ('all:"tool use" AND all:LLM', 'cs.CL'),
    ('all:"tool learning" AND all:"language model"', 'cs.AI'),
    ('all:"planning agent" AND all:LLM', 'cs.AI'),
    ('all:"memory augmented" AND all:agent', 'cs.CL'),
    ('all:"multi-agent" AND all:LLM', 'cs.AI'),
    ('all:"multi-agent system" AND all:"language model"', 'cs.CL'),
    ('all:"computer use" AND all:agent', 'cs.AI'),
    ('all:"web agent" AND all:LLM', 'cs.CL'),
    ('all:"code agent" AND all:LLM', 'cs.CL'),
    # Additional relevant topics
    ('all:"autonomous agent" AND all:"large language"', 'cs.AI'),
    ('all:"agent framework" AND all:LLM', 'cs.CL'),
    ('all:"reasoning agent" AND all:LLM', 'cs.CL'),
    ('all:"task planning" AND all:"language model"', 'cs.AI'),
    ('all:"chain of thought" AND all:agent', 'cs.CL'),
    ('all:"function calling" AND all:LLM', 'cs.CL'),
    ('all:"API agent" AND all:LLM', 'cs.CL'),
    ('all:AutoGPT OR all:BabyAGI OR all:AgentGPT', 'cs.AI'),
    ('all:"instruction following" AND all:agent', 'cs.CL'),
    ('all:"dialogue agent" AND all:LLM', 'cs.CL'),
]

def load_existing_arxiv_ids():
    """Load existing arXiv IDs from papers directory."""
    existing = set()
    for f in PAPERS_DIR.glob('*.pdf'):
        if not f.stem.startswith('W'):  # Skip OpenAlex IDs
            existing.add(f.stem)
    return existing

def load_progress():
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {'collected': [], 'failed': [], 'query_index': 0, 'offset': 0}

def save_progress(progress):
    """Save progress to file."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

def fetch_with_retry(url, max_retries=3, timeout=60):
    """Fetch URL with retry logic for 429 errors."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AIMS-Research-Agent/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read(), response.status, None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 60 * (attempt + 1)
                print(f'  429 Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries})')
                time.sleep(wait_time)
            else:
                return None, e.code, str(e)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'  Error: {e}, retrying...')
                time.sleep(10)
            else:
                return None, 0, str(e)
    return None, 429, 'Max retries exceeded'

def search_arxiv(query, category, start=0, max_results=50):
    """Search arXiv API."""
    full_query = f'{query} AND cat:{category}'
    params = {
        'search_query': full_query,
        'start': start,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    url = f'{BASE_URL}?{urllib.parse.urlencode(params)}'
    
    data, status, error = fetch_with_retry(url)
    if not data:
        return [], error
    
    # Parse XML
    try:
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', ns)
        
        papers = []
        for entry in entries:
            id_elem = entry.find('atom:id', ns)
            title_elem = entry.find('atom:title', ns)
            abstract_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            
            if id_elem is None:
                continue
            
            arxiv_url = id_elem.text
            # Extract clean arxiv_id (without version)
            arxiv_id = arxiv_url.split('/')[-1]
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
            
            papers.append({
                'arxiv_id': arxiv_id,
                'title': title_elem.text.strip().replace('\n', ' ') if title_elem is not None else '',
                'abstract': abstract_elem.text.strip() if abstract_elem is not None else '',
                'published': published_elem.text if published_elem is not None else '',
                'url': arxiv_url
            })
        
        return papers, None
    except ET.ParseError as e:
        return [], str(e)

def download_pdf(arxiv_id):
    """Download PDF and verify integrity."""
    pdf_url = f'{PDF_BASE}/{arxiv_id}.pdf'
    pdf_path = PAPERS_DIR / f'{arxiv_id}.pdf'
    
    if pdf_path.exists():
        return True, 'Already exists'
    
    data, status, error = fetch_with_retry(pdf_url, timeout=120)
    
    if not data:
        return False, error or f'HTTP {status}'
    
    # Verify PDF header
    if not data[:4] == b'%PDF':
        return False, 'Invalid PDF header'
    
    # Save PDF
    pdf_path.write_bytes(data)
    return True, f'{len(data)/1024:.1f} KB'

def main():
    print('=' * 70)
    print('TARGETED ARXIV EXPANSION')
    print('=' * 70)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # Load existing papers
    existing_ids = load_existing_arxiv_ids()
    print(f'Existing arXiv papers: {len(existing_ids)}')
    print(f'Target range: {MIN_TARGET}-{MAX_TARGET}')
    print()
    
    if len(existing_ids) >= MAX_TARGET:
        print('Already at or above maximum target. Nothing to do.')
        return
    
    # Load progress
    progress = load_progress()
    collected_ids = set(progress['collected'])
    
    # Track stats
    new_papers = 0
    failed_downloads = 0
    skipped_duplicates = 0
    rate_limits = 0
    
    # Process each query
    for q_idx, (query, category) in enumerate(QUERIES):
        if len(existing_ids) + new_papers >= MAX_TARGET:
            print(f'\nReached target ({len(existing_ids) + new_papers} papers). Stopping.')
            break
        
        if q_idx < progress['query_index']:
            continue  # Skip already completed queries
        
        print(f'\n[Query {q_idx+1}/{len(QUERIES)}] {category}: {query[:50]}...')
        
        offset = progress['offset'] if q_idx == progress['query_index'] else 0
        
        while True:
            if len(existing_ids) + new_papers >= MAX_TARGET:
                break
            
            # Search
            time.sleep(REQUEST_DELAY)
            papers, error = search_arxiv(query, category, start=offset, max_results=25)
            
            if error:
                print(f'  Search error: {error}')
                if '429' in str(error):
                    rate_limits += 1
                    time.sleep(60)
                    continue
                break
            
            if not papers:
                print(f'  No more results at offset {offset}')
                break
            
            print(f'  Found {len(papers)} papers at offset {offset}')
            
            for paper in papers:
                arxiv_id = paper['arxiv_id']
                
                # Check for duplicates
                if arxiv_id in existing_ids or arxiv_id in collected_ids:
                    skipped_duplicates += 1
                    continue
                
                if len(existing_ids) + new_papers >= MAX_TARGET:
                    break
                
                # Download PDF
                print(f'    [{len(existing_ids) + new_papers + 1}] {arxiv_id}: {paper["title"][:40]}...')
                time.sleep(PDF_DELAY)
                
                success, msg = download_pdf(arxiv_id)
                
                if success:
                    new_papers += 1
                    collected_ids.add(arxiv_id)
                    progress['collected'].append(arxiv_id)
                    print(f'        OK - {msg}')
                else:
                    failed_downloads += 1
                    progress['failed'].append(arxiv_id)
                    print(f'        FAILED - {msg}')
                
                # Save progress
                progress['query_index'] = q_idx
                progress['offset'] = offset
                save_progress(progress)
            
            offset += 25
            
            # Limit per query to avoid exhausting one topic
            if offset >= 200:
                print(f'  Reached offset limit for this query')
                break
    
    # Final report
    print()
    print('=' * 70)
    print('EXPANSION COMPLETE')
    print('=' * 70)
    print(f'New papers added: {new_papers}')
    print(f'Failed downloads: {failed_downloads}')
    print(f'Skipped duplicates: {skipped_duplicates}')
    print(f'Rate limit events: {rate_limits}')
    print()
    
    # Count final totals
    final_arxiv = len(load_existing_arxiv_ids())
    total_pdfs = len(list(PAPERS_DIR.glob('*.pdf')))
    
    print(f'Total arXiv papers: {final_arxiv}')
    print(f'Total PDFs (including OpenAlex): {total_pdfs}')
    print(f'Target achieved: {"YES" if MIN_TARGET <= final_arxiv <= MAX_TARGET else "NO"}')
    print()
    print(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
