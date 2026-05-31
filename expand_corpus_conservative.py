"""
Conservative corpus expansion script.
- Very small batches (15 papers)
- Long delays between requests (8 seconds)
- Save progress after every successful batch
- Resume from last successful offset
- Simple, reliable approach
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("data")
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PROGRESS_FILE = DATA_DIR / "conservative_progress.json"

# Conservative settings
BATCH_SIZE = 15  # Very small batches
DELAY_BETWEEN_BATCHES = 8  # 8 seconds between requests
TARGET_PAPERS = 500
MAX_RETRIES = 5
RETRY_DELAY = 30  # 30 seconds on error

# arXiv API
ARXIV_API = "http://export.arxiv.org/api/query"
SEARCH_QUERY = "cat:cs.CL+AND+(abs:LLM+OR+abs:language+model+OR+abs:agent)"


def load_progress():
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"offset": 0, "collected_ids": [], "last_success": None}


def save_progress(progress):
    """Save progress to file."""
    progress["last_success"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def load_existing_papers():
    """Load existing papers from metadata."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_papers(papers):
    """Save papers to metadata file."""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


def fetch_batch(offset, batch_size):
    """Fetch a batch of papers from arXiv."""
    params = {
        "search_query": SEARCH_QUERY,
        "start": offset,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    response = requests.get(ARXIV_API, params=params, timeout=60)
    
    if response.status_code == 429:
        raise Exception("Rate limited (429)")
    
    response.raise_for_status()
    return response.text


def parse_arxiv_response(xml_text):
    """Parse arXiv API XML response."""
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('atom:entry', ns):
        arxiv_id_full = entry.find('atom:id', ns).text
        arxiv_id = arxiv_id_full.split('/abs/')[-1].split('v')[0]
        
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        # Get PDF link
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "pdf_url": pdf_url
        })
    
    return papers


def download_pdf(paper, retry=3):
    """Download PDF for a paper."""
    pdf_path = PAPERS_DIR / f"{paper['arxiv_id'].replace('/', '_')}.pdf"
    
    if pdf_path.exists():
        return True
    
    for attempt in range(retry):
        try:
            response = requests.get(paper['pdf_url'], timeout=60)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(5)
    
    return False


def main():
    print("=" * 60)
    print("CONSERVATIVE CORPUS EXPANSION")
    print(f"Target: {TARGET_PAPERS} papers")
    print(f"Batch size: {BATCH_SIZE}, Delay: {DELAY_BETWEEN_BATCHES}s")
    print("=" * 60)
    
    # Create directories
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data
    papers = load_existing_papers()
    existing_ids = {p['arxiv_id'] for p in papers}
    progress = load_progress()
    
    print(f"\nExisting papers: {len(papers)}")
    print(f"Starting from offset: {progress['offset']}")
    
    if len(papers) >= TARGET_PAPERS:
        print(f"Already have {len(papers)} papers. Target met!")
        return True
    
    offset = progress['offset']
    consecutive_empty = 0
    new_papers_count = 0
    
    while len(papers) < TARGET_PAPERS:
        print(f"\n--- Batch at offset {offset} ---")
        print(f"Total papers: {len(papers)}, New this session: {new_papers_count}")
        
        # Fetch batch with retries
        batch_papers = None
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Fetching... (attempt {attempt + 1}/{MAX_RETRIES})")
                xml_text = fetch_batch(offset, BATCH_SIZE)
                batch_papers = parse_arxiv_response(xml_text)
                print(f"Got {len(batch_papers)} papers from API")
                break
            except Exception as e:
                print(f"Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Waiting {RETRY_DELAY}s before retry...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached for this batch.")
                    # Save progress and continue from next offset
                    progress['offset'] = offset + BATCH_SIZE
                    save_progress(progress)
                    offset += BATCH_SIZE
                    consecutive_empty += 1
                    if consecutive_empty >= 5:
                        print("Too many consecutive failures. Stopping.")
                        return False
                    continue
        
        if batch_papers is None:
            continue
        
        # Process papers
        added = 0
        for paper in batch_papers:
            if paper['arxiv_id'] not in existing_ids:
                papers.append(paper)
                existing_ids.add(paper['arxiv_id'])
                added += 1
                new_papers_count += 1
        
        print(f"Added {added} new papers")
        
        if added == 0:
            consecutive_empty += 1
        else:
            consecutive_empty = 0
        
        # Save progress after every batch
        save_papers(papers)
        progress['offset'] = offset + BATCH_SIZE
        progress['collected_ids'] = list(existing_ids)
        save_progress(progress)
        print(f"Progress saved. Total: {len(papers)} papers")
        
        # Move to next batch
        offset += BATCH_SIZE
        
        # Check if we've exhausted results
        if len(batch_papers) < BATCH_SIZE:
            print("Received fewer papers than batch size - may have reached end of results")
            if consecutive_empty >= 3:
                print("Multiple empty batches. Stopping.")
                break
        
        # Delay before next request
        print(f"Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
        time.sleep(DELAY_BETWEEN_BATCHES)
    
    # Download PDFs for new papers
    print("\n" + "=" * 60)
    print("DOWNLOADING PDFs")
    print("=" * 60)
    
    downloaded = 0
    failed = 0
    for i, paper in enumerate(papers):
        pdf_path = PAPERS_DIR / f"{paper['arxiv_id'].replace('/', '_')}.pdf"
        if not pdf_path.exists():
            print(f"[{i+1}/{len(papers)}] Downloading {paper['arxiv_id']}...")
            if download_pdf(paper):
                downloaded += 1
            else:
                failed += 1
                print(f"  Failed to download {paper['arxiv_id']}")
            time.sleep(1)  # 1 second delay between downloads
    
    print(f"\nDownloaded: {downloaded}, Failed: {failed}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total papers: {len(papers)}")
    print(f"New papers this session: {new_papers_count}")
    print(f"PDFs on disk: {len(list(PAPERS_DIR.glob('*.pdf')))}")
    
    return len(papers) >= TARGET_PAPERS


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
