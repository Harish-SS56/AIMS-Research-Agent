#!/usr/bin/env python3
"""
Fix Corpus Pipeline

Downloads missing PDFs, parses them, and rebuilds chunks/index.
"""

import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_DIR = SCRIPT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_pdf(arxiv_id: str, pdf_url: str = None) -> Path | None:
    """Download a single PDF with retry logic."""
    pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"
    
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path
    
    url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    headers = {
        "User-Agent": "AIMSResearchAgent/1.0 (academic research; contact@example.com)"
    }
    
    wait = 5
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, timeout=90, stream=True)
            if resp.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                if pdf_path.stat().st_size > 1000:
                    return pdf_path
                else:
                    pdf_path.unlink()  # Remove invalid file
            elif resp.status_code == 429:
                print(f"  Rate limited, sleeping {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                print(f"  HTTP {resp.status_code} for {arxiv_id}")
                return None
        except Exception as e:
            print(f"  Error downloading {arxiv_id}: {e}")
            time.sleep(wait)
            wait = min(wait * 2, 60)
    
    return None


def download_all_pdfs(papers: list, delay: float = 3.0) -> dict:
    """Download all missing PDFs."""
    print(f"\nDownloading PDFs for {len(papers)} papers...")
    
    results = {"success": [], "failed": []}
    
    for paper in tqdm(papers, desc="Downloading"):
        arxiv_id = paper['arxiv_id']
        pdf_url = paper.get('pdf_url')
        
        pdf_path = download_pdf(arxiv_id, pdf_url)
        
        if pdf_path:
            paper['pdf_path'] = str(pdf_path)
            results["success"].append(arxiv_id)
        else:
            paper['pdf_path'] = None
            results["failed"].append(arxiv_id)
        
        time.sleep(delay)  # Polite delay
    
    print(f"\n✓ Downloaded: {len(results['success'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    
    return results


def main():
    print("=" * 70)
    print("CORPUS PIPELINE FIX")
    print("=" * 70)
    
    # Load metadata
    metadata_path = PROCESSED_DIR / "papers_metadata.json"
    if not metadata_path.exists():
        print("❌ No papers_metadata.json found!")
        return
    
    papers = json.load(open(metadata_path, 'r', encoding='utf-8'))
    print(f"Loaded {len(papers)} papers from metadata")
    
    # Download PDFs
    results = download_all_pdfs(papers)
    
    # Save updated metadata with pdf_path
    print("\nSaving updated metadata...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    if results["failed"]:
        print(f"\nFailed to download ({len(results['failed'])}):")
        for arxiv_id in results["failed"][:20]:
            print(f"  - {arxiv_id}")
        if len(results["failed"]) > 20:
            print(f"  ... and {len(results['failed']) - 20} more")
    
    # Now parse the PDFs
    print("\n" + "=" * 70)
    print("PARSING PDFs")
    print("=" * 70)
    
    from src.corpus.pdf_parser import PDFParser
    
    parser = PDFParser()
    parsed_papers = []
    stats = {"parsed": 0, "failed": 0, "full_text_avg": []}
    
    for paper in tqdm(papers, desc="Parsing"):
        pdf_path = paper.get('pdf_path')
        
        if pdf_path and Path(pdf_path).exists():
            text = parser.extract_text(Path(pdf_path))
            if text and len(text) > 500:
                paper_data = {
                    "arxiv_id": paper['arxiv_id'],
                    "title": paper['title'],
                    "abstract": paper.get('abstract', ''),
                    "full_text": text,
                    "sections": parser.extract_sections(text),
                    "word_count": len(text.split()),
                    "char_count": len(text)
                }
                parsed_papers.append(paper_data)
                stats["parsed"] += 1
                stats["full_text_avg"].append(len(text))
            else:
                # Fall back to abstract
                paper_data = {
                    "arxiv_id": paper['arxiv_id'],
                    "title": paper['title'],
                    "abstract": paper.get('abstract', ''),
                    "full_text": f"{paper['title']}\n\n{paper.get('abstract', '')}",
                    "sections": {"abstract": paper.get('abstract', '')},
                    "word_count": len(paper.get('abstract', '').split()),
                    "char_count": len(paper.get('abstract', ''))
                }
                parsed_papers.append(paper_data)
                stats["failed"] += 1
        else:
            # No PDF - use abstract
            paper_data = {
                "arxiv_id": paper['arxiv_id'],
                "title": paper['title'],
                "abstract": paper.get('abstract', ''),
                "full_text": f"{paper['title']}\n\n{paper.get('abstract', '')}",
                "sections": {"abstract": paper.get('abstract', '')},
                "word_count": len(paper.get('abstract', '').split()),
                "char_count": len(paper.get('abstract', ''))
            }
            parsed_papers.append(paper_data)
            stats["failed"] += 1
    
    avg_len = sum(stats["full_text_avg"]) / len(stats["full_text_avg"]) if stats["full_text_avg"] else 0
    print(f"\n✓ Parsed with full text: {stats['parsed']}")
    print(f"✗ Abstract only: {stats['failed']}")
    print(f"Average full text length: {avg_len:.0f} chars")
    
    # Save parsed corpus
    texts_dir = PROCESSED_DIR / "texts"
    texts_dir.mkdir(exist_ok=True)
    
    for paper in parsed_papers:
        txt_path = texts_dir / f"{paper['arxiv_id']}.txt"
        txt_path.write_text(paper['full_text'], encoding='utf-8')
    
    # Save papers_texts.json (metadata)
    texts_meta = []
    for paper in parsed_papers:
        meta = {k: v for k, v in paper.items() if k != 'full_text'}
        meta['has_full_text'] = paper['char_count'] > 1000
        texts_meta.append(meta)
    
    with open(PROCESSED_DIR / "papers_texts.json", 'w', encoding='utf-8') as f:
        json.dump(texts_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(parsed_papers)} parsed papers")
    
    # Chunk the corpus
    print("\n" + "=" * 70)
    print("CHUNKING CORPUS")
    print("=" * 70)
    
    from src.corpus.chunker import TextChunker
    
    chunker = TextChunker()
    all_chunks = []
    
    for paper in tqdm(parsed_papers, desc="Chunking"):
        chunks = chunker.chunk_paper(paper)
        all_chunks.extend(chunks)
    
    print(f"\n✓ Created {len(all_chunks)} chunks from {len(parsed_papers)} papers")
    print(f"Average chunks per paper: {len(all_chunks)/len(parsed_papers):.1f}")
    
    # Analyze chunks by section
    from collections import defaultdict
    by_section = defaultdict(int)
    for c in all_chunks:
        by_section[c.section or 'unknown'] += 1
    
    print("\nChunks by section:")
    for section, count in sorted(by_section.items()):
        print(f"  {section}: {count}")
    
    # Save chunks
    chunker.save_chunks(all_chunks)
    
    # Build indexes
    print("\n" + "=" * 70)
    print("BUILDING INDEXES")
    print("=" * 70)
    
    # Clear old index
    import shutil
    chroma_dir = INDEX_DIR / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print("Cleared old ChromaDB index")
    
    from src.retrieval import build_index, build_bm25_index
    
    print("Building vector index...")
    build_index(all_chunks)
    
    print("Building BM25 index...")
    build_bm25_index(all_chunks)
    
    print("\n" + "=" * 70)
    print("PIPELINE FIX COMPLETE")
    print("=" * 70)
    print(f"Total papers: {len(papers)}")
    print(f"PDFs downloaded: {len(results['success'])}")
    print(f"Papers with full text: {stats['parsed']}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Average chunks per paper: {len(all_chunks)/len(parsed_papers):.1f}")


if __name__ == "__main__":
    main()
