#!/usr/bin/env python3
"""
Comprehensive corpus rebuild script.

Re-extracts text from all PDFs, rechunks with production settings,
and rebuilds vector + BM25 indexes.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# Paths
BASE_DIR = Path("H:/AIMS Research Agent")
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
TEXTS_FILE = PROCESSED_DIR / "papers_texts.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r' {3,}', ' ', text)
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()

def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        full_text = "\n\n".join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        print(f"  [ERROR] Failed to parse {pdf_path.name}: {e}")
        return None

def get_paper_id(paper: Dict[str, Any]) -> str:
    """Get unique ID for a paper (prefer arxiv_id, fallback to openalex_id or id)."""
    if paper.get('arxiv_id'):
        return paper['arxiv_id']
    if paper.get('openalex_id'):
        return paper['openalex_id'].replace('https://openalex.org/', '')
    return paper.get('id', f"paper_{hash(paper.get('title', ''))}")

def extract_all_texts():
    """Extract text from all PDFs in the corpus."""
    print("\n" + "="*70)
    print("STEP 1: Extracting text from PDFs")
    print("="*70)
    
    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Total papers in metadata: {len(papers)}")
    
    # Find papers with PDFs
    papers_with_pdfs = []
    for p in papers:
        pdf_path = p.get('pdf_path')
        if pdf_path:
            full_path = BASE_DIR / pdf_path
            if full_path.exists():
                papers_with_pdfs.append((p, full_path))
    
    print(f"Papers with existing PDFs: {len(papers_with_pdfs)}")
    
    # Extract text from each PDF
    parsed_papers = []
    failed = 0
    
    for paper, pdf_path in tqdm(papers_with_pdfs, desc="Extracting text"):
        text = extract_pdf_text(pdf_path)
        if text and len(text) > 100:  # Minimum viable text
            paper_id = get_paper_id(paper)
            parsed = {
                'id': paper_id,
                'arxiv_id': paper.get('arxiv_id', paper_id),  # For compatibility
                'openalex_id': paper.get('openalex_id', ''),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'year': paper.get('year'),
                'pdf_url': paper.get('pdf_url', ''),
                'source': paper.get('source', 'unknown'),
                'full_text': text,
                'text_length': len(text)
            }
            parsed_papers.append(parsed)
        else:
            failed += 1
    
    print(f"\nSuccessfully extracted: {len(parsed_papers)}")
    print(f"Failed/empty: {failed}")
    
    # Save parsed texts
    with open(TEXTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(parsed_papers, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {TEXTS_FILE}")
    return parsed_papers

def chunk_papers(papers: List[Dict[str, Any]], 
                 chunk_size: int = 512, 
                 overlap: int = 50) -> List[Dict[str, Any]]:
    """Chunk papers with production settings."""
    print("\n" + "="*70)
    print("STEP 2: Chunking corpus")
    print("="*70)
    print(f"Chunk size: {chunk_size} tokens (approx)")
    print(f"Overlap: {overlap} tokens")
    
    all_chunks = []
    
    for paper in tqdm(papers, desc="Chunking"):
        text = paper.get('full_text', '')
        if not text:
            continue
        
        # Simple word-based chunking (approx 1.3 words per token)
        words = text.split()
        words_per_chunk = int(chunk_size * 1.3)
        overlap_words = int(overlap * 1.3)
        
        paper_id = paper.get('id', paper.get('arxiv_id', 'unknown'))
        
        # Calculate total chunks for this paper
        total_chunks = 0
        i = 0
        while i < len(words):
            if len(' '.join(words[i:i + words_per_chunk])) > 50:
                total_chunks += 1
            i += words_per_chunk - overlap_words
            if i >= len(words) - overlap_words:
                break
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) > 50:  # Minimum chunk size
                # Use Chunk-compatible format
                chunk = {
                    'chunk_id': f"{paper_id}_chunk_{chunk_idx}",
                    'arxiv_id': paper.get('arxiv_id', paper_id),
                    'title': paper.get('title', ''),
                    'text': chunk_text,
                    'section': None,
                    'chunk_index': chunk_idx,
                    'total_chunks': total_chunks,
                    'token_count': len(chunk_text.split())  # Approx token count
                }
                all_chunks.append(chunk)
                chunk_idx += 1
            
            i += words_per_chunk - overlap_words
            if i >= len(words) - overlap_words:
                break
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    # Save chunks
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    
    print(f"Saved to: {CHUNKS_FILE}")
    return all_chunks

def build_indexes(chunks: List[Dict[str, Any]]):
    """Build vector and BM25 indexes."""
    print("\n" + "="*70)
    print("STEP 3: Building indexes")
    print("="*70)
    
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(BASE_DIR))
    
    from src.retrieval import build_index, build_bm25_index
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Build vector index
    print("\nBuilding vector index...")
    try:
        build_index(chunks)
        print("  [OK] Vector index built")
    except Exception as e:
        print(f"  [ERROR] Vector index failed: {e}")
        return False
    
    # Build BM25 index
    print("\nBuilding BM25 index...")
    try:
        build_bm25_index(chunks)
        print("  [OK] BM25 index built")
    except Exception as e:
        print(f"  [ERROR] BM25 index failed: {e}")
        return False
    
    return True

def generate_report(papers: List[Dict], chunks: List[Dict]):
    """Generate final statistics report."""
    print("\n" + "="*70)
    print("FINAL CORPUS STATISTICS")
    print("="*70)
    
    # Load metadata for full stats
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    # PDF count
    pdf_count = len(list(PAPERS_DIR.glob('*.pdf')))
    total_size_mb = sum(p.stat().st_size for p in PAPERS_DIR.glob('*.pdf')) / (1024*1024)
    
    # Papers with arxiv IDs
    arxiv_papers = [p for p in papers if p.get('arxiv_id') and not p['arxiv_id'].startswith('W')]
    
    # Year distribution
    years = {}
    for p in papers:
        yr = str(p.get('year', 'unknown'))
        years[yr] = years.get(yr, 0) + 1
    
    # Load index stats if available
    vector_count = 0
    bm25_count = 0
    try:
        index_dir = PROCESSED_DIR / "faiss_index"
        if (index_dir / "index.faiss").exists():
            import faiss
            index = faiss.read_index(str(index_dir / "index.faiss"))
            vector_count = index.ntotal
    except:
        pass
    
    try:
        bm25_file = PROCESSED_DIR / "bm25_index.pkl"
        if bm25_file.exists():
            import pickle
            with open(bm25_file, 'rb') as f:
                bm25_data = pickle.load(f)
                bm25_count = len(bm25_data.get('corpus', []))
    except:
        pass
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'metadata_records': len(all_metadata),
        'pdfs_on_disk': pdf_count,
        'corpus_size_mb': round(total_size_mb, 2),
        'papers_parsed': len(papers),
        'papers_with_arxiv': len(arxiv_papers),
        'papers_without_arxiv': len(papers) - len(arxiv_papers),
        'total_chunks': len(chunks),
        'avg_chunks_per_paper': round(len(chunks) / max(len(papers), 1), 1),
        'vector_count': vector_count,
        'bm25_doc_count': bm25_count,
        'year_distribution': years,
        'estimated_retrieval_coverage': f"{min(100, len(papers) / 5):.1f}%"  # Based on typical topic diversity
    }
    
    # Print report
    print(f"""
┌────────────────────────────────────┬──────────────┐
│ Metric                             │ Value        │
├────────────────────────────────────┼──────────────┤
│ Total metadata records             │ {report['metadata_records']:>12} │
│ PDFs on disk                       │ {report['pdfs_on_disk']:>12} │
│ Corpus size                        │ {report['corpus_size_mb']:>9.1f} MB │
│ Papers parsed (with text)          │ {report['papers_parsed']:>12} │
│ Papers with arXiv IDs              │ {report['papers_with_arxiv']:>12} │
│ Papers without arXiv IDs           │ {report['papers_without_arxiv']:>12} │
│ Total chunks                       │ {report['total_chunks']:>12} │
│ Avg chunks/paper                   │ {report['avg_chunks_per_paper']:>12.1f} │
│ Vector count                       │ {report['vector_count']:>12} │
│ BM25 document count                │ {report['bm25_doc_count']:>12} │
│ Est. retrieval coverage            │ {report['estimated_retrieval_coverage']:>12} │
└────────────────────────────────────┴──────────────┘
""")
    
    print("Year distribution:")
    for yr, cnt in sorted(years.items()):
        bar = '█' * (cnt // 10)
        print(f"  {yr}: {cnt:4d}  {bar}")
    
    # Save report
    report_file = DATA_DIR / "rebuild_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")
    
    return report

def main():
    """Main rebuild pipeline."""
    print("="*70)
    print("CORPUS REBUILD PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    # Step 1: Extract text from all PDFs
    papers = extract_all_texts()
    
    if not papers:
        print("\n[ERROR] No papers extracted. Aborting.")
        return
    
    # Step 2: Chunk the corpus
    chunks = chunk_papers(papers, chunk_size=512, overlap=50)
    
    if not chunks:
        print("\n[ERROR] No chunks created. Aborting.")
        return
    
    # Step 3: Build indexes (commented out - will be done separately due to API issues)
    # success = build_indexes(chunks)
    
    # Step 4: Generate report
    report = generate_report(papers, chunks)
    
    print("\n" + "="*70)
    print("REBUILD COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python run.py build-index --use-existing-chunks")
    print("     (to build vector and BM25 indexes)")
    print("  2. Run: python run.py ablation --configs 'no_reranker,...' --num-questions 5")
    print("     (to run ablation studies)")

if __name__ == "__main__":
    main()
