#!/usr/bin/env python3
"""
Corpus Diagnostics Script

Audits the entire ingestion pipeline to identify issues:
- arXiv Collection
- PDF Download
- PDF Parsing
- Chunking
- Indexing
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

def load_json(path):
    """Load JSON file if exists."""
    if path.exists():
        return json.load(open(path, 'r', encoding='utf-8'))
    return None

def diagnose():
    """Run full corpus diagnostics."""
    print("=" * 70)
    print("CORPUS DIAGNOSTICS REPORT")
    print("=" * 70)
    
    # 1. Check papers metadata
    print("\n[1] ARXIV COLLECTION")
    print("-" * 50)
    metadata_path = PROCESSED_DIR / "papers_metadata.json"
    metadata = load_json(metadata_path)
    
    if not metadata:
        print("❌ No papers_metadata.json found!")
        return
    
    print(f"✓ Papers in metadata: {len(metadata)}")
    
    # 2. Check PDF downloads
    print("\n[2] PDF DOWNLOAD STATUS")
    print("-" * 50)
    
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"PDFs in data/papers/: {len(pdf_files)}")
    
    papers_with_pdf_path = sum(1 for p in metadata if p.get('pdf_path'))
    print(f"Papers with pdf_path in metadata: {papers_with_pdf_path}")
    
    # Check each paper
    download_status = {"downloaded": [], "missing": [], "failed": []}
    for paper in metadata:
        arxiv_id = paper['arxiv_id']
        pdf_path = paper.get('pdf_path')
        expected_pdf = PAPERS_DIR / f"{arxiv_id}.pdf"
        
        if pdf_path and Path(pdf_path).exists():
            download_status["downloaded"].append(arxiv_id)
        elif expected_pdf.exists():
            download_status["downloaded"].append(arxiv_id)
        else:
            download_status["missing"].append(arxiv_id)
    
    print(f"✓ Downloaded: {len(download_status['downloaded'])}")
    print(f"✗ Missing: {len(download_status['missing'])}")
    
    if download_status['missing']:
        print(f"  First 10 missing: {download_status['missing'][:10]}")
    
    # 3. Check parsed texts
    print("\n[3] PDF PARSING STATUS")
    print("-" * 50)
    
    texts_path = PROCESSED_DIR / "papers_texts.json"
    texts_meta = load_json(texts_path)
    texts_dir = PROCESSED_DIR / "texts"
    
    if texts_meta:
        print(f"Papers in papers_texts.json: {len(texts_meta)}")
        with_full = sum(1 for t in texts_meta if t.get('has_full_text'))
        print(f"Papers marked has_full_text=True: {with_full}")
    else:
        print("❌ No papers_texts.json found")
    
    # Check actual text files
    if texts_dir.exists():
        txt_files = list(texts_dir.glob("*.txt"))
        print(f"Text files in texts/: {len(txt_files)}")
        
        # Analyze text lengths
        char_counts = []
        suspiciously_short = []
        for tf in txt_files:
            content = tf.read_text(encoding='utf-8')
            char_counts.append(len(content))
            if len(content) < 5000:
                suspiciously_short.append((tf.stem, len(content)))
        
        if char_counts:
            avg_chars = sum(char_counts) / len(char_counts)
            print(f"Average text length: {avg_chars:.0f} chars")
            print(f"Min/Max: {min(char_counts)} / {max(char_counts)} chars")
            print(f"Papers with < 5000 chars: {len(suspiciously_short)}")
    else:
        print("❌ texts/ directory does not exist")
    
    # 4. Check chunks
    print("\n[4] CHUNKING STATUS")
    print("-" * 50)
    
    chunks_path = PROCESSED_DIR / "chunks.json"
    chunks = load_json(chunks_path)
    
    if chunks:
        print(f"Total chunks: {len(chunks)}")
        
        # Analyze by section
        by_section = defaultdict(int)
        by_paper = defaultdict(int)
        for c in chunks:
            by_section[c.get('section', 'unknown')] += 1
            by_paper[c['arxiv_id']] += 1
        
        print(f"Chunks by section:")
        for section, count in sorted(by_section.items()):
            print(f"  {section}: {count}")
        
        avg_chunks = sum(by_paper.values()) / len(by_paper) if by_paper else 0
        print(f"Average chunks per paper: {avg_chunks:.1f}")
        print(f"Papers with only 1 chunk: {sum(1 for c in by_paper.values() if c == 1)}")
    else:
        print("❌ No chunks.json found")
    
    # 5. Check index
    print("\n[5] INDEX STATUS")
    print("-" * 50)
    
    chroma_dir = INDEX_DIR / "chroma"
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()
            for col in collections:
                print(f"Collection '{col.name}': {col.count()} vectors")
        except Exception as e:
            print(f"Error reading ChromaDB: {e}")
    else:
        print("❌ ChromaDB index not found")
    
    bm25_path = INDEX_DIR / "bm25_index.pkl"
    print(f"BM25 index exists: {bm25_path.exists()}")
    
    # 6. Summary & Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if len(download_status['missing']) > 0:
        issues.append(f"❌ {len(download_status['missing'])} PDFs not downloaded")
    
    if texts_meta and texts_dir.exists():
        txt_files = list(texts_dir.glob("*.txt"))
        short_count = sum(1 for tf in txt_files if len(tf.read_text(encoding='utf-8')) < 5000)
        if short_count == len(txt_files):
            issues.append("❌ ALL text files are <5000 chars (likely abstracts only)")
    
    if chunks:
        abstract_only = sum(1 for c in chunks if c.get('section') == 'abstract')
        if abstract_only == len(chunks):
            issues.append("❌ ALL chunks are abstracts (no full-text chunks)")
    
    if not issues:
        print("✓ No major issues detected!")
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\nROOT CAUSE:")
        print("  PDFs were never downloaded during corpus building.")
        print("  The pipeline fell back to abstract-only mode.")
        print("  This results in ~1 chunk per paper instead of 10-50+ chunks.")
        
        print("\nRECOMMENDED FIX:")
        print("  1. Download PDFs for all papers")
        print("  2. Parse PDFs to extract full text")
        print("  3. Rechunk the corpus")
        print("  4. Rebuild the index")
        print("\n  Run: python run.py build-corpus --max-papers 100")
        print("  Then: python run.py build-index")


if __name__ == "__main__":
    diagnose()
