#!/usr/bin/env python
"""
Expand the corpus to 500-700 papers while preserving existing papers.

This script:
1. Loads existing paper metadata
2. Searches arXiv for new papers (avoiding duplicates)
3. Downloads new PDFs
4. Parses new PDFs
5. Rechunks the full corpus
6. Rebuilds vector and BM25 indexes
"""

import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import config, get_logger, print_header, print_info, print_success, print_error, PAPERS_DIR, PROCESSED_DIR, INDEX_DIR
from src.corpus.arxiv_collector import ArxivCollector
from src.corpus.pdf_parser import PDFParser

logger = get_logger(__name__)

# Target settings
TARGET_PAPERS = 700
START_DATE = "2024-01-01"
END_DATE = "2026-04-30"


def load_existing_papers():
    """Load existing papers metadata."""
    metadata_file = PROCESSED_DIR / "papers_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            print_info(f"Loaded {len(papers)} existing papers")
            return {p["arxiv_id"]: p for p in papers}
    return {}


def save_papers_metadata(papers: dict):
    """Save merged papers metadata."""
    metadata_file = PROCESSED_DIR / "papers_metadata.json"
    papers_list = list(papers.values())
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(papers_list, f, indent=2, ensure_ascii=False)
    print_success(f"Saved metadata for {len(papers_list)} papers")


def collect_new_papers(existing_ids: set, target_total: int):
    """Search arXiv for new papers not in existing set."""
    collector = ArxivCollector()
    
    # Override config for expansion
    config.corpus.start_date = START_DATE
    config.corpus.end_date = END_DATE
    config.corpus.max_papers = target_total * 2  # Fetch more to filter
    
    new_papers = []
    count = len(existing_ids)
    
    print_info(f"Searching arXiv for papers from {START_DATE} to {END_DATE}")
    print_info(f"Starting from {count} existing papers, targeting {target_total} total")
    
    for paper in collector.search_papers():
        if paper["arxiv_id"] in existing_ids:
            continue
        
        new_papers.append(paper)
        count += 1
        
        if count >= target_total:
            break
        
        if len(new_papers) % 50 == 0:
            print_info(f"Found {len(new_papers)} new papers so far...")
    
    print_success(f"Found {len(new_papers)} new papers to add")
    return new_papers


def download_pdfs(papers: list):
    """Download PDFs for new papers."""
    collector = ArxivCollector()
    downloaded = 0
    
    for paper in tqdm(papers, desc="Downloading PDFs"):
        pdf_path = collector.download_paper(paper)
        paper["pdf_path"] = str(pdf_path) if pdf_path else None
        if pdf_path:
            downloaded += 1
        time.sleep(5)  # Polite delay
    
    print_success(f"Downloaded {downloaded}/{len(papers)} PDFs")
    return papers


def parse_new_pdfs(papers: dict):
    """Parse PDFs for papers that don't have full text yet."""
    parser = PDFParser()
    
    # Load existing parsed texts
    texts_file = PROCESSED_DIR / "papers_texts.json"
    if texts_file.exists():
        with open(texts_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
            existing_texts = {t["arxiv_id"]: t for t in texts}
    else:
        existing_texts = {}
    
    # Parse papers that need parsing
    papers_needing_parsing = []
    for arxiv_id, paper in papers.items():
        if arxiv_id not in existing_texts or not existing_texts[arxiv_id].get("full_text"):
            if paper.get("pdf_path") and Path(paper["pdf_path"]).exists():
                papers_needing_parsing.append(paper)
    
    print_info(f"Parsing {len(papers_needing_parsing)} PDFs...")
    
    for paper in tqdm(papers_needing_parsing, desc="Parsing PDFs"):
        try:
            text = parser.extract_text(Path(paper["pdf_path"]))
            if text:
                existing_texts[paper["arxiv_id"]] = {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "full_text": text,
                    "has_full_text": True
                }
                paper["has_full_text"] = True
        except Exception as e:
            logger.error(f"Failed to parse {paper['arxiv_id']}: {e}")
            # Still add with abstract-only fallback
            if paper["arxiv_id"] not in existing_texts:
                existing_texts[paper["arxiv_id"]] = {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "full_text": "",
                    "has_full_text": False
                }
    
    # Save texts
    texts_list = list(existing_texts.values())
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts_list, f, indent=2, ensure_ascii=False)
    
    print_success(f"Saved parsed texts for {len(texts_list)} papers")
    return existing_texts


def rechunk_corpus(texts: dict):
    """Rechunk all papers using TextChunker."""
    from src.corpus.chunker import TextChunker
    
    chunker = TextChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_chunk_size=config.chunking.min_chunk_size
    )
    
    all_chunks = []
    
    for arxiv_id, paper in tqdm(texts.items(), desc="Chunking papers"):
        # Chunk abstract
        abstract_chunks = chunker.chunk_text(
            text=paper.get("abstract", ""),
            metadata={
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "section": "abstract"
            }
        )
        all_chunks.extend(abstract_chunks)
        
        # Chunk full text if available
        full_text = paper.get("full_text", "")
        if full_text and len(full_text) > 100:
            full_text_chunks = chunker.chunk_text(
                text=full_text,
                metadata={
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "section": "full_text"
                }
            )
            all_chunks.extend(full_text_chunks)
    
    # Save chunks
    chunks_file = PROCESSED_DIR / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
    
    print_success(f"Created {len(all_chunks)} chunks from {len(texts)} papers")
    return all_chunks


def rebuild_indexes(chunks: list):
    """Rebuild vector and BM25 indexes."""
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_index import BM25Index
    
    # Clear old indexes
    chroma_dir = INDEX_DIR / "chroma"
    if chroma_dir.exists():
        print_info("Removing old vector index...")
        shutil.rmtree(chroma_dir)
    
    bm25_file = INDEX_DIR / "bm25_index.pkl"
    if bm25_file.exists():
        print_info("Removing old BM25 index...")
        bm25_file.unlink()
    
    # Build vector index
    print_info("Building vector index...")
    vs = VectorStore()
    
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Vector indexing", total=total_batches):
        batch = chunks[i:i+batch_size]
        try:
            vs.add_chunks(batch)
        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {e}")
            # Retry once
            time.sleep(5)
            try:
                vs.add_chunks(batch)
            except Exception as e2:
                logger.error(f"Batch {i//batch_size} failed again: {e2}")
    
    print_success(f"Indexed {len(chunks)} chunks in vector store")
    
    # Build BM25 index
    print_info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build_index(chunks)
    print_success(f"Built BM25 index with {len(chunks)} documents")


def main():
    """Main expansion workflow."""
    print_header("CORPUS EXPANSION")
    print_info(f"Target: {TARGET_PAPERS} papers from {START_DATE} to {END_DATE}")
    
    # Step 1: Load existing papers
    existing_papers = load_existing_papers()
    existing_ids = set(existing_papers.keys())
    
    if len(existing_papers) >= TARGET_PAPERS:
        print_success(f"Already have {len(existing_papers)} papers (>= target {TARGET_PAPERS})")
        proceed = input("Proceed anyway to reindex? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Step 2: Collect new papers
    new_papers = collect_new_papers(existing_ids, TARGET_PAPERS)
    
    if new_papers:
        # Step 3: Download PDFs
        new_papers = download_pdfs(new_papers)
        
        # Step 4: Merge into existing
        for paper in new_papers:
            existing_papers[paper["arxiv_id"]] = paper
        
        # Save merged metadata
        save_papers_metadata(existing_papers)
    
    # Step 5: Parse all PDFs
    texts = parse_new_pdfs(existing_papers)
    
    # Step 6: Rechunk corpus
    chunks = rechunk_corpus(texts)
    
    # Step 7: Rebuild indexes
    rebuild_indexes(chunks)
    
    # Final stats
    print_header("EXPANSION COMPLETE")
    print_info(f"Total papers: {len(existing_papers)}")
    print_info(f"Total chunks: {len(chunks)}")
    print_info(f"Average chunks per paper: {len(chunks)/len(existing_papers):.1f}")


if __name__ == "__main__":
    main()
