"""
Process newly collected papers: merge metadata, parse PDFs, generate chunks,
rebuild indexes, and verify synchronization.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent))

from src.corpus import TextChunker, Chunk
from src.retrieval import build_bm25_index
from src.retrieval.vector_store import VectorStore
import time

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
PDF_DIR = DATA_DIR / 'papers'
PROGRESS_FILE = DATA_DIR / 'final_expansion_progress.json'

def main():
    print("=" * 70)
    print("CORPUS PROCESSING AND INDEX REBUILD")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load newly collected papers
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📁 Step 1: Loading newly collected papers...")
    
    with open(PROGRESS_FILE, encoding='utf-8') as f:
        progress = json.load(f)
    
    new_papers = progress.get('new_papers', [])
    print(f"  New papers to process: {len(new_papers)}")
    
    if not new_papers:
        print("  ❌ No new papers found in progress file!")
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Merge into metadata
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📝 Step 2: Merging into papers_metadata.json...")
    
    with open(PROCESSED_DIR / 'papers_metadata.json', encoding='utf-8') as f:
        metadata = json.load(f)
    
    existing_ids = set(str(m.get('arxiv_id', '')) for m in metadata)
    added = 0
    
    for paper in new_papers:
        if paper['arxiv_id'] not in existing_ids:
            metadata.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'published': paper.get('published', ''),
                'categories': paper.get('categories', []),
                'authors': paper.get('authors', []),
                'pdf_path': str(PDF_DIR / f"{paper['arxiv_id']}.pdf"),
                'source': 'arxiv_expansion_final',
            })
            added += 1
    
    with open(PROCESSED_DIR / 'papers_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Added {added} new entries to metadata (total: {len(metadata)})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Parse new PDFs
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📄 Step 3: Parsing PDFs...")
    
    # Load existing texts
    texts_file = PROCESSED_DIR / 'papers_texts.json'
    if texts_file.exists():
        with open(texts_file, encoding='utf-8') as f:
            texts = json.load(f)
    else:
        texts = []
    
    existing_text_ids = set(t.get('arxiv_id', '') for t in texts)
    
    parsed = 0
    failed = 0
    
    for paper in new_papers:
        arxiv_id = paper['arxiv_id']
        if arxiv_id in existing_text_ids:
            continue
        
        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"  ⚠️  PDF not found: {arxiv_id}")
            failed += 1
            texts.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'full_text': '',
                'has_full_text': False
            })
            continue
        
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            full_text = '\n'.join(text_parts)
            doc.close()
            
            texts.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'full_text': full_text,
                'has_full_text': len(full_text) > 1000
            })
            parsed += 1
            
            if parsed % 20 == 0:
                print(f"    Parsed {parsed}/{len(new_papers)}...")
                
        except Exception as e:
            print(f"  ⚠️  Error parsing {arxiv_id}: {e}")
            failed += 1
            texts.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'full_text': '',
                'has_full_text': False
            })
    
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    
    print(f"  Parsed: {parsed}, Failed: {failed}")
    print(f"  Total texts: {len(texts)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Generate chunks for ALL papers
    # ═══════════════════════════════════════════════════════════════════════
    print("\n✂️  Step 4: Generating chunks...")
    
    chunker = TextChunker()
    all_chunks = []
    
    for i, paper in enumerate(texts):
        try:
            paper_chunks = chunker.chunk_paper(paper)
            all_chunks.extend(paper_chunks)
        except Exception as e:
            print(f"  ⚠️  Error chunking {paper.get('arxiv_id', '?')}: {e}")
            # Create at least an abstract chunk
            all_chunks.append(Chunk(
                chunk_id=f"{paper.get('arxiv_id', 'unknown')}_abstract",
                arxiv_id=paper.get('arxiv_id', 'unknown'),
                title=paper.get('title', ''),
                text=paper.get('abstract', '')[:2000],
                section='abstract'
            ))
        
        if (i + 1) % 100 == 0:
            print(f"    Chunked {i + 1}/{len(texts)} papers...")
    
    # Convert to dicts
    chunk_dicts = [
        {
            'chunk_id': c.chunk_id,
            'arxiv_id': c.arxiv_id,
            'title': c.title,
            'text': c.text,
            'section': c.section
        } for c in all_chunks
    ]
    
    with open(PROCESSED_DIR / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
    
    unique_papers = len(set(c['arxiv_id'] for c in chunk_dicts))
    print(f"  Total chunks: {len(chunk_dicts)}")
    print(f"  Unique papers: {unique_papers}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Rebuild BM25 index
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🔍 Step 5: Rebuilding BM25 index...")
    
    chunks_for_bm25 = [
        Chunk(
            chunk_id=c['chunk_id'],
            arxiv_id=c['arxiv_id'],
            title=c['title'],
            text=c['text'],
            section=c.get('section', 'abstract')
        ) for c in chunk_dicts
    ]
    
    build_bm25_index(chunks_for_bm25)
    print(f"  BM25 index rebuilt with {len(chunks_for_bm25)} documents")
    
    # Also rebuild BM25 metadata with text field
    import pickle
    bm25_meta = []
    for c in chunk_dicts:
        bm25_meta.append({
            'chunk_id': c['chunk_id'],
            'arxiv_id': c['arxiv_id'],
            'title': c['title'],
            'section': c.get('section', 'abstract'),
            'text': c['text']
        })
    
    with open(DATA_DIR / 'index' / 'bm25_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(bm25_meta, f, indent=2, ensure_ascii=False)
    
    print(f"  BM25 metadata rebuilt with {len(bm25_meta)} entries (including text field)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Rebuild ChromaDB vector index
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🧠 Step 6: Rebuilding ChromaDB vector index...")
    
    vector_store = VectorStore()
    
    # Use smaller batch size to avoid timeouts
    batch_size = 50
    max_retries = 3
    
    for i in range(0, len(chunks_for_bm25), batch_size):
        batch = chunks_for_bm25[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks_for_bm25) + batch_size - 1) // batch_size
        
        for attempt in range(max_retries):
            try:
                print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                vector_store.add_chunks(batch, batch_size=len(batch))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"    ❌ Failed batch {batch_num}: {e}")
                    raise
    
    print(f"  ChromaDB index rebuilt with {len(chunks_for_bm25)} vectors")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Verification
    # ═══════════════════════════════════════════════════════════════════════
    print("\n✅ Step 7: Verifying index synchronization...")
    
    # Reload and verify
    with open(PROCESSED_DIR / 'chunks.json', encoding='utf-8') as f:
        chunks_verify = json.load(f)
    
    with open(DATA_DIR / 'index' / 'bm25_metadata.json', encoding='utf-8') as f:
        bm25_verify = json.load(f)
    
    with open(DATA_DIR / 'index' / 'bm25_index.pkl', 'rb') as f:
        bm25_pkl = pickle.load(f)
    
    # ChromaDB count
    import chromadb
    from chromadb.config import Settings
    chroma_client = chromadb.PersistentClient(
        path=str(DATA_DIR / 'index' / 'chroma'),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection('arxiv_papers')
    chroma_count = collection.count()
    
    chunks_count = len(chunks_verify)
    chunks_papers = len(set(c['arxiv_id'] for c in chunks_verify))
    bm25_meta_count = len(bm25_verify)
    bm25_meta_papers = len(set(m['arxiv_id'] for m in bm25_verify))
    bm25_pkl_count = len(bm25_pkl.get('tokenized_corpus', []))
    
    print(f"\n  {'Component':<25} {'Count':<10} {'Papers':<10}")
    print(f"  {'-' * 45}")
    print(f"  {'chunks.json':<25} {chunks_count:<10} {chunks_papers:<10}")
    print(f"  {'bm25_metadata.json':<25} {bm25_meta_count:<10} {bm25_meta_papers:<10}")
    print(f"  {'bm25_index.pkl':<25} {bm25_pkl_count:<10} {'—':<10}")
    print(f"  {'ChromaDB vectors':<25} {chroma_count:<10} {'—':<10}")
    
    # Check sync
    all_synced = (
        chunks_count == bm25_meta_count == bm25_pkl_count == chroma_count
    )
    
    if all_synced:
        print(f"\n  ✅ ALL INDEXES SYNCHRONIZED!")
    else:
        print(f"\n  ⚠️  INDEX MISMATCH DETECTED!")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION REPORT")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'new_papers_added': len(new_papers),
        'total_arxiv_papers': chunks_papers,
        'total_chunks': chunks_count,
        'bm25_documents': bm25_pkl_count,
        'chromadb_vectors': chroma_count,
        'failed_downloads': len(progress.get('failed_downloads', [])),
        'failed_parses': failed,
        'duplicates_skipped': progress.get('skipped_duplicates', 0),
        'indexes_synchronized': all_synced,
        'corpus_status': 'FINAL',
    }
    
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    # Save report
    with open(DATA_DIR / 'final_corpus_verification.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to: data/final_corpus_verification.json")
    print("\n" + "=" * 70)
    print("CORPUS COLLECTION MARKED AS FINAL")
    print("Project status: Evaluation and Report phase")
    print("=" * 70)

if __name__ == '__main__':
    main()
