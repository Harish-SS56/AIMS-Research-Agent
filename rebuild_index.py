"""Rebuild vector and BM25 indexes from chunks."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.corpus import Chunk
from src.retrieval import build_bm25_index
from src.retrieval.vector_store import VectorStore
from src.utils import print_info, print_success

PROCESSED_DIR = Path('data/processed')

def main():
    # Load chunks
    chunk_dicts = json.load(open(PROCESSED_DIR / 'chunks.json', encoding='utf-8'))
    print(f'Loaded {len(chunk_dicts)} chunks')
    
    # Convert to Chunk objects
    chunks = [
        Chunk(
            chunk_id=c['chunk_id'],
            arxiv_id=c['arxiv_id'],
            title=c['title'],
            text=c['text'],
            section=c.get('section', 'abstract')
        ) for c in chunk_dicts
    ]
    
    # Build vector index with smaller batches and retry logic
    print('\nBuilding vector index...')
    vector_store = VectorStore()
    
    # Use smaller batch size to avoid timeouts
    batch_size = 50
    max_retries = 3
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for attempt in range(max_retries):
            try:
                print(f'  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...')
                vector_store.add_chunks(batch, batch_size=len(batch))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'    Retry {attempt + 1}/{max_retries} after error: {e}')
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    raise
    
    print_success(f'Indexed {len(chunks)} chunks in vector store')
    
    # Build BM25 index
    print('\nBuilding BM25 index...')
    build_bm25_index(chunks)
    
    print('\nDone! Indexes rebuilt with full corpus.')

if __name__ == '__main__':
    main()
